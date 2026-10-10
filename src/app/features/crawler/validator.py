"""URL validation and SSRF protection for crawler."""

import asyncio
import ipaddress
import re
import socket
from urllib.parse import urlparse, urlunparse

MAX_URL_LENGTH = 4_096

PRIVATE_IP_RANGES: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),  # Carrier-grade NAT
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),  # Documentation
    ipaddress.ip_network("198.51.100.0/24"),  # Documentation
    ipaddress.ip_network("203.0.113.0/24"),  # Documentation
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("ff00::/8"),  # IPv6 multicast
]

BLOCKED_DOMAINS: set[str] = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.google",
    "169.254.169.254",  # Cloud metadata
    "metadata.aws.internal",
    "kubernetes.default.svc",
}

BLOCKED_HOSTNAME_PATTERNS: list[str] = [
    r".*\.local$",
    r".*\.localhost$",
    r".*\.internal$",
    r".*\.corp$",
    r".*\.lan$",
    r"^localhost$",
]


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private."""
    try:
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(ip_str)
        for network in PRIVATE_IP_RANGES:
            if ip in network:
                return True
    except ValueError:
        return False
    return False


def validate_url(url: str) -> tuple[bool, str]:  # noqa: PLR0912
    """
    Validate URL for security (SSRF protection).

    Returns:
        tuple: (is_valid, error_message)
    """
    if not url:
        return False, "URL cannot be empty"
    if len(url) > MAX_URL_LENGTH:
        return False, "URL is too long"
    if any(ord(character) < 0x20 for character in url):
        return False, "URL contains control characters"
    if "\\" in url:
        return False, "URL contains an invalid path separator"

    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "Invalid URL format"

    if not parsed.scheme:
        return False, "URL must include a scheme (http:// or https://)"

    if parsed.scheme.lower() not in {"http", "https"}:
        return False, "Only http and https schemes are allowed"

    if not parsed.netloc:
        return False, "URL must include a domain"

    try:
        hostname = parsed.hostname
    except ValueError:
        return False, "Invalid hostname"
    if not hostname:
        return False, "Invalid hostname"

    try:
        hostname = hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError:
        return False, "Invalid internationalized hostname"
    if parsed.username is not None or parsed.password is not None:
        return False, "URLs with embedded credentials are not allowed"
    if hostname in BLOCKED_DOMAINS:
        return False, f"Domain '{hostname}' is not allowed"

    for pattern in BLOCKED_HOSTNAME_PATTERNS:
        if re.match(pattern, hostname):
            return False, f"Hostname '{hostname}' is not allowed"

    if hostname is not None:
        ip_candidate = hostname
        if ip_candidate.isdigit():
            try:
                ip_candidate = str(ipaddress.ip_address(int(ip_candidate)))
            except ValueError:
                return False, "Invalid numeric hostname"
        if is_private_ip(ip_candidate):
            return False, f"Private IP addresses are not allowed: {hostname}"

    try:
        port = parsed.port
    except ValueError:
        return False, "Invalid port"
    if port is not None and port not in {80, 443}:
        return False, f"Port {port} is not allowed for security reasons"

    return True, ""


async def validate_url_for_fetch(url: str) -> tuple[bool, str]:
    """Validate a URL and all addresses returned by DNS before navigation."""
    return await validate_navigation_destination(url, phase="pre-navigation")


async def validate_navigation_destination(url: str, *, phase: str) -> tuple[bool, str]:
    """Enforce the browser network policy before and after navigation.

    Pre-navigation DNS validation cannot close DNS rebinding on its own: the
    browser resolves the hostname again at connection time. Callers must
    therefore invoke this helper for the seed URL *and* for the final
    browser-reported URL (including any ``redirected_url``). Post-navigation
    failures are treated as denials, not as retryable crawl errors.
    """
    valid, message = validate_url(url)
    if not valid:
        return valid, message

    parsed = urlparse(sanitize_url(url))
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, f"Browser navigation blocked during {phase}: unsupported scheme"
    hostname = parsed.hostname
    if hostname is None:
        return False, "Invalid hostname"

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except (OSError, ValueError):
        return False, "Unable to resolve hostname"

    for address in addresses:
        ip_text = str(address[4][0])
        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError:
            return False, "Hostname resolved to an invalid address"
        is_non_public = is_private_ip(ip_text) or any(
            (
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            )
        )
        if is_non_public:
            return False, f"Hostname resolves to a private address: {ip_text}"

    return True, ""


def validate_browser_result_urls(
    result: object,
) -> tuple[bool, str]:
    """Validate every URL the browser reports after navigation.

    Crawl4AI exposes the final navigation target as ``result.url`` and, when a
    redirect occurred, as ``result.redirected_url``. Both must satisfy the
    network policy synchronously (scheme + string policy); DNS revalidation
    happens in the async caller via :func:`validate_navigation_destination`.
    """
    candidates: list[str] = []
    for attr in ("url", "redirected_url"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            candidates.append(value)
    for candidate in candidates:
        valid, message = validate_url(candidate)
        if not valid:
            return False, f"Browser navigation destination is not allowed: {message}"
        try:
            scheme = urlparse(sanitize_url(candidate)).scheme.lower()
        except ValueError:
            return False, "Browser navigation destination is not allowed: invalid URL"
        if scheme not in {"http", "https"}:
            return False, "Browser navigation destination is not allowed: unsupported scheme"
    return True, ""


def sanitize_url(url: str) -> str:
    """Sanitize and normalize a URL."""
    url = url.strip()

    try:
        has_http_scheme = urlparse(url).scheme.lower() in {"http", "https"}
    except ValueError:
        return url
    if not has_http_scheme:
        url = f"https://{url}"
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        fragment="",
    )
    return urlunparse(normalized)


def get_domain_from_url(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(url)
    return parsed.hostname or ""


def is_valid_url(url: str) -> bool:
    """Quick check if URL is valid and allowed."""
    is_valid, _ = validate_url(url)
    return is_valid
