"""URL validation and SSRF protection for crawler."""

import ipaddress
import re
from urllib.parse import urlparse

PRIVATE_IP_RANGES = [
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

BLOCKED_DOMAINS = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.google",
    "169.254.169.254",  # Cloud metadata
    "metadata.aws.internal",
    "kubernetes.default.svc",
}

BLOCKED_HOSTNAME_PATTERNS = [
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
        ip = ipaddress.ip_address(ip_str)
        for network in PRIVATE_IP_RANGES:
            if ip in network:
                return True
    except ValueError:
        return False
    return False


def validate_url(url: str) -> tuple[bool, str]:
    """
    Validate URL for security (SSRF protection).

    Returns:
        tuple: (is_valid, error_message)
    """
    if not url:
        return False, "URL cannot be empty"

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"

    if not parsed.scheme:
        return False, "URL must include a scheme (http:// or https://)"

    if parsed.scheme not in ("http", "https"):
        return False, "Only http and https schemes are allowed"

    if not parsed.netloc:
        return False, "URL must include a domain"

    hostname = parsed.hostname.lower()
    if not hostname:
        return False, "Invalid hostname"

    if hostname in BLOCKED_DOMAINS:
        return False, f"Domain '{hostname}' is not allowed"

    for pattern in BLOCKED_HOSTNAME_PATTERNS:
        if re.match(pattern, hostname):
            return False, f"Hostname '{hostname}' is not allowed"

    try:
        ip_str = parsed.hostname
        if is_private_ip(ip_str):
            return False, f"Private IP addresses are not allowed: {ip_str}"
    except Exception:
        pass

    port = parsed.port
    if port and port in (22, 23, 25, 3306, 5432, 6379, 27017, 11211):
        return False, f"Port {port} is not allowed for security reasons"

    return True, ""


def sanitize_url(url: str) -> str:
    """Sanitize and normalize a URL."""
    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    return url


def get_domain_from_url(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(url)
    return parsed.hostname or ""


def is_valid_url(url: str) -> bool:
    """Quick check if URL is valid and allowed."""
    is_valid, _ = validate_url(url)
    return is_valid
