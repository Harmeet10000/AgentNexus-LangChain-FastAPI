# API Reference: Server Auth

Source lines: 34658-38288 from the original FastMCP documentation dump.

Package-level API reference for server auth, auth providers, settings, telemetry, redirect validation, and SSRF helpers.

---

# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-__init__



# `fastmcp.server`

*This module is empty or contains only private/internal implementations.*


# apps
Source: https://gofastmcp.com/python-sdk/fastmcp-server-apps



# `fastmcp.server.apps`

MCP Apps support — extension negotiation and typed UI metadata models.

Provides constants and Pydantic models for the MCP Apps extension
(io.modelcontextprotocol/ui), enabling tools and resources to carry
UI metadata for clients that support interactive app rendering.

## Functions

### `app_config_to_meta_dict` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/apps.py#L115"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
app_config_to_meta_dict(app: AppConfig | dict[str, Any]) -> dict[str, Any]
```

Convert an AppConfig or dict to the wire-format dict for `meta["ui"]`.

### `resolve_ui_mime_type` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/apps.py#L122"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resolve_ui_mime_type(uri: str, explicit_mime_type: str | None) -> str | None
```

Return the appropriate MIME type for a resource URI.

For `ui://` scheme resources, defaults to `UI_MIME_TYPE` when no
explicit MIME type is provided. This ensures UI resources are correctly
identified regardless of how they're registered (via FastMCP.resource,
the standalone @resource decorator, or resource templates).

**Args:**

* `uri`: The resource URI string
* `explicit_mime_type`: The MIME type explicitly provided by the user

**Returns:**

* The resolved MIME type (explicit value, UI default, or None)

## Classes

### `ResourceCSP` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/apps.py#L18"><Icon icon="github" /></a></sup>

Content Security Policy for MCP App resources.

Declares which external origins the app is allowed to connect to or
load resources from.  Hosts use these declarations to build the
`Content-Security-Policy` header for the sandboxed iframe.

### `ResourcePermissions` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/apps.py#L50"><Icon icon="github" /></a></sup>

Iframe sandbox permissions for MCP App resources.

Each field, when set (typically to `{}`), requests that the host
grant the corresponding Permission Policy feature to the sandboxed
iframe.  Hosts MAY honour these; apps should use JS feature detection
as a fallback.

### `AppConfig` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/apps.py#L77"><Icon icon="github" /></a></sup>

Configuration for MCP App tools and resources.

Controls how a tool or resource participates in the MCP Apps extension.
On tools, `resource_uri` and `visibility` specify which UI resource
to render and where the tool appears.  On resources, those fields must
be left unset (the resource itself is the UI).

All fields use `exclude_none` serialization so only explicitly-set
values appear on the wire.  Aliases match the MCP Apps wire format
(camelCase).


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-__init__



# `fastmcp.server.auth`

*This module is empty or contains only private/internal implementations.*


# auth
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-auth



# `fastmcp.server.auth.auth`

## Classes

### `AccessToken` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L54"><Icon icon="github" /></a></sup>

AccessToken that includes all JWT claims.

### `TokenHandler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L60"><Icon icon="github" /></a></sup>

TokenHandler that returns MCP-compliant error responses.

This handler addresses two SDK issues:

1. Error code: The SDK returns `unauthorized_client` for client authentication
   failures, but RFC 6749 Section 5.2 requires `invalid_client` with HTTP 401.
   This distinction matters for client re-registration behavior.

2. Status code: The SDK returns HTTP 400 for all token errors including
   `invalid_grant` (expired/invalid tokens). However, the MCP spec requires:
   "Invalid or expired tokens MUST receive a HTTP 401 response."

This handler transforms responses to be compliant with both OAuth 2.1 and MCP specs.

**Methods:**

#### `handle` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L76"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
handle(self, request: Any)
```

Wrap SDK handle() and transform auth error responses.

### `PrivateKeyJWTClientAuthenticator` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L126"><Icon icon="github" /></a></sup>

Client authenticator with private\_key\_jwt support for CIMD clients.

Extends the SDK's ClientAuthenticator to add support for the `private_key_jwt`
authentication method per RFC 7523. This is required for CIMD (Client ID Metadata
Document) clients that use asymmetric keys for authentication.

The authenticator:

1. Delegates to SDK for standard methods (client\_secret\_basic, client\_secret\_post, none)
2. Adds private\_key\_jwt handling for CIMD clients
3. Validates JWT assertions against client's JWKS

**Methods:**

#### `authenticate_request` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L156"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
authenticate_request(self, request: Request) -> OAuthClientInformationFull
```

Authenticate a client from an HTTP request.

Extends SDK authentication to support private\_key\_jwt for CIMD clients.
Delegates to SDK for client\_secret\_basic (Authorization header) and
client\_secret\_post (form body) authentication.

### `AuthProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L207"><Icon icon="github" /></a></sup>

Base class for all FastMCP authentication providers.

This class provides a unified interface for all authentication providers,
whether they are simple token verifiers or full OAuth authorization servers.
All providers must be able to verify tokens and can optionally provide
custom authentication routes.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L236"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify a bearer token and return access info if valid.

All auth providers must implement token verification.

**Args:**

* `token`: The token string to validate

**Returns:**

* AccessToken object if valid, None if invalid or expired

#### `set_mcp_path` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L249"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_mcp_path(self, mcp_path: str | None) -> None
```

Set the MCP endpoint path and compute resource URL.

This method is called by get\_routes() to configure the expected
resource URL before route creation. Subclasses can override to
perform additional initialization that depends on knowing the
MCP endpoint path.

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L263"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get all routes for this authentication provider.

This includes both well-known discovery routes and operational routes.
Each provider is responsible for creating whatever routes it needs:

* TokenVerifier: typically no routes (default implementation)
* RemoteAuthProvider: protected resource metadata routes
* OAuthProvider: full OAuth authorization server routes
* Custom providers: whatever routes they need

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")
  This is used to advertise the resource URL in metadata, but the
  provider does not create the actual MCP endpoint route.

**Returns:**

* List of all routes for this provider (excluding the MCP endpoint itself)

#### `get_well_known_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L286"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_well_known_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get well-known discovery routes for this authentication provider.

This is a utility method that filters get\_routes() to return only
well-known discovery routes (those starting with /.well-known/).

Well-known routes provide OAuth metadata and discovery endpoints that
clients use to discover authentication capabilities. These routes should
be mounted at the root level of the application to comply with RFC 8414
and RFC 9728.

Common well-known routes:

* /.well-known/oauth-authorization-server (authorization server metadata)
* /.well-known/oauth-protected-resource/\* (protected resource metadata)

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")
  This is used to construct path-scoped well-known URLs.

**Returns:**

* List of well-known discovery routes (typically mounted at root level)

#### `get_middleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L318"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_middleware(self) -> list
```

Get HTTP application-level middleware for this auth provider.

**Returns:**

* List of Starlette Middleware instances to apply to the HTTP app

### `TokenVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L352"><Icon icon="github" /></a></sup>

Base class for token verifiers (Resource Servers).

This class provides token verification capability without OAuth server functionality.
Token verifiers typically don't provide authentication routes by default.

**Methods:**

#### `scopes_supported` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L374"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
scopes_supported(self) -> list[str]
```

Scopes to advertise in OAuth metadata.

Defaults to required\_scopes. Override in subclasses when the
advertised scopes differ from the validation scopes (e.g., Azure AD
where tokens contain short-form scopes but clients request full URI
scopes).

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L384"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify a bearer token and return access info if valid.

### `RemoteAuthProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L389"><Icon icon="github" /></a></sup>

Authentication provider for resource servers that verify tokens from known authorization servers.

This provider composes a TokenVerifier with authorization server metadata to create
standardized OAuth 2.0 Protected Resource endpoints (RFC 9728). Perfect for:

* JWT verification with known issuers
* Remote token introspection services
* Any resource server that knows where its tokens come from

Use this when you have token verification logic and want to advertise
the authorization servers that issue valid tokens.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L436"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify token using the configured token verifier.

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L440"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get routes for this provider.

Creates protected resource metadata routes (RFC 9728).

### `MultiAuth` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L472"><Icon icon="github" /></a></sup>

Composes an optional auth server with additional token verifiers.

Use this when a single server needs to accept tokens from multiple sources.
For example, an OAuth proxy for interactive clients combined with a JWT
verifier for machine-to-machine tokens.

Token verification tries the server first (if present), then each verifier
in order, returning the first successful result. Routes and OAuth metadata
come from the server; verifiers contribute only token verification.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L537"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify a token by trying the server, then each verifier in order.

Each source is tried independently. If a source raises an exception,
it is logged and treated as a non-match so that remaining sources
still get a chance to verify the token.

#### `set_mcp_path` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L558"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_mcp_path(self, mcp_path: str | None) -> None
```

Propagate MCP path to the server and all verifiers.

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L566"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Delegate route creation to the server.

#### `get_well_known_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L572"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_well_known_routes(self, mcp_path: str | None = None) -> list[Route]
```

Delegate well-known route creation to the server.

This ensures that server-specific well-known route logic (e.g.,
OAuthProvider's RFC 8414 path-aware discovery) is preserved.

### `OAuthProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L583"><Icon icon="github" /></a></sup>

OAuth Authorization Server provider.

This class provides full OAuth server functionality including client registration,
authorization flows, token issuance, and token verification.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L646"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify a bearer token and return access info if valid.

This method implements the TokenVerifier protocol by delegating
to our existing load\_access\_token method.

**Args:**

* `token`: The token string to validate

**Returns:**

* AccessToken object if valid, None if invalid or expired

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L661"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get OAuth authorization server routes and optional protected resource routes.

This method creates the full set of OAuth routes including:

* Standard OAuth authorization server routes (/.well-known/oauth-authorization-server, /authorize, /token, etc.)
* Optional protected resource routes

**Returns:**

* List of OAuth routes

#### `get_well_known_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/auth.py#L740"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_well_known_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get well-known discovery routes with RFC 8414 path-aware support.

Overrides the base implementation to support path-aware authorization
server metadata discovery per RFC 8414. If issuer\_url has a path component,
the authorization server metadata route is adjusted to include that path.

For example, if issuer\_url is "[http://example.com/api](http://example.com/api)", the discovery
endpoint will be at "/.well-known/oauth-authorization-server/api" instead
of just "/.well-known/oauth-authorization-server".

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")

**Returns:**

* List of well-known discovery routes


# authorization
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-authorization



# `fastmcp.server.auth.authorization`

Authorization checks for FastMCP components.

This module provides callable-based authorization for tools, resources, and prompts.
Auth checks are functions that receive an AuthContext and return True to allow access
or False to deny.

Auth checks can also raise exceptions:

* AuthorizationError: Propagates with the custom message for explicit denial
* Other exceptions: Masked for security (logged, treated as auth failure)

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth import require_scopes

mcp = FastMCP()

@mcp.tool(auth=require_scopes("write"))
def protected_tool(): ...

@mcp.resource("data://secret", auth=require_scopes("read"))
def secret_data(): ...

@mcp.prompt(auth=require_scopes("admin"))
def admin_prompt(): ...
```

## Functions

### `require_scopes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/authorization.py#L78"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
require_scopes(*scopes: str) -> AuthCheck
```

Require specific OAuth scopes.

Returns an auth check that requires ALL specified scopes to be present
in the token (AND logic).

**Args:**

* `*scopes`: One or more scope strings that must all be present.

### `restrict_tag` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/authorization.py#L106"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
restrict_tag(tag: str) -> AuthCheck
```

Restrict components with a specific tag to require certain scopes.

If the component has the specified tag, the token must have ALL the
required scopes. If the component doesn't have the tag, access is allowed.

**Args:**

* `tag`: The tag that triggers the scope requirement.
* `scopes`: List of scopes required when the tag is present.

### `run_auth_checks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/authorization.py#L134"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_auth_checks(checks: AuthCheck | list[AuthCheck], ctx: AuthContext) -> bool
```

Run auth checks with AND logic.

All checks must pass for authorization to succeed. Checks can be
synchronous or asynchronous functions.

Auth checks can:

* Return True to allow access
* Return False to deny access
* Raise AuthorizationError to deny with a custom message (propagates)
* Raise other exceptions (masked for security, treated as denial)

**Args:**

* `checks`: A single check function or list of check functions.
  Each check can be sync (returns bool) or async (returns Awaitable\[bool]).
* `ctx`: The auth context to pass to each check.

**Returns:**

* True if all checks pass, False if any check fails.

**Raises:**

* `AuthorizationError`: If an auth check explicitly raises it.

## Classes

### `AuthContext` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/authorization.py#L48"><Icon icon="github" /></a></sup>

Context passed to auth check callables.

This object is passed to each auth check function and provides
access to the current authentication token and the component being accessed.

**Attributes:**

* `token`: The current access token, or None if unauthenticated.
* `component`: The component (tool, resource, or prompt) being accessed.
* `tool`: Backwards-compatible alias for component when it's a Tool.

**Methods:**

#### `tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/authorization.py#L64"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool(self) -> Tool | None
```

Backwards-compatible access to the component as a Tool.

Returns the component if it's a Tool, None otherwise.


# cimd
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-cimd



# `fastmcp.server.auth.cimd`

CIMD (Client ID Metadata Document) support for FastMCP.

.. warning::
**Beta Feature**: CIMD support is currently in beta. The API may change
in future releases. Please report any issues you encounter.

CIMD is a simpler alternative to Dynamic Client Registration where clients
host a static JSON document at an HTTPS URL, and that URL becomes their
client\_id. See the IETF draft: draft-parecki-oauth-client-id-metadata-document

This module provides:

* CIMDDocument: Pydantic model for CIMD document validation
* CIMDFetcher: Fetch and validate CIMD documents with SSRF protection
* CIMDClientManager: Manages CIMD client operations

## Classes

### `CIMDDocument` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L45"><Icon icon="github" /></a></sup>

CIMD document per draft-parecki-oauth-client-id-metadata-document.

The client metadata document is a JSON document containing OAuth client
metadata. The client\_id property MUST match the URL where this document
is hosted.

Key constraint: token\_endpoint\_auth\_method MUST NOT use shared secrets
(client\_secret\_post, client\_secret\_basic, client\_secret\_jwt).

redirect\_uris is required and must contain at least one entry.

**Methods:**

#### `validate_auth_method` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L125"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_auth_method(cls, v: str) -> str
```

Ensure no shared-secret auth methods are used.

#### `validate_redirect_uris` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L137"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_redirect_uris(cls, v: list[str]) -> list[str]
```

Ensure redirect\_uris is non-empty and each entry is a valid URI.

### `CIMDValidationError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L154"><Icon icon="github" /></a></sup>

Raised when CIMD document validation fails.

### `CIMDFetchError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L158"><Icon icon="github" /></a></sup>

Raised when CIMD document fetching fails.

### `CIMDFetcher` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L186"><Icon icon="github" /></a></sup>

Fetch and validate CIMD documents with SSRF protection.

Delegates HTTP fetching to ssrf\_safe\_fetch\_response, which provides DNS
pinning, IP validation, size limits, and timeout enforcement. Documents are
cached using HTTP caching semantics (Cache-Control/ETag/Last-Modified), with
a TTL fallback when response headers do not define caching behavior.

**Methods:**

#### `is_cimd_client_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L270"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_cimd_client_id(self, client_id: str) -> bool
```

Check if a client\_id looks like a CIMD URL.

CIMD URLs must be HTTPS with a host and non-root path.

#### `fetch` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L287"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fetch(self, client_id_url: str) -> CIMDDocument
```

Fetch and validate a CIMD document with SSRF protection.

Uses ssrf\_safe\_fetch\_response for the HTTP layer, which provides:

* HTTPS only, DNS resolution with IP validation
* DNS pinning (connects to validated IP directly)
* Blocks private/loopback/link-local/multicast IPs
* Response size limit and timeout enforcement
* Redirects disabled

**Args:**

* `client_id_url`: The URL to fetch (also the expected client\_id)

**Returns:**

* Validated CIMDDocument

**Raises:**

* `CIMDValidationError`: If document is invalid or URL blocked
* `CIMDFetchError`: If document cannot be fetched

#### `validate_redirect_uri` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L422"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_redirect_uri(self, doc: CIMDDocument, redirect_uri: str) -> bool
```

Validate that a redirect\_uri is allowed by the CIMD document.

**Args:**

* `doc`: The CIMD document
* `redirect_uri`: The redirect URI to validate

**Returns:**

* True if valid, False otherwise

### `CIMDAssertionValidator` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L452"><Icon icon="github" /></a></sup>

Validates JWT assertions for private\_key\_jwt CIMD clients.

Implements RFC 7523 (JSON Web Token (JWT) Profile for OAuth 2.0 Client
Authentication and Authorization Grants) for CIMD client authentication.

JTI replay protection uses TTL-based caching to ensure proper security:

* JTIs are cached with expiration matching the JWT's exp claim
* Expired JTIs are automatically cleaned up
* Maximum assertion lifetime is enforced (5 minutes)

**Methods:**

#### `validate_assertion` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L495"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_assertion(self, assertion: str, client_id: str, token_endpoint: str, cimd_doc: CIMDDocument) -> bool
```

Validate JWT assertion from client.

**Args:**

* `assertion`: The JWT assertion string
* `client_id`: Expected client\_id (must match iss and sub claims)
* `token_endpoint`: Token endpoint URL (must match aud claim)
* `cimd_doc`: CIMD document containing JWKS for key verification

**Returns:**

* True if valid

**Raises:**

* `ValueError`: If validation fails

### `CIMDClientManager` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L677"><Icon icon="github" /></a></sup>

Manages all CIMD client operations for OAuth proxy.

This class encapsulates:

* CIMD client detection
* Document fetching and validation
* Synthetic OAuth client creation
* Private key JWT assertion validation

This allows the OAuth proxy to delegate all CIMD-specific logic to a
single, focused manager class.

**Methods:**

#### `is_cimd_client_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L711"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_cimd_client_id(self, client_id: str) -> bool
```

Check if client\_id is a CIMD URL.

**Args:**

* `client_id`: Client ID to check

**Returns:**

* True if client\_id is an HTTPS URL (CIMD format)

#### `get_client` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L722"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_client(self, client_id_url: str)
```

Fetch CIMD document and create synthetic OAuth client.

**Args:**

* `client_id_url`: HTTPS URL pointing to CIMD document

**Returns:**

* OAuthProxyClient with CIMD document attached, or None if fetch fails

#### `validate_private_key_jwt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/cimd.py#L771"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_private_key_jwt(self, assertion: str, client, token_endpoint: str) -> bool
```

Validate JWT assertion for private\_key\_jwt auth.

**Args:**

* `assertion`: JWT assertion string from client
* `client`: OAuth proxy client (must have cimd\_document)
* `token_endpoint`: Token endpoint URL for aud validation

**Returns:**

* True if assertion is valid

**Raises:**

* `ValueError`: If client doesn't have CIMD document or validation fails


# jwt_issuer
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-jwt_issuer



# `fastmcp.server.auth.jwt_issuer`

JWT token issuance and verification for FastMCP OAuth Proxy.

This module implements the token factory pattern for OAuth proxies, where the proxy
issues its own JWT tokens to clients instead of forwarding upstream provider tokens.
This maintains proper OAuth 2.0 token audience boundaries.

## Functions

### `derive_jwt_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/jwt_issuer.py#L37"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
derive_jwt_key() -> bytes
```

Derive JWT signing key from a high-entropy or low-entropy key material and server salt.

## Classes

### `JWTIssuer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/jwt_issuer.py#L74"><Icon icon="github" /></a></sup>

Issues and validates FastMCP-signed JWT tokens using HS256.

This issuer creates JWT tokens for MCP clients with proper audience claims,
maintaining OAuth 2.0 token boundaries. Tokens are signed with HS256 using
a key derived from the upstream client secret.

**Methods:**

#### `issue_access_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/jwt_issuer.py#L100"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
issue_access_token(self, client_id: str, scopes: list[str], jti: str, expires_in: int = 3600, upstream_claims: dict[str, Any] | None = None) -> str
```

Issue a minimal FastMCP access token.

FastMCP tokens are reference tokens containing only the minimal claims
needed for validation and lookup. The JTI maps to the upstream token
which contains actual user identity and authorization data.

**Args:**

* `client_id`: MCP client ID
* `scopes`: Token scopes
* `jti`: Unique token identifier (maps to upstream token)
* `expires_in`: Token lifetime in seconds
* `upstream_claims`: Optional claims from upstream IdP token to include

**Returns:**

* Signed JWT token

#### `issue_refresh_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/jwt_issuer.py#L152"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
issue_refresh_token(self, client_id: str, scopes: list[str], jti: str, expires_in: int, upstream_claims: dict[str, Any] | None = None) -> str
```

Issue a minimal FastMCP refresh token.

FastMCP refresh tokens are reference tokens containing only the minimal
claims needed for validation and lookup. The JTI maps to the upstream
token which contains actual user identity and authorization data.

**Args:**

* `client_id`: MCP client ID
* `scopes`: Token scopes
* `jti`: Unique token identifier (maps to upstream token)
* `expires_in`: Token lifetime in seconds (should match upstream refresh expiry)
* `upstream_claims`: Optional claims from upstream IdP token to include

**Returns:**

* Signed JWT token

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/jwt_issuer.py#L205"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> dict[str, Any]
```

Verify and decode a FastMCP token.

Validates JWT signature, expiration, issuer, and audience.

**Args:**

* `token`: JWT token to verify

**Returns:**

* Decoded token payload

**Raises:**

* `JoseError`: If token is invalid, expired, or has wrong claims


# middleware
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-middleware



# `fastmcp.server.auth.middleware`

Enhanced authentication middleware with better error messages.

This module provides enhanced versions of MCP SDK authentication middleware
that return more helpful error messages for developers troubleshooting
authentication issues.

## Classes

### `RequireAuthMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/middleware.py#L22"><Icon icon="github" /></a></sup>

Enhanced authentication middleware with detailed error messages.

Extends the SDK's RequireAuthMiddleware to provide more actionable
error messages when authentication fails. This helps developers
understand what went wrong and how to fix it.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-oauth_proxy-__init__



# `fastmcp.server.auth.oauth_proxy`

OAuth Proxy Provider for FastMCP.

This package provides OAuth proxy functionality split across multiple modules:

* models: Pydantic models and constants
* ui: HTML generation functions
* consent: Consent management mixin
* proxy: Main OAuthProxy class


# consent
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-oauth_proxy-consent



# `fastmcp.server.auth.oauth_proxy.consent`

OAuth Proxy Consent Management.

This module contains consent management functionality for the OAuth proxy.
The ConsentMixin class provides methods for handling user consent flows,
cookie management, and consent page rendering.

## Classes

### `ConsentMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/consent.py#L35"><Icon icon="github" /></a></sup>

Mixin class providing consent management functionality for OAuthProxy.

This mixin contains all methods related to:

* Cookie signing and verification
* Consent page rendering
* Consent approval/denial handling
* URI normalization for consent tracking


# models
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-oauth_proxy-models



# `fastmcp.server.auth.oauth_proxy.models`

OAuth Proxy Models and Constants.

This module contains all Pydantic models and constants used by the OAuth proxy.

## Classes

### `OAuthTransaction` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/models.py#L40"><Icon icon="github" /></a></sup>

OAuth transaction state for consent flow.

Stored server-side to track active authorization flows with client context.
Includes CSRF tokens for consent protection per MCP security best practices.

### `ClientCode` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/models.py#L62"><Icon icon="github" /></a></sup>

Client authorization code with PKCE and upstream tokens.

Stored server-side after upstream IdP callback. Contains the upstream
tokens bound to the client's PKCE challenge for secure token exchange.

### `UpstreamTokenSet` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/models.py#L80"><Icon icon="github" /></a></sup>

Stored upstream OAuth tokens from identity provider.

These tokens are obtained from the upstream provider (Google, GitHub, etc.)
and stored in plaintext within this model. Encryption is handled transparently
at the storage layer via FernetEncryptionWrapper. Tokens are never exposed to MCP clients.

### `JTIMapping` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/models.py#L102"><Icon icon="github" /></a></sup>

Maps FastMCP token JTI to upstream token ID.

This allows stateless JWT validation while still being able to look up
the corresponding upstream token when tools need to access upstream APIs.

### `RefreshTokenMetadata` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/models.py#L114"><Icon icon="github" /></a></sup>

Metadata for a refresh token, stored keyed by token hash.

We store only metadata (not the token itself) for security - if storage
is compromised, attackers get hashes they can't reverse into usable tokens.

### `ProxyDCRClient` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/models.py#L136"><Icon icon="github" /></a></sup>

Client for DCR proxy with configurable redirect URI validation.

This special client class is critical for the OAuth proxy to work correctly
with Dynamic Client Registration (DCR). Here's why it exists:

## Problem:

When MCP clients use OAuth, they dynamically register with random localhost
ports (e.g., [http://localhost:55454/callback](http://localhost:55454/callback)). The OAuth proxy needs to:

1. Accept these dynamic redirect URIs from clients based on configured patterns
2. Use its own fixed redirect URI with the upstream provider (Google, GitHub, etc.)
3. Forward the authorization code back to the client's dynamic URI

## Solution:

This class validates redirect URIs against configurable patterns,
while the proxy internally uses its own fixed redirect URI with the upstream
provider. This allows the flow to work even when clients reconnect with
different ports or when tokens are cached.

Without proper validation, clients could get "Redirect URI not registered" errors
when trying to authenticate with cached tokens, or security vulnerabilities could
arise from accepting arbitrary redirect URIs.

**Methods:**

#### `validate_redirect_uri` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/models.py#L167"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_redirect_uri(self, redirect_uri: AnyUrl | None) -> AnyUrl
```

Validate redirect URI against proxy patterns and optionally CIMD redirect\_uris.

For CIMD clients: validates against BOTH the CIMD document's redirect\_uris
AND the proxy's allowed patterns (if configured). Both must pass.

For DCR clients: validates against proxy patterns first, falling back to
base validation (registered redirect\_uris) if patterns don't match.


# proxy
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-oauth_proxy-proxy



# `fastmcp.server.auth.oauth_proxy.proxy`

OAuth Proxy Provider for FastMCP.

This provider acts as a transparent proxy to an upstream OAuth Authorization Server,
handling Dynamic Client Registration locally while forwarding all other OAuth flows.
This enables authentication with upstream providers that don't support DCR or have
restricted client registration policies.

Key features:

* Proxies authorization and token endpoints to upstream server
* Implements local Dynamic Client Registration with fixed upstream credentials
* Validates tokens using upstream JWKS
* Maintains minimal local state for bookkeeping
* Enhanced logging with request correlation

This implementation is based on the OAuth 2.1 specification and is designed for
production use with enterprise identity providers.

## Classes

### `OAuthProxy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L118"><Icon icon="github" /></a></sup>

OAuth provider that presents a DCR-compliant interface while proxying to non-DCR IDPs.

## Purpose

MCP clients expect OAuth providers to support Dynamic Client Registration (DCR),
where clients can register themselves dynamically and receive unique credentials.
Most enterprise IDPs (Google, GitHub, Azure AD, etc.) don't support DCR and require
pre-registered OAuth applications with fixed credentials.

This proxy bridges that gap by:

* Presenting a full DCR-compliant OAuth interface to MCP clients
* Translating DCR registration requests to use pre-configured upstream credentials
* Proxying all OAuth flows to the upstream IDP with appropriate translations
* Managing the state and security requirements of both protocols

## Architecture Overview

The proxy maintains a single OAuth app registration with the upstream provider
while allowing unlimited MCP clients to register and authenticate dynamically.
It implements the complete OAuth 2.1 + DCR specification for clients while
translating to whatever OAuth variant the upstream provider requires.

## Key Translation Challenges Solved

1. Dynamic Client Registration:
   * MCP clients expect to register dynamically and get unique credentials
   * Upstream IDPs require pre-registered apps with fixed credentials
   * Solution: Accept DCR requests, return shared upstream credentials

2. Dynamic Redirect URIs:
   * MCP clients use random localhost ports that change between sessions
   * Upstream IDPs require fixed, pre-registered redirect URIs
   * Solution: Use proxy's fixed callback URL with upstream, forward to client's dynamic URI

3. Authorization Code Mapping:
   * Upstream returns codes for the proxy's redirect URI
   * Clients expect codes for their own redirect URIs
   * Solution: Exchange upstream code server-side, issue new code to client

4. State Parameter Collision:
   * Both client and proxy need to maintain state through the flow
   * Only one state parameter available in OAuth
   * Solution: Use transaction ID as state with upstream, preserve client's state

5. Token Management:
   * Clients may expect different token formats/claims than upstream provides
   * Need to track tokens for revocation and refresh
   * Solution: Store token relationships, forward upstream tokens transparently

## OAuth Flow Implementation

1. Client Registration (DCR):
   * Accept any client registration request
   * Store ProxyDCRClient that accepts dynamic redirect URIs

2. Authorization:
   * Store transaction mapping client details to proxy flow
   * Redirect to upstream with proxy's fixed redirect URI
   * Use transaction ID as state parameter with upstream

3. Upstream Callback:
   * Exchange upstream authorization code for tokens (server-side)
   * Generate new authorization code bound to client's PKCE challenge
   * Redirect to client's original dynamic redirect URI

4. Token Exchange:
   * Validate client's code and PKCE verifier
   * Return previously obtained upstream tokens
   * Clean up one-time use authorization code

5. Token Refresh:
   * Forward refresh requests to upstream using authlib
   * Handle token rotation if upstream issues new refresh token
   * Update local token mappings

## State Management

The proxy maintains minimal but crucial state via pluggable storage (client\_storage):

* \_oauth\_transactions: Active authorization flows with client context
* \_client\_codes: Authorization codes with PKCE challenges and upstream tokens
* \_jti\_mapping\_store: Maps FastMCP token JTIs to upstream token IDs
* \_refresh\_token\_store: Refresh token metadata (keyed by token hash)

All state is stored in the configured client\_storage backend (Redis, disk, etc.)
enabling horizontal scaling across multiple instances.

## Security Considerations

* Refresh tokens stored by hash only (defense in depth if storage compromised)
* PKCE enforced end-to-end (client to proxy, proxy to upstream)
* Authorization codes are single-use with short expiry
* Transaction IDs are cryptographically random
* All state is cleaned up after use to prevent replay
* Token validation delegates to upstream provider

## Provider Compatibility

Works with any OAuth 2.0 provider that supports:

* Authorization code flow
* Fixed redirect URI (configured in provider's app settings)
* Standard token endpoint

Handles provider-specific requirements:

* Google: Ensures minimum scope requirements
* GitHub: Compatible with OAuth Apps and GitHub Apps
* Azure AD: Handles tenant-specific endpoints
* Generic: Works with any spec-compliant provider

**Methods:**

#### `set_mcp_path` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L540"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_mcp_path(self, mcp_path: str | None) -> None
```

Set the MCP endpoint path and create JWTIssuer with correct audience.

This method is called by get\_routes() to configure the resource URL
and create the JWTIssuer. The JWT audience is set to the full resource
URL (e.g., [http://localhost:8000/mcp](http://localhost:8000/mcp)) to ensure tokens are bound to
this specific MCP endpoint.

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")

#### `jwt_issuer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L564"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
jwt_issuer(self) -> JWTIssuer
```

Get the JWT issuer, ensuring it has been initialized.

The JWT issuer is created when set\_mcp\_path() is called (via get\_routes()).
This property ensures a clear error if used before initialization.

#### `get_client` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L601"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_client(self, client_id: str) -> OAuthClientInformationFull | None
```

Get client information by ID. This is generally the random ID
provided to the DCR client during registration, not the upstream client ID.

For unregistered clients, returns None (which will raise an error in the SDK).
CIMD clients (URL-based client IDs) are looked up and cached automatically.

#### `register_client` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L645"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_client(self, client_info: OAuthClientInformationFull) -> None
```

Register a client locally

When a client registers, we create a ProxyDCRClient that is more
forgiving about validating redirect URIs, since the DCR client's
redirect URI will likely be localhost or unknown to the proxied IDP. The
proxied IDP only knows about this server's fixed redirect URI.

#### `authorize` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L698"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str
```

Start OAuth transaction and route through consent interstitial.

Flow:

1. Validate client's resource matches server's resource URL (security check)
2. Store transaction with client details and PKCE (if forwarding)
3. Return local /consent URL; browser visits consent first
4. Consent handler redirects to upstream IdP if approved/already approved

If consent is disabled (require\_authorization\_consent=False), skip the consent screen
and redirect directly to the upstream IdP.

#### `load_authorization_code` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L817"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str) -> AuthorizationCode | None
```

Load authorization code for validation.

Look up our client code and return authorization code object
with PKCE challenge for validation.

#### `exchange_authorization_code` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L865"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode) -> OAuthToken
```

Exchange authorization code for FastMCP-issued tokens.

Implements the token factory pattern:

1. Retrieves upstream tokens from stored authorization code
2. Extracts user identity from upstream token
3. Encrypts and stores upstream tokens
4. Issues FastMCP-signed JWT tokens
5. Returns FastMCP tokens (NOT upstream tokens)

PKCE validation is handled by the MCP framework before this method is called.

#### `load_refresh_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L1111"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None
```

Load refresh token metadata from distributed storage.

Looks up by token hash and reconstructs the RefreshToken object.
Validates that the token belongs to the requesting client.

#### `exchange_refresh_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L1140"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]) -> OAuthToken
```

Exchange FastMCP refresh token for new FastMCP access token.

Implements two-tier refresh:

1. Verify FastMCP refresh token
2. Look up upstream token via JTI mapping
3. Refresh upstream token with upstream provider
4. Update stored upstream token
5. Issue new FastMCP access token
6. Keep same FastMCP refresh token (unless upstream rotates)

#### `load_access_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L1384"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_access_token(self, token: str) -> AccessToken | None
```

Validate FastMCP JWT by swapping for upstream token.

This implements the token swap pattern:

1. Verify FastMCP JWT signature (proves it's our token)
2. Look up upstream token via JTI mapping
3. Decrypt upstream token
4. Validate upstream token with provider (GitHub API, JWT validation, etc.)
5. Return upstream validation result

The FastMCP JWT is a reference token - all authorization data comes
from validating the upstream token via the TokenVerifier.

#### `revoke_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L1460"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
revoke_token(self, token: AccessToken | RefreshToken) -> None
```

Revoke token locally and with upstream server if supported.

For refresh tokens, removes from local storage by hash.
For all tokens, attempts upstream revocation if endpoint is configured.
Access token JTI mappings expire via TTL.

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/proxy.py#L1493"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get OAuth routes with custom handlers for better error UX.

This method creates standard OAuth routes and replaces:

* /authorize endpoint: Enhanced error responses for unregistered clients
* /token endpoint: OAuth 2.1 compliant error codes

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")
  This is used to advertise the resource URL in metadata.


# ui
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-oauth_proxy-ui



# `fastmcp.server.auth.oauth_proxy.ui`

OAuth Proxy UI Generation Functions.

This module contains HTML generation functions for consent and error pages.

## Functions

### `create_consent_html` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/ui.py#L20"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_consent_html(client_id: str, redirect_uri: str, scopes: list[str], txn_id: str, csrf_token: str, client_name: str | None = None, title: str = 'Application Access Request', server_name: str | None = None, server_icon_url: str | None = None, server_website_url: str | None = None, client_website_url: str | None = None, csp_policy: str | None = None, is_cimd_client: bool = False, cimd_domain: str | None = None) -> str
```

Create a styled HTML consent page for OAuth authorization requests.

**Args:**

* `csp_policy`: Content Security Policy override.
  If None, uses the built-in CSP policy with appropriate directives.
  If empty string "", disables CSP entirely (no meta tag is rendered).
  If a non-empty string, uses that as the CSP policy value.

### `create_error_html` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oauth_proxy/ui.py#L215"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_error_html(error_title: str, error_message: str, error_details: dict[str, str] | None = None, server_name: str | None = None, server_icon_url: str | None = None) -> str
```

Create a styled HTML error page for OAuth errors.

**Args:**

* `error_title`: The error title (e.g., "OAuth Error", "Authorization Failed")
* `error_message`: The main error message to display
* `error_details`: Optional dictionary of error details to show (e.g., `{"Error Code"\: "invalid_client"}`)
* `server_name`: Optional server name to display
* `server_icon_url`: Optional URL to server icon/logo

**Returns:**

* Complete HTML page as a string


# oidc_proxy
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-oidc_proxy



# `fastmcp.server.auth.oidc_proxy`

OIDC Proxy Provider for FastMCP.

This provider acts as a transparent proxy to an upstream OIDC compliant Authorization
Server. It leverages the OAuthProxy class to handle Dynamic Client Registration and
forwarding of all OAuth flows.

This implementation is based on:
OpenID Connect Discovery 1.0 - [https://openid.net/specs/openid-connect-discovery-1\_0.html](https://openid.net/specs/openid-connect-discovery-1_0.html)
OAuth 2.0 Authorization Server Metadata - [https://datatracker.ietf.org/doc/html/rfc8414](https://datatracker.ietf.org/doc/html/rfc8414)

## Classes

### `OIDCConfiguration` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oidc_proxy.py#L28"><Icon icon="github" /></a></sup>

OIDC Configuration.

**Methods:**

#### `get_oidc_configuration` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oidc_proxy.py#L143"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_oidc_configuration(cls, config_url: AnyHttpUrl) -> Self
```

Get the OIDC configuration for the specified config URL.

**Args:**

* `config_url`: The OIDC config URL
* `strict`: The strict flag for the configuration
* `timeout_seconds`: HTTP request timeout in seconds

### `OIDCProxy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oidc_proxy.py#L173"><Icon icon="github" /></a></sup>

OAuth provider that wraps OAuthProxy to provide configuration via an OIDC configuration URL.

This provider makes it easier to add OAuth protection for any upstream provider
that is OIDC compliant.

**Methods:**

#### `get_oidc_configuration` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oidc_proxy.py#L432"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_oidc_configuration(self, config_url: AnyHttpUrl, strict: bool | None, timeout_seconds: int | None) -> OIDCConfiguration
```

Gets the OIDC configuration for the specified configuration URL.

**Args:**

* `config_url`: The OIDC configuration URL
* `strict`: The strict flag for the configuration
* `timeout_seconds`: HTTP request timeout in seconds

#### `get_token_verifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/oidc_proxy.py#L449"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_token_verifier(self) -> TokenVerifier
```

Creates the token verifier for the specified OIDC configuration and arguments.

**Args:**

* `algorithm`: Optional token verifier algorithm
* `audience`: Optional token verifier audience
* `required_scopes`: Optional token verifier required\_scopes
* `timeout_seconds`: HTTP request timeout in seconds


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-__init__



# `fastmcp.server.auth.providers`

*This module is empty or contains only private/internal implementations.*


# auth0
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-auth0



# `fastmcp.server.auth.providers.auth0`

Auth0 OAuth provider for FastMCP.

This module provides a complete Auth0 integration that's ready to use with
just the configuration URL, client ID, client secret, audience, and base URL.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.auth0 import Auth0Provider

# Simple Auth0 OAuth protection
auth = Auth0Provider(
    config_url="https://auth0.config.url",
    client_id="your-auth0-client-id",
    client_secret="your-auth0-client-secret",
    audience="your-auth0-api-audience",
    base_url="http://localhost:8000",
)

mcp = FastMCP("My Protected Server", auth=auth)
```

## Classes

### `Auth0Provider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/auth0.py#L34"><Icon icon="github" /></a></sup>

An Auth0 provider implementation for FastMCP.

This provider is a complete Auth0 integration that's ready to use with
just the configuration URL, client ID, client secret, audience, and base URL.


# aws
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-aws



# `fastmcp.server.auth.providers.aws`

AWS Cognito OAuth provider for FastMCP.

This module provides a complete AWS Cognito OAuth integration that's ready to use
with a user pool ID, domain prefix, client ID and client secret. It handles all
the complexity of AWS Cognito's OAuth flow, token validation, and user management.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.aws_cognito import AWSCognitoProvider

# Simple AWS Cognito OAuth protection
auth = AWSCognitoProvider(
    user_pool_id="your-user-pool-id",
    aws_region="eu-central-1",
    client_id="your-cognito-client-id",
    client_secret="your-cognito-client-secret"
)

mcp = FastMCP("My Protected Server", auth=auth)
```

## Classes

### `AWSCognitoTokenVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/aws.py#L39"><Icon icon="github" /></a></sup>

Token verifier that filters claims to Cognito-specific subset.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/aws.py#L42"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify token and filter claims to Cognito-specific subset.

### `AWSCognitoProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/aws.py#L66"><Icon icon="github" /></a></sup>

Complete AWS Cognito OAuth provider for FastMCP.

This provider makes it trivial to add AWS Cognito OAuth protection to any
FastMCP server using OIDC Discovery. Just provide your Cognito User Pool details,
client credentials, and a base URL, and you're ready to go.

Features:

* Automatic OIDC Discovery from AWS Cognito User Pool
* Automatic JWT token validation via Cognito's public keys
* Cognito-specific claim filtering (sub, username, cognito:groups)
* Support for Cognito User Pools

**Methods:**

#### `get_token_verifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/aws.py#L174"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_token_verifier(self) -> TokenVerifier
```

Creates a Cognito-specific token verifier with claim filtering.

**Args:**

* `algorithm`: Optional token verifier algorithm
* `audience`: Optional token verifier audience
* `required_scopes`: Optional token verifier required\_scopes
* `timeout_seconds`: HTTP request timeout in seconds


# azure
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-azure



# `fastmcp.server.auth.providers.azure`

Azure (Microsoft Entra) OAuth provider for FastMCP.

This provider implements Azure/Microsoft Entra ID OAuth authentication
using the OAuth Proxy pattern for non-DCR OAuth flows.

## Functions

### `EntraOBOToken` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/azure.py#L683"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
EntraOBOToken(scopes: list[str]) -> str
```

Exchange the user's Entra token for a downstream API token via OBO.

This dependency performs a Microsoft Entra On-Behalf-Of (OBO) token exchange,
allowing your MCP server to call downstream APIs (like Microsoft Graph) on
behalf of the authenticated user.

**Args:**

* `scopes`: The scopes to request for the downstream API. For Microsoft Graph,
  use scopes like \["[https://graph.microsoft.com/Mail.Read](https://graph.microsoft.com/Mail.Read)"] or
  \["[https://graph.microsoft.com/.default](https://graph.microsoft.com/.default)"].

**Returns:**

* A dependency that resolves to the downstream API access token string

**Raises:**

* `ImportError`: If fastmcp\[azure] is not installed
* `RuntimeError`: If no access token is available, provider is not Azure,
  or OBO exchange fails

## Classes

### `AzureProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/azure.py#L35"><Icon icon="github" /></a></sup>

Azure (Microsoft Entra) OAuth provider for FastMCP.

This provider implements Azure/Microsoft Entra ID authentication using the
OAuth Proxy pattern. It supports both organizational accounts and personal
Microsoft accounts depending on the tenant configuration.

Scope Handling:

* required\_scopes: Provide unprefixed scope names (e.g., \["read", "write"])
  → Automatically prefixed with identifier\_uri during initialization
  → Validated on all tokens and advertised to MCP clients
* additional\_authorize\_scopes: Provide full format (e.g., \["User.Read"])
  → NOT prefixed, NOT validated, NOT advertised to clients
  → Used to request Microsoft Graph or other upstream API permissions

Features:

* OAuth proxy to Azure/Microsoft identity platform
* JWT validation using tenant issuer and JWKS
* Supports tenant configurations: specific tenant ID, "organizations", or "consumers"
* Custom API scopes and Microsoft Graph scopes in a single provider

Setup:

1. Create an App registration in Azure Portal
2. Configure Web platform redirect URI: [http://localhost:8000/auth/callback](http://localhost:8000/auth/callback) (or your custom path)
3. Add an Application ID URI under "Expose an API" (defaults to api://)
4. Add custom scopes (e.g., "read", "write") under "Expose an API"
5. Set access token version to 2 in the App manifest: "requestedAccessTokenVersion": 2
6. Create a client secret
7. Get Application (client) ID, Directory (tenant) ID, and client secret

**Methods:**

#### `authorize` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/azure.py#L252"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str
```

Start OAuth transaction and redirect to Azure AD.

Override parent's authorize method to filter out the 'resource' parameter
which is not supported by Azure AD v2.0 endpoints. The v2.0 endpoints use
scopes to determine the resource/audience instead of a separate parameter.

**Args:**

* `client`: OAuth client information
* `params`: Authorization parameters from the client

**Returns:**

* Authorization URL to redirect the user to Azure AD

#### `get_obo_credential` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/azure.py#L478"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_obo_credential(self, user_assertion: str) -> OnBehalfOfCredential
```

Get a cached or new OnBehalfOfCredential for OBO token exchange.

Credentials are cached by user assertion so the Azure SDK's internal
token cache can avoid redundant OBO exchanges when the same user
calls multiple tools with the same scopes.

**Args:**

* `user_assertion`: The user's access token to exchange via OBO.

**Returns:**

* A configured OnBehalfOfCredential ready for get\_token() calls.

**Raises:**

* `ImportError`: If azure-identity is not installed (requires fastmcp\[azure]).

#### `close_obo_credentials` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/azure.py#L519"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
close_obo_credentials(self) -> None
```

Close all cached OBO credentials.

### `AzureJWTVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/azure.py#L530"><Icon icon="github" /></a></sup>

JWT verifier pre-configured for Azure AD / Microsoft Entra ID.

Auto-configures JWKS URI, issuer, audience, and scope handling from your
Azure app registration details. Designed for Managed Identity and other
token-verification-only scenarios where AzureProvider's full OAuth proxy
isn't needed.

Handles Azure's scope format automatically:

* Validates tokens using short-form scopes (what Azure puts in `scp` claims)
* Advertises full-URI scopes in OAuth metadata (what clients need to request)

Example::

from fastmcp.server.auth import RemoteAuthProvider
from fastmcp.server.auth.providers.azure import AzureJWTVerifier
from pydantic import AnyHttpUrl

verifier = AzureJWTVerifier(
client\_id="your-client-id",
tenant\_id="your-tenant-id",
required\_scopes=\["access\_as\_user"],
)

auth = RemoteAuthProvider(
token\_verifier=verifier,
authorization\_servers=\[
AnyHttpUrl("[https://login.microsoftonline.com/your-tenant-id/v2.0](https://login.microsoftonline.com/your-tenant-id/v2.0)")
],
base\_url="[https://my-server.com](https://my-server.com)",
)

**Methods:**

#### `scopes_supported` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/azure.py#L610"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
scopes_supported(self) -> list[str]
```

Return scopes with Azure URI prefix for OAuth metadata.

Azure tokens contain short-form scopes (e.g., `read`) in the `scp`
claim, but clients must request full URI scopes (e.g.,
`api://client-id/read`) from the Azure authorization endpoint. This
property returns the full-URI form for OAuth metadata while
`required_scopes` retains the short form for token validation.


# debug
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-debug



# `fastmcp.server.auth.providers.debug`

Debug token verifier for testing and special cases.

This module provides a flexible token verifier that delegates validation
to a custom callable. Useful for testing, development, or scenarios where
standard verification isn't possible (like opaque tokens without introspection).

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.debug import DebugTokenVerifier

# Accept all tokens (default - useful for testing)
auth = DebugTokenVerifier()

# Custom sync validation logic
auth = DebugTokenVerifier(validate=lambda token: token.startswith("valid-"))

# Custom async validation logic
async def check_cache(token: str) -> bool:
    return await redis.exists(f"token:{token}")

auth = DebugTokenVerifier(validate=check_cache)

mcp = FastMCP("My Server", auth=auth)
```

## Classes

### `DebugTokenVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/debug.py#L40"><Icon icon="github" /></a></sup>

Token verifier with custom validation logic.

This verifier delegates token validation to a user-provided callable.
By default, it accepts all non-empty tokens (useful for testing).

Use cases:

* Testing: Accept any token without real verification
* Development: Custom validation logic for prototyping
* Opaque tokens: When you have tokens with no introspection endpoint

WARNING: This bypasses standard security checks. Only use in controlled
environments or when you understand the security implications.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/debug.py#L77"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify token using custom validation logic.

**Args:**

* `token`: The token string to validate

**Returns:**

* AccessToken if validation succeeds, None otherwise


# descope
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-descope



# `fastmcp.server.auth.providers.descope`

Descope authentication provider for FastMCP.

This module provides DescopeProvider - a complete authentication solution that integrates
with Descope's OAuth 2.1 and OpenID Connect services, supporting Dynamic Client Registration (DCR)
for seamless MCP client authentication.

## Classes

### `DescopeProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/descope.py#L25"><Icon icon="github" /></a></sup>

Descope metadata provider for DCR (Dynamic Client Registration).

This provider implements Descope integration using metadata forwarding.
This is the recommended approach for Descope DCR
as it allows Descope to handle the OAuth flow directly while FastMCP acts
as a resource server.

IMPORTANT SETUP REQUIREMENTS:

1. Create an MCP Server in Descope Console:
   * Go to the [MCP Servers page](https://app.descope.com/mcp-servers) of the Descope Console
   * Create a new MCP Server
   * Ensure that **Dynamic Client Registration (DCR)** is enabled
   * Note your Well-Known URL

2. Note your Well-Known URL:
   * Save your Well-Known URL from [MCP Server Settings](https://app.descope.com/mcp-servers)
   * Format: `https://.../v1/apps/agentic/P.../M.../.well-known/openid-configuration`

For detailed setup instructions, see:
[https://docs.descope.com/identity-federation/inbound-apps/creating-inbound-apps#method-2-dynamic-client-registration-dcr](https://docs.descope.com/identity-federation/inbound-apps/creating-inbound-apps#method-2-dynamic-client-registration-dcr)

**Methods:**

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/descope.py#L154"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get OAuth routes including Descope authorization server metadata forwarding.

This returns the standard protected resource routes plus an authorization server
metadata endpoint that forwards Descope's OAuth metadata to clients.

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")
  This is used to advertise the resource URL in metadata.


# discord
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-discord



# `fastmcp.server.auth.providers.discord`

Discord OAuth provider for FastMCP.

This module provides a complete Discord OAuth integration that's ready to use
with just a client ID and client secret. It handles all the complexity of
Discord's OAuth flow, token validation, and user management.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.discord import DiscordProvider

# Simple Discord OAuth protection
auth = DiscordProvider(
    client_id="your-discord-client-id",
    client_secret="your-discord-client-secret"
)

mcp = FastMCP("My Protected Server", auth=auth)
```

## Classes

### `DiscordTokenVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/discord.py#L41"><Icon icon="github" /></a></sup>

Token verifier for Discord OAuth tokens.

Discord OAuth tokens are opaque (not JWTs), so we verify them
by calling Discord's tokeninfo API to check if they're valid and get user info.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/discord.py#L68"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify Discord OAuth token by calling Discord's tokeninfo API.

### `DiscordProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/discord.py#L154"><Icon icon="github" /></a></sup>

Complete Discord OAuth provider for FastMCP.

This provider makes it trivial to add Discord OAuth protection to any
FastMCP server. Just provide your Discord OAuth app credentials and
a base URL, and you're ready to go.

Features:

* Transparent OAuth proxy to Discord
* Automatic token validation via Discord's API
* User information extraction from Discord APIs
* Minimal configuration required


# settings
Source: https://gofastmcp.com/python-sdk/fastmcp-settings



# `fastmcp.settings`

## Classes

### `DocketSettings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/settings.py#L29"><Icon icon="github" /></a></sup>

Docket worker configuration.

### `Settings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/settings.py#L117"><Icon icon="github" /></a></sup>

FastMCP settings.

**Methods:**

#### `get_setting` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/settings.py#L129"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_setting(self, attr: str) -> Any
```

Get a setting. If the setting contains one or more `__`, it will be
treated as a nested setting.

#### `set_setting` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/settings.py#L142"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_setting(self, attr: str, value: Any) -> None
```

Set a setting. If the setting contains one or more `__`, it will be
treated as a nested setting.

#### `normalize_log_level` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/settings.py#L164"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
normalize_log_level(cls, v)
```


# telemetry
Source: https://gofastmcp.com/python-sdk/fastmcp-telemetry



# `fastmcp.telemetry`

OpenTelemetry instrumentation for FastMCP.

This module provides native OpenTelemetry integration for FastMCP servers and clients.
It uses only the opentelemetry-api package, so telemetry is a no-op unless the user
installs an OpenTelemetry SDK and configures exporters.

Example usage with SDK:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Configure the SDK (user responsibility)
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Now FastMCP will emit traces
from fastmcp import FastMCP
mcp = FastMCP("my-server")
```

## Functions

### `get_tracer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/telemetry.py#L38"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tracer(version: str | None = None) -> Tracer
```

Get the FastMCP tracer for creating spans.

**Args:**

* `version`: Optional version string for the instrumentation

**Returns:**

* A tracer instance. Returns a no-op tracer if no SDK is configured.

### `inject_trace_context` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/telemetry.py#L50"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
inject_trace_context(meta: dict[str, Any] | None = None) -> dict[str, Any] | None
```

Inject current trace context into a meta dict for MCP request propagation.

**Args:**

* `meta`: Optional existing meta dict to merge with trace context

**Returns:**

* A new dict containing the original meta (if any) plus trace context keys,
* or None if no trace context to inject and meta was None

### `record_span_error` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/telemetry.py#L76"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
record_span_error(span: Span, exception: BaseException) -> None
```

Record an exception on a span and set error status.

### `extract_trace_context` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/telemetry.py#L82"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
extract_trace_context(meta: dict[str, Any] | None) -> Context
```

Extract trace context from an MCP request meta dict.

If already in a valid trace (e.g., from HTTP propagation), the existing
trace context is preserved and meta is not used.

**Args:**

* `meta`: The meta dict from an MCP request (ctx.request\_context.meta)

**Returns:**

* An OpenTelemetry Context with the extracted trace context,
* or the current context if no trace context found or already in a trace


# FastMCP Updates
Source: https://gofastmcp.com/updates



<Update label="FastMCP 3.0.2" description="February 22, 2026">
  <Card title="FastMCP v3.0.2: Threecovery Mode II" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v3.0.2">
    Two community-contributed fixes: auth headers from MCP transport no longer leak through to downstream OpenAPI APIs, and background task workers now correctly receive the originating request ID. Plus a new docs example for context-aware tool factories.
  </Card>
</Update>

<Update label="FastMCP 3.0.1" description="February 20, 2026">
  <Card title="FastMCP v3.0.1: Three-covery Mode" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v3.0.1">
    First patch after 3.0 — mostly smoothing out rough edges discovered in the wild. The big ones: middleware state that wasn't surviving the trip to tool handlers now does, `Tool.from_tool()` accepts callables again, OpenAPI schemas with circular references no longer crash discovery, and decorator overloads now return the correct types in function mode.

    🔐 **OIDC `verify_id_token`** — New option for providers that issue opaque access tokens but standard JWT id\_tokens. Verifies identity via the id\_token while using the access\_token for upstream API calls.

    🐞 **11 bug fixes** — State serialization, future annotations with `Context`/`Depends`, OpenAI handler deprecation warnings, type checker compatibility, and more.
  </Card>
</Update>

<Update label="FastMCP 3.0.0" description="February 18, 2026">
  <Card title="FastMCP v3.0.0: Three at Last" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v3.0.0">
    FastMCP 3.0 is stable. Two betas, two release candidates, 21 new contributors, and more than 100,000 pre-release installs later — the architecture held up, the upgrade path was smooth, and we're shipping it.

    The surface API is largely unchanged — `@mcp.tool()` still works exactly as before. What changed is everything underneath: a provider/transform architecture that makes FastMCP extensible, observable, and composable in ways v2 couldn't support.

    🔌 **Build servers from anything** — `FileSystemProvider`, `OpenAPIProvider`, `ProxyProvider`, `SkillsProvider`, and composable transforms that rename, namespace, filter, version, and secure components as they flow to clients.

    🔐 **Ship to production** — Component versioning, granular authorization with async auth checks, CIMD, Static Client Registration, Azure OBO, OpenTelemetry tracing, and background tasks with distributed Redis notification.

    💾 **Adapt per session** — Session state persists across requests, and `ctx.enable_components()` / `ctx.disable_components()` let servers adapt dynamically per client.

    ⚡ **Develop faster** — `--reload`, standalone decorators, automatic threadpool dispatch, tool timeouts, pagination, and concurrent tool execution.

    🖥️ **CLI** — `fastmcp list`, `fastmcp call`, `fastmcp discover`, `fastmcp generate-cli`, and `fastmcp install` for Claude Desktop, Cursor, and Goose.
  </Card>
</Update>

<Update label="FastMCP 3.0.0rc1" description="February 12, 2026">
  <Card title="FastMCP v3.0.0rc1: RC-ing is Believing" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v3.0.0rc1">
    FastMCP 3 RC1 means we believe the API is stable. Beta 2 drew a wave of real-world adoption — production deployments, migration reports, integration testing — and the feedback overwhelmingly confirmed that the architecture works. This release closes gaps that surfaced under load: auth flows that needed to be async, background tasks that needed reliable notification delivery, and APIs still carrying beta-era naming. If nothing unexpected surfaces, this is what 3.0.0 looks like.

    🚨 **Breaking Changes** — The `ui=` parameter is now `app=` with a unified `AppConfig` class, and 16 `FastMCP()` constructor kwargs have been removed after months of deprecation warnings.

    🔐 **Auth Improvements** — Async `auth=` checks, Static Client Registration for servers without DCR, and declarative Azure OBO flows via dependency injection.

    ⚡ **Concurrent Sampling** — `context.sample()` can now execute multiple tool calls in parallel with `tool_concurrency=0`.

    📡 **Background Task Notifications** — A distributed Redis queue replaces polling for progress updates and elicitation relay.

    ✅ **OpenAPI Output Validation** — `validate_output=False` disables strict schema checking for imperfect backend APIs.
  </Card>
</Update>

<Update label="FastMCP 3.0.0b2" description="February 7, 2026">
  <Card title="FastMCP v3.0.0b2: 2 Fast 2 Beta" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v3.0.0b2">
    Beta 2 reflects the huge number of people that kicked the tires on Beta 1. Seven new contributors landed changes, and early migration reports went smoother than expected. Most of Beta 2 is refinement — fixing what people found, filling gaps from real usage, hardening edges — but a few new features landed along the way.

    🖥️ **Client CLI** — `fastmcp list`, `fastmcp call`, `fastmcp discover`, and `fastmcp generate-cli` turn any MCP server into something you can poke at from a terminal.

    🔐 **CIMD** (Client ID Metadata Documents) adds an alternative to Dynamic Client Registration for OAuth.

    📱 **MCP Apps** — Spec-level compliance for the MCP Apps extension with `ui://` resource scheme and typed UI metadata.

    ⏳ **Background Task Context** — `Context` now works transparently in Docket workers with Redis-based coordination.

    🛡️ **ResponseLimitingMiddleware** caps tool response sizes with UTF-8-safe truncation.

    🪿 **Goose Integration** — `fastmcp install goose` for one-command server installation into Goose.
  </Card>
</Update>

<Update label="FastMCP 3.0.0b1" description="January 20, 2026">
  <Card title="FastMCP 3.0.0b1: This Beta Work" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v3.0.0b1">
    FastMCP 3.0 rebuilds the framework around three primitives: components, providers, and transforms. Providers source components dynamically—from decorators, filesystems, OpenAPI specs, remote servers, or anywhere else. Transforms modify components as they flow to clients. The features that required specialized subsystems in v2 now compose naturally from these building blocks.

    🔌 **Provider Architecture** unifies how components are sourced with `FileSystemProvider`, `SkillsProvider`, `OpenAPIProvider`, and `ProxyProvider`.

    🔄 **Transforms** add middleware for components—namespace, rename, filter by version, control visibility.

    📋 **Component Versioning** lets you register multiple versions of the same tool with automatic highest-version selection.

    💾 **Session-Scoped State** persists across requests, with per-session visibility control.

    ⚡ **DX Improvements** include `--reload` for development, automatic threadpool dispatch, tool timeouts, pagination, and OpenTelemetry tracing.
  </Card>
</Update>

<Update label="FastMCP 2.14.5" description="February 3, 2026">
  <Card title="FastMCP 2.14.5: Sealed Docket" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.14.5">
    Fixes a memory leak in the memory:// docket broker where cancelled tasks accumulated instead of being cleaned up. Bumps pydocket to ≥0.17.2.
  </Card>
</Update>

<Update label="FastMCP 2.14.4" description="January 22, 2026">
  <Card title="FastMCP 2.14.4: Package Deal" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.14.4">
    Fixes a fresh install bug where the packaging library was missing as a direct dependency, plus backports \$ref dereferencing in tool schemas and a task capabilities location fix.
  </Card>
</Update>

<Update label="FastMCP 2.14.3" description="January 12, 2026">
  <Card title="FastMCP 2.14.3: Time After Timeout" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.14.3">
    Sometimes five seconds just isn't enough. This release fixes an HTTP transport bug that was cutting connections short, along with OAuth and Redis fixes, better ASGI support, and CLI update notifications so you never miss a beat.

    ⏱️ **HTTP transport timeout fix** restores MCP's 30-second default connect timeout, which was incorrectly defaulting to 5 seconds.

    🔧 **Infrastructure fixes** including OAuth token storage TTL, Redis key prefixing for ACL isolation, and ContextVar propagation for ASGI-mounted servers with background tasks.
  </Card>
</Update>

<Update label="FastMCP 2.14.2" description="December 31, 2025">
  <Card title="FastMCP 2.14.2: Port Authority" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.14.2">
    A wave of community contributions arrives safely in the 2.x line. Important backports from 3.0 improve OpenAPI 3.1 compatibility, MCP spec compliance for output schemas and elicitation, and correct a subtle base\_url fallback issue.

    🔧 **OpenAPI 3.1 support** fixes version detection to properly handle 3.1 specs alongside 3.0.

    📋 **MCP spec compliance** for root-level `$ref` resolution in output schemas and titled enum elicitation schemas.
  </Card>
</Update>

<Update label="FastMCP 2.14.1" description="December 15, 2025">
  <Card title="FastMCP 2.14.1: 'Tis a Gift to Be Sample" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.14.1">
    FastMCP 2.14.1 introduces sampling with tools (SEP-1577), enabling servers to pass tools to `ctx.sample()` for agentic workflows where the LLM can automatically execute tool calls in a loop.

    🤖 **Sampling with tools** lets servers leverage client LLM capabilities for multi-step agentic workflows. The new `ctx.sample_step()` method provides single LLM calls with tool inspection, while `result_type` enables structured outputs via validated Pydantic models.

    🔧 **AnthropicSamplingHandler** joins the existing OpenAI handler, and both are now promoted from experimental to production-ready status with a unified API.
  </Card>
</Update>

<Update label="FastMCP 2.14.0" description="December 11, 2025">
  <Card title="FastMCP 2.14.0: Task and You Shall Receive" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.14.0">
    FastMCP 2.14 begins adopting the MCP 2025-11-25 specification, introducing protocol-native background tasks that enable long-running operations to report progress without blocking clients.

    ⏳ **Background Tasks (SEP-1686)** let you add `task=True` to any async tool decorator. Powered by [Docket](https://github.com/chrisguidry/docket) for enterprise task scheduling—in-memory backends work out-of-the-box, Redis enables persistence and horizontal scaling.

    🔧 **OpenAPI Parser Promoted** from experimental to standard with improved performance through single-pass schema processing.

    📋 **MCP Spec Updates** including SSE polling (SEP-1699), multi-select elicitation (SEP-1330), and tool name validation (SEP-986). Also removes deprecated APIs accumulated across 2.x.
  </Card>
</Update>

<Update label="FastMCP 2.13.3" description="December 3, 2025">
  <Card title="FastMCP 2.13.3: Pin-ish Line" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.13.3">
    Pins `mcp<1.23` as a precaution due to MCP SDK changes related to the 11/25/25 protocol update that break certain FastMCP patches and workarounds. FastMCP 2.14 introduces proper support for the updated protocol.
  </Card>
</Update>

<Update label="FastMCP 2.13.2" description="December 1, 2025">
  <Card title="FastMCP 2.13.2: Refreshing Changes" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.13.2">
    Polishes the authentication stack with improvements to token refresh, scope handling, and multi-instance deployments.

    🎮 **Discord OAuth provider** added as a built-in authentication option.

    🔄 **Token refresh fixes** for Azure and Google providers, plus OAuth proxy improvements for multi-instance deployments.

    🎨 **Icon support** added to proxy classes for richer UX.
  </Card>
</Update>

<Update label="FastMCP 2.13.1" description="November 15, 2025">
  <Card title="FastMCP 2.13.1: Heavy Meta" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.13.1">
    Introduces meta parameter support for `ToolResult`, enabling tools to return supplementary metadata alongside results for patterns like OpenAI's Apps SDK.

    🏷️ **Meta parameters** let tools return supplementary metadata alongside results.

    🔐 **New auth providers** for OCI and Supabase, plus custom token verifiers with DebugTokenVerifier for development.

    🔒 **Security fixes** for CVE-2025-61920 and safer Cursor deeplink URL validation on Windows.
  </Card>
</Update>

<Update label="FastMCP 2.13.0" description="October 25, 2025">
  <Card title="FastMCP 2.13.0: Cache Me If You Can" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.13.0">
    FastMCP 2.13 "Cache Me If You Can" represents a fundamental maturation of the framework. After months of community feedback on authentication and state management, this release delivers the infrastructure FastMCP needs to handle production workloads: persistent storage, response caching, and pragmatic OAuth improvements that reflect real-world deployment challenges.

    💾 **Pluggable storage backends** bring persistent state to FastMCP servers. Built on [py-key-value-aio](https://github.com/strawgate/py-key-value), a new library from FastMCP maintainer Bill Easton ([@strawgate](https://github.com/strawgate)), the storage layer provides encrypted disk storage by default, platform-aware token management, and a simple key-value interface for application state. We're excited to bring this elegantly designed library into the FastMCP ecosystem - it's both powerful and remarkably easy to use, including wrappers to add encryption, TTLs, caching, and more to backends ranging from Elasticsearch, Redis, DynamoDB, filesystem, in-memory, and more!

    🔐 **OAuth maturity** brings months of production learnings into the framework. The new consent screen prevents confused deputy and authorization bypass attacks discovered in earlier versions, while the OAuth proxy now issues its own tokens with automatic key derivation. RFC 7662 token introspection support enables enterprise auth flows, and path prefix mounting enables OAuth-protected servers to integrate into existing web applications. FastMCP now supports out-of-the-box authentication with [WorkOS](https://gofastmcp.com/integrations/workos) and [AuthKit](https://gofastmcp.com/integrations/authkit), [GitHub](https://gofastmcp.com/integrations/github), [Google](https://gofastmcp.com/integrations/google), [Azure](https://gofastmcp.com/integrations/azure) (Entra ID), [AWS Cognito](https://gofastmcp.com/integrations/aws-cognito), [Auth0](https://gofastmcp.com/integrations/auth0), [Descope](https://gofastmcp.com/integrations/descope), [Scalekit](https://gofastmcp.com/integrations/scalekit), [JWTs](https://gofastmcp.com/servers/auth/token-verification#jwt-token-verification), and [RFC 7662 token introspection](https://gofastmcp.com/servers/auth/token-verification#token-introspection-protocol).

    ⚡ **Response Caching Middleware** dramatically improves performance for expensive operations, while **Server lifespans** provide proper initialization and cleanup hooks that run once per server instance instead of per client session.

    ✨ **Developer experience improvements** include Pydantic input validation, icon support, RFC 6570 query parameters for resource templates, improved Context API methods, and async file/directory resources.
  </Card>
</Update>

<Update label="FastMCP 2.12.5" description="October 17, 2025">
  <Card title="FastMCP 2.12.5: Safety Pin" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.12.5">
    Pins MCP SDK version below 1.17 to ensure the `.well-known` payload appears in the expected location when using FastMCP auth providers with composite applications.
  </Card>
</Update>

<Update label="FastMCP 2.12.4" description="September 26, 2025">
  <Card title="FastMCP 2.12.4: OIDC What You Did There" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.12.4">
    FastMCP 2.12.4 adds comprehensive OIDC support and expands authentication options with AWS Cognito and Descope providers. The release also includes improvements to logging middleware, URL handling for nested resources, persistent OAuth client registration storage, and various fixes to the experimental OpenAPI parser.

    🔐 **OIDC Configuration** brings native support for OpenID Connect, enabling seamless integration with enterprise identity providers.

    🏢 **Enterprise Authentication** expands with AWS Cognito and Descope providers, broadening the authentication ecosystem.

    🛠️ **Improved Reliability** through enhanced URL handling, persistent OAuth storage, and numerous parser fixes based on community feedback.
  </Card>
</Update>

<Update label="FastMCP 2.12.3" description="September 17, 2025">
  <Card title="FastMCP 2.12.3: Double Time" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.12.3">
    FastMCP 2.12.3 focuses on performance and developer experience improvements. This release includes optimized auth provider imports that reduce server startup time, enhanced OIDC authentication flows, and automatic inline snapshot creation for testing.
  </Card>
</Update>

<Update label="FastMCP 2.12.2" description="September 3, 2025">
  <Card title="FastMCP 2.12.2: Perchance to Stream" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.12.2">
    Hotfix for streamable-http transport validation in fastmcp.json configuration files, resolving a parsing error when CLI arguments were merged against the configuration spec.
  </Card>
</Update>

<Update label="FastMCP 2.12.1" description="September 3, 2025">
  <Card title="FastMCP 2.12.1: OAuth to Joy" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.12.1">
    FastMCP 2.12.1 strengthens OAuth proxy implementation with improved client storage reliability, PKCE forwarding, configurable token endpoint authentication methods, and expanded scope handling based on extensive community testing.
  </Card>
</Update>

<Update label="FastMCP 2.12" description="August 31, 2025">
  <Card title="FastMCP 2.12: Auth to the Races" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.12.0">
    FastMCP 2.12 represents one of our most significant releases to date. After extensive testing and iteration with the community, we're shipping major improvements to authentication, configuration, and MCP feature adoption.

    🔐 **OAuth Proxy** bridges the gap for providers that don't support Dynamic Client Registration, enabling authentication with GitHub, Google, WorkOS, and Azure through minimal configuration.

    📋 **Declarative JSON Configuration** introduces `fastmcp.json` as the single source of truth for server settings, making MCP servers as portable and shareable as container images.

    🧠 **Sampling API Fallback** tackles adoption challenges by letting servers generate completions server-side when clients don't support the feature, encouraging innovation while maintaining compatibility.
  </Card>
</Update>

<Update label="FastMCP 2.11" description="August 1, 2025">
  <Card title="FastMCP 2.11: Auth to a Good Start" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.11.0">
    FastMCP 2.11 brings enterprise-ready authentication and dramatic performance improvements.

    🔒 **Comprehensive OAuth 2.1 Support** with WorkOS AuthKit integration, Dynamic Client Registration, and support for separate resource and authorization servers.

    ⚡ **Experimental OpenAPI Parser** delivers dramatic performance gains through single-pass schema processing and optimized memory usage (enable with environment variable).

    💾 **Enhanced State Management** provides persistent state across tool calls with a simple dictionary interface, improving context handling and type annotations.

    This release emphasizes speed and simplicity while setting the foundation for future enterprise features.
  </Card>
</Update>

<Update label="FastMCP 2.10" description="July 2, 2025">
  <Card title="FastMCP 2.10: Great Spec-tations" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.10.0">
    FastMCP 2.10 achieves full compliance with the 6/18/2025 MCP specification update, introducing powerful new communication patterns.

    💬 **Elicitation Support** enables dynamic server-client communication and "human-in-the-loop" workflows, allowing servers to request additional information during execution.

    📊 **Output Schemas** provide structured outputs for tools, making results more predictable and easier to parse programmatically.

    🛠️ **Enhanced HTTP Routing** with OpenAPI extensions support and configurable algorithms for more flexible API integration.

    This release includes a breaking change to `client.call_tool()` return signatures but significantly expands the interaction capabilities of MCP servers.
  </Card>
</Update>

<Update label="FastMCP 2.9" description="June 23, 2025">
  <Card title="FastMCP 2.9: MCP-Native Middleware" href="https://www.jlowin.dev/blog/fastmcp-2-9-middleware">
    FastMCP 2.9 is a major release that, among other things, introduces two important features that push beyond the basic MCP protocol.

    🤝 *MCP Middleware* brings a flexible middleware system for intercepting and controlling server operations - think authentication, logging, rate limiting, and custom business logic without touching core protocol code.

    ✨ *Server-side type conversion* for prompts solves a major developer pain point: while MCP requires string arguments, your functions can now work with native Python types like lists and dictionaries, with automatic conversion handling the complexity.

    These features transform FastMCP from a simple protocol implementation into a powerful framework for building sophisticated MCP applications. Combined with the new `File` utility for binary data and improvements to authentication and serialization, this release makes FastMCP significantly more flexible and developer-friendly while maintaining full protocol compliance.
  </Card>
</Update>

<Update label="FastMCP 2.8" description="June 11, 2025">
  <Card title="FastMCP 2.8: Transform and Roll Out" href="https://www.jlowin.dev/blog/fastmcp-2-8-tool-transformation">
    FastMCP 2.8 is here, and it's all about taking control of your tools.

    This release is packed with new features for curating the perfect LLM experience:

    🛠️ Tool Transformation

    The headline feature lets you wrap any tool—from your own code, a third-party library, or an OpenAPI spec—to create an enhanced, LLM-friendly version. You can rename arguments, rewrite descriptions, and hide parameters without touching the original code.

    This feature was developed in close partnership with Bill Easton. As Bill brilliantly [put it](https://www.linkedin.com/posts/williamseaston_huge-thanks-to-william-easton-for-providing-activity-7338011349525983232-Mw6T?utm_source=share\&utm_medium=member_desktop\&rcm=ACoAAAAd6d0B3uL9zpCsq9eYWKi3HIvb8eN_r_Q), "Tool transformation flips Prompt Engineering on its head: stop writing tool-friendly LLM prompts and start providing LLM-friendly tools."

    🏷️ Component Control

    Now that you're transforming tools, you need a way to hide the old ones! In FastMCP 2.8 you can programmatically enable/disable any component, and for everyone who's been asking what FastMCP's tags are for—they finally have a purpose! You can now use tags to declaratively filter which components are exposed to your clients.

    🚀 Pragmatic by Default

    Lastly, to ensure maximum compatibility with the ecosystem, we've made the pragmatic decision to default all OpenAPI routes to Tools, making your entire API immediately accessible to any tool-using agent. When the industry catches up and supports resources, we'll restore the old default -- but no reason you should do extra work before OpenAI, Anthropic, or Google!
  </Card>
</Update>

<Update label="FastMCP 2.7" description="June 6, 2025">
  <Card title="FastMCP 2.7: Pare Programming" href="https://github.com/PrefectHQ/fastmcp/releases/tag/v2.7.0">
    FastMCP 2.7 has been released!

    Most notably, it introduces the highly requested (and Pythonic) "naked" decorator usage:

    ```python {3} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
    mcp = FastMCP()

    @mcp.tool
    def add(a: int, b: int) -> int:
        return a + b
    ```

    In addition, decorators now return the objects they create, instead of the decorated function. This is an important usability enhancement.

    The bulk of the update is focused on improving the FastMCP internals, including a few breaking internal changes to private APIs. A number of functions that have clung on since 1.0 are now deprecated.
  </Card>
</Update>

<Update label="FastMCP 2.6" description="June 2, 2025">
  <Card title="FastMCP 2.6: Blast Auth" href="https://www.jlowin.dev/blog/fastmcp-2-6">
    FastMCP 2.6 is here!

    This release introduces first-class authentication for MCP servers and clients, including pragmatic Bearer token support and seamless OAuth 2.1 integration. This release aligns with how major AI platforms are adopting MCP today, making it easier than ever to securely connect your tools to real-world AI models. Dive into the update and secure your stack with minimal friction.
  </Card>
</Update>

<Update description="May 21, 2025" label="Vibe-Testing">
  <Card title="Stop Vibe-Testing Your MCP Server" href="https://www.jlowin.dev/blog/stop-vibe-testing-mcp-servers">
    Your tests are bad and you should feel bad.

    Stop vibe-testing your MCP server through LLM guesswork. FastMCP 2.0 introduces in-memory testing for fast, deterministic, and fully Pythonic validation of your MCP logic—no network, no subprocesses, no vibes.
  </Card>
</Update>

<Update description="May 8, 2025" label="10,000 Stars">
  <Card title="Reflecting on FastMCP at 10k stars 🌟" href="https://www.jlowin.dev/blog/fastmcp-2-10k-stars">
    In just six weeks since its relaunch, FastMCP has surpassed 10,000 GitHub stars—becoming the fastest-growing OSS project in our orbit. What started as a personal itch has become the backbone of Python-based MCP servers, powering a rapidly expanding ecosystem. While the protocol itself evolves, FastMCP continues to lead with clarity, developer experience, and opinionated tooling. Here’s to what’s next.
  </Card>
</Update>

<Update description="May 8, 2025" label="FastMCP 2.3">
  <Card title="Now Streaming: FastMCP 2.3" href="https://www.jlowin.dev/blog/fastmcp-2-3-streamable-http">
    FastMCP 2.3 introduces full support for Streamable HTTP, a modern alternative to SSE that simplifies MCP deployments over the web. It’s efficient, reliable, and now the default HTTP transport. Just run your server with transport="http" and connect clients via a standard URL—FastMCP handles the rest. No special setup required. This release makes deploying MCP servers easier and more portable than ever.
  </Card>
</Update>

<Update description="April 23, 2025" label="Proxy Servers">
  <Card title="MCP Proxy Servers with FastMCP 2.0" href="https://www.jlowin.dev/blog/fastmcp-proxy">
    Even AI needs a good travel adapter 🔌

    FastMCP now supports proxying arbitrary MCP servers, letting you run a local FastMCP instance that transparently forwards requests to any remote or third-party server—regardless of transport. This enables transport bridging (e.g., stdio ⇄ SSE), simplified client configuration, and powerful gateway patterns. Proxies are fully composable with other FastMCP servers, letting you mount or import them just like local servers. Use `FastMCP.from_client()` to wrap any backend in a clean, Pythonic proxy.
  </Card>
</Update>

<Update label="FastMCP 2.0" description="April 16, 2025">
  <Card title="Introducing FastMCP 2.0 🚀" href="https://www.jlowin.dev/blog/fastmcp-2">
    This major release reimagines FastMCP as a full ecosystem platform, with powerful new features for composition, integration, and client interaction. You can now compose local and remote servers, proxy arbitrary MCP servers (with transport translation), and generate MCP servers from OpenAPI or FastAPI apps. A new client infrastructure supports advanced workflows like LLM sampling.

    FastMCP 2.0 builds on the success of v1 with a cleaner, more flexible foundation—try it out today!
  </Card>
</Update>

<Update label="Official SDK" description="December 3, 2024">
  <Card title="FastMCP is joining the official MCP Python SDK!" href="https://bsky.app/profile/jlowin.dev/post/3lch4xk5cf22c" icon="sparkles">
    FastMCP 1.0 will become part of the official MCP Python SDK!
  </Card>
</Update>

<Update label="FastMCP 1.0" description="December 1, 2024">
  <Card title="Introducing FastMCP 🚀" href="https://www.jlowin.dev/blog/introducing-fastmcp">
    Because life's too short for boilerplate.

    This is where it all started. FastMCP’s launch post introduced a clean, Pythonic way to build MCP servers without the protocol overhead. Just write functions; FastMCP handles the rest. What began as a weekend project quickly became the foundation of a growing ecosystem.
  </Card>
</Update>


# github
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-github



# `fastmcp.server.auth.providers.github`

GitHub OAuth provider for FastMCP.

This module provides a complete GitHub OAuth integration that's ready to use
with just a client ID and client secret. It handles all the complexity of
GitHub's OAuth flow, token validation, and user management.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider

# Simple GitHub OAuth protection
auth = GitHubProvider(
    client_id="your-github-client-id",
    client_secret="your-github-client-secret"
)

mcp = FastMCP("My Protected Server", auth=auth)
```

## Classes

### `GitHubTokenVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/github.py#L39"><Icon icon="github" /></a></sup>

Token verifier for GitHub OAuth tokens.

GitHub OAuth tokens are opaque (not JWTs), so we verify them
by calling GitHub's API to check if they're valid and get user info.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/github.py#L66"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify GitHub OAuth token by calling GitHub API.

### `GitHubProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/github.py#L153"><Icon icon="github" /></a></sup>

Complete GitHub OAuth provider for FastMCP.

This provider makes it trivial to add GitHub OAuth protection to any
FastMCP server. Just provide your GitHub OAuth app credentials and
a base URL, and you're ready to go.

Features:

* Transparent OAuth proxy to GitHub
* Automatic token validation via GitHub API
* User information extraction
* Minimal configuration required


# google
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-google



# `fastmcp.server.auth.providers.google`

Google OAuth provider for FastMCP.

This module provides a complete Google OAuth integration that's ready to use
with just a client ID and client secret. It handles all the complexity of
Google's OAuth flow, token validation, and user management.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.google import GoogleProvider

# Simple Google OAuth protection
auth = GoogleProvider(
    client_id="your-google-client-id.apps.googleusercontent.com",
    client_secret="your-google-client-secret"
)

mcp = FastMCP("My Protected Server", auth=auth)
```

## Classes

### `GoogleTokenVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/google.py#L40"><Icon icon="github" /></a></sup>

Token verifier for Google OAuth tokens.

Google OAuth tokens are opaque (not JWTs), so we verify them
by calling Google's tokeninfo API to check if they're valid and get user info.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/google.py#L67"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify Google OAuth token by calling Google's tokeninfo API.

### `GoogleProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/google.py#L168"><Icon icon="github" /></a></sup>

Complete Google OAuth provider for FastMCP.

This provider makes it trivial to add Google OAuth protection to any
FastMCP server. Just provide your Google OAuth app credentials and
a base URL, and you're ready to go.

Features:

* Transparent OAuth proxy to Google
* Automatic token validation via Google's tokeninfo API
* User information extraction from Google APIs
* Minimal configuration required


# in_memory
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-in_memory



# `fastmcp.server.auth.providers.in_memory`

## Classes

### `InMemoryOAuthProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L31"><Icon icon="github" /></a></sup>

An in-memory OAuth provider for testing purposes.
It simulates the OAuth 2.1 flow locally without external calls.

**Methods:**

#### `get_client` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L65"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_client(self, client_id: str) -> OAuthClientInformationFull | None
```

#### `register_client` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L68"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_client(self, client_info: OAuthClientInformationFull) -> None
```

#### `authorize` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L92"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str
```

Simulates user authorization and generates an authorization code.
Returns a redirect URI with the code and state.

#### `load_authorization_code` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L149"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str) -> AuthorizationCode | None
```

#### `exchange_authorization_code` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L162"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode) -> OAuthToken
```

#### `load_refresh_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L215"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None
```

#### `exchange_refresh_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L230"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]) -> OAuthToken
```

#### `load_access_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L287"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_access_token(self, token: str) -> AccessToken | None
```

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L298"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify a bearer token and return access info if valid.

This method implements the TokenVerifier protocol by delegating
to our existing load\_access\_token method.

**Args:**

* `token`: The token string to validate

**Returns:**

* AccessToken object if valid, None if invalid or expired

#### `revoke_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/in_memory.py#L355"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
revoke_token(self, token: AccessToken | RefreshToken) -> None
```

Revokes an access or refresh token and its counterpart.


# introspection
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-introspection



# `fastmcp.server.auth.providers.introspection`

OAuth 2.0 Token Introspection (RFC 7662) provider for FastMCP.

This module provides token verification for opaque tokens using the OAuth 2.0
Token Introspection protocol defined in RFC 7662. It allows FastMCP servers to
validate tokens issued by authorization servers that don't use JWT format.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.introspection import IntrospectionTokenVerifier

# Verify opaque tokens via RFC 7662 introspection
verifier = IntrospectionTokenVerifier(
    introspection_url="https://auth.example.com/oauth/introspect",
    client_id="your-client-id",
    client_secret="your-client-secret",
    required_scopes=["read", "write"]
)

mcp = FastMCP("My Protected Server", auth=verifier)
```

## Classes

### `IntrospectionTokenVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/introspection.py#L54"><Icon icon="github" /></a></sup>

OAuth 2.0 Token Introspection verifier (RFC 7662).

This verifier validates opaque tokens by calling an OAuth 2.0 token introspection
endpoint. Unlike JWT verification which is stateless, token introspection requires
a network call to the authorization server for each token validation.

The verifier authenticates to the introspection endpoint using either:

* HTTP Basic Auth (client\_secret\_basic, default): credentials in Authorization header
* POST body authentication (client\_secret\_post): credentials in request body

Both methods are specified in RFC 6749 (OAuth 2.0) and RFC 7662 (Token Introspection).

Use this when:

* Your authorization server issues opaque (non-JWT) tokens
* You need to validate tokens from Auth0, Okta, Keycloak, or other OAuth servers
* Your tokens require real-time revocation checking
* Your authorization server supports RFC 7662 introspection

Caching is disabled by default to preserve real-time revocation semantics.
Set `cache_ttl_seconds` to enable caching and reduce load on the
introspection endpoint (e.g., `cache_ttl_seconds=300` for 5 minutes).

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/introspection.py#L278"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify a bearer token using OAuth 2.0 Token Introspection (RFC 7662).

This method makes a POST request to the introspection endpoint with the token,
authenticated using the configured client authentication method (client\_secret\_basic
or client\_secret\_post).

Results are cached in-memory to reduce load on the introspection endpoint.
Cache TTL and size are configurable via constructor parameters.

**Args:**

* `token`: The opaque token string to validate

**Returns:**

* AccessToken object if valid and active, None if invalid, inactive, or expired


# jwt
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-jwt



# `fastmcp.server.auth.providers.jwt`

TokenVerifier implementations for FastMCP.

## Classes

### `JWKData` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L27"><Icon icon="github" /></a></sup>

JSON Web Key data structure.

### `JWKSData` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L40"><Icon icon="github" /></a></sup>

JSON Web Key Set data structure.

### `RSAKeyPair` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L47"><Icon icon="github" /></a></sup>

RSA key pair for JWT testing.

**Methods:**

#### `generate` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L54"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate(cls) -> RSAKeyPair
```

Generate an RSA key pair for testing.

**Returns:**

* Generated key pair

#### `create_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L89"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_token(self, subject: str = 'fastmcp-user', issuer: str = 'https://fastmcp.example.com', audience: str | list[str] | None = None, scopes: list[str] | None = None, expires_in_seconds: int = 3600, additional_claims: dict[str, Any] | None = None, kid: str | None = None) -> str
```

Generate a test JWT token for testing purposes.

**Args:**

* `subject`: Subject claim (usually user ID)
* `issuer`: Issuer claim
* `audience`: Audience claim - can be a string or list of strings (optional)
* `scopes`: List of scopes to include
* `expires_in_seconds`: Token expiration time in seconds
* `additional_claims`: Any additional claims to include
* `kid`: Key ID to include in header

### `JWTVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L142"><Icon icon="github" /></a></sup>

JWT token verifier supporting both asymmetric (RSA/ECDSA) and symmetric (HMAC) algorithms.

This verifier validates JWT tokens using various signing algorithms:

* **Asymmetric algorithms** (RS256/384/512, ES256/384/512, PS256/384/512):
  Uses public/private key pairs. Ideal for external clients and services where
  only the authorization server has the private key.
* **Symmetric algorithms** (HS256/384/512): Uses a shared secret for both
  signing and verification. Perfect for internal microservices and trusted
  environments where the secret can be securely shared.

Use this when:

* You have JWT tokens issued by an external service (asymmetric)
* You need JWKS support for automatic key rotation (asymmetric)
* You have internal microservices sharing a secret key (symmetric)
* Your tokens contain standard OAuth scopes and claims

**Methods:**

#### `load_access_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L372"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_access_token(self, token: str) -> AccessToken | None
```

Validate a JWT bearer token and return an AccessToken when the token is valid.

**Args:**

* `token`: The JWT bearer token string to validate.

**Returns:**

* AccessToken | None: An AccessToken populated from token claims if the token is valid; `None` if the token is expired, has an invalid signature or format, fails issuer/audience/scope validation, or any other validation error occurs.

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L490"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify a bearer token and return access info if valid.

This method implements the TokenVerifier protocol by delegating
to our existing load\_access\_token method.

**Args:**

* `token`: The JWT token string to validate

**Returns:**

* AccessToken object if valid, None if invalid or expired

### `StaticTokenVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L506"><Icon icon="github" /></a></sup>

Simple static token verifier for testing and development.

This verifier validates tokens against a predefined dictionary of valid token
strings and their associated claims. When a token string matches a key in the
dictionary, the verifier returns the corresponding claims as if the token was
validated by a real authorization server.

Use this when:

* You're developing or testing locally without a real OAuth server
* You need predictable tokens for automated testing
* You want to simulate different users/scopes without complex setup
* You're prototyping and need simple API key-style authentication

WARNING: Never use this in production - tokens are stored in plain text!

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/jwt.py#L540"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify token against static token dictionary.


# oci
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-oci



# `fastmcp.server.auth.providers.oci`

OCI OIDC provider for FastMCP.

The pull request for the provider is submitted to fastmcp.

This module provides OIDC Implementation to integrate MCP servers with OCI.
You only need OCI Identity Domain's discovery URL, client ID, client secret, and base URL.

Post Authentication, you get OCI IAM domain access token. That is not authorized to invoke OCI control plane.
You need to exchange the IAM domain access token for OCI UPST token to invoke OCI control plane APIs.
The sample code below has get\_oci\_signer function that returns OCI TokenExchangeSigner object.
You can use the signer object to create OCI service object.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.oci import OCIProvider
from fastmcp.server.dependencies import get_access_token
from fastmcp.utilities.logging import get_logger

import os

import oci
from oci.auth.signers import TokenExchangeSigner

logger = get_logger(__name__)

# Load configuration from environment
config_url = os.environ.get("OCI_CONFIG_URL")  # OCI IAM Domain OIDC discovery URL
client_id = os.environ.get("OCI_CLIENT_ID")  # Client ID configured for the OCI IAM Domain Integrated Application
client_secret = os.environ.get("OCI_CLIENT_SECRET")  # Client secret configured for the OCI IAM Domain Integrated Application
iam_guid = os.environ.get("OCI_IAM_GUID")  # IAM GUID configured for the OCI IAM Domain

# Simple OCI OIDC protection
auth = OCIProvider(
    config_url=config_url,  # config URL is the OCI IAM Domain OIDC discovery URL
    client_id=client_id,  # This is same as the client ID configured for the OCI IAM Domain Integrated Application
    client_secret=client_secret,  # This is same as the client secret configured for the OCI IAM Domain Integrated Application
    required_scopes=["openid", "profile", "email"],
    redirect_path="/auth/callback",
    base_url="http://localhost:8000",
)

# NOTE: For production use, replace this with a thread-safe cache implementation
# such as threading.Lock-protected dict or a proper caching library
_global_token_cache = {}  # In memory cache for OCI session token signer

def get_oci_signer() -> TokenExchangeSigner:

    authntoken = get_access_token()
    tokenID = authntoken.claims.get("jti")
    token = authntoken.token

    # Check if the signer exists for the token ID in memory cache
    cached_signer = _global_token_cache.get(tokenID)
    logger.debug(f"Global cached signer: {cached_signer}")
    if cached_signer:
        logger.debug(f"Using globally cached signer for token ID: {tokenID}")
        return cached_signer

    # If the signer is not yet created for the token then create new OCI signer object
    logger.debug(f"Creating new signer for token ID: {tokenID}")
    signer = TokenExchangeSigner(
        jwt_or_func=token,
        oci_domain_id=iam_guid.split(".")[0] if iam_guid else None,  # This is same as IAM GUID configured for the OCI IAM Domain
        client_id=client_id,  # This is same as the client ID configured for the OCI IAM Domain Integrated Application
        client_secret=client_secret,  # This is same as the client secret configured for the OCI IAM Domain Integrated Application
    )
    logger.debug(f"Signer {signer} created for token ID: {tokenID}")

    #Cache the signer object in memory cache
    _global_token_cache[tokenID] = signer
    logger.debug(f"Signer cached for token ID: {tokenID}")

    return signer

mcp = FastMCP("My Protected Server", auth=auth)
```

## Classes

### `OCIProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/oci.py#L90"><Icon icon="github" /></a></sup>

An OCI IAM Domain provider implementation for FastMCP.

This provider is a complete OCI integration that's ready to use with
just the configuration URL, client ID, client secret, and base URL.


# propelauth
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-propelauth



# `fastmcp.server.auth.providers.propelauth`

PropelAuth authentication provider for FastMCP.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.propelauth import PropelAuthProvider

auth = PropelAuthProvider(
    auth_url="https://auth.yourdomain.com",
    introspection_client_id="your-client-id",
    introspection_client_secret="your-client-secret",
    base_url="https://your-fastmcp-server.com",
    required_scopes=["read:user_data"],
)

mcp = FastMCP("My App", auth=auth)
```

## Classes

### `PropelAuthTokenIntrospectionOverrides` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/propelauth.py#L36"><Icon icon="github" /></a></sup>

### `PropelAuthProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/propelauth.py#L43"><Icon icon="github" /></a></sup>

PropelAuth resource server provider using OAuth 2.1 token introspection.

This provider validates access tokens via PropelAuth's introspection endpoint
and forwards authorization server metadata for OAuth discovery.

For detailed setup instructions, see:
[https://docs.propelauth.com/mcp-authentication/overview](https://docs.propelauth.com/mcp-authentication/overview)

**Methods:**

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/propelauth.py#L130"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get routes for this provider.

Includes the standard routes from the RemoteAuthProvider (protected resource metadata routes (RFC 9728)),
and creates an authorization server metadata route that forwards to PropelAuth's route

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")
  This is used to advertise the resource URL in metadata.

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/propelauth.py#L174"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify token and check the `aud` claim against the configured resource.


# scalekit
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-scalekit



# `fastmcp.server.auth.providers.scalekit`

Scalekit authentication provider for FastMCP.

This module provides ScalekitProvider - a complete authentication solution that integrates
with Scalekit's OAuth 2.1 and OpenID Connect services, supporting Resource Server
authentication for seamless MCP client authentication.

## Classes

### `ScalekitProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/scalekit.py#L23"><Icon icon="github" /></a></sup>

Scalekit resource server provider for OAuth 2.1 authentication.

This provider implements Scalekit integration using resource server pattern.
FastMCP acts as a protected resource server that validates access tokens issued
by Scalekit's authorization server.

IMPORTANT SETUP REQUIREMENTS:

1. Create an MCP Server in Scalekit Dashboard:
   * Go to your [Scalekit Dashboard](https://app.scalekit.com/)
   * Navigate to MCP Servers section
   * Register a new MCP Server with appropriate scopes
   * Ensure the Resource Identifier matches exactly what you configure as MCP URL
   * Note the Resource ID

2. Environment Configuration:
   * Set SCALEKIT\_ENVIRONMENT\_URL (e.g., [https://your-env.scalekit.com](https://your-env.scalekit.com))
   * Set SCALEKIT\_RESOURCE\_ID from your created resource
   * Set BASE\_URL to your FastMCP server's public URL

For detailed setup instructions, see:
[https://docs.scalekit.com/mcp/overview/](https://docs.scalekit.com/mcp/overview/)

**Methods:**

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/scalekit.py#L145"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get OAuth routes including Scalekit authorization server metadata forwarding.

This returns the standard protected resource routes plus an authorization server
metadata endpoint that forwards Scalekit's OAuth metadata to clients.

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")
  This is used to advertise the resource URL in metadata.


# supabase
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-supabase



# `fastmcp.server.auth.providers.supabase`

Supabase authentication provider for FastMCP.

This module provides SupabaseProvider - a complete authentication solution that integrates
with Supabase Auth's JWT verification, supporting Dynamic Client Registration (DCR)
for seamless MCP client authentication.

## Classes

### `SupabaseProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/supabase.py#L25"><Icon icon="github" /></a></sup>

Supabase metadata provider for DCR (Dynamic Client Registration).

This provider implements Supabase Auth integration using metadata forwarding.
This approach allows Supabase to handle the OAuth flow directly while FastMCP acts
as a resource server, verifying JWTs issued by Supabase Auth.

IMPORTANT SETUP REQUIREMENTS:

1. Supabase Project Setup:
   * Create a Supabase project at [https://supabase.com](https://supabase.com)
   * Note your project URL (e.g., "[https://abc123.supabase.co](https://abc123.supabase.co)")
   * Configure your JWT algorithm in Supabase Auth settings (HS256, RS256, or ES256)
   * Asymmetric keys (RS256/ES256) are recommended for production

2. JWT Verification:
   * FastMCP verifies JWTs using the JWKS endpoint at /.well-known/jwks.json
   * JWTs are issued by&#x20;
   * Default auth\_route is "/auth/v1" (can be customized for self-hosted setups)
   * Tokens are cached for up to 10 minutes by Supabase's edge servers
   * Algorithm must match your Supabase Auth configuration

3. Authorization:
   * Supabase uses Row Level Security (RLS) policies for database authorization
   * OAuth-level scopes are an upcoming feature in Supabase Auth
   * Both approaches will be supported once scope handling is available

For detailed setup instructions, see:
[https://supabase.com/docs/guides/auth/jwts](https://supabase.com/docs/guides/auth/jwts)

**Methods:**

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/supabase.py#L126"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get OAuth routes including Supabase authorization server metadata forwarding.

This returns the standard protected resource routes plus an authorization server
metadata endpoint that forwards Supabase's OAuth metadata to clients.

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")
  This is used to advertise the resource URL in metadata.


# workos
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-providers-workos



# `fastmcp.server.auth.providers.workos`

WorkOS authentication providers for FastMCP.

This module provides two WorkOS authentication strategies:

1. WorkOSProvider - OAuth proxy for WorkOS Connect applications (non-DCR)
2. AuthKitProvider - DCR-compliant provider for WorkOS AuthKit

Choose based on your WorkOS setup and authentication requirements.

## Classes

### `WorkOSTokenVerifier` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/workos.py#L30"><Icon icon="github" /></a></sup>

Token verifier for WorkOS OAuth tokens.

WorkOS AuthKit tokens are opaque, so we verify them by calling
the /oauth2/userinfo endpoint to check validity and get user info.

**Methods:**

#### `verify_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/workos.py#L60"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
verify_token(self, token: str) -> AccessToken | None
```

Verify WorkOS OAuth token by calling userinfo endpoint.

### `WorkOSProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/workos.py#L111"><Icon icon="github" /></a></sup>

Complete WorkOS OAuth provider for FastMCP.

This provider implements WorkOS AuthKit OAuth using the OAuth Proxy pattern.
It provides OAuth2 authentication for users through WorkOS Connect applications.

Features:

* Transparent OAuth proxy to WorkOS AuthKit
* Automatic token validation via userinfo endpoint
* User information extraction from ID tokens
* Support for standard OAuth scopes (openid, profile, email)

Setup Requirements:

1. Create a WorkOS Connect application in your dashboard
2. Note your AuthKit domain (e.g., "[https://your-app.authkit.app](https://your-app.authkit.app)")
3. Configure redirect URI as: [http://localhost:8000/auth/callback](http://localhost:8000/auth/callback)
4. Note your Client ID and Client Secret

### `AuthKitProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/workos.py#L232"><Icon icon="github" /></a></sup>

AuthKit metadata provider for DCR (Dynamic Client Registration).

This provider implements AuthKit integration using metadata forwarding
instead of OAuth proxying. This is the recommended approach for WorkOS DCR
as it allows WorkOS to handle the OAuth flow directly while FastMCP acts
as a resource server.

IMPORTANT SETUP REQUIREMENTS:

1. Enable Dynamic Client Registration in WorkOS Dashboard:
   * Go to Applications → Configuration
   * Toggle "Dynamic Client Registration" to enabled

2. Configure your FastMCP server URL as a callback:
   * Add your server URL to the Redirects tab in WorkOS dashboard
   * Example: [https://your-fastmcp-server.com/oauth2/callback](https://your-fastmcp-server.com/oauth2/callback)

For detailed setup instructions, see:
[https://workos.com/docs/authkit/mcp/integrating/token-verification](https://workos.com/docs/authkit/mcp/integrating/token-verification)

**Methods:**

#### `get_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/providers/workos.py#L319"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_routes(self, mcp_path: str | None = None) -> list[Route]
```

Get OAuth routes including AuthKit authorization server metadata forwarding.

This returns the standard protected resource routes plus an authorization server
metadata endpoint that forwards AuthKit's OAuth metadata to clients.

**Args:**

* `mcp_path`: The path where the MCP endpoint is mounted (e.g., "/mcp")
  This is used to advertise the resource URL in metadata.


# redirect_validation
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-redirect_validation



# `fastmcp.server.auth.redirect_validation`

Utilities for validating client redirect URIs in OAuth flows.

This module provides secure redirect URI validation with wildcard support,
protecting against userinfo-based bypass attacks like [http://localhost@evil.com](http://localhost@evil.com).

## Functions

### `matches_allowed_pattern` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/redirect_validation.py#L121"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
matches_allowed_pattern(uri: str, pattern: str) -> bool
```

Securely check if a URI matches an allowed pattern with wildcard support.

This function parses both the URI and pattern as URLs, comparing each
component separately to prevent bypass attacks like userinfo injection.

Patterns support wildcards:

* [http://localhost](http://localhost):\* matches any localhost port
* [http://127.0.0.1](http://127.0.0.1):\* matches any 127.0.0.1 port
* https\://*.example.com/* matches any subdomain of example.com
* [https://app.example.com/auth/](https://app.example.com/auth/)\* matches any path under /auth/

Security: Rejects URIs with userinfo (user:pass\@host) which could bypass
naive string matching (e.g., [http://localhost@evil.com](http://localhost@evil.com)).

**Args:**

* `uri`: The redirect URI to validate
* `pattern`: The allowed pattern (may contain wildcards)

**Returns:**

* True if the URI matches the pattern

### `validate_redirect_uri` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/redirect_validation.py#L175"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_redirect_uri(redirect_uri: str | AnyUrl | None, allowed_patterns: list[str] | None) -> bool
```

Validate a redirect URI against allowed patterns.

**Args:**

* `redirect_uri`: The redirect URI to validate
* `allowed_patterns`: List of allowed patterns. If None, all URIs are allowed (for DCR compatibility).
  If empty list, no URIs are allowed.
  To restrict to localhost only, explicitly pass DEFAULT\_LOCALHOST\_PATTERNS.

**Returns:**

* True if the redirect URI is allowed


# ssrf
Source: https://gofastmcp.com/python-sdk/fastmcp-server-auth-ssrf



# `fastmcp.server.auth.ssrf`

SSRF-safe HTTP utilities for FastMCP.

This module provides SSRF-protected HTTP fetching with:

* DNS resolution and IP validation before requests
* DNS pinning to prevent rebinding TOCTOU attacks
* Support for both CIMD and JWKS fetches

## Functions

### `format_ip_for_url` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L26"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
format_ip_for_url(ip_str: str) -> str
```

Format IP address for use in URL (bracket IPv6 addresses).

IPv6 addresses must be bracketed in URLs to distinguish the address from
the port separator. For example: https\://\[2001:db8::1]:443/path

**Args:**

* `ip_str`: IP address string

**Returns:**

* IP string suitable for URL (IPv6 addresses are bracketed)

### `is_ip_allowed` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L55"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_ip_allowed(ip_str: str) -> bool
```

Check if an IP address is allowed (must be globally routable unicast).

Uses ip.is\_global which catches:

* Private (10.x, 172.16-31.x, 192.168.x)
* Loopback (127.x, ::1)
* Link-local (169.254.x, fe80::) - includes AWS metadata!
* Reserved, unspecified
* RFC6598 Carrier-Grade NAT (100.64.0.0/10) - can point to internal networks

Additionally blocks multicast addresses (not caught by is\_global).

**Args:**

* `ip_str`: IP address string to check

**Returns:**

* True if the IP is allowed (public unicast internet), False if blocked

### `resolve_hostname` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L98"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resolve_hostname(hostname: str, port: int = 443) -> list[str]
```

Resolve hostname to IP addresses using DNS.

**Args:**

* `hostname`: Hostname to resolve
* `port`: Port number (used for getaddrinfo)

**Returns:**

* List of resolved IP addresses

**Raises:**

* `SSRFError`: If resolution fails

### `validate_url` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L147"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_url(url: str, require_path: bool = False) -> ValidatedURL
```

Validate URL for SSRF and resolve to IPs.

**Args:**

* `url`: URL to validate
* `require_path`: If True, require non-root path (for CIMD)

**Returns:**

* ValidatedURL with resolved IPs

**Raises:**

* `SSRFError`: If URL is invalid or resolves to blocked IPs

### `ssrf_safe_fetch` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L196"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ssrf_safe_fetch(url: str) -> bytes
```

Fetch URL with comprehensive SSRF protection and DNS pinning.

Security measures:

1. HTTPS only
2. DNS resolution with IP validation
3. Connects to validated IP directly (DNS pinning prevents rebinding)
4. Response size limit
5. Redirects disabled
6. Overall timeout

**Args:**

* `url`: URL to fetch
* `require_path`: If True, require non-root path
* `max_size`: Maximum response size in bytes (default 5KB)
* `timeout`: Per-operation timeout in seconds
* `overall_timeout`: Overall timeout for entire operation

**Returns:**

* Response body as bytes

**Raises:**

* `SSRFError`: If SSRF validation fails
* `SSRFFetchError`: If fetch fails

### `ssrf_safe_fetch_response` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L239"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ssrf_safe_fetch_response(url: str) -> SSRFFetchResponse
```

Fetch URL with SSRF protection and return response metadata.

This is equivalent to :func:`ssrf_safe_fetch` but returns response headers
and status code, and supports conditional request headers.

## Classes

### `SSRFError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L47"><Icon icon="github" /></a></sup>

Raised when an SSRF protection check fails.

### `SSRFFetchError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L51"><Icon icon="github" /></a></sup>

Raised when SSRF-safe fetch fails.

### `ValidatedURL` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L128"><Icon icon="github" /></a></sup>

A URL that has been validated for SSRF with resolved IPs.

### `SSRFFetchResponse` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/auth/ssrf.py#L139"><Icon icon="github" /></a></sup>

Response payload from an SSRF-safe fetch.
