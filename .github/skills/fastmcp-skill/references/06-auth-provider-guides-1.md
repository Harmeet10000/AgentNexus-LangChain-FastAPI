# Auth Provider Guides I

Source lines: 6313-7346 from the original FastMCP documentation dump.

Auth0, AuthKit, AWS Cognito, and Azure provider setup and production notes.

---

# Auth0 OAuth 🤝 FastMCP
Source: https://gofastmcp.com/integrations/auth0

Secure your FastMCP server with Auth0 OAuth

<VersionBadge />

This guide shows you how to secure your FastMCP server using **Auth0 OAuth**. While Auth0 does have support for Dynamic Client Registration, it is not enabled by default so this integration uses the [**OIDC Proxy**](/servers/auth/oidc-proxy) pattern to bridge Auth0's dynamic OIDC configuration with MCP's authentication requirements.

## Configuration

### Prerequisites

Before you begin, you will need:

1. An **[Auth0 Account](https://auth0.com/)** with access to create Applications
2. Your FastMCP server's URL (can be localhost for development, e.g., `http://localhost:8000`)

### Step 1: Create an Auth0 Application

Create an Application in your Auth0 settings to get the credentials needed for authentication:

<Steps>
  <Step title="Navigate to Applications">
    Go to **Applications → Applications** in your Auth0 account.

    Click **"+ Create Application"** to create a new application.
  </Step>

  <Step title="Create Your Application">
    * **Name**: Choose a name users will recognize (e.g., "My FastMCP Server")
    * **Choose an application type**: Choose "Single Page Web Applications"
    * Click **Create** to create the application
  </Step>

  <Step title="Configure Your Application">
    Select the "Settings" tab for your application, then find the "Application URIs" section.

    * **Allowed Callback URLs**: Your server URL + `/auth/callback` (e.g., `http://localhost:8000/auth/callback`)
    * Click **Save** to save your changes

    <Warning>
      The callback URL must match exactly. The default path is `/auth/callback`, but you can customize it using the `redirect_path` parameter.
    </Warning>

    <Tip>
      If you want to use a custom callback path (e.g., `/auth/auth0/callback`), make sure to set the same path in both your Auth0 Application settings and the `redirect_path` parameter when configuring the Auth0Provider.
    </Tip>
  </Step>

  <Step title="Save Your Credentials">
    After creating the app, in the "Basic Information" section you'll see:

    * **Client ID**: A public identifier like `tv2ObNgaZAWWhhycr7Bz1LU2mxlnsmsB`
    * **Client Secret**: A private hidden value that should always be stored securely

    <Tip>
      Store these credentials securely. Never commit them to version control. Use environment variables or a secrets manager in production.
    </Tip>
  </Step>

  <Step title="Select Your Audience">
    Go to **Applications → APIs** in your Auth0 account.

    * Find the API that you want to use for your application
    * **API Audience**: A URL that uniquely identifies the API

    <Tip>
      Store this along with of the credentials above. Never commit this to version control. Use environment variables or a secrets manager in production.
    </Tip>
  </Step>
</Steps>

### Step 2: FastMCP Configuration

Create your FastMCP server using the `Auth0Provider`.

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.auth0 import Auth0Provider

# The Auth0Provider utilizes Auth0 OIDC configuration
auth_provider = Auth0Provider(
    config_url="https://.../.well-known/openid-configuration",  # Your Auth0 configuration URL
    client_id="tv2ObNgaZAWWhhycr7Bz1LU2mxlnsmsB",               # Your Auth0 application Client ID
    client_secret="vPYqbjemq...",                               # Your Auth0 application Client Secret
    audience="https://...",                                     # Your Auth0 API audience
    base_url="http://localhost:8000",                           # Must match your application configuration
    # redirect_path="/auth/callback"                            # Default value, customize if needed
)

mcp = FastMCP(name="Auth0 Secured App", auth=auth_provider)

# Add a protected tool to test authentication
@mcp.tool
async def get_token_info() -> dict:
    """Returns information about the Auth0 token."""
    from fastmcp.server.dependencies import get_access_token

    token = get_access_token()

    return {
        "issuer": token.claims.get("iss"),
        "audience": token.claims.get("aud"),
        "scope": token.claims.get("scope")
    }
```

## Testing

### Running the Server

Start your FastMCP server with HTTP transport to enable OAuth flows:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py --transport http --port 8000
```

Your server is now running and protected by Auth0 authentication.

### Testing with a Client

Create a test client that authenticates with your Auth0-protected server:

```python test_client.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
import asyncio

async def main():
    # The client will automatically handle Auth0 OAuth flows
    async with Client("http://localhost:8000/mcp", auth="oauth") as client:
        # First-time connection will open Auth0 login in your browser
        print("✓ Authenticated with Auth0!")

        # Test the protected tool
        result = await client.call_tool("get_token_info")
        print(f"Auth0 audience: {result['audience']}")

if __name__ == "__main__":
    asyncio.run(main())
```

When you run the client for the first time:

1. Your browser will open to Auth0's authorization page
2. After you authorize the app, you'll be redirected back
3. The client receives the token and can make authenticated requests

## Production Configuration

<VersionBadge />

For production deployments with persistent token management across server restarts, configure `jwt_signing_key`, and `client_storage`:

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import os
from fastmcp import FastMCP
from fastmcp.server.auth.providers.auth0 import Auth0Provider
from key_value.aio.stores.redis import RedisStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from cryptography.fernet import Fernet

# Production setup with encrypted persistent token storage
auth_provider = Auth0Provider(
    config_url="https://.../.well-known/openid-configuration",
    client_id="tv2ObNgaZAWWhhycr7Bz1LU2mxlnsmsB",
    client_secret="vPYqbjemq...",
    audience="https://...",
    base_url="https://your-production-domain.com",

    # Production token management
    jwt_signing_key=os.environ["JWT_SIGNING_KEY"],
    client_storage=FernetEncryptionWrapper(
        key_value=RedisStore(
            host=os.environ["REDIS_HOST"],
            port=int(os.environ["REDIS_PORT"])
        ),
        fernet=Fernet(os.environ["STORAGE_ENCRYPTION_KEY"])
    )
)

mcp = FastMCP(name="Production Auth0 App", auth=auth_provider)
```

<Note>
  Parameters (`jwt_signing_key` and `client_storage`) work together to ensure tokens and client registrations survive server restarts. **Wrap your storage in `FernetEncryptionWrapper` to encrypt sensitive OAuth tokens at rest** - without it, tokens are stored in plaintext. Store secrets in environment variables and use a persistent storage backend like Redis for distributed deployments.

  For complete details on these parameters, see the [OAuth Proxy documentation](/servers/auth/oauth-proxy#configuration-parameters).
</Note>

<Info>
  The client caches tokens locally, so you won't need to re-authenticate for subsequent runs unless the token expires or you explicitly clear the cache.
</Info>


# AuthKit 🤝 FastMCP
Source: https://gofastmcp.com/integrations/authkit

Secure your FastMCP server with AuthKit by WorkOS

<VersionBadge />

This guide shows you how to secure your FastMCP server using WorkOS's **AuthKit**, a complete authentication and user management solution. This integration uses the [**Remote OAuth**](/servers/auth/remote-oauth) pattern, where AuthKit handles user login and your FastMCP server validates the tokens.

<Warning>
  AuthKit does not currently support [RFC 8707](https://www.rfc-editor.org/rfc/rfc8707.html) resource indicators, so FastMCP cannot validate that tokens were issued for the specific resource server. If you need resource-specific audience validation, consider using [WorkOSProvider](/integrations/workos) (OAuth proxy pattern) instead.
</Warning>

## Configuration

### Prerequisites

Before you begin, you will need:

1. A **[WorkOS Account](https://workos.com/)** and a new **Project**.
2. An **[AuthKit](https://www.authkit.com/)** instance configured within your WorkOS project.
3. Your FastMCP server's URL (can be localhost for development, e.g., `http://localhost:8000`).

### Step 1: AuthKit Configuration

In your WorkOS Dashboard, enable AuthKit and configure the following settings:

<Steps>
  <Step title="Enable Dynamic Client Registration">
    Go to **Applications → Configuration** and enable **Dynamic Client Registration**. This allows MCP clients register with your application automatically.

    <img alt="Enable Dynamic Client Registration" />
  </Step>

  <Step title="Note Your AuthKit Domain">
    Find your **AuthKit Domain** on the configuration page. It will look like `https://your-project-12345.authkit.app`. You'll need this for your FastMCP server configuration.
  </Step>
</Steps>

### Step 2: FastMCP Configuration

Create your FastMCP server file and use the `AuthKitProvider` to handle all the OAuth integration automatically:

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.workos import AuthKitProvider

# The AuthKitProvider automatically discovers WorkOS endpoints
# and configures JWT token validation
auth_provider = AuthKitProvider(
    authkit_domain="https://your-project-12345.authkit.app",
    base_url="http://localhost:8000"  # Use your actual server URL
)

mcp = FastMCP(name="AuthKit Secured App", auth=auth_provider)
```

## Testing

To test your server, you can use the `fastmcp` CLI to run it locally. Assuming you've saved the above code to `server.py` (after replacing the `authkit_domain` and `base_url` with your actual values!), you can run the following command:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py --transport http --port 8000
```

AuthKit defaults DCR clients to `client_secret_basic` for token exchange, which conflicts with how some MCP clients send credentials. To avoid token exchange errors, register as a public client by setting `token_endpoint_auth_method` to `"none"`:

```python client.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.auth import OAuth
import asyncio

auth = OAuth(additional_client_metadata={"token_endpoint_auth_method": "none"})

async def main():
    async with Client("http://localhost:8000/mcp", auth=auth) as client:
        assert await client.ping()

if __name__ == "__main__":
    asyncio.run(main())
```

## Production Configuration

For production deployments, load sensitive configuration from environment variables:

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import os
from fastmcp import FastMCP
from fastmcp.server.auth.providers.workos import AuthKitProvider

# Load configuration from environment variables
auth = AuthKitProvider(
    authkit_domain=os.environ.get("AUTHKIT_DOMAIN"),
    base_url=os.environ.get("BASE_URL", "https://your-server.com")
)

mcp = FastMCP(name="AuthKit Secured App", auth=auth)
```


# AWS Cognito OAuth 🤝 FastMCP
Source: https://gofastmcp.com/integrations/aws-cognito

Secure your FastMCP server with AWS Cognito user pools

<VersionBadge />

This guide shows you how to secure your FastMCP server using **AWS Cognito user pools**. Since AWS Cognito doesn't support Dynamic Client Registration, this integration uses the [**OAuth Proxy**](/servers/auth/oauth-proxy) pattern to bridge AWS Cognito's traditional OAuth with MCP's authentication requirements. It also includes robust JWT token validation, ensuring enterprise-grade authentication.

## Configuration

### Prerequisites

Before you begin, you will need:

1. An **[AWS Account](https://aws.amazon.com/)** with access to create AWS Cognito user pools
2. Basic familiarity with AWS Cognito concepts (user pools, app clients)
3. Your FastMCP server's URL (can be localhost for development, e.g., `http://localhost:8000`)

### Step 1: Create an AWS Cognito User Pool and App Client

Set up AWS Cognito user pool with an app client to get the credentials needed for authentication:

<Steps>
  <Step title="Navigate to AWS Cognito">
    Go to the **[AWS Cognito Console](https://console.aws.amazon.com/cognito/)** and ensure you're in your desired AWS region.

    Select **"User pools"** from the side navigation (click on the hamburger icon at the top left in case you don't see any), and click **"Create user pool"** to create a new user pool.
  </Step>

  <Step title="Define Your Application">
    AWS Cognito now provides a streamlined setup experience:

    1. **Application type**: Select **"Traditional web application"** (this is the correct choice for FastMCP server-side authentication)
    2. **Name your application**: Enter a descriptive name (e.g., `FastMCP Server`)

    The traditional web application type automatically configures:

    * Server-side authentication with client secrets
    * Authorization code grant flow
    * Appropriate security settings for confidential clients

    <Info>
      Choose "Traditional web application" rather than SPA, Mobile app, or Machine-to-machine options. This ensures proper OAuth 2.0 configuration for FastMCP.
    </Info>
  </Step>

  <Step title="Configure Options">
    AWS will guide you through configuration options:

    * **Sign-in identifiers**: Choose how users will sign in (email, username, or phone)
    * **Required attributes**: Select any additional user information you need
    * **Return URL**: Add your callback URL (e.g., `http://localhost:8000/auth/callback` for development)

    <Tip>
      The simplified interface handles most OAuth security settings automatically based on your application type selection.
    </Tip>
  </Step>

  <Step title="Review and Create">
    Review your configuration and click **"Create user pool"**.

    After creation, you'll see your user pool details. Save these important values:

    * **User pool ID** (format: `eu-central-1_XXXXXXXXX`)
    * **Client ID** (found under → "Applications" → "App clients" in the side navigation → \<Your application name, e.g., `FastMCP Server`> → "App client information")
    * **Client Secret** (found under → "Applications" → "App clients" in the side navigation → \<Your application name, e.g., `FastMCP Server`> → "App client information")

    <Tip>
      The user pool ID and app client credentials are all you need for FastMCP configuration.
    </Tip>
  </Step>

  <Step title="Configure OAuth Settings">
    Under "Login pages" in your app client's settings, you can double check and adjust the OAuth configuration:

    * **Allowed callback URLs**: Add your server URL + `/auth/callback` (e.g., `http://localhost:8000/auth/callback`)
    * **Allowed sign-out URLs**: Optional, for logout functionality
    * **OAuth 2.0 grant types**: Ensure "Authorization code grant" is selected
    * **OpenID Connect scopes**: Select scopes your application needs (e.g., `openid`, `email`, `profile`)

    <Tip>
      For local development, you can use `http://localhost` URLs. For production, you must use HTTPS.
    </Tip>
  </Step>

  <Step title="Configure Resource Server">
    AWS Cognito requires a resource server entry to support OAuth with protected resources. Without this, token exchange will fail with an `invalid_grant` error.

    Navigate to **"Branding" → "Domain"** in the side navigation, then:

    1. Click **"Create resource server"**
    2. **Resource server name**: Enter a descriptive name (e.g., `My MCP Server`)
    3. **Resource server identifier**: Enter your MCP endpoint URL exactly as it will be accessed (e.g., `http://localhost:8000/mcp` for development, or `https://your-server.com/mcp` for production)
    4. Click **"Create resource server"**

    <Warning>
      The resource server identifier must exactly match your `base_url + mcp_path`. For the default configuration with `base_url="http://localhost:8000"` and `path="/mcp"`, use `http://localhost:8000/mcp`.
    </Warning>
  </Step>

  <Step title="Save Your Credentials">
    After setup, you'll have:

    * **User Pool ID**: Format like `eu-central-1_XXXXXXXXX`
    * **Client ID**: Your application's client identifier
    * **Client Secret**: Generated client secret (keep secure)
    * **AWS Region**: Where Your AWS Cognito user pool is located

    <Tip>
      Store these credentials securely. Never commit them to version control. Use environment variables or AWS Secrets Manager in production.
    </Tip>
  </Step>
</Steps>

### Step 2: FastMCP Configuration

Create your FastMCP server using the `AWSCognitoProvider`, which handles AWS Cognito's JWT tokens and user claims automatically:

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.aws import AWSCognitoProvider
from fastmcp.server.dependencies import get_access_token

# The AWSCognitoProvider handles JWT validation and user claims
auth_provider = AWSCognitoProvider(
    user_pool_id="eu-central-1_XXXXXXXXX",   # Your AWS Cognito user pool ID
    aws_region="eu-central-1",               # AWS region (defaults to eu-central-1)
    client_id="your-app-client-id",          # Your app client ID
    client_secret="your-app-client-secret",  # Your app client Secret
    base_url="http://localhost:8000",        # Must match your callback URL
    # redirect_path="/auth/callback"         # Default value, customize if needed
)

mcp = FastMCP(name="AWS Cognito Secured App", auth=auth_provider)

# Add a protected tool to test authentication
@mcp.tool
async def get_access_token_claims() -> dict:
    """Get the authenticated user's access token claims."""
    token = get_access_token()
    return {
        "sub": token.claims.get("sub"),
        "username": token.claims.get("username"),
        "cognito:groups": token.claims.get("cognito:groups", []),
    }
```

## Testing

### Running the Server

Start your FastMCP server with HTTP transport to enable OAuth flows:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py --transport http --port 8000
```

Your server is now running and protected by AWS Cognito OAuth authentication.

### Testing with a Client

Create a test client that authenticates with Your AWS Cognito-protected server:

```python test_client.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
import asyncio

async def main():
    # The client will automatically handle AWS Cognito OAuth
    async with Client("http://localhost:8000/mcp", auth="oauth") as client:
        # First-time connection will open AWS Cognito login in your browser
        print("✓ Authenticated with AWS Cognito!")

        # Test the protected tool
        print("Calling protected tool: get_access_token_claims")
        result = await client.call_tool("get_access_token_claims")
        user_data = result.data
        print("Available access token claims:")
        print(f"- sub: {user_data.get('sub', 'N/A')}")
        print(f"- username: {user_data.get('username', 'N/A')}")
        print(f"- cognito:groups: {user_data.get('cognito:groups', [])}")

if __name__ == "__main__":
    asyncio.run(main())
```

When you run the client for the first time:

1. Your browser will open to AWS Cognito's hosted UI login page
2. After you sign in (or sign up), you'll be redirected back to your MCP server
3. The client receives the JWT token and can make authenticated requests

<Info>
  The client caches tokens locally, so you won't need to re-authenticate for subsequent runs unless the token expires or you explicitly clear the cache.
</Info>

## Production Configuration

<VersionBadge />

For production deployments with persistent token management across server restarts, configure `jwt_signing_key`, and `client_storage`:

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import os
from fastmcp import FastMCP
from fastmcp.server.auth.providers.aws import AWSCognitoProvider
from key_value.aio.stores.redis import RedisStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from cryptography.fernet import Fernet

# Production setup with encrypted persistent token storage
auth_provider = AWSCognitoProvider(
    user_pool_id="eu-central-1_XXXXXXXXX",
    aws_region="eu-central-1",
    client_id="your-app-client-id",
    client_secret="your-app-client-secret",
    base_url="https://your-production-domain.com",

    # Production token management
    jwt_signing_key=os.environ["JWT_SIGNING_KEY"],
    client_storage=FernetEncryptionWrapper(
        key_value=RedisStore(
            host=os.environ["REDIS_HOST"],
            port=int(os.environ["REDIS_PORT"])
        ),
        fernet=Fernet(os.environ["STORAGE_ENCRYPTION_KEY"])
    )
)

mcp = FastMCP(name="Production AWS Cognito App", auth=auth_provider)
```

<Note>
  Parameters (`jwt_signing_key` and `client_storage`) work together to ensure tokens and client registrations survive server restarts. **Wrap your storage in `FernetEncryptionWrapper` to encrypt sensitive OAuth tokens at rest** - without it, tokens are stored in plaintext. Store secrets in environment variables and use a persistent storage backend like Redis for distributed deployments.

  For complete details on these parameters, see the [OAuth Proxy documentation](/servers/auth/oauth-proxy#configuration-parameters).
</Note>

## Features

### JWT Token Validation

The AWS Cognito provider includes robust JWT token validation:

* **Signature Verification**: Validates tokens against AWS Cognito's public keys (JWKS)
* **Expiration Checking**: Automatically rejects expired tokens
* **Issuer Validation**: Ensures tokens come from your specific AWS Cognito user pool
* **Scope Enforcement**: Verifies required OAuth scopes are present

### User Claims and Groups

Access rich user information from AWS Cognito JWT tokens:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.dependencies import get_access_token

@mcp.tool
async def admin_only_tool() -> str:
    """A tool only available to admin users."""
    token = get_access_token()
    user_groups = token.claims.get("cognito:groups", [])

    if "admin" not in user_groups:
        raise ValueError("This tool requires admin access")

    return "Admin access granted!"
```

### Enterprise Integration

Perfect for enterprise environments with:

* **Single Sign-On (SSO)**: Integrate with corporate identity providers
* **Multi-Factor Authentication (MFA)**: Leverage AWS Cognito's built-in MFA
* **User Groups**: Role-based access control through AWS Cognito groups
* **Custom Attributes**: Access custom user attributes defined in your AWS Cognito user pool
* **Compliance**: Meet enterprise security and compliance requirements


# Azure (Microsoft Entra ID) OAuth 🤝 FastMCP
Source: https://gofastmcp.com/integrations/azure

Secure your FastMCP server with Azure/Microsoft Entra OAuth

<VersionBadge />

This guide shows you how to secure your FastMCP server using **Azure OAuth** (Microsoft Entra ID). Since Azure doesn't support Dynamic Client Registration, this integration uses the [**OAuth Proxy**](/servers/auth/oauth-proxy) pattern to bridge Azure's traditional OAuth with MCP's authentication requirements. FastMCP validates Azure JWTs against your application's client\_id.

## Configuration

### Prerequisites

Before you begin, you will need:

1. An **[Azure Account](https://portal.azure.com/)** with access to create App registrations
2. Your FastMCP server's URL (can be localhost for development, e.g., `http://localhost:8000`)
3. Your Azure tenant ID (found in Azure Portal under Microsoft Entra ID)

### Step 1: Create an Azure App Registration

Create an App registration in Azure Portal to get the credentials needed for authentication:

<Steps>
  <Step title="Navigate to App registrations">
    Go to the [Azure Portal](https://portal.azure.com) and navigate to **Microsoft Entra ID → App registrations**.

    Click **"New registration"** to create a new application.
  </Step>

  <Step title="Configure Your Application">
    Fill in the application details:

    * **Name**: Choose a name users will recognize (e.g., "My FastMCP Server")
    * **Supported account types**: Choose based on your needs:
      * **Single tenant**: Only users in your organization
      * **Multitenant**: Users in any Microsoft Entra directory
      * **Multitenant + personal accounts**: Any Microsoft account
    * **Redirect URI**: Select "Web" and enter your server URL + `/auth/callback` (e.g., `http://localhost:8000/auth/callback`)

    <Warning>
      The redirect URI must match exactly. The default path is `/auth/callback`, but you can customize it using the `redirect_path` parameter. For local development, Azure allows `http://localhost` URLs. For production, you must use HTTPS.
    </Warning>

    <Tip>
      If you want to use a custom callback path (e.g., `/auth/azure/callback`), make sure to set the same path in both your Azure App registration and the `redirect_path` parameter when configuring the AzureProvider.
    </Tip>

    * **Expose an API**: Configure your Application ID URI and define scopes
      * Go to **Expose an API** in the App registration sidebar.
      * Click **Set** next to "Application ID URI" and choose one of:
        * Keep the default `api://{client_id}`
        * Set a custom value, following the supported formats (see [Identifier URI restrictions](https://learn.microsoft.com/en-us/entra/identity-platform/identifier-uri-restrictions))
      * Click **Add a scope** and create a scope your app will require, for example:
        * Scope name: `read` (or `write`, etc.)
        * Admin consent display name/description: as appropriate for your org
        * Who can consent: as needed (Admins only or Admins and users)

    * **Configure Access Token Version**: Ensure your app uses access token v2
      * Go to **Manifest** in the App registration sidebar.
      * Find the `requestedAccessTokenVersion` property and set it to `2`:
        ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
        "api": {
            "requestedAccessTokenVersion": 2
        }
        ```
      * Click **Save** at the top of the manifest editor.

    <Warning>
      Access token v2 is required for FastMCP's Azure integration to work correctly. If this is not set, you may encounter authentication errors.
    </Warning>

    <Note>
      In FastMCP's `AzureProvider`, set `identifier_uri` to your Application ID URI (optional; defaults to `api://{client_id}`) and set `required_scopes` to the unprefixed scope names (e.g., `read`, `write`). During authorization, FastMCP automatically prefixes scopes with your `identifier_uri`.
    </Note>
  </Step>

  <Step title="Create Client Secret">
    After registration, navigate to **Certificates & secrets** in your app's settings.

    * Click **"New client secret"**
    * Add a description (e.g., "FastMCP Server")
    * Choose an expiration period
    * Click **"Add"**

    <Warning>
      Copy the secret value immediately - it won't be shown again! You'll need to create a new secret if you lose it.
    </Warning>
  </Step>

  <Step title="Note Your Credentials">
    From the **Overview** page of your app registration, note:

    * **Application (client) ID**: A UUID like `835f09b6-0f0f-40cc-85cb-f32c5829a149`
    * **Directory (tenant) ID**: A UUID like `08541b6e-646d-43de-a0eb-834e6713d6d5`
    * **Client Secret**: The value you copied in the previous step

    <Tip>
      Store these credentials securely. Never commit them to version control. Use environment variables or a secrets manager in production.
    </Tip>
  </Step>
</Steps>

### Step 2: FastMCP Configuration

Create your FastMCP server using the `AzureProvider`, which handles Azure's OAuth flow automatically:

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.azure import AzureProvider

# The AzureProvider handles Azure's token format and validation
auth_provider = AzureProvider(
    client_id="835f09b6-0f0f-40cc-85cb-f32c5829a149",  # Your Azure App Client ID
    client_secret="your-client-secret",                 # Your Azure App Client Secret
    tenant_id="08541b6e-646d-43de-a0eb-834e6713d6d5", # Your Azure Tenant ID (REQUIRED)
    base_url="http://localhost:8000",                   # Must match your App registration
    required_scopes=["your-scope"],                 # At least one scope REQUIRED - name of scope from your App
    # identifier_uri defaults to api://{client_id}
    # identifier_uri="api://your-api-id",
    # Optional: request additional upstream scopes in the authorize request
    # additional_authorize_scopes=["User.Read", "openid", "email"],
    # redirect_path="/auth/callback"                  # Default value, customize if needed
    # base_authority="login.microsoftonline.us"      # For Azure Government (default: login.microsoftonline.com)
)

mcp = FastMCP(name="Azure Secured App", auth=auth_provider)

# Add a protected tool to test authentication
@mcp.tool
async def get_user_info() -> dict:
    """Returns information about the authenticated Azure user."""
    from fastmcp.server.dependencies import get_access_token
    
    token = get_access_token()
    # The AzureProvider stores user data in token claims
    return {
        "azure_id": token.claims.get("sub"),
        "email": token.claims.get("email"),
        "name": token.claims.get("name"),
        "job_title": token.claims.get("job_title"),
        "office_location": token.claims.get("office_location")
    }
```

<Note>
  **Important**: The `tenant_id` parameter is **REQUIRED**. Azure no longer supports using "common" for new applications due to security requirements. You must use one of:

  * **Your specific tenant ID**: Found in Azure Portal (e.g., `08541b6e-646d-43de-a0eb-834e6713d6d5`)
  * **"organizations"**: For work and school accounts only
  * **"consumers"**: For personal Microsoft accounts only

  Using your specific tenant ID is recommended for better security and control.
</Note>

<Note>
  **Important**: The `required_scopes` parameter is **REQUIRED** and must include at least one scope. Azure's OAuth API requires the `scope` parameter in all authorization requests - you cannot authenticate without specifying at least one scope. Use the unprefixed scope names from your Azure App registration (e.g., `["read", "write"]`). These scopes must be created under **Expose an API** in your App registration.
</Note>

### Scope Handling

FastMCP automatically prefixes `required_scopes` with your `identifier_uri` (e.g., `api://your-client-id`) since these are your custom API scopes. Scopes in `additional_authorize_scopes` are sent as-is since they target external resources like Microsoft Graph.

**`required_scopes`** — Your custom API scopes, defined in Azure "Expose an API":

| You write        | Sent to Azure        | Validated on tokens |
| ---------------- | -------------------- | ------------------- |
| `mcp-read`       | `api://xxx/mcp-read` | ✓                   |
| `my.scope`       | `api://xxx/my.scope` | ✓                   |
| `openid`         | `openid`             | ✗ (OIDC scope)      |
| `api://xxx/read` | `api://xxx/read`     | ✓                   |

**`additional_authorize_scopes`** — External scopes (e.g., Microsoft Graph) for server-side use:

| You write   | Sent to Azure | Validated on tokens |
| ----------- | ------------- | ------------------- |
| `User.Read` | `User.Read`   | ✗                   |
| `Mail.Send` | `Mail.Send`   | ✗                   |

<Note>
  `offline_access` is automatically included to obtain refresh tokens. FastMCP manages token refreshing automatically.
</Note>

<Info>
  **Why aren't `additional_authorize_scopes` validated?** Azure issues separate tokens per resource. The access token FastMCP receives is for *your API*—Graph scopes aren't in its `scp` claim. To call Graph APIs, your server uses the upstream Azure token in an on-behalf-of (OBO) flow.
</Info>

<Note>
  OIDC scopes (`openid`, `profile`, `email`, `offline_access`) are never prefixed and excluded from validation because Azure doesn't include them in access token `scp` claims.
</Note>

## Testing

### Running the Server

Start your FastMCP server with HTTP transport to enable OAuth flows:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py --transport http --port 8000
```

Your server is now running and protected by Azure OAuth authentication.

### Testing with a Client

Create a test client that authenticates with your Azure-protected server:

```python test_client.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
import asyncio

async def main():
    # The client will automatically handle Azure OAuth
    async with Client("http://localhost:8000/mcp", auth="oauth") as client:
        # First-time connection will open Azure login in your browser
        print("✓ Authenticated with Azure!")
        
        # Test the protected tool
        result = await client.call_tool("get_user_info")
        print(f"Azure user: {result['email']}")
        print(f"Name: {result['name']}")

if __name__ == "__main__":
    asyncio.run(main())
```

When you run the client for the first time:

1. Your browser will open to Microsoft's authorization page
2. Sign in with your Microsoft account (work, school, or personal based on your tenant configuration)
3. Grant the requested permissions
4. After authorization, you'll be redirected back
5. The client receives the token and can make authenticated requests

<Info>
  The client caches tokens locally, so you won't need to re-authenticate for subsequent runs unless the token expires or you explicitly clear the cache.
</Info>

## Production Configuration

<VersionBadge />

For production deployments with persistent token management across server restarts, configure `jwt_signing_key` and `client_storage`:

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import os
from fastmcp import FastMCP
from fastmcp.server.auth.providers.azure import AzureProvider
from key_value.aio.stores.redis import RedisStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from cryptography.fernet import Fernet

# Production setup with encrypted persistent token storage
auth_provider = AzureProvider(
    client_id="835f09b6-0f0f-40cc-85cb-f32c5829a149",
    client_secret="your-client-secret",
    tenant_id="08541b6e-646d-43de-a0eb-834e6713d6d5",
    base_url="https://your-production-domain.com",
    required_scopes=["your-scope"],

    # Production token management
    jwt_signing_key=os.environ["JWT_SIGNING_KEY"],
    client_storage=FernetEncryptionWrapper(
        key_value=RedisStore(
            host=os.environ["REDIS_HOST"],
            port=int(os.environ["REDIS_PORT"])
        ),
        fernet=Fernet(os.environ["STORAGE_ENCRYPTION_KEY"])
    )
)

mcp = FastMCP(name="Production Azure App", auth=auth_provider)
```

<Note>
  Parameters (`jwt_signing_key` and `client_storage`) work together to ensure tokens and client registrations survive server restarts. **Wrap your storage in `FernetEncryptionWrapper` to encrypt sensitive OAuth tokens at rest** - without it, tokens are stored in plaintext. Store secrets in environment variables and use a persistent storage backend like Redis for distributed deployments.

  For complete details on these parameters, see the [OAuth Proxy documentation](/servers/auth/oauth-proxy#configuration-parameters).
</Note>

## Token Verification Only (Managed Identity)

<VersionBadge />

For deployments where your server only needs to **validate incoming tokens** — such as Azure Container Apps with Managed Identity — use `AzureJWTVerifier` with `RemoteAuthProvider` instead of the full `AzureProvider`.

This pattern is ideal when:

* Your infrastructure handles authentication (e.g., Managed Identity)
* You don't need the OAuth proxy flow (no `client_secret` required)
* You just need to verify that incoming Azure AD tokens are valid

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth import RemoteAuthProvider
from fastmcp.server.auth.providers.azure import AzureJWTVerifier
from pydantic import AnyHttpUrl

tenant_id = "your-tenant-id"
client_id = "your-client-id"

# AzureJWTVerifier auto-configures JWKS, issuer, and audience
verifier = AzureJWTVerifier(
    client_id=client_id,
    tenant_id=tenant_id,
    required_scopes=["access_as_user"],  # Scope names from Azure Portal
)

auth = RemoteAuthProvider(
    token_verifier=verifier,
    authorization_servers=[
        AnyHttpUrl(f"https://login.microsoftonline.com/{tenant_id}/v2.0")
    ],
    base_url="https://your-container-app.azurecontainerapps.io",
)

mcp = FastMCP(name="Azure MI App", auth=auth)
```

`AzureJWTVerifier` handles Azure's scope format automatically. You write scope names exactly as they appear in Azure Portal under **Expose an API** (e.g., `access_as_user`). The verifier validates tokens using the short-form scopes that Azure puts in the `scp` claim, while advertising the full URI scopes (e.g., `api://your-client-id/access_as_user`) in OAuth metadata so MCP clients know what to request.

<Note>
  For Azure Government, pass `base_authority="login.microsoftonline.us"` to `AzureJWTVerifier`.
</Note>

## On-Behalf-Of (OBO)

<VersionBadge />

The On-Behalf-Of (OBO) flow allows your FastMCP server to call downstream Microsoft APIs—like Microsoft Graph—using the authenticated user's identity. When a user authenticates to your MCP server, you receive a token for your API. OBO exchanges that token for a new token that can call other services, maintaining the user's identity and permissions throughout the chain.

This pattern is useful when your tools need to access user-specific data from Microsoft services: reading emails, accessing calendar events, querying SharePoint, or any other Graph API operation that requires user context.

<Note>
  OBO features require the `azure` extra:

  ```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  pip install 'fastmcp[azure]'
  ```
</Note>

### Azure Portal Setup

OBO requires additional configuration in your Azure App registration beyond basic authentication.

<Steps>
  <Step title="Add API Permissions">
    In your App registration, navigate to **API permissions** and add the Microsoft Graph permissions your tools will need.

    * Click **Add a permission** → **Microsoft Graph** → **Delegated permissions**
    * Select the permissions required for your use case (e.g., `Mail.Read`, `Calendars.Read`, `User.Read`)
    * Repeat for any other APIs you need to call

    <Warning>
      Only add delegated permissions for OBO. Application permissions bypass user context entirely and are inappropriate for the OBO flow.
    </Warning>
  </Step>

  <Step title="Grant Admin Consent">
    OBO requires admin consent for the permissions you've added. In the **API permissions** page, click **Grant admin consent for \[Your Organization]**.

    Without admin consent, OBO token exchanges will fail with an `AADSTS65001` error indicating the user or administrator hasn't consented to use the application.

    <Tip>
      For development, you can grant consent for just your own account. For production, an Azure AD administrator must grant tenant-wide consent.
    </Tip>
  </Step>
</Steps>

### Configure AzureProvider for OBO

The `additional_authorize_scopes` parameter tells Azure which downstream API permissions to include during the initial authorization. These scopes establish what your server can request through OBO later.

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.azure import AzureProvider

auth_provider = AzureProvider(
    client_id="your-client-id",
    client_secret="your-client-secret",
    tenant_id="your-tenant-id",
    base_url="http://localhost:8000",
    required_scopes=["mcp-access"],  # Your API scope
    # Include Graph scopes for OBO
    additional_authorize_scopes=[
        "https://graph.microsoft.com/Mail.Read",
        "https://graph.microsoft.com/User.Read",
        "offline_access",  # Enables refresh tokens
    ],
)

mcp = FastMCP(name="Graph-Enabled Server", auth=auth_provider)
```

Scopes listed in `additional_authorize_scopes` are requested during the initial OAuth flow but aren't validated on incoming tokens. They establish permission for your server to later exchange the user's token for downstream API access.

<Info>
  Use fully-qualified scope URIs for downstream APIs (e.g., `https://graph.microsoft.com/Mail.Read`). Short forms like `Mail.Read` work for authorization requests, but fully-qualified URIs are clearer and avoid ambiguity.
</Info>

### EntraOBOToken Dependency

The `EntraOBOToken` dependency handles the complete OBO flow automatically. Declare it as a parameter default with the scopes you need, and FastMCP exchanges the user's token for a downstream API token before your function runs.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.azure import AzureProvider, EntraOBOToken
import httpx

auth_provider = AzureProvider(
    client_id="your-client-id",
    client_secret="your-client-secret",
    tenant_id="your-tenant-id",
    base_url="http://localhost:8000",
    required_scopes=["mcp-access"],
    additional_authorize_scopes=[
        "https://graph.microsoft.com/Mail.Read",
        "https://graph.microsoft.com/User.Read",
    ],
)

mcp = FastMCP(name="Email Reader", auth=auth_provider)

@mcp.tool
async def get_recent_emails(
    count: int = 10,
    graph_token: str = EntraOBOToken(["https://graph.microsoft.com/Mail.Read"]),
) -> list[dict]:
    """Get the user's recent emails from Microsoft Graph."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://graph.microsoft.com/v1.0/me/messages?$top={count}",
            headers={"Authorization": f"Bearer {graph_token}"},
        )
        response.raise_for_status()
        data = response.json()

    return [
        {"subject": msg["subject"], "from": msg["from"]["emailAddress"]["address"]}
        for msg in data.get("value", [])
    ]
```

The `graph_token` parameter receives a ready-to-use access token for Microsoft Graph. FastMCP handles the OBO exchange transparently—your function just uses the token to call the API.

<Warning>
  **Scope alignment is critical.** The scopes passed to `EntraOBOToken` must be a subset of the scopes in `additional_authorize_scopes`. If you request a scope during OBO that wasn't included in the initial authorization, the exchange will fail.
</Warning>

<Tip>
  For advanced OBO scenarios, use `CurrentAccessToken()` to get the user's token, then construct an `azure.identity.aio.OnBehalfOfCredential` directly with your Azure credentials.
</Tip>

<Tip>
  For a complete working example of Azure OBO with FastMCP, see [Pamela Fox's blog post on OBO flow for Entra-based MCP servers](https://blog.pamelafox.org/2026/01/using-on-behalf-of-flow-for-entra-based.html).
</Tip>
