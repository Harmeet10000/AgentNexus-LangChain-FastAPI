# Caddy Configuration

## Files

- **Caddyfile.http** - HTTP-only configuration for development
- **Caddyfile.https** - HTTPS configuration with automatic SSL for production

## Usage

### Development (HTTP)
```bash
docker-compose up
```
Access at: http://localhost:5000

### Production (HTTPS)
1. Update `Caddyfile.https` with your domain
2. Update docker-compose.yml to use `Caddyfile.https`:
   ```yaml
   volumes:
     - ./caddy/Caddyfile.https:/etc/caddy/Caddyfile:ro
   ```
3. Run:
   ```bash
   docker-compose up -d
   ```

## Features

✅ **Automatic HTTPS** - Let's Encrypt integration (production)  
✅ **Load Balancing** - Least connections algorithm  
✅ **Health Checks** - Automatic failover  
✅ **Compression** - Gzip encoding  
✅ **Security Headers** - HSTS, CSP, X-Frame-Options  
✅ **Static Caching** - 1 year cache for assets  
✅ **Logging** - Access logs in /var/log/caddy

## Advantages over Nginx

- **Automatic HTTPS** - No manual certificate management
- **Simpler config** - More readable syntax
- **Zero downtime reloads** - Graceful config updates
- **Built-in rate limiting** - No external modules needed
- **HTTP/2 & HTTP/3** - Enabled by default
