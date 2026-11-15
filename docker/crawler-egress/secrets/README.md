# Crawler Redis relay certificates

Provision these files outside source control before starting the crawler worker:

- `relay-server.pem`: HAProxy server certificate and private key bundle for `crawler-redis-relay`.
- `client-ca.pem`: CA used by HAProxy to authenticate crawler-worker client certificates.
- `relay-ca.pem`: CA used by the crawler worker to authenticate the relay certificate.
- `client.crt` and `client.key`: crawler-worker client certificate and private key.

Production may point the corresponding `CRAWLER_REDIS_*_FILE` Compose variables at a deployment secret store instead. Private keys must never be committed.
