# Crawler egress boundary

The crawler worker is safe to enable only when its network namespace has no direct public route. In Compose this is represented by the internal `crawler-control` network: the worker attaches only to that network, RabbitMQ attaches for local control-plane traffic, and exactly two constrained gateways are dual-homed.

- `crawler-egress-proxy` is the only arbitrary HTTP(S) path and applies Smokescreen's connection-time address policy.
- `crawler-redis-relay` exposes mTLS internally and has exactly one TLS-verified managed-Redis upstream. It exposes no caller-selected destination protocol.

For Kubernetes, Nomad, or a host-managed worker, deployment acceptance requires equivalent evidence:

1. The worker namespace has no default public route and an explicit deny-all egress policy.
2. Egress allows only RabbitMQ, the internal Smokescreen listener, and the internal Redis-relay listener.
3. Smokescreen and the Redis relay are isolated workloads; only they receive the narrowly required external routes.
4. The crawler capability's positive public probe succeeds through Smokescreen and its loopback probe receives HTTP 407 with `X-Smokescreen-Error`.
5. A raw client in the worker namespace cannot reach a public fixture with proxy use disabled.
6. The Redis relay rejects clients without a trusted certificate, verifies the upstream CA/SNI/hostname, and contains exactly one configured backend.

Rollback disables crawler task acceptance or stops the crawler workers. It must not attach the worker to a public network, enable private ranges, remove TLS verification, or add a direct managed-Redis route.
