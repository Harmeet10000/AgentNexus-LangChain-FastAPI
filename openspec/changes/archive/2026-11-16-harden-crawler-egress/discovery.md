# Discovery: Event Storm harden-crawler-egress

## Actors

| Actor | System role |
|---|---|
| API client / agent | Submits caller-controlled crawl destinations and reads job outcomes |
| Crawler API | Records and dispatches the existing crawler command |
| Crawler worker | Provisions the browser capability and executes crawl jobs without direct internet reachability |
| Smokescreen proxy | Resolves requested destinations, applies address policy, selects an approved address, and dials it |
| Fixed Redis relay | Exposes one authenticated internal TLS listener and forwards only to the configured managed Redis endpoint over verified TLS |
| Managed Redis / local RabbitMQ | Existing job read model and message broker reached through distinct, non-general control-plane paths |
| Deployment operator | Supplies pinned proxy configuration and enforces equivalent network topology outside Compose |

## Commands (blue commands)

| Command | Actor | Payload essentials | Produces events |
|---|---|---|---|
| Provision crawler capability | Crawler worker | Proxy URL, policy-probe targets, browser configuration | No domain event; succeeds or fails closed before accepting execution |
| Execute crawl | Crawler worker | Existing `crawl_id` and request payload | No new event; existing Redis job read model reaches a truthful terminal state |
| Resolve and authorize destination | Smokescreen proxy | Requested host and port, DNS answers, configured address policy | No domain event; emits operational logs/metrics only |
| Relay Redis connection | Fixed Redis relay | Authenticated worker connection, one operator-configured upstream host/port | No domain event; the worker cannot select or encode an alternate destination |
| Retrieve crawl status/results | API client / agent | Existing owner-scoped `crawl_id` | None |

## Domain Events (orange stickies — past tense)

No domain events are added or altered. Proxy decisions are operational security signals, not subscriber-facing domain events; the existing crawler task command and Redis job lifecycle remain unchanged.

## Policies (lilac — WHEN/THEN)

| WHEN | THEN | Kind (reactive / command / timer+expiry) |
|---|---|---|
| The worker provisions its crawler capability | It proves the configured proxy is reachable, permits a controlled public destination, and denies a controlled private destination before exposing the capability | Command |
| A browser navigation, redirect, or subresource selects a destination | Every connection crosses Smokescreen, which rejects disallowed addresses before dialing and connects only to the approved resolved address | Reactive |
| Browser code requests an unsupported scheme or port | Browser interception denies it before proxying; proxy/network isolation remains the independent non-public-address boundary | Reactive |
| Smokescreen or its policy is unavailable | Capability provisioning and subsequent crawl execution fail closed; the worker never falls back to direct egress | Command |
| A worker process attempts to bypass proxy configuration | The internal-only network provides no direct internet route | Reactive |
| The isolated worker needs the existing managed Redis | It connects to an authenticated fixed-destination relay; the relay verifies TLS/SNI/hostname to exactly one operator-configured Redis upstream and exposes no CONNECT, SOCKS, or caller-selected target | Reactive |
| The Redis relay is unavailable or its TLS verification fails | Redis-dependent crawler execution remains unavailable; no direct Redis fallback or general TCP route is introduced | Command |

## Read Models

| Read model | Serves | Derives from | Staleness tolerance |
|---|---|---|---|
| Redis crawl job metadata and bounded result | Existing owner-scoped status/result retrieval | Existing API creation and worker writes | Existing eventual consistency; no new guarantee |
| Smokescreen decision logs and metrics | Operator evidence for allowed, denied, failed, and saturated proxy operations | Proxy decisions and connection outcomes | Near-real-time operational telemetry; not used to authorize a request after the fact |

## Hotspots

| # | Question | Who decides | Resolved in |
|---|---|---|---|
| 1 | Which implementation provides connection-time resolution and IP policy? | Architecture owner | `design.md`: pinned Stripe Smokescreen `v0.1.0` |
| 2 | How does an internal-only worker retain the repository's externally managed Redis connection? | Architecture owner | `design.md`: pinned HAProxy 3.4 mTLS listener with one static, TLS-verified Redis upstream; no caller-selected destination protocol |
| 3 | How are HTTP(S) model/provider calls from the crawler worker routed without restoring a default route? | Implementation owner | `design.md`: explicit Smokescreen proxy environment plus `NO_PROXY` limited to RabbitMQ and the Redis relay; unsupported transports fail acceptance |
| 4 | What constitutes equivalent enforcement outside Compose? | Deployment owner | `design.md` and tasks: no direct worker route, mandatory Smokescreen preflight, fixed-destination Redis relay, and deployment evidence |
