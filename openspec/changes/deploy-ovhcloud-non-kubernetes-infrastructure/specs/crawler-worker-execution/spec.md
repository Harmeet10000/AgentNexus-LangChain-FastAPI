# crawler-worker-execution

## ADDED Requirements

### Requirement: The released crawler artifact owns its browser runtime
The immutable crawler artifact SHALL contain a browser version compatible with the installed crawler library and every required runtime library. It SHALL launch as the production user without downloading executable code or installing operating-system packages at startup.

#### Scenario: Final artifact is tested offline from package registries
- **WHEN** the released crawler artifact starts with package and browser download endpoints unavailable
- **THEN** it launches the pinned browser and completes a controlled local navigation without runtime installation

#### Scenario: Browser and crawler library drift
- **WHEN** the crawler library or browser revision changes incompatibly
- **THEN** the image contract test fails before the artifact is promoted to an environment

### Requirement: Browser smoke testing uses the production security settings
The deployment SHALL prove browser startup and controlled navigation using the same non-root user, filesystem permissions, sandbox decision, certificate mounts, proxy configuration, resource limits, and network policy used by the deployed worker.

#### Scenario: Test-only privilege hides a production failure
- **WHEN** the browser starts only as root, with broader filesystem access, without the production proxy, or outside the production network boundary
- **THEN** that result does not satisfy deployment acceptance
