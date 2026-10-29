"""Authentication bounded context.

Import from explicit modules (``app.features.auth.api.http.router``,
``app.features.auth.api.http.dependencies``, ``app.features.auth.application.service``,
``app.features.auth.adapters.persistence.mongodb.account_repository``, ``app.features.auth.domain.user``,
``app.features.auth.api.http.dto``, ``app.features.auth.application.oauth``,
``app.security.tokens``, ``app.security.passwords``) rather than from
this package root: root re-exports pull the HTTP stack into every importer.
"""
