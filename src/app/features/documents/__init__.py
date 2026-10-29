"""Documents bounded context.

Import from explicit modules (``app.features.documents.api.http.router``,
``app.features.documents.api.http.dependencies``, ``app.features.documents.api.http.dto``,
``app.features.documents.domain.errors``, ``app.features.documents.domain.constants``,
``app.features.documents.application.commands.upload_document``,
``app.features.documents.application.queries.documents``,
``app.features.documents.application.evidence.service``, ``app.features.documents.application.tools.*``,
``app.features.documents.ingestion.*``, ``app.features.documents.retrieval.*``,
``app.features.documents.adapters.persistence.postgres.document_repository``)
rather than from this package root: root re-exports pull the HTTP stack and
LangChain into every importer.

Legacy compatibility routes and their fail-closed dependency live at
``app.features.documents.api.http.legacy_ingest_router`` and
``app.features.documents.api.http.legacy_ingest_dependencies`` (the former
``app.features.ingestion`` package, now folded in here).
"""
