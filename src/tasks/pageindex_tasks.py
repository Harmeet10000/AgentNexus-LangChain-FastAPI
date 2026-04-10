from app.connections import celery_app


@celery_app.task
def ingest_pageindex_document(file_path: str, user_id: str) -> str:
    # run in worker – fully async inside via the client above
    ...
