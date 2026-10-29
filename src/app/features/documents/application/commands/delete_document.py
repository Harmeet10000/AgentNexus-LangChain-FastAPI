"""Delete use case for the documents slice.

Reserved: no delete operation exists today. When document removal lands
(hard delete vs retention-window tombstone), its service entry point belongs
here, reusing the repository and outbox primitives.
"""
