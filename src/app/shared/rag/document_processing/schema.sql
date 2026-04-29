CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP INDEX IF EXISTS idx_chunks_embedding;
DROP INDEX IF EXISTS idx_chunks_document_id;
DROP INDEX IF EXISTS idx_documents_metadata;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_documents_created_at ON documents (created_at DESC);

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1);
CREATE INDEX idx_chunks_document_id ON chunks (document_id);
CREATE INDEX idx_chunks_chunk_index ON chunks (document_id, chunk_index);

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.content,
        1 - (c.embedding <=> query_embedding) AS similarity,
        c.metadata,
        d.title AS document_title,
        d.source AS document_source
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RAG data table for extracted content (tables, code, images, entities)
DROP TABLE IF EXISTS rag_data CASCADE;
DROP INDEX IF EXISTS idx_rag_data_document_id;
DROP INDEX IF EXISTS idx_rag_data_content_type;
DROP INDEX IF EXISTS idx_rag_data_metadata;

CREATE TABLE rag_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL CHECK (content_type IN ('markdown', 'json', 'doctags', 'table', 'code', 'image', 'entity')),
    data JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rag_data_document_id ON rag_data (document_id);
CREATE INDEX idx_rag_data_content_type ON rag_data (content_type);
CREATE INDEX idx_rag_data_metadata ON rag_data USING GIN (metadata);
CREATE INDEX idx_rag_data_created_at ON rag_data (created_at DESC);

-- Function to search rag_data by content
CREATE OR REPLACE FUNCTION search_rag_data(
    search_query TEXT,
    content_type_filter TEXT DEFAULT NULL,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    content_type TEXT,
    data JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.document_id,
        r.content,
        r.content_type,
        r.data,
        r.metadata,
        r.created_at
    FROM rag_data r
    WHERE r.content ILIKE '%' || search_query || '%'
      AND (content_type_filter IS NULL OR r.content_type = content_type_filter)
    ORDER BY r.created_at DESC
    LIMIT match_count;
END;
$$;
