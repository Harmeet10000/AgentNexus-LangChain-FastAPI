"""Prompts for contract KB ingestion nodes."""

from app.shared.langchain_layer.prompts import render_prompt_sections

EXTRACT_SCHEMA_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a contract metadata extraction engine."),
    (
        "OBJECTIVE",
        "Extract canonical contract metadata from the full layout-aware document.",
    ),
    (
        "EXECUTION POLICY",
        "Prefer exact values from the document for dates, party names, governing law, "
        "jurisdiction, contract value, notice days, liability caps, and event dates.",
    ),
    (
        "CONSTRAINTS",
        "Do not infer facts that are not present. Return only the structured output schema.",
    ),
)

SEGMENT_DOCUMENT_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a legal clause segmentation engine."),
    (
        "OBJECTIVE",
        "Segment the document into retrieval-ready legal clause chunks.",
    ),
    (
        "EXECUTION POLICY",
        "Preserve schedule, annexure, and table meaning. Keep table rows readable. "
        "Use chunk_index values in reading order starting at 0.",
    ),
    (
        "CONSTRAINTS",
        "Each segment must have a stable clause_id, clause_type, text, page_no, chunk_index, "
        "chunk_faqs, and chunk_keywords.",
    ),
)

CONTEXTUALIZE_CHUNK_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a legal chunk contextualization engine."),
    ("OBJECTIVE", "Return a ContextualizedChunk."),
    (
        "EXECUTION POLICY",
        "Use the required_preamble exactly unless it is grammatically broken.",
    ),
    (
        "CONSTRAINTS",
        "The text field must be the original chunk text, not a summary. tokens is the estimated "
        "token count of preamble plus text.",
    ),
)

CLASSIFY_EXTRACT_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a legal entity and relationship extraction engine."),
    (
        "OBJECTIVE",
        "Extract legal entities and relationships from the contextualized chunks.",
    ),
    (
        "CONSTRAINTS",
        "Use only these entity types when applicable: PARTY, PERSON, ORG, CONTRACT, CLAUSE, "
        "OBLIGATION, RIGHT_OR_PERMISSION, PENALTY_CLAUSE, DATE, JURISDICTION. Use only these "
        "relationship types when applicable: SIGNED_BY, SUBSIDIARY_OF, OBLIGATED_TO, "
        "GOVERNED_BY, SUPERSEDES, REFERENCES_CLAUSE. Return only the structured output schema.",
    ),
)
