"""Prompts for contract KB ingestion nodes."""

EXTRACT_SCHEMA_SYSTEM_PROMPT = """
Extract canonical contract metadata from the full layout-aware document.
Return only the structured output schema. Do not infer facts that are not present.
Prefer exact values from the document for dates, party names, governing law,
jurisdiction, contract value, notice days, liability caps, and event dates.
"""

SEGMENT_DOCUMENT_SYSTEM_PROMPT = """
Segment the document into retrieval-ready legal clause chunks.
Preserve schedule, annexure, and table meaning. Keep table rows readable.
Each segment must have a stable clause_id, clause_type, text, page_no,
chunk_index, chunk_faqs, and chunk_keywords.
Use chunk_index values in reading order starting at 0.
"""

CONTEXTUALIZE_CHUNK_SYSTEM_PROMPT = """
Return a ContextualizedChunk. Use the required_preamble exactly unless it is
grammatically broken. The text field must be the original chunk text, not a
summary. tokens is the estimated token count of preamble plus text.
"""

CLASSIFY_EXTRACT_SYSTEM_PROMPT = """
Extract legal entities and relationships from the contextualized chunks.
Use only these entity types when applicable: PARTY, PERSON, ORG, CONTRACT,
CLAUSE, OBLIGATION, RIGHT_OR_PERMISSION, PENALTY_CLAUSE, DATE, JURISDICTION.
Use only these relationship types when applicable: SIGNED_BY, SUBSIDIARY_OF,
OBLIGATED_TO, GOVERNED_BY, SUPERSEDES, REFERENCES_CLAUSE.
Return only the structured output schema.
"""
