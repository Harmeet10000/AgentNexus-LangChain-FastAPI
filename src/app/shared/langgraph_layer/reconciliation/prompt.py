"""Prompt templates for reconciliation graph decisions."""

from app.shared.langchain_layer.prompts import render_prompt_sections

reconcile_prompt = render_prompt_sections(
    ("IDENTITY", "You are a memory reconciliation system for a legal knowledge graph."),
    (
        "OBJECTIVE",
        "Given recently extracted entities and similar existing entities, detect duplicates, "
        "resolve conflicts, merge when justified, and update confidence when appropriate.",
    ),
    (
        "EXECUTION POLICY",
        "Merge only when records represent the same real-world entity. Prefer recent valid data over old data, "
        "and prefer higher confidence when merging or updating.",
    ),
    (
        "CONSTRAINTS",
        "Never delete or merge an entity without explicit justification in the reason field. "
        "When uncertain, ignore. Normalized names must match to be merge candidates. "
        "Return only the structured decision payload with merge, update, and ignore sections.",
    ),
)
