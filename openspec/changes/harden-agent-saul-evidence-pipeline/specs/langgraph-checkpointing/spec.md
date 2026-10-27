## ADDED Requirements

### Requirement: Saul checkpoints contain references rather than legal payloads

Agent Saul checkpoints MUST store document IDs, segment/chunk IDs, evidence handles, findings, and small state scalars,
but MUST NOT store full document text, raw bytes, or full clause text. Evidence-tool responses MUST preserve opaque
handles and bounded metadata through a content-and-artifact result channel (such as `content_and_artifact`) without
placing canonical chunk text into checkpointed state. Old state versions MUST be upgraded before use.

#### Scenario: Large document text is absent from state
- **WHEN** a Saul checkpoint is serialized for a large document
- **THEN** it MUST contain references and handles only and remain within its stated size budget

#### Scenario: Evidence tool output carries handles without legal text
- **WHEN** `search_evidence` returns canonical chunks for a Saul run
- **THEN** the tool result MUST expose handles/metadata through its content-and-artifact channel while checkpoint
  state contains no full chunk text

#### Scenario: Legacy thread resumes
- **WHEN** an old checkpoint uses single-document scope or obsolete placeholder fields
- **THEN** the state hydrator MUST upgrade it to the versioned scope model or fail with a typed, actionable reason

### Requirement: Evidence state survives pause without trusting stale authority

After a HITL pause, resumption MUST revalidate authorization, document versions, and evidence handles before reading
stored chunks or approving findings. Checkpoint retention MUST prune expired paused threads and obsolete tool outputs.

#### Scenario: Version mismatch is not silently cited
- **WHEN** the document version changes before resume
- **THEN** stale handles MUST be rejected or reissued and MUST NOT silently support a final finding

#### Scenario: Expired pause is collected
- **WHEN** a paused thread exceeds its retention window
- **THEN** its checkpoint and obsolete evidence outputs MUST be pruned and the run MUST not be resumable as approved
