## Purpose

Record the served fusion's branch composition as a contract: two mandatory legs
(bm25, vector), per-plan adaptive weights normalized identically on both doors.

## ADDED Requirements

### Requirement: The served fusion comprises the bm25 and vector branches

The served RRF fusion SHALL run the bm25 and vector branches. The trigram leg
SHALL NOT participate in the served fusion; it SHALL remain available as an
unfused repository capability for measurement sweeps. The optional graphiti
leg SHALL join only behind its settings flag, unchanged.

#### Scenario: The two-branch fusion is the served contract

- **WHEN** a query is served through either door (graph node or service search)
- **THEN** the fusion runs exactly the bm25 and vector branches with
  per-plan adaptive weights
- **AND** both doors yield the identical chunk-id order

#### Scenario: The removal is evidence-cited

- **WHEN** the removal decision is read
- **THEN** it cites the E4 branch sweep (trigram weakest leg solo at 0.5097
  recall@10; the fused pair traded recall for MRR marginally) as its evidence

### Requirement: Plan weights normalize identically on both doors

The graph's plan normalization SHALL total-normalize the plan's vector and
keyword weights — identical to the inline plan contract — and SHALL NOT force
weights by query type. Both fusion doors SHALL apply the same effective
weights for the same plan.

#### Scenario: The doors never diverge on plan weights

- **WHEN** the same query plan flows through the graph node and the service
- **THEN** both doors fuse with identical effective weights
- **AND** the parity suite's two-door order assertion holds on the
  two-branch contract
