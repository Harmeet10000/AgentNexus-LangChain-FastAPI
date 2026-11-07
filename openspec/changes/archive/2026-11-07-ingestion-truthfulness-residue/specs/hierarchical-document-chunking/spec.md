## ADDED Requirements

### Requirement: Classification and extraction respect multi-contract document boundaries

For a document containing more than one contract (a bundled packet), classification and
legal-metadata extraction SHALL attribute each chunk's classified content to exactly one contract
boundary, and the extracted content of one boundary SHALL NOT mix another boundary's parties,
dates, or governing law. The boundary attribution SHALL be verified against the golden corpus's
recorded multi-contract documents, whose boundary tables are the normative ground truth.

#### Scenario: Extracted content does not leak across contract boundaries

- **WHEN** the cross-contract leakage assertion runs over extracted content from SRC-04 (the
  golden corpus's 5-boundary NHAI HAM packet)
- **THEN** no extracted party, date, or governing-law value attributed to one contract boundary
  SHALL originate from a different boundary, and a value that appears only in another boundary's
  pages SHALL NOT be attributed to the boundary under test

#### Scenario: A dedicated multi-contract classification suite exists

- **WHEN** the classification-boundary suite runs against the corpus's multi-contract documents
  (SRC-04 with 5 boundaries, SRC-14 with 2, SYN-02 with 2)
- **THEN** each document's chunks classified into one contract boundary SHALL match that
  boundary's recorded page ranges, and the suite SHALL be part of the deterministic test selection
  that runs without a database or model provider

#### Scenario: An ambiguous boundary is declared, not guessed

- **WHEN** a packet page belongs to no contract boundary or to an ambiguous one (the corpus's
  flagged front matter and embedded templates)
- **THEN** the classification output SHALL mark it ambiguous rather than assigning it to the
  nearest boundary
