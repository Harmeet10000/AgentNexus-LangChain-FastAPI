## Purpose

Define the key-set contract at the boundary between every compiled graph's input channel and its
declared state: what a caller may place into an invocation's initial state, and what the system
must do about a key the state does not declare. The contract exists because LangGraph maps an
invocation's input onto declared channels and discards the rest, and a `total=False` TypedDict
makes every channel optional — so a disjoint state dict type-checks clean while silently
 discarding the caller's data.

## ADDED Requirements

### Requirement: An invocation's initial state keys are declared by the graph's state

For every compiled graph, the key set of an invocation's initial state SHALL be a subset of the
graph state declaration's annotated keys. A contract test SHALL assert this boundary for each
compiled graph, in the form of a key-set comparison (for example
`set(initial_state) <= set(State.__annotations__)`), and SHALL NOT rely on type checking alone.

#### Scenario: The documents ingestion graph honours its declared channels

- **WHEN** the key-set contract test runs against the document ingestion graph's input channel and
  `DocumentIngestionState`
- **THEN** every key the declared input channel accepts SHALL be an annotated key of
  `DocumentIngestionState`, and the test SHALL fail if the channel and the declaration diverge

#### Scenario: Every compiled graph carries the seatbelt

- **WHEN** a new compiled graph is added to the application
- **THEN** it SHALL gain a key-set boundary contract test of the same form, and the suite SHALL
  enumerate the graphs it covers so an untested graph is visible

#### Scenario: A disjoint state dict does not type-check clean by accident

- **WHEN** a caller constructs an initial state whose keys are disjoint from the graph's declared
  channels
- **THEN** the boundary contract test SHALL catch the mismatch even though the
  `total=False` TypedDict accepts it statically

### Requirement: A silent key drop is not an acceptable boundary behaviour

Where a caller's state keys cannot satisfy the declared channels, the boundary SHALL either map
each key explicitly onto a declared channel or fail loudly with a typed error naming the unmapped
keys. Silent dropping by the graph runtime SHALL NOT be the mechanism by which a boundary resolves
a key mismatch.

#### Scenario: The legacy compatibility path maps its keys

- **WHEN** the legacy compatibility ingestion path invokes the document ingestion graph with its
  historical state shape (`doc_id`, `raw_bytes`, `thread_id`, `source`, `document_type`,
  `jurisdiction` — none of which `DocumentIngestionState` declares)
- **THEN** each key SHALL be mapped explicitly onto a declared channel (or routed through the
  invocation config where it is invocation metadata), or the invocation SHALL fail with a typed
  error naming every unmapped key

#### Scenario: A regression reintroduces a silent drop

- **WHEN** a change reverts the explicit mapping so the legacy path again sends undeclared keys
- **THEN** the boundary contract test SHALL fail, naming the edge and the undeclared keys

#### Scenario: An unmapped key is named, not summarised

- **WHEN** a boundary fails loudly on undeclared keys
- **THEN** the error SHALL enumerate the offending key names so the fix is mechanical
