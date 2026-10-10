"""Reserved: time abstractions.

No caller needs a Clock abstraction today — call sites use
`datetime.now(UTC)` directly. If a caller ever needs injectable time
(frozen clocks in tests, logical timestamps), it belongs here, behind a
`Protocol`, with that caller named in this docstring.
"""
