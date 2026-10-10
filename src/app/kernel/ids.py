"""Reserved: identity value-objects.

No caller needs shared ID types today — features use `uuid4()` and their own
key formats directly. If a cross-feature identifier ever needs one spelling
(e.g. a typed request/correlation id), it belongs here, with that caller
named in this docstring.
"""
