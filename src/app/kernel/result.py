"""Result conventions for expected recoverable failures.

Single import site for the `returns` primitives plus the kernel error spine
they are parameterized with. Transport rendering (`render_result`,
`http_error`) lives in `app.shared.result` and stays out of the kernel.
"""

from returns.result import Failure, Result, Success

from .errors import ErrorDetails, ErrorKind, FeatureError

__all__ = [
    "ErrorDetails",
    "ErrorKind",
    "Failure",
    "FeatureError",
    "Result",
    "Success",
]
