"""2.5 AUTH 401/403 coverage — no AppError could express these."""

from app.shared.result.errors import ErrorKind, http_status_for_kind, STATUS_BY_KIND


def test_status_by_kind_covers_seven():
    assert set(STATUS_BY_KIND.keys()) == set(ErrorKind)
    assert len(STATUS_BY_KIND) == 7


def test_authentication_maps_to_401():
    assert STATUS_BY_KIND[ErrorKind.AUTHENTICATION] == 401
    assert http_status_for_kind(ErrorKind.AUTHENTICATION) == 401


def test_authorization_maps_to_403():
    assert STATUS_BY_KIND[ErrorKind.AUTHORIZATION] == 403
    assert http_status_for_kind(ErrorKind.AUTHORIZATION) == 403


def test_infrastructure_retryable_logic():
    assert http_status_for_kind(ErrorKind.INFRASTRUCTURE, retryable=False) == 500
    assert http_status_for_kind(ErrorKind.INFRASTRUCTURE, retryable=True) == 503
    # Base mapping for infrastructure is 500 (dead)
    assert STATUS_BY_KIND[ErrorKind.INFRASTRUCTURE] == 500


def test_other_kinds():
    assert http_status_for_kind(ErrorKind.VALIDATION) == 422
    assert http_status_for_kind(ErrorKind.NOT_FOUND) == 404
    assert http_status_for_kind(ErrorKind.CONFLICT) == 409
    assert http_status_for_kind(ErrorKind.EXTERNAL_SERVICE) == 502
