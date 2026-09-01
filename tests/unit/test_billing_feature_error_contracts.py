from fastapi import Response
from pydantic import ValidationError
from returns.result import Failure

from app.features.credits.errors import CreditInfrastructureError
from app.features.invoices.errors import InvoiceNotFoundError
from app.features.payments.errors import PaymentProviderUnavailableError
from app.features.webhooks.errors import WebhookVerificationError
from app.shared.result import render_result


def test_feature_classification_is_constant_and_not_serialized() -> None:
    error = InvoiceNotFoundError(message="missing", details={"invoice_id": "inv-1"})

    assert error.model_dump() == {
        "message": "missing",
        "details": {"invoice_id": "inv-1"},
        "source": None,
    }
    assert "kind" not in InvoiceNotFoundError.model_fields
    assert "code" not in InvoiceNotFoundError.model_fields


def test_feature_classification_cannot_be_overridden() -> None:
    try:
        WebhookVerificationError(message="bad signature", code="OTHER")
    except ValidationError:
        pass
    else:
        message = "classification override must be rejected"
        raise AssertionError(message)


def test_each_feature_failure_renders_its_transport_status() -> None:
    cases = (
        (InvoiceNotFoundError(message="missing"), 404),
        (PaymentProviderUnavailableError(message="down"), 502),
        (WebhookVerificationError(message="bad signature"), 401),
        (CreditInfrastructureError(message="dead transaction"), 500),
    )

    for error, expected_status in cases:
        response = Response()
        rendered = render_result(Failure(error), response, message="unused")

        assert response.status_code == expected_status
        assert rendered.success is False
