"""G6b regression: a description-only PATCH must not brick the plan.

The 2026-11-15 sweep found that ``PATCH /plans/{id}`` with only a description
archived the addressed row (isActive true -> false) and minted a successor
version under a NEW id, so every further PATCH on the id the client holds
returned INVALID_PLAN_UPDATE and subscription creation returned "Plan is not
active" — one description edit silently killed the plan.

Pricing changes keep the versioning semantics (Requirement 24: updates never
mutate the original record, existing subscribers keep their version).
Presentation-only changes (name, description, features, metadata, trial,
tax, refund policy) now mutate the active row in place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from app.features.audit.public import AuditAction
from app.features.billing.plans.dto import PlanUpdateDTO
from app.features.billing.plans.models import Plan
from app.features.billing.plans.service import PlanService

if TYPE_CHECKING:
    from typing import Any

    from app.features.audit.public import RecordAuditEntryInput

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_provider_sync(monkeypatch) -> None:
    """The provider mirror is orthogonal to local update semantics."""
    monkeypatch.setattr(PlanService, "_sync_razorpay_enabled", staticmethod(lambda: False))


class _FakePlanRepository:
    """Records what the service does to the plan row."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.archive_calls: list[str] = []
        self.create_calls: list[Plan] = []
        self.update_calls: list[tuple[str, dict[str, Any]]] = []

    async def find_by_id(self, plan_id: str) -> object:
        plan = self.plan if str(self.plan.id) == plan_id else None
        from returns.result import Failure, Success

        from app.features.billing.plans.exceptions import PlanNotFoundError

        if plan is None:
            return Failure(
                PlanNotFoundError(
                    message="Plan not found", details={"plan_id": plan_id}, source="test"
                )
            )
        return Success(plan)

    async def archive(self, plan_id: str) -> object:
        self.archive_calls.append(plan_id)
        self.plan.is_active = False
        from returns.result import Success

        return Success(self.plan)

    async def create(self, plan: Plan) -> object:
        self.create_calls.append(plan)
        # Mirror the column defaults the database applies at flush time —
        # the repository returns the flushed row.
        plan.created_at = plan.created_at or datetime.now(tz=UTC)
        plan.updated_at = plan.updated_at or datetime.now(tz=UTC)
        from returns.result import Success

        return Success(plan)

    async def update(self, plan: Plan, *, values: dict[str, Any]) -> object:
        self.update_calls.append((str(plan.id), values))
        for key, value in values.items():
            setattr(plan, key, value)
        from returns.result import Success

        return Success(plan)


class _FakeAuditRepository:
    def __init__(self) -> None:
        self.entries: list[RecordAuditEntryInput] = []

    async def record(self, entry: RecordAuditEntryInput, **_: Any) -> object:
        self.entries.append(entry)
        from returns.result import Success

        return Success(entry)


def _active_plan() -> Plan:
    now = datetime.now(tz=UTC)
    return Plan(
        name="pro-monthly",
        description="before",
        amount=199_900,
        currency="INR",
        interval="monthly",
        interval_count=1,
        trial_period_days=0,
        tax_rate=Decimal("0.18"),
        refund_policy="PRO_RATA",
        features={},
        metadata_={},
        is_active=True,
        razorpay_plan_id="plan_existing",
        created_at=now,
        updated_at=now,
    )


def _service(
    repository: _FakePlanRepository,
) -> tuple[PlanService, _FakePlanRepository, _FakeAuditRepository]:
    audit = _FakeAuditRepository()
    service = PlanService(session=None, plans=repository, audit=audit)  # type: ignore[arg-type]
    return service, repository, audit


def _assert_success(result: object) -> Any:
    from returns.result import Success

    assert isinstance(result, Success), f"expected Success, got {result!r}"
    return result.unwrap()


async def test_description_only_patch_updates_in_place_and_stays_active() -> None:
    repository = _FakePlanRepository(_active_plan())
    service, repository, audit = _service(repository)
    dto = PlanUpdateDTO(description="presentation-only change")

    response = _assert_success(
        await service.update_plan(str(repository.plan.id), dto, user_id=str(uuid4()))
    )

    # The plan the client already holds stays active and addressable.
    assert repository.plan.is_active is True
    assert repository.plan.description == "presentation-only change"
    assert response.is_active is True
    assert str(repository.plan.id) == str(response.id)
    assert repository.archive_calls == []
    assert repository.create_calls == []
    assert len(repository.update_calls) == 1
    assert audit.entries[0].action == AuditAction.PLAN_UPDATED.value


async def test_description_only_patch_keeps_provider_binding_and_further_patches_work() -> None:
    repository = _FakePlanRepository(_active_plan())
    service, repository, _ = _service(repository)
    dto = PlanUpdateDTO(description="first edit")

    _assert_success(await service.update_plan(str(repository.plan.id), dto, user_id=str(uuid4())))

    # The successor PATCH on the same id must still succeed (not INVALID_PLAN_UPDATE).
    second = _assert_success(
        await service.update_plan(
            str(repository.plan.id), PlanUpdateDTO(description="second edit"), user_id=str(uuid4())
        )
    )
    assert second.is_active is True
    assert second.description == "second edit"
    assert repository.plan.razorpay_plan_id == "plan_existing"


async def test_financial_change_still_creates_a_new_version() -> None:
    """Requirement 24 is untouched: pricing edits version, they never mutate."""
    repository = _FakePlanRepository(_active_plan())
    service, repository, _ = _service(repository)
    dto = PlanUpdateDTO(amount=299_900)

    _assert_success(await service.update_plan(str(repository.plan.id), dto, user_id=str(uuid4())))

    assert repository.archive_calls == [str(repository.plan.id)]
    assert len(repository.create_calls) == 1
    new_version = repository.create_calls[0]
    assert new_version.is_active is True
    assert new_version.amount == 299_900
    assert new_version.parent_plan_id == repository.plan.id
