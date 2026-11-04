"""Tests for support-ticket signal ingestion models and parsers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.ai.evaluation.signal_ingestion import (
    GoldenQueryCandidate,
    JiraTicketParser,
    SalesforceTicketParser,
    SignalIngestionError,
    SupportTicket,
    TaxonomyCandidate,
    TaxonomyCandidateRecord,
    TicketCluster,
    TicketParseError,
    ZendeskTicketParser,
    parser_for_system,
)

pytestmark = pytest.mark.unit

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "evals"
    / "fixtures"
    / "tickets"
    / "retrieval_failures.json"
)


def _load_fixtures() -> list[dict[str, object]]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


_JIRA_TICKET: dict[str, object] = {
    "key": "SUP-101",
    "reporter": {"accountId": "acct-9"},
    "fields": {
        "issuetype": {"name": "retrieval_failure"},
        "description": "Search for 'indemnification clause' returns no results",
        "resolution": {"description": "Re-triggered the indexing pipeline."},
        "created": "2024-01-20T10:30:00Z",
        "resolutiondate": "2024-01-20T14:45:00Z",
    },
}

_ZENDESK_TICKET: dict[str, object] = {
    "ticket": {
        "id": 9001,
        "requester_id": 42,
        "type": "retrieval_failure",
        "description": "Cannot find the termination notice period",
        "comment": {"body": "Re-ingested with the enhanced OCR profile."},
        "created_at": "2024-01-22T09:15:00Z",
        "solved_at": "2024-01-23T11:20:00Z",
    },
}

_SALESFORCE_CASE: dict[str, object] = {
    "CaseNumber": "00001234",
    "ContactId": "003AA000001",
    "Type": "retrieval_failure",
    "Description": "Query returns irrelevant chunks about insurance coverage",
    "Resolution__c": "Updated the semantic chunking strategy.",
    "CreatedDate": "2024-01-25T14:00:00Z",
    "ClosedDate": "2024-01-26T16:30:00Z",
}


async def test_jira_parser_maps_all_fields() -> None:
    parsed = await JiraTicketParser().parse(_JIRA_TICKET)

    assert parsed == SupportTicket(
        ticket_id="SUP-101",
        customer_id="acct-9",
        issue_category="retrieval_failure",
        symptom_description="Search for 'indemnification clause' returns no results",
        resolution_note="Re-triggered the indexing pipeline.",
        created_at="2024-01-20T10:30:00Z",
        resolved_at="2024-01-20T14:45:00Z",
    )


async def test_jira_parser_normalizes_epoch_timestamps() -> None:
    raw = {
        **_JIRA_TICKET,
        "fields": {
            **_JIRA_TICKET["fields"],  # type: ignore[dict-item]
            "created": 1705746600,
            "resolutiondate": 1705761900,
        },
    }

    parsed = await JiraTicketParser().parse(raw)

    assert parsed["created_at"] == "2024-01-20T10:30:00Z"
    assert parsed["resolved_at"] == "2024-01-20T14:45:00Z"


async def test_jira_parser_allows_missing_optional_fields() -> None:
    raw = {
        "key": "SUP-102",
        "reporter": {"accountId": "acct-10"},
        "fields": {
            "issuetype": {"name": "retrieval_failure"},
            "description": "No results for liability cap",
            "created": "2024-02-01T00:00:00Z",
        },
    }

    parsed = await JiraTicketParser().parse(raw)

    assert parsed["resolution_note"] is None
    assert parsed["resolved_at"] is None


async def test_zendesk_parser_maps_wrapped_payload() -> None:
    parsed = await ZendeskTicketParser().parse(_ZENDESK_TICKET)

    assert parsed == SupportTicket(
        ticket_id="9001",
        customer_id="42",
        issue_category="retrieval_failure",
        symptom_description="Cannot find the termination notice period",
        resolution_note="Re-ingested with the enhanced OCR profile.",
        created_at="2024-01-22T09:15:00Z",
        resolved_at="2024-01-23T11:20:00Z",
    )


async def test_zendesk_parser_falls_back_to_tags_and_notes() -> None:
    raw = {
        "id": 9002,
        "requester_id": 43,
        "tags": ["retrieval_failure", "search"],
        "description": "Confidentiality clause search fails",
        "notes": "Refreshed permission sync.",
        "created_at": 1707033600,
    }

    parsed = await ZendeskTicketParser().parse(raw)

    assert parsed["ticket_id"] == "9002"
    assert parsed["customer_id"] == "43"
    assert parsed["issue_category"] == "retrieval_failure"
    assert parsed["resolution_note"] == "Refreshed permission sync."
    assert parsed["created_at"] == "2024-02-04T08:00:00Z"
    assert parsed["resolved_at"] is None


async def test_zendesk_parser_prefers_comment_body_over_notes() -> None:
    raw = {
        **_ZENDESK_TICKET,
        "ticket": {
            **_ZENDESK_TICKET["ticket"],  # type: ignore[dict-item]
            "notes": "legacy note",
        },
    }

    parsed = await ZendeskTicketParser().parse(raw)

    assert parsed["resolution_note"] == "Re-ingested with the enhanced OCR profile."


async def test_salesforce_parser_maps_all_fields() -> None:
    parsed = await SalesforceTicketParser().parse(_SALESFORCE_CASE)

    assert parsed == SupportTicket(
        ticket_id="00001234",
        customer_id="003AA000001",
        issue_category="retrieval_failure",
        symptom_description="Query returns irrelevant chunks about insurance coverage",
        resolution_note="Updated the semantic chunking strategy.",
        created_at="2024-01-25T14:00:00Z",
        resolved_at="2024-01-26T16:30:00Z",
    )


async def test_salesforce_parser_falls_back_to_id_and_reason() -> None:
    raw = {
        "Id": "500AA000002",
        "ContactId": "003AA000002",
        "Reason": "retrieval_failure",
        "Description": "Stale version returned from the index",
        "CreatedDate": 1707487200,
    }

    parsed = await SalesforceTicketParser().parse(raw)

    assert parsed["ticket_id"] == "500AA000002"
    assert parsed["issue_category"] == "retrieval_failure"
    assert parsed["resolution_note"] is None
    assert parsed["created_at"] == "2024-02-09T14:00:00Z"
    assert parsed["resolved_at"] is None


@pytest.mark.parametrize(
    ("parser", "raw", "field"),
    [
        (JiraTicketParser(), {"reporter": {}, "fields": {}}, "key"),
        (
            JiraTicketParser(),
            {
                "key": "SUP-1",
                "reporter": {"accountId": "a"},
                "fields": {"issuetype": {"name": "x"}, "created": "t"},
            },
            "fields.description",
        ),
        (
            ZendeskTicketParser(),
            {"id": 1, "type": "x", "description": "d", "created_at": "t"},
            "requester_id",
        ),
        (
            SalesforceTicketParser(),
            {"CaseNumber": "1", "ContactId": "c", "Type": "x", "CreatedDate": "t"},
            "Description",
        ),
    ],
)
async def test_missing_required_fields_raise(
    parser: JiraTicketParser | ZendeskTicketParser | SalesforceTicketParser,
    raw: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(TicketParseError) as exc_info:
        await parser.parse(raw)

    assert isinstance(exc_info.value, SignalIngestionError)
    assert exc_info.value.field == field
    assert field in str(exc_info.value)


@pytest.mark.parametrize(
    "parser",
    [JiraTicketParser(), ZendeskTicketParser(), SalesforceTicketParser()],
)
async def test_non_mapping_payload_raises(
    parser: JiraTicketParser | ZendeskTicketParser | SalesforceTicketParser,
) -> None:
    with pytest.raises(TicketParseError, match="expected a mapping"):
        await parser.parse(["not", "a", "mapping"])


async def test_wrong_field_type_raises() -> None:
    raw = {
        **_JIRA_TICKET,
        "fields": {
            **_JIRA_TICKET["fields"],  # type: ignore[dict-item]
            "description": ["not", "a", "string"],
        },
    }

    with pytest.raises(TicketParseError, match="expected a string"):
        await JiraTicketParser().parse(raw)


def test_parser_factory_selects_each_system() -> None:
    assert isinstance(parser_for_system("jira"), JiraTicketParser)
    assert isinstance(parser_for_system("zendesk"), ZendeskTicketParser)
    assert isinstance(parser_for_system("salesforce"), SalesforceTicketParser)


async def test_fixtures_round_trip_through_all_parsers() -> None:
    fixtures = _load_fixtures()
    assert fixtures

    for fixture in fixtures:
        jira_raw = {
            "key": fixture["ticket_id"],
            "reporter": {"accountId": fixture["customer_id"]},
            "fields": {
                "issuetype": {"name": fixture["issue_category"]},
                "description": fixture["symptom_description"],
                "resolution": {"description": fixture["resolution_note"]},
                "created": fixture["created_at"],
                "resolutiondate": fixture["resolved_at"],
            },
        }
        zendesk_raw = {
            "ticket": {
                "id": fixture["ticket_id"],
                "requester_id": fixture["customer_id"],
                "type": fixture["issue_category"],
                "description": fixture["symptom_description"],
                "comment": {"body": fixture["resolution_note"]},
                "created_at": fixture["created_at"],
                "solved_at": fixture["resolved_at"],
            }
        }
        salesforce_raw = {
            "CaseNumber": fixture["ticket_id"],
            "ContactId": fixture["customer_id"],
            "Type": fixture["issue_category"],
            "Description": fixture["symptom_description"],
            "Resolution__c": fixture["resolution_note"],
            "CreatedDate": fixture["created_at"],
            "ClosedDate": fixture["resolved_at"],
        }

        for parser, raw in (
            (JiraTicketParser(), jira_raw),
            (ZendeskTicketParser(), zendesk_raw),
            (SalesforceTicketParser(), salesforce_raw),
        ):
            assert await parser.parse(raw) == fixture


def test_ticket_cluster_is_frozen_and_forbids_extra_fields() -> None:
    cluster = TicketCluster(
        cluster_id="cluster-1",
        ticket_ids=["TICKET-001", "TICKET-002"],
        centroid_similarity_score=0.92,
        symptom_variations=["indemnification clause", "indemnity provisions"],
        failure_count=2,
        first_seen="2024-01-20T10:30:00Z",
        last_seen="2024-02-03T11:20:00Z",
    )

    assert cluster.model_config["frozen"] is True
    assert cluster.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        TicketCluster(
            **cluster.model_dump(),
            unexpected="field",
        )


def test_taxonomy_candidate_record_validates_approval_status() -> None:
    record = TaxonomyCandidateRecord(
        candidate_id="candidate-1",
        query_template="What are the indemnification obligations?",
        recommended_case_type="no_evidence_vs_ingestion_failure",
        source_cluster_id="cluster-1",
        symptom_variations=["indemnification clause"],
        cluster_similarity_score=0.92,
        approval_status="pending",
        created_at="2024-03-01T00:00:00Z",
    )

    assert record.approved_at is None
    with pytest.raises(ValidationError):
        TaxonomyCandidateRecord(
            **record.model_dump(exclude={"approval_status"}),
            approval_status="maybe",
        )


def test_taxonomy_candidate_and_query_candidate_shapes() -> None:
    candidate = TaxonomyCandidate(
        query_template="What are the indemnification obligations?",
        recommended_case_type="no_evidence_vs_ingestion_failure",
        source_ticket_ids=["TICKET-001"],
        cluster_similarity_score=0.92,
        symptom_variations=["indemnification clause"],
    )
    query_candidate = GoldenQueryCandidate(
        query="What are the indemnification obligations?",
        source_ticket_ids=["TICKET-001"],
        failure_count=1,
        cluster_id=None,
    )

    assert candidate.source_ticket_ids == ["TICKET-001"]
    assert query_candidate["failure_count"] == 1
