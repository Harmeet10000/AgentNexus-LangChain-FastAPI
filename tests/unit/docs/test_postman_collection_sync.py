from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SYNC_SCRIPT = REPOSITORY_ROOT / "docs-site" / "scripts" / "sync_postman_collection.py"
sync_collection = cast(
    "Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]",
    runpy.run_path(str(SYNC_SCRIPT))["sync_collection"],
)
is_destructive_request = cast(
    "Callable[[dict[str, Any]], bool]",
    runpy.run_path(str(SYNC_SCRIPT))["is_destructive_request"],
)


def sample_spec() -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Test API", "version": "1"},
        "components": {"schemas": {}},
        "paths": {
            "/api/v2/things/{thing_id}": {
                "parameters": [
                    {
                        "name": "thing_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "get": {
                    "tags": ["things"],
                    "summary": "Read thing",
                    "security": [{"HTTPBearer": []}],
                    "parameters": [
                        {
                            "name": "include",
                            "in": "query",
                            "required": True,
                            "description": "Related data to include.",
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "201": {"description": "Found"},
                        "422": {"description": "Invalid input"},
                    },
                },
            },
            "/api/v2/things": {
                "post": {
                    "tags": ["things"],
                    "summary": "Create thing",
                    "security": [{"HTTPBearer": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                    "required": ["name"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {"description": "Created"},
                        "422": {"description": "Invalid input"},
                    },
                }
            },
        },
    }


def sample_collection() -> dict[str, Any]:
    return {
        "info": {
            "name": "Test collection",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "auth": {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{access_token}}", "type": "string"}],
        },
        "variable": [
            {
                "key": "baseUrl",
                "value": "http://localhost:5000",
                "type": "string",
                "description": "API origin.",
            },
            {
                "key": "access_token",
                "value": "replace-after-login",
                "type": "string",
                "description": "Test token.",
            },
            {
                "key": "allowClientError",
                "value": "false",
                "type": "string",
                "description": "Negative-test switch.",
            },
            {
                "key": "thing_id",
                "value": "replace-with-thing_id",
                "type": "string",
                "description": "Thing ID.",
            },
        ],
        "item": [
            {
                "name": "things",
                "description": "Curated folder documentation.",
                "item": [
                    {
                        "name": "GET /api/v2/things/{thing_id} - Read thing",
                        "request": {
                            "method": "GET",
                            "description": "Curated operation documentation.",
                            "header": [{"key": "Accept", "value": "application/json"}],
                            "url": {
                                "raw": "{{baseUrl}}/api/v2/things/{{thing_id}}",
                                "query": [],
                            },
                            "auth": {
                                "type": "bearer",
                                "bearer": [
                                    {
                                        "key": "token",
                                        "value": "{{admin_access_token}}",
                                        "type": "string",
                                    }
                                ],
                            },
                        },
                        "event": [
                            {
                                "listen": "prerequest",
                                "script": {"type": "text/javascript", "exec": ["// curated guard"]},
                            },
                            {
                                "listen": "test",
                                "script": {
                                    "type": "text/javascript",
                                    "exec": [
                                        "const expectedSuccessStatuses = [200];",
                                        "pm.test('curated assertion', function () {});",
                                    ],
                                },
                            },
                        ],
                    }
                ],
            }
        ],
    }


def requests_by_key(collection: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    requests: dict[tuple[str, str], dict[str, Any]] = {}
    for folder in collection["item"]:
        for item in folder["item"]:
            request = item["request"]
            raw_path = (
                request["url"]["raw"]
                .removeprefix("{{baseUrl}}")
                .replace("{{thing_id}}", "{thing_id}")
            )
            requests[request["method"].lower(), raw_path] = item
    return requests


def test_sync_preserves_curation_and_updates_schema_owned_fields() -> None:
    synced = sync_collection(sample_spec(), sample_collection())
    request = requests_by_key(synced)["get", "/api/v2/things/{thing_id}"]["request"]

    assert request["description"] == "Curated operation documentation."
    assert request["auth"]["bearer"][0]["value"] == "{{admin_access_token}}"
    assert request["url"]["query"] == [
        {
            "key": "include",
            "value": "{{include}}",
            "disabled": False,
            "description": "Related data to include.",
        }
    ]
    events = {
        event["listen"]: event["script"]["exec"] for event in synced["item"][0]["item"][0]["event"]
    }
    assert events["prerequest"] == ["// curated guard"]
    assert events["test"][0] == "const expectedSuccessStatuses = [201];"
    assert "pm.test('curated assertion'" in events["test"][1]
    assert any(variable["key"] == "include" for variable in synced["variable"])


def test_sync_scaffolds_new_operation_and_is_idempotent() -> None:
    spec = sample_spec()
    first = sync_collection(spec, sample_collection())
    second = sync_collection(spec, first)
    created = requests_by_key(first)["post", "/api/v2/things"]

    assert first == second
    assert created["request"]["description"].startswith("## Create thing")
    assert json_body(created["request"]) == {"name": "replace-me"}
    events = {event["listen"]: event["script"]["exec"] for event in created["event"]}
    assert (
        'const sampleBody = /"replace-me"|"replace-with-[A-Za-z0-9_-]+"/i.test(expandedBody);'
        in events["prerequest"]
    )
    assert "pm.execution.skipRequest()" in "\n".join(events["prerequest"])
    assert "requiresDestructiveOptIn = false" in "\n".join(events["prerequest"])
    assert events["test"][0] == "const expectedSuccessStatuses = [201];"


def test_destructive_routes_are_classified_for_opt_in_guards() -> None:
    assert is_destructive_request(
        {"method": "DELETE", "url": {"raw": "{{baseUrl}}/api/v2/things/{{thing_id}}"}}
    )
    assert is_destructive_request(
        {"method": "POST", "url": {"raw": "{{baseUrl}}/api/v2/things/{{thing_id}}/cancel"}}
    )
    assert is_destructive_request(
        {"method": "POST", "url": {"raw": "{{baseUrl}}/api/v2/subscriptions"}}
    )
    assert is_destructive_request(
        {"method": "PATCH", "url": {"raw": "{{baseUrl}}/api/v2/users/{{user_id}}/role"}}
    )
    assert not is_destructive_request(
        {"method": "GET", "url": {"raw": "{{baseUrl}}/api/v2/things/{{thing_id}}"}}
    )


def test_sync_updates_json_body_fields_without_losing_curated_values() -> None:
    spec = sample_spec()
    schema = spec["paths"]["/api/v2/things"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ]
    schema["properties"]["enabled"] = {"type": "boolean"}
    schema["required"].append("enabled")
    schema["additionalProperties"] = False
    collection = sample_collection()
    collection["item"][0]["item"].append(
        {
            "name": "POST /api/v2/things - Create thing",
            "request": {
                "method": "POST",
                "description": "Existing curated create request.",
                "header": [{"key": "Content-Type", "value": "application/json"}],
                "url": {"raw": "{{baseUrl}}/api/v2/things"},
                "auth": {
                    "type": "bearer",
                    "bearer": [{"key": "token", "value": "{{access_token}}", "type": "string"}],
                },
                "body": {"mode": "raw", "raw": '{"name":"keep this value","legacy":true}'},
            },
        }
    )

    synced = sync_collection(spec, collection)
    request = requests_by_key(synced)["post", "/api/v2/things"]["request"]

    assert json_body(request) == {"name": "keep this value", "enabled": False}


def json_body(request: dict[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(request["body"]["raw"])
