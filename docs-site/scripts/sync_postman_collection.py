"""Synchronize the Postman collection with the checked-in OpenAPI document.

The collection keeps curated descriptions, auth overrides, pre-request guards,
and test scripts for existing operations. OpenAPI owns the operation set,
methods, paths, parameters, request-body media types, and expected success
statuses. New operations receive a documented, editable request scaffold.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

DOCS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DOCS_ROOT.parent
SPEC_PATH = DOCS_ROOT / "openapi.json"
COLLECTION_PATH = REPO_ROOT / "postman" / "agentnexus-local.postman_collection.json"
HTTP_METHODS = frozenset({"delete", "get", "head", "options", "patch", "post", "put"})
TAG_FOLDERS = {
    "crawler": "Crawler",
    "monitoring": "Monitoring",
    "root": "Root",
}


def operation_entries(spec: dict[str, Any]) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
    """Index HTTP operations by method and OpenAPI path."""
    operations: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            normalized_method = method.lower()
            if normalized_method in HTTP_METHODS and isinstance(operation, dict):
                operations[normalized_method, path] = (path, operation)
    return operations


def collection_operation_key(item: dict[str, Any]) -> tuple[str, str] | None:
    """Convert a Postman request URL back to an OpenAPI method/path key."""
    request = item.get("request", {})
    url = request.get("url", {})
    raw_url = url if isinstance(url, str) else url.get("raw", "")
    if not isinstance(raw_url, str) or not raw_url:
        return None
    path = raw_url.removeprefix("{{baseUrl}}")
    path = re.sub(r"\{\{([^}]+)\}\}", r"{\1}", path)
    path = path.split("?", 1)[0]
    method = request.get("method", "GET").lower()
    return (method, path)


def resolve_schema(
    schema: Any, components: dict[str, Any], seen: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Resolve local schema references and merge simple union branches."""
    if not isinstance(schema, dict):
        return {}
    result = dict(schema)
    reference = result.pop("$ref", None)
    if (
        isinstance(reference, str)
        and reference.startswith("#/components/")
        and reference not in seen
    ):
        target: Any = components
        for segment in reference.removeprefix("#/components/").split("/"):
            target = target.get(segment, {}) if isinstance(target, dict) else {}
        resolved = resolve_schema(target, components, seen | {reference})
        resolved.update(result)
        result = resolved

    for union_key in ("allOf", "anyOf", "oneOf"):
        branches = result.pop(union_key, None)
        if not isinstance(branches, list):
            continue
        merged: dict[str, Any] = {}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for branch in branches:
            resolved = resolve_schema(branch, components, seen)
            merged.update(resolved)
            if isinstance(resolved.get("properties"), dict):
                properties.update(resolved["properties"])
            if isinstance(resolved.get("required"), list):
                required.extend(resolved["required"])
        if properties:
            merged["properties"] = properties
        if required:
            merged["required"] = list(dict.fromkeys(required))
        merged.update(result)
        result = merged
        break
    return result


def is_destructive_request(request: dict[str, Any]) -> bool:
    """Identify requests that can revoke, delete, charge, or otherwise mutate state."""
    method = request.get("method", "GET").upper()
    raw_url = request.get("url", {}).get("raw", "")
    path = re.sub(r"\{\{([^}]+)\}\}", r"{\1}", raw_url.removeprefix("{{baseUrl}}"))
    path = path.split("?", 1)[0]
    mutating_action = re.search(
        r"/(?:cancel|pause|resume|change-plan|trial-extension|refund|void|archive|replay|deactivate|activate|impersonate|logout|change-password)(?:/|$)",
        path,
    )
    return (
        method == "DELETE"
        or (method in {"POST", "PATCH", "PUT"} and mutating_action is not None)
        or path
        in {
            "/api/v2/webhooks/razorpay",
            "/api/v2/admin/billing/dunning/run",
            "/api/v2/credits",
            "/api/v2/credits/apply-to-invoice",
        }
        or (method == "POST" and path in {"/api/v2/plans", "/api/v2/subscriptions"})
        or (
            method in {"PATCH", "PUT"}
            and path.startswith("/api/v2/users/")
            and path.endswith("/role")
        )
    )


def variable_name(name: str) -> str:
    """Return the collection-variable spelling for an OpenAPI parameter."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def sample_string(
    schema: dict[str, Any],
    name: str,
    variables: dict[str, dict[str, Any]],
) -> str:
    """Build a useful editable sample for string-like schema values."""
    fmt = schema.get("format")
    lowered = name.lower()
    if fmt == "binary":
        return ""
    if fmt == "email" or "email" in lowered:
        return "{{email}}"
    if fmt in {"uri", "url"} or lowered == "url":
        return "https://example.com/"
    if fmt == "date-time":
        return "2026-10-18T12:00:00Z"
    if "password" in lowered:
        return "{{password}}"
    if "token" in lowered:
        return "replace-with-token"
    if lowered.endswith("_id") or lowered == "id":
        variable = variable_name(name)
        variables.setdefault(
            variable,
            {
                "key": variable,
                "value": f"replace-with-{variable}",
                "type": "string",
                "description": f"Set this variable to the ID required for `{name}`.",
            },
        )
        return "{{" + variable + "}}"
    return "replace-me"


def sample_value(
    schema: Any,
    name: str,
    variables: dict[str, dict[str, Any]],
    components: dict[str, Any],
    depth: int = 0,
) -> Any:
    """Build a minimal editable example from an OpenAPI schema."""
    resolved = resolve_schema(schema, components)
    if depth > 6:
        return "replace-me"
    for key in ("example", "default", "const"):
        if key in resolved:
            return resolved[key]
    enum_values = resolved.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]
    kind = resolved.get("type")
    if isinstance(kind, list):
        kind = next((candidate for candidate in kind if candidate != "null"), "string")
    if kind == "object" or "properties" in resolved:
        properties = resolved.get("properties", {})
        required = resolved.get("required", [])
        if not isinstance(properties, dict):
            return {}
        return {
            field_name: sample_value(field_schema, field_name, variables, components, depth + 1)
            for field_name, field_schema in properties.items()
            if field_name in required
        }
    if kind == "array":
        return []
    if kind == "boolean":
        return False
    if kind in {"integer", "number"}:
        return resolved.get("minimum", 1)
    if kind == "string" or kind is None:
        return sample_string(resolved, name, variables)
    return None


def ensure_variable(variables: dict[str, dict[str, Any]], name: str, description: str) -> None:
    key = variable_name(name)
    variables.setdefault(
        key,
        {"key": key, "value": f"replace-with-{key}", "type": "string", "description": description},
    )


def operation_parameters(
    path_item: dict[str, Any], operation: dict[str, Any]
) -> list[dict[str, Any]]:
    """Combine path and operation parameters, with operation values taking precedence."""
    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for parameter in [*path_item.get("parameters", []), *operation.get("parameters", [])]:
        if not isinstance(parameter, dict):
            continue
        name = parameter.get("name")
        location = parameter.get("in")
        if isinstance(name, str) and isinstance(location, str):
            combined[location, name] = parameter
    return list(combined.values())


def update_url(
    request: dict[str, Any],
    path: str,
    parameters: list[dict[str, Any]],
    variables: dict[str, dict[str, Any]],
    components: dict[str, Any],
) -> None:
    """Synchronize path and query parameters without discarding curated values."""
    old_url = request.get("url", {})
    old_url = old_url if isinstance(old_url, dict) else {"raw": old_url}
    old_query = {
        entry.get("key"): entry for entry in old_url.get("query", []) if isinstance(entry, dict)
    }
    query: list[dict[str, Any]] = []
    for parameter in parameters:
        if parameter.get("in") != "query":
            continue
        name = parameter["name"]
        existing = copy.deepcopy(old_query.get(name, {}))
        required = bool(parameter.get("required"))
        schema = resolve_schema(parameter.get("schema", {}), components)
        if not existing:
            variable = variable_name(name)
            if required:
                ensure_variable(variables, name, f"Required query value for `{name}`.")
                value = "{{" + variable + "}}"
            else:
                value = ""
            existing = {"key": name, "value": value, "disabled": not required}
        existing["key"] = name
        existing["description"] = parameter.get("description", existing.get("description", ""))
        existing["disabled"] = False if required else existing.get("disabled", True)
        if isinstance(existing.get("value"), str) and existing["value"] == "replace-me":
            example = sample_value(schema, name, variables, components)
            existing["value"] = str(example)
        query.append(existing)

    for name in re.findall(r"\{([^}]+)\}", path):
        ensure_variable(variables, name, f"Path ID for `{name}` in `{path}`.")
    raw_path = re.sub(r"\{([^}]+)\}", r"{{\1}}", path)
    old_url["raw"] = "{{baseUrl}}" + raw_path
    if query:
        old_url["query"] = query
    else:
        old_url.pop("query", None)
    request["url"] = old_url


def build_body(
    desired_mode: str,
    schema: dict[str, Any],
    variables: dict[str, dict[str, Any]],
    components: dict[str, Any],
) -> dict[str, Any]:
    """Create a basic Postman body for a newly discovered request body."""
    if desired_mode == "raw":
        payload = sample_value(schema, "body", variables, components)
        return {
            "mode": "raw",
            "raw": json.dumps(payload, indent=2),
            "options": {"raw": {"language": "json"}},
        }
    if desired_mode == "formdata":
        return {"mode": "formdata", "formdata": sync_formdata(schema, {}, variables, components)}
    return {"mode": "urlencoded", "urlencoded": []}


def sync_formdata(
    schema: dict[str, Any],
    old_body: dict[str, Any],
    variables: dict[str, dict[str, Any]],
    components: dict[str, Any],
) -> list[dict[str, Any]]:
    """Preserve selected files while matching form fields to the current schema."""
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return []
    old_fields = {
        field.get("key"): field for field in old_body.get("formdata", []) if isinstance(field, dict)
    }
    required = set(schema.get("required", []))
    formdata: list[dict[str, Any]] = []
    for name, field_schema in properties.items():
        field = resolve_schema(field_schema, components)
        existing = copy.deepcopy(old_fields.get(name, {}))
        if field.get("format") == "binary":
            existing.update(
                {
                    "key": name,
                    "type": "file",
                    "src": existing.get("src"),
                    "description": field.get("description", "Select a local file in Postman."),
                }
            )
            formdata.append(existing)
        elif name in required:
            existing.update(
                {
                    "key": name,
                    "type": "text",
                    "value": existing.get(
                        "value", str(sample_value(field, name, variables, components))
                    ),
                }
            )
            formdata.append(existing)
    return formdata


def sync_json_body(
    body: dict[str, Any],
    schema: dict[str, Any],
    variables: dict[str, dict[str, Any]],
    components: dict[str, Any],
) -> dict[str, Any]:
    """Preserve valid sample values while matching JSON fields to the schema."""
    try:
        payload = json.loads(body.get("raw", ""))
    except json.JSONDecodeError:
        return body
    properties = schema.get("properties", {})
    if not isinstance(payload, dict) or not isinstance(properties, dict) or not properties:
        return body

    if schema.get("additionalProperties") is False:
        payload = {key: value for key, value in payload.items() if key in properties}
    required = set(schema.get("required", []))
    for name, field_schema in properties.items():
        if name not in payload and name in required:
            payload[name] = sample_value(field_schema, name, variables, components)
    body["raw"] = json.dumps(payload, indent=2)
    return body


def update_body(
    request: dict[str, Any],
    operation: dict[str, Any],
    variables: dict[str, dict[str, Any]],
    components: dict[str, Any],
) -> None:
    """Keep valid curated samples while aligning the body's media type with OpenAPI."""
    request_body = operation.get("requestBody")
    content = request_body.get("content", {}) if isinstance(request_body, dict) else {}
    if not content:
        request.pop("body", None)
        return
    media_type = next(
        (
            kind
            for kind in (
                "application/json",
                "multipart/form-data",
                "application/x-www-form-urlencoded",
            )
            if kind in content
        ),
        next(iter(content)),
    )
    old_body = request.get("body", {})
    old_mode = old_body.get("mode")
    desired_mode = (
        "formdata"
        if media_type == "multipart/form-data"
        else "urlencoded"
        if media_type == "application/x-www-form-urlencoded"
        else "raw"
    )
    schema = resolve_schema(content[media_type].get("schema", {}), components)

    if old_mode != desired_mode:
        old_body = build_body(desired_mode, schema, variables, components)
    elif desired_mode == "raw":
        old_body = sync_json_body(old_body, schema, variables, components)
    elif desired_mode == "formdata":
        old_body["formdata"] = sync_formdata(schema, old_body, variables, components)
    request["body"] = old_body

    headers = request.setdefault("header", [])
    headers[:] = [header for header in headers if header.get("key", "").lower() != "content-type"]
    if desired_mode in {"raw", "urlencoded"}:
        headers.append(
            {"key": "Content-Type", "value": media_type, "description": "Request body media type."}
        )


def default_events(request: dict[str, Any], operation: dict[str, Any]) -> list[dict[str, Any]]:
    """Create basic safety and success checks for newly discovered operations."""
    success_codes = sorted(
        int(code)
        for code in operation.get("responses", {})
        if isinstance(code, str) and code.isdigit() and code.startswith("2")
    ) or [200]
    raw = request["url"]["raw"]
    placeholders = set(re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", raw))
    for bearer in request.get("auth", {}).get("bearer", []):
        placeholders.update(re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", str(bearer.get("value", ""))))
    for parameter in request["url"].get("query", []):
        if not parameter.get("disabled"):
            placeholders.update(
                re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", str(parameter.get("value", "")))
            )
    placeholders_list = json.dumps(sorted(placeholders))
    test_script = [
        f"const expectedSuccessStatuses = {json.dumps(success_codes)};",
        "pm.test('Expected documented success status', function () { pm.expect(expectedSuccessStatuses).to.include(pm.response.code); });",
        "const contentType = pm.response.headers.get('Content-Type') || '';",
        "if (contentType.includes('application/json')) {",
        "  const body = pm.response.json();",
        "  if (Object.prototype.hasOwnProperty.call(body, 'statusCode')) pm.test('Envelope status matches HTTP status', function () { pm.expect(body.statusCode).to.eql(pm.response.code); });",
        "  if (body.success === false) pm.test('Error envelope has a code', function () { pm.expect(body.error.code).to.be.a('string').and.not.empty; });",
        "  if (body.success === true && body.data) {",
        "    for (const key of ['access_token', 'refresh_token', 'user_id', 'doc_id', 'thread_id', 'session_id', 'crawl_id', 'plan_id', 'subscription_id', 'payment_id', 'invoice_id', 'event_id', 'ws_url']) { if (body.data[key]) pm.collectionVariables.set(key, body.data[key]); }",
        "  }",
        "}",
    ]
    body = request.get("body", {})
    has_raw_body = body.get("mode") == "raw"
    destructive = is_destructive_request(request)
    guard_script = [
        f"const unresolvedNames = {placeholders_list};",
        "const allowClientError = String(pm.variables.get('allowClientError') || 'false').toLowerCase() === 'true';",
        f"const requiresDestructiveOptIn = {str(destructive).lower()};",
        "const allowDestructive = String(pm.variables.get('allowDestructive') || 'false').toLowerCase() === 'true';",
        "const unresolvedVariable = unresolvedNames.some((name) => { const value = pm.variables.get(name); return value == null || /^(replace-with-|replace-after-login)/i.test(String(value)); });",
        *(
            [
                "const expandedBody = pm.variables.replaceIn(pm.request.body && pm.request.body.raw || '');",
                'const sampleBody = /"replace-me"|"replace-with-[A-Za-z0-9_-]+"/i.test(expandedBody);',
            ]
            if has_raw_body
            else ["const sampleBody = false;"]
        ),
        "if (requiresDestructiveOptIn && !allowDestructive) { pm.execution.skipRequest(); } else if (!allowClientError && (unresolvedVariable || sampleBody)) { pm.execution.skipRequest(); }",
    ]
    return [
        {"listen": "prerequest", "script": {"type": "text/javascript", "exec": guard_script}},
        {"listen": "test", "script": {"type": "text/javascript", "exec": test_script}},
    ]


def operation_description(
    path: str, operation: dict[str, Any], parameters: list[dict[str, Any]]
) -> str:
    """Document a newly discovered operation without overwriting curated descriptions."""
    lines = [
        f"## {operation.get('summary', 'API operation')}",
        "",
        f"**Request:** OpenAPI `{path}`.",
    ]
    description = operation.get("description")
    if description:
        lines.extend(["", description.strip()])
    if parameters:
        lines.extend(["", "### Parameters"])
        for parameter in parameters:
            required = "required" if parameter.get("required") else "optional"
            lines.append(
                f"- `{parameter.get('name')}` ({parameter.get('in')}, {required}): {parameter.get('description', 'No description provided.')}"
            )
    request_body = operation.get("requestBody", {})
    if request_body:
        media_types = ", ".join(f"`{media}`" for media in request_body.get("content", {}))
        lines.extend(["", f"### Request body: {media_types}"])
    success_codes = [code for code in operation.get("responses", {}) if code.startswith("2")]
    lines.extend(
        [
            "",
            "### Expected success",
            "Documented success statuses: "
            + ", ".join(f"`{code}`" for code in success_codes or ["2xx"])
            + ".",
        ]
    )
    return "\n".join(lines)


def collection_state(
    original: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[tuple[str, str], tuple[dict[str, Any], str]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    """Copy a collection and index its variables, requests, and folders."""
    collection = copy.deepcopy(original)
    variables = {
        entry["key"]: entry
        for entry in collection.get("variable", [])
        if isinstance(entry, dict) and isinstance(entry.get("key"), str)
    }
    existing_items: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    folder_items: dict[str, list[dict[str, Any]]] = {}
    folder_templates: dict[str, dict[str, Any]] = {}

    for folder in collection.get("item", []):
        folder_name = folder.get("name", "Other")
        folder_templates[folder_name] = folder
        folder_items[folder_name] = []
        for item in folder.get("item", []):
            key = collection_operation_key(item)
            if key is not None:
                existing_items[key] = (item, folder_name)
    return collection, variables, existing_items, folder_items, folder_templates


def operation_item(
    key: tuple[str, str],
    operation: dict[str, Any],
    spec: dict[str, Any],
    existing_items: dict[tuple[str, str], tuple[dict[str, Any], str]],
    folder_items: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]:
    """Get a curated request or initialize a scaffold for a new operation."""
    path = key[1]
    parameters = operation_parameters(spec["paths"][path], operation)
    tags = operation.get("tags", ["Other"])
    raw_tag = tags[0] if tags else "Other"
    folder_name = TAG_FOLDERS.get(raw_tag.lower(), raw_tag)
    existing = existing_items.get(key)
    if existing is not None:
        item, folder_name = existing
        request = item.setdefault("request", {})
        request["method"] = key[0].upper()
        return item, request, folder_name, parameters

    item: dict[str, Any] = {
        "name": f"{key[0].upper()} {path} - {operation.get('summary', 'Request')}"
    }
    security = operation.get("security", spec.get("security", []))
    auth = (
        {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{access_token}}", "type": "string"}],
        }
        if security
        else {"type": "noauth"}
    )
    request: dict[str, Any] = {
        "method": key[0].upper(),
        "header": [{"key": "Accept", "value": "application/json"}],
        "auth": auth,
    }
    item["request"] = request
    folder_items.setdefault(folder_name, [])
    return item, request, folder_name, parameters


def update_expected_status(
    item: dict[str, Any], request: dict[str, Any], operation: dict[str, Any]
) -> None:
    """Refresh both the test assertion and its documented success status."""
    test_event = next(
        (event for event in item.get("event", []) if event.get("listen") == "test"), None
    )
    expected = sorted(
        int(code)
        for code in operation.get("responses", {})
        if isinstance(code, str) and code.isdigit() and code.startswith("2")
    ) or [200]
    if test_event:
        script = test_event.get("script", {}).get("exec", [])
        for index, line in enumerate(script):
            if line.startswith("const expectedSuccessStatuses = "):
                script[index] = f"const expectedSuccessStatuses = {json.dumps(expected)};"
                break
    status_text = ", ".join(f"`{code}`" for code in expected)
    description = request.get("description", "")
    description = re.sub(
        r"Default expected success status: `[^`]*`\.",
        f"Default expected success status: `{status_text}`.",
        description,
    )
    description = re.sub(
        r"Documented success statuses: .*?\.",
        f"Documented success statuses: {status_text}.",
        description,
    )
    request["description"] = description


def update_destructive_guard(item: dict[str, Any], request: dict[str, Any]) -> None:
    """Refresh destructive-operation policy in a curated pre-request guard."""
    guard = next(
        (
            event
            for event in item.get("event", [])
            if event.get("listen") == "prerequest"
            and "requiresDestructiveOptIn" in "\n".join(event.get("script", {}).get("exec", []))
        ),
        None,
    )
    if not guard:
        return
    expected = str(is_destructive_request(request)).lower()
    script = guard.get("script", {}).get("exec", [])
    for index, line in enumerate(script):
        if line.startswith("const requiresDestructiveOptIn = "):
            script[index] = f"const requiresDestructiveOptIn = {expected};"
            return


def assemble_folders(
    folder_items: dict[str, list[dict[str, Any]]],
    folder_templates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return nonempty folders while retaining their curated descriptions."""
    synced_folders: list[dict[str, Any]] = []
    for folder_name, items in folder_items.items():
        if not items:
            continue
        folder = folder_templates.get(folder_name, {"name": folder_name})
        folder["item"] = items
        folder.setdefault(
            "description",
            f"{folder_name} API operations; see each request for auth, parameters, payloads, and response details.",
        )
        synced_folders.append(folder)
    return synced_folders


def sync_collection(spec: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    """Reconcile request coverage and OpenAPI-owned metadata while preserving curation."""
    collection, variables, existing_items, folder_items, folder_templates = collection_state(
        original
    )
    components = spec.get("components", {})
    for key, (_, operation) in operation_entries(spec).items():
        item, request, folder_name, parameters = operation_item(
            key, operation, spec, existing_items, folder_items
        )
        update_url(request, key[1], parameters, variables, components)
        update_body(request, operation, variables, components)
        request.setdefault("description", operation_description(key[1], operation, parameters))
        if not item.get("event"):
            item["event"] = default_events(request, operation)
        else:
            update_expected_status(item, request, operation)
            update_destructive_guard(item, request)
        folder_items.setdefault(folder_name, []).append(item)

    collection["item"] = assemble_folders(folder_items, folder_templates)
    collection["variable"] = list(variables.values())
    return collection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="check for drift without writing the collection"
    )
    args = parser.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    collection = json.loads(COLLECTION_PATH.read_text(encoding="utf-8"))
    synced = sync_collection(spec, collection)
    if args.check:
        if synced != collection:
            print(
                "Postman collection is out of sync. Run: uv run python docs-site/scripts/sync_postman_collection.py"
            )
            return 1
        print("Postman collection matches docs-site/openapi.json.")
        return 0

    COLLECTION_PATH.write_text(
        json.dumps(synced, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    operation_count = len(operation_entries(spec))
    print(f"Synchronized {operation_count} OpenAPI operations to {COLLECTION_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
