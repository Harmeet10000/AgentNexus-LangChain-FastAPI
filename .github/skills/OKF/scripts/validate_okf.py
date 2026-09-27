#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML>=6,<7", "markdown-it-py>=3,<5", "mdit-py-plugins>=0.4,<1"]
# ///
"""Read-only OKF v0.2 validation; --strict adds producer hygiene checks.

Custom types and missing optional metadata are valid. Broken links are advisory
in default mode. Imported source links are reported separately. No remote URL,
document command, executor, or attester is fetched or executed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import operator
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

import yaml
from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin

if TYPE_CHECKING:
    from typing import Any

PARSER = MarkdownIt("commonmark").use(footnote_plugin)
START = "<!-- okf:original:start -->\n"
END = "\n<!-- okf:original:end -->"
RESERVED = {"index.md", "log.md"}


class UniqueLoader(yaml.SafeLoader):
    """Safe YAML loading without silently overwritten duplicate keys."""


def unique_mapping(loader, node, deep=False) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int, float, bool, type(None))):
            message = "YAML mapping keys must be scalars"
            raise TypeError(message)
        if key in result:
            message = f"Duplicate YAML key: {key}"
            raise ValueError(message)
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def timestamp(value) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str):
        # Python 3.10 requires +00:00; 3.11+ also accepts the ISO Z form.
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = dt.datetime.fromisoformat(normalized)
    else:
        message = "Expected ISO datetime with explicit UTC offset"
        raise TypeError(message)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        message = "Datetime lacks an explicit UTC offset"
        raise ValueError(message)
    return parsed


def actor(value) -> bool:
    return nonempty(value) and bool(
        re.fullmatch(r"(?:human:[^\s]+|process:[^\s]+|[^\s/:]+/[^\s]+)", value)
    )


def frontmatter(text) -> tuple[dict | None, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text
    for i, line in enumerate(lines[1:], 1):
        if line.strip() != "---":
            continue
        loader = UniqueLoader("".join(lines[1:i]))
        try:
            meta = loader.get_single_data()
        finally:
            loader.dispose()
        if not isinstance(meta, dict):
            message = "Frontmatter must be a YAML mapping"
            raise TypeError(message)
        return meta, "".join(lines[i + 1 :])
    message = "Unterminated frontmatter"
    raise ValueError(message)


def links(body) -> list[str]:
    result = []
    for token in PARSER.parse(body):
        for child in token.children or []:
            if child.type == "link_open":
                result.append(child.attrGet("href"))
            elif child.type == "image":
                result.append(child.attrGet("src"))
    return result


def visible_text(body) -> str:
    parts = []
    for token in PARSER.parse(body):
        for child in token.children or []:
            if child.type == "text":
                parts.append(child.content)
            elif child.type == "footnote_ref":
                parts.append("[^" + child.meta["label"] + "]")
    return "\n".join(parts)


def local_target(root, document, resource) -> Path | None:
    parsed = urlsplit(resource)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path)
    return (root / path.lstrip("/") if path.startswith("/") else document.parent / path).resolve()


def authored_body(body) -> tuple[str, str]:
    authored = []
    inherited = []
    remaining = body
    while START in remaining:
        before, rest = remaining.split(START, 1)
        if END not in rest:
            break
        passage, remaining = rest.split(END, 1)
        authored.append(before)
        inherited.append(passage)
    authored.append(remaining)
    return "".join(authored), "\n\n".join(inherited)


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.report: dict[str, Any] = {
            "okf_version": "0.2",
            "checked_at": dt.datetime.now().astimezone().isoformat(),
            "root": str(root),
            "documents": 0,
            "conformance_errors": [],
            "producer_issues": [],
            "inherited_link_issues": [],
            "preservation_errors": [],
            "originals_checked": 0,
            "segments_checked": 0,
            "trust_tiers": {},
        }

    def issue(self, path, message, bucket="producer_issues") -> None:
        self.report[bucket].append({"path": str(path.relative_to(self.root)), "message": message})

    def check_time(self, path, value, name) -> dt.datetime | None:
        try:
            return timestamp(value)
        except (ValueError, TypeError, OverflowError) as exc:
            self.issue(path, f"{name}: {exc}")
            return None

    def check_link(self, path, value, inherited=False) -> None:
        bucket = "inherited_link_issues" if inherited else "producer_issues"
        try:
            target = local_target(self.root, path, value)
            if target is not None and (not target.is_relative_to(self.root) or not target.exists()):
                self.issue(path, f"Unresolved or external-to-bundle local link: {value}", bucket)
        except (ValueError, TypeError, OSError) as exc:
            self.issue(path, f"Invalid link {value!r}: {exc}", bucket)

    def index(self, path, meta, body) -> None:
        if meta is not None and path != self.root / "index.md":
            self.issue(path, "Only bundle-root index may have frontmatter", "conformance_errors")
        elif meta is not None and set(meta) != {"okf_version"}:
            self.issue(path, "Root index frontmatter should contain only okf_version")
        if meta and str(meta.get("okf_version")) != "0.2":
            self.issue(path, 'This producer profile targets okf_version: "0.2"')
        if not re.search(r"^#{1,6}\s+\S", body, re.MULTILINE):
            self.issue(path, "Index lacks section headings", "conformance_errors")
        if not links(body):
            self.issue(path, "Index has no linked entries")

    def log(self, path, meta, body) -> None:
        if meta is not None:
            self.issue(path, "Logs do not have frontmatter", "conformance_errors")
        dates = re.findall(r"^##\s+(.+?)\s*$", body, re.MULTILINE)
        if not dates:
            self.issue(path, "Log lacks date-grouped entries", "conformance_errors")
        for date in dates:
            valid = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date))
            try:
                dt.date.fromisoformat(date)
            except ValueError:
                valid = False
            if not valid:
                self.issue(path, f"Invalid log date: {date}", "conformance_errors")
        if dates != sorted(dates, reverse=True):
            self.issue(path, "Log dates should be newest first")

    def trust(self, path, meta) -> None:
        if "generated" in meta:
            generation = meta["generated"]
            if not isinstance(generation, dict) or not actor(generation.get("by")):
                self.issue(path, "generated requires by in the actor convention")
            elif "at" in generation:
                self.check_time(path, generation["at"], "generated.at")
        events = meta.get("verified", [])
        if isinstance(events, dict):
            events = [events]
        if not isinstance(events, list):
            self.issue(path, "verified must be a mapping or list of mappings")
            events = []
        valid_events = []
        for event in events:
            if not isinstance(event, dict) or not actor(event.get("by")) or "at" not in event:
                self.issue(path, "Each verified event needs by and at")
            elif self.check_time(path, event["at"], "verified.at"):
                valid_events.append(event)
        tier = "machine-confirmed" if valid_events else "unverified"
        if any(event["by"].startswith("human:") for event in valid_events):
            tier = "human-reviewed"
        self.report["trust_tiers"][tier] = self.report["trust_tiers"].get(tier, 0) + 1

    def source_record(self, path, source, source_ids) -> None:
        if not isinstance(source, dict) or not nonempty(source.get("resource")):
            self.issue(path, "Each source requires a non-empty resource")
            return
        key = source.get("id")
        if key is not None:
            if not nonempty(key) or key in source_ids:
                self.issue(path, "Source ids must be non-empty and unique when present")
            else:
                source_ids.add(key)
        resource = source["resource"]
        explicit_path = resource.startswith(("/", "./", "../"))
        if explicit_path or (" " not in resource and "/" in resource):
            self.check_link(path, resource)
        if "last_modified" in source:
            self.check_time(path, source["last_modified"], "sources.last_modified")
        if "usage_count" in source and (
            type(source["usage_count"]) is not int or source["usage_count"] < 0
        ):
            self.issue(path, "usage_count should be a non-negative integer")

    def window(self, path, window) -> None:
        if window is None:
            return
        if not isinstance(window, dict) or not {"from", "to"} <= window.keys():
            self.issue(path, "usage_window requires from and to")
            return
        start = self.check_time(path, window["from"], "usage_window.from")
        end = self.check_time(path, window["to"], "usage_window.to")
        if start and end and end < start:
            self.issue(path, "usage_window.to precedes from")

    def provenance(self, path, meta, body) -> None:
        sources = meta.get("sources", [])
        if not isinstance(sources, list):
            self.issue(path, "sources should be a list")
            sources = []
        source_ids: set[str] = set()
        for source in sources:
            self.source_record(path, source, source_ids)
            if isinstance(source, dict):
                self.window(path, source.get("usage_window"))
        self.window(path, meta.get("usage_window"))
        cited = set(re.findall(r"\[\^([^\]\s]+)\]", visible_text(body)))
        definitions = set(re.findall(r"^\[\^([^\]\s]+)\]:", body, re.MULTILINE))
        for key in cited | definitions:
            if key not in source_ids:
                self.issue(path, f"Footnote {key} lacks a matching sources[].id")
        for key in cited - definitions:
            self.issue(path, f"Footnote {key} lacks a body definition")

    def parameters(self, path, parameters) -> None:
        if not isinstance(parameters, list):
            self.issue(path, "parameters should be a list")
            return
        names = set()
        for param in parameters:
            if (
                not isinstance(param, dict)
                or not nonempty(param.get("name"))
                or not nonempty(param.get("type"))
                or type(param.get("required")) is not bool
            ):
                self.issue(path, "Each parameter needs name, type, and boolean required")
            elif param["name"] in names:
                self.issue(path, "Duplicate computation parameter")
            else:
                names.add(param["name"])

    def computation(self, path, meta, body) -> None:
        if meta.get("type") != "Attested Computation":
            return
        if not nonempty(meta.get("runtime")):
            self.issue(path, "Attested Computation requires runtime in its producer contract")
        if meta.get("computation"):
            self.check_link(path, meta["computation"])
        elif not re.search(r"^# Computation\s*$[\s\S]*?(?:^```|^~~~)", body, re.MULTILINE):
            self.issue(path, "Provide a computation path or a fenced block under # Computation")
        self.parameters(path, meta.get("parameters", []))
        for name in ("executor", "attester"):
            if name not in meta:
                continue
            obj = meta[name]
            if not isinstance(obj, dict) or not nonempty(obj.get("resource")):
                self.issue(path, f"{name} requires resource when supplied")
            else:
                self.check_link(path, obj["resource"])

    def metadata(self, path, meta, body) -> None:
        if "tags" in meta and (
            not isinstance(meta["tags"], list) or not all(nonempty(x) for x in meta["tags"])
        ):
            self.issue(path, "tags should be a list of non-empty strings")
        if "status" in meta and (
            not isinstance(meta["status"], str)
            or meta["status"] not in {"draft", "stable", "deprecated"}
        ):
            self.issue(path, "Producer status should be draft, stable, or deprecated")
        if "timestamp" in meta:
            self.issue(path, "Use generated.at rather than legacy timestamp")
        if "stale_after" in meta:
            self.check_time(path, meta["stale_after"], "stale_after")
        self.trust(path, meta)
        self.provenance(path, meta, body)
        self.computation(path, meta, body)

    def document(self, path) -> None:
        if path.is_symlink():
            self.issue(path, "Symlink document is not portable; not followed")
            return
        self.report["documents"] += 1
        try:
            meta, body = frontmatter(path.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError, yaml.YAMLError, OSError, TypeError) as exc:
            self.issue(path, str(exc), "conformance_errors")
            return
        if path.name == "index.md":
            self.index(path, meta, body)
        elif path.name == "log.md":
            self.log(path, meta, body)
        elif meta is None or not nonempty(meta.get("type")):
            self.issue(
                path,
                "Concept requires frontmatter with non-empty string type",
                "conformance_errors",
            )
            return
        if START in body and END not in body:
            self.issue(path, "Unterminated preservation marker")
        authored, inherited = authored_body(body)
        for href in links(authored):
            self.check_link(path, href)
        for href in links(inherited):
            self.check_link(path, href, inherited=True)
        if path.name not in RESERVED:
            self.metadata(path, meta, authored)


def safe_relative(value) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        message = "Unsafe path in preservation manifest"
        raise ValueError(message)
    return relative


def segment_payload(root, segment, copied, position, problems) -> bytes:
    raw = (root / safe_relative(segment["path"])).read_bytes()
    start, end = segment["start"], segment["end"]
    offset, size = segment["payload_offset"], segment["payload_bytes"]
    if not all(type(v) is int and v >= 0 for v in (start, end, offset, size)):
        message = "Invalid segment offsets"
        raise ValueError(message)
    payload = raw[offset : offset + size]
    if start != position or end < start or end - start != size:
        problems.append(f"Gap, overlap, or invalid range: {segment['path']}")
    if payload != copied[start:end] or hashlib.sha256(payload).hexdigest() != segment["sha256"]:
        problems.append(f"Altered source passage: {segment['path']}")
    if raw[max(0, offset - len(START.encode())) : offset] != START.encode():
        problems.append(f"Payload start marker mismatch: {segment['path']}")
    if not raw[offset + size :].startswith(END.encode()):
        problems.append(f"Payload end marker mismatch: {segment['path']}")
    return payload


def reconstruct(root, selected, copied, report, problems) -> None:
    if not selected:
        problems.append("Markdown source has no preserved segments")
    position = 0
    reconstructed = bytearray()
    for segment in selected:
        reconstructed.extend(segment_payload(root, segment, copied, position, problems))
        position = segment["end"]
        report["segments_checked"] += 1
    if bytes(reconstructed) != copied or position != len(copied):
        problems.append("Segments do not reconstruct the complete original")


def check_preservation(root, manifest_path, originals, report) -> None:
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        name = str(safe_relative(entry["path"]))
        default_asset = "assets/originals/" + (name + ".txt" if name.endswith(".md") else name)
        asset = root / safe_relative(entry.get("asset", default_asset))
        if not asset.resolve().is_relative_to(root.resolve()):
            message = "Preserved asset escapes bundle root"
            raise ValueError(message)
        copied = asset.read_bytes()
        problems: list[str] = []
        if len(copied) != entry["bytes"] or hashlib.sha256(copied).hexdigest() != entry["sha256"]:
            problems.append("Preserved asset differs from baseline")
        if originals is not None:
            current = (originals / name).read_bytes()
            report["originals_checked"] += 1
            if current != copied:
                problems.append("Original file differs from preserved asset")
        if name.endswith(".md"):
            selected = sorted(
                (s for s in manifest["segments"] if s["source"] == name),
                key=operator.itemgetter("start"),
            )
            reconstruct(root, selected, copied, report, problems)
        for problem in problems:
            report["preservation_errors"].append({"path": name, "message": problem})


def validate(root, strict=False, manifest=None, originals=None) -> dict[str, Any]:
    root = Path(root).resolve()
    checker = Checker(root)
    if not root.is_dir():
        checker.issue(root, "Bundle directory does not exist", "conformance_errors")
    for path in sorted(root.rglob("*.md")):
        checker.document(path)
    if manifest is not None:
        try:
            check_preservation(
                root,
                Path(manifest),
                Path(originals).resolve() if originals else None,
                checker.report,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            checker.report["preservation_errors"].append(
                {"path": str(manifest), "message": f"Invalid manifest: {exc}"}
            )
    report = checker.report
    report["ok"] = not (
        report["conformance_errors"]
        or report["preservation_errors"]
        or (strict and report["producer_issues"])
    )
    return report


def summary(result) -> str:
    rows = [
        ("PASS" if result["ok"] else "FAIL") + f": {result['documents']} Markdown documents; "
        f"{len(result['conformance_errors'])} conformance errors; {len(result['producer_issues'])} producer issues; "
        f"{len(result['preservation_errors'])} preservation errors.",
        f"Preservation: {result['originals_checked']} originals, {result['segments_checked']} source passages checked.",
        f"Inherited unresolved links: {len(result['inherited_link_issues'])} (preserved and reported).",
    ]
    for bucket in ("conformance_errors", "producer_issues", "preservation_errors"):
        rows.extend(f"{bucket}: {item['path']}: {item['message']}" for item in result[bucket][:20])
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: uv run validate_okf.py knowledge --strict --json. "
        "Exit codes: 0 pass; 1 validation failed; 2 invalid invocation or report I/O. "
        "Document content is never changed. Only --output writes a report.",
    )
    parser.add_argument(
        "bundle", type=Path, help="Actual OKF bundle root, not its originals workspace"
    )
    parser.add_argument("--strict", action="store_true", help="Fail on producer-profile issues too")
    parser.add_argument("--manifest", type=Path, help="Optional preservation manifest JSON")
    parser.add_argument(
        "--originals", type=Path, help="Compare the manifest against untouched originals"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the complete machine-readable report"
    )
    parser.add_argument(
        "--output", type=Path, help="Write the report to a new or explicitly selected report file"
    )
    args = parser.parse_args()
    if args.originals and not args.manifest:
        parser.error("--originals requires --manifest")
    if args.output and args.output.suffix.lower() != ".json":
        parser.error("--output must be a .json report, never a source document")
    if args.output and args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot replace a non-report file: {exc}")
        if (
            not isinstance(existing, dict)
            or not {"okf_version", "conformance_errors", "preservation_errors", "ok"}
            <= existing.keys()
        ):
            parser.error("--output exists and is not a validation report; choose a new path")
    result = validate(args.bundle, args.strict, args.manifest, args.originals)
    if args.output:
        try:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        except OSError as exc:
            parser.error(f"Cannot write --output report: {exc}")
    sys.stdout.write((json.dumps(result, indent=2) if args.json else summary(result)) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
