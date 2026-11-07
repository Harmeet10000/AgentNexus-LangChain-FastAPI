#!/usr/bin/env python3
"""Compose the effective spec for the behave run."""

import re
import sys
from pathlib import Path

RULE_RE = re.compile(r"^\s*Rule:\s*(.+)$")
MARKER_RE = re.compile(r"@openspec:\s*(ADDED|MODIFIED|REMOVED|RENAMED)")
SCENARIO_RE = re.compile(r"^\s*Scenario(?: Outline)?:\s*(.*)$")

HERE = Path(__file__).resolve().parent


class CompositionError(Exception):
    """Two active changes supersede the same rule."""


def _read_lines(feature_path):
    return re.split(r"\r?\n", (HERE / feature_path).read_text(encoding="utf-8"))


def _source_of(feature_path):
    out = re.sub(r"^\.extracted/", "../openspec/", str(feature_path))
    return re.sub(r"spec\.feature$", "spec.md", out)


def _capability_of(feature_path):
    parts = str(feature_path).split("/")
    return parts[len(parts) - 1 - parts[::-1].index("specs") + 1]


def _change_id_of(delta_path):
    parts = str(delta_path).split("/")
    return parts[parts.index("changes") + 1]


def collect_superseded_rules(delta_paths):
    superseded = {}
    for delta_path in delta_paths:
        capability = _capability_of(delta_path)
        change_id = _change_id_of(delta_path)
        pending_op = None
        for line in _read_lines(delta_path):
            marker = MARKER_RE.search(line)
            if marker:
                pending_op = marker.group(1)
                continue
            rule = RULE_RE.match(line)
            if not rule:
                continue
            name = rule.group(1).strip()
            if pending_op in ("MODIFIED", "REMOVED"):
                by_rule = superseded.setdefault(capability, {})
                other_change = by_rule.get(name)
                if other_change and other_change != change_id:
                    raise CompositionError(
                        'Active changes "%s" and "%s" both supersede rule "%s" of capability "%s".'
                        % (other_change, change_id, name, capability)
                    )
                by_rule[name] = change_id
    return superseded


def _rule_blocks(lines):
    starts = [(i, m.group(1).strip()) for i, m in ((i, RULE_RE.match(line)) for i, line in enumerate(lines)) if m]
    blocks = []
    for pos, (idx, name) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        blocks.append((name, idx, end))
    return blocks


def prune_source_of_truth_spec(spec_path, superseded_by_rule):
    lines = _read_lines(spec_path)
    seen_rules = set()
    excluded = []
    blanked = set()

    for name, start, end in _rule_blocks(lines):
        seen_rules.add(name)
        if name not in superseded_by_rule:
            continue
        entry = {"rule": name, "change_id": superseded_by_rule[name], "scenarios": []}
        for idx in range(start, end):
            scenario = SCENARIO_RE.match(lines[idx])
            if scenario:
                entry["scenarios"].append({"name": scenario.group(1).strip() or "(unnamed scenario)", "line": idx + 1})
            blanked.add(idx)
        excluded.append(entry)

    if not excluded:
        kept = sum(1 for line in lines if SCENARIO_RE.match(line))
        return kept, seen_rules, excluded

    pruned = ["" if i in blanked else line for i, line in enumerate(lines)]
    if len(pruned) != len(lines):
        raise CompositionError("%s: line-count invariant violated (pruning bug)" % spec_path)
    (HERE / spec_path).write_text("\n".join(pruned), encoding="utf-8")

    kept = sum(1 for line in pruned if SCENARIO_RE.match(line))
    return kept, seen_rules, excluded


def _print_composition_report(exclusions):
    left_out = 0
    for spec_path, capability, rules in exclusions:
        for entry in rules:
            sys.stderr.write("[effective-spec] %s / Rule: %s\n" % (capability, entry["rule"]))
            sys.stderr.write("[effective-spec]   superseded by change: %s\n" % entry["change_id"])
            for scenario in entry["scenarios"]:
                sys.stderr.write(
                    "[effective-spec]   left out: %s (%s:%d)\n"
                    % (scenario["name"], _source_of(spec_path), scenario["line"])
                )
                left_out += 1
    sys.stderr.write(
        "[effective-spec] %d source-of-truth scenario(s) excluded; delta versions run from openspec/changes/\n"
        % left_out
    )


def _rel(path):
    return path.relative_to(HERE).as_posix()


def effective_locations():
    delta_paths = sorted(
        _rel(p) for p in (HERE / ".extracted" / "changes").glob("*/specs/**/*.feature") if "changes/archive/" not in _rel(p)
    )
    sot_paths = sorted(_rel(p) for p in (HERE / ".extracted" / "specs").glob("**/*.feature"))

    superseded = collect_superseded_rules(delta_paths)
    locations = []
    seen_rules_by_capability = {}
    exclusions = []

    for spec_path in sot_paths:
        capability = _capability_of(spec_path)
        superseded_by_rule = superseded.get(capability)
        if not superseded_by_rule:
            locations.append(spec_path)
            continue
        kept, seen_rules, excluded = prune_source_of_truth_spec(spec_path, superseded_by_rule)
        if kept > 0:
            locations.append(spec_path)
        seen_rules_by_capability[capability] = seen_rules
        if excluded:
            exclusions.append((spec_path, capability, excluded))

    if exclusions:
        _print_composition_report(exclusions)

    for capability, by_rule in superseded.items():
        seen_rules = seen_rules_by_capability.get(capability, set())
        for name, change_id in by_rule.items():
            if name not in seen_rules:
                sys.stderr.write(
                    '[effective-spec] WARNING: change "%s" marks rule "%s" of capability "%s" as MODIFIED/REMOVED, but no such rule exists in openspec/specs.\n'
                    % (change_id, name, capability)
                )

    return locations + delta_paths


def source_of_truth_locations():
    return sorted(_rel(p) for p in (HERE / ".extracted" / "specs").glob("**/*.feature"))
