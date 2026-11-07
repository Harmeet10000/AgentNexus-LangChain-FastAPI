#!/usr/bin/env python3
"""Extracts classic Gherkin from Markdown specs (spec.md) into .feature files,
synthesizing Feature:/Rule:/Scenario: from the Markdown headings and copying
fenced step lines verbatim.

../EXTRACTION.md IS THE DEFINITION -- the line-by-line mapping, the fence
mechanics, the line-fidelity invariant and every hard error live there. This
file is one binding of it; javascript/extract-gherkin.cjs is the other, and the
two must stay behaviourally identical -- same regexes, same hard errors, same
blanking. Change the doc first, then both implementations, then re-verify (see
the skill's "Port parity" note).

Deliberately dependency-free (stdlib only) and 3.8+ compatible: the CLI form
must run from the skill's references/python/ directory, where no virtualenv
exists.
"""

import re
import shutil
import sys
from pathlib import Path

GHERKIN_OPEN_RE = re.compile(r"^(`{3,})gherkin\s*$")
ANY_OPEN_RE = re.compile(r"^(`{3,})\S*\s*$")
INDENTED_GHERKIN_RE = re.compile(r"^\s+`{3,}gherkin\s*$")

HEADING_RE = re.compile(r"^#{1,6}\s+")
H1_RE = re.compile(r"^#\s+(.+?)\s*$")
DELTA_SECTION_RE = re.compile(
    r"^##\s+(ADDED|MODIFIED|REMOVED|RENAMED)\s+Requirements\s*$", re.IGNORECASE
)
REQUIREMENT_RE = re.compile(r"^###\s+Requirement:\s*(.+?)\s*$", re.IGNORECASE)
SCENARIO_RE = re.compile(r"^####\s+(Scenario(?:\s+Outline)?):\s*(.+?)\s*$", re.IGNORECASE)

# Structure keywords are illegal inside a fence -- they come from the headings.
# `Examples:` (the Scenario Outline table) and `Background:` are deliberately
# absent: both legitimately live in a fence.
STRUCTURE_IN_FENCE_RE = re.compile(r"^\s*(Feature|Rule|Scenario\s+Outline|Scenario|Example):")


class ExtractionError(Exception):
    """A spec.md could not be extracted. Always names file:line."""


def extract_file(md_path):
    """Returns the extracted .feature text for one spec.md, line count preserved."""
    md_path = Path(md_path)
    lines = re.split(r"\r?\n", md_path.read_text(encoding="utf-8"))
    out = []
    state = "prose"  # 'prose' | 'gherkin' | 'other-fence'
    fence_ticks = 0
    open_line = 0
    h1_count = 0
    pending_scenario = None  # scenario heading still awaiting its fence
    close_re = None

    for i, line in enumerate(lines):
        if state == "prose":
            m = GHERKIN_OPEN_RE.match(line)
            if m:
                state = "gherkin"
                fence_ticks = len(m.group(1))
                close_re = re.compile(r"^`{%d,}\s*$" % fence_ticks)
                open_line = i + 1
                pending_scenario = None
                out.append("")
                continue
            if INDENTED_GHERKIN_RE.match(line):
                raise ExtractionError(
                    "%s:%d: indented ```gherkin fence — gherkin fences must "
                    "start at column 0" % (md_path, i + 1)
                )
            m = ANY_OPEN_RE.match(line)
            if m:
                state = "other-fence"
                fence_ticks = len(m.group(1))
                close_re = re.compile(r"^`{%d,}\s*$" % fence_ticks)
                open_line = i + 1
                out.append("")
                continue
            if not HEADING_RE.match(line):
                out.append("")
                continue
            # A heading ends any scenario still waiting for its steps.
            if pending_scenario:
                raise ExtractionError(
                    '%s:%d: "#### %s: %s" has no ```gherkin fence before the next heading'
                    % (md_path, pending_scenario[0], pending_scenario[1], pending_scenario[2])
                )
            m = H1_RE.match(line)
            if m:
                h1_count += 1
                if h1_count > 1:
                    raise ExtractionError(
                        '%s:%d: more than one H1 — a spec.md has exactly one '
                        '"# <capability>" title' % (md_path, i + 1)
                    )
                out.append("Feature: %s" % m.group(1))
                continue
            m = DELTA_SECTION_RE.match(line)
            if m:
                out.append("  # @openspec: %s" % m.group(1).upper())
                continue
            m = REQUIREMENT_RE.match(line)
            if m:
                out.append("  Rule: %s" % m.group(1))
                continue
            m = SCENARIO_RE.match(line)
            if m:
                keyword = "Scenario Outline" if re.search(r"outline", m.group(1), re.I) else "Scenario"
                pending_scenario = (i + 1, keyword, m.group(2))
                out.append("    %s: %s" % (keyword, m.group(2)))
                continue
            out.append("")
            continue

        if close_re.match(line):
            state = "prose"
            out.append("")
            continue
        if state != "gherkin":
            out.append("")
            continue
        kw = STRUCTURE_IN_FENCE_RE.match(line)
        if kw:
            raise ExtractionError(
                '%s:%d: "%s:" inside a ```gherkin fence — structure comes from Markdown '
                'headings ("# title", "### Requirement:", "#### Scenario:"); fences hold '
                "only steps" % (md_path, i + 1, kw.group(1))
            )
        out.append(line)

    if state != "prose":
        raise ExtractionError("%s:%d: unclosed fence" % (md_path, open_line))
    if pending_scenario:
        raise ExtractionError(
            '%s:%d: "#### %s: %s" has no ```gherkin fence before the end of the file'
            % (md_path, pending_scenario[0], pending_scenario[1], pending_scenario[2])
        )
    if h1_count == 0:
        raise ExtractionError(
            '%s: no H1 title — a spec.md must start with "# <capability>"' % md_path
        )
    if len(out) != len(lines):
        raise ExtractionError("%s: line-count invariant violated (extractor bug)" % md_path)
    return "\n".join(out)


def _walk(root, directory, basename, found):
    """Collects files named <basename> under <directory>, as posix paths
    relative to <root>. '*.ext' is supported as a suffix match."""
    if not directory.is_dir():
        return found
    for entry in sorted(directory.iterdir()):
        if entry.is_dir():
            _walk(root, entry, basename, found)
        elif entry.name == basename or (
            basename.startswith("*.") and entry.name.endswith(basename[1:])
        ):
            found.append(entry.relative_to(root).as_posix())
    return found


def collect_spec_sources(openspec_dir, basename):
    """Spec roots: specs/ (source of truth) and each active change's specs/ --
    changes/archive/ is excluded structurally (archive nests one level deeper
    than changes/<id>/) plus a defensive filter on the collected paths."""
    openspec_dir = Path(openspec_dir)
    found = _walk(openspec_dir, openspec_dir / "specs", basename, [])
    changes_dir = openspec_dir / "changes"
    if changes_dir.is_dir():
        for entry in sorted(changes_dir.iterdir()):
            if not entry.is_dir() or entry.name == "archive":
                continue
            _walk(openspec_dir, entry / "specs", basename, found)
    return sorted(p for p in found if "changes/archive/" not in p)


def extract_all(openspec_dir=None, out_dir=None):
    """Extracts every spec.md under <openspec_dir> (source of truth + active
    change deltas, archive excluded) into <out_dir>, mirroring the
    openspec-relative path with spec.md -> spec.feature. The output dir is
    wiped first -- a stale extraction would keep deleted or renamed
    capabilities executing."""
    here = Path(__file__).resolve().parent
    openspec_dir = Path(openspec_dir).resolve() if openspec_dir else (here / ".." / "openspec").resolve()
    out_dir = Path(out_dir).resolve() if out_dir else (here / ".extracted").resolve()

    shutil.rmtree(out_dir, ignore_errors=True)

    sources = collect_spec_sources(openspec_dir, "spec.md")

    # Legacy-format tripwire: raw .feature files under openspec/ no longer run
    # anywhere -- flag them instead of letting them silently drop out.
    legacy = collect_spec_sources(openspec_dir, "*.feature")
    if legacy:
        sys.stderr.write(
            "[extract-gherkin] WARNING: legacy .feature file(s) under openspec/ "
            "are ignored (specs are spec.md now): %s\n" % ", ".join(legacy)
        )

    written = []
    for rel in sources:
        dest = out_dir / re.sub(r"spec\.md$", "spec.feature", rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(extract_file(openspec_dir / rel), encoding="utf-8")
        written.append(dest)
    return out_dir, written


# CLI: python extract_gherkin.py [openspecDir] [outDir]
if __name__ == "__main__":
    try:
        out, written_files = extract_all(
            sys.argv[1] if len(sys.argv) > 1 else None,
            sys.argv[2] if len(sys.argv) > 2 else None,
        )
        sys.stderr.write(
            "[extract-gherkin] %d spec.md file(s) extracted to %s\n" % (len(written_files), out)
        )
    except ExtractionError as err:
        sys.stderr.write("[extract-gherkin] %s\n" % err)
        sys.exit(1)
