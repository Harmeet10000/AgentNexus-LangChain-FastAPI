# Python Stack - behave

For `stack: python` in `openspec/config.yaml`. Read the skill's `SKILL.md` first for the spec format and runner invariants, plus [../EXTRACTION.md](../EXTRACTION.md) and [../COMPOSITION.md](../COMPOSITION.md) — the two normative contracts this pack implements.

Requires `behave>=1.2.7` for Gherkin v6 `Rule:` support.

## Files To Copy

Copy into `acceptance-tests/` at the repo root. Destination filenames are load-bearing.

| Source | Destination | Why the name is fixed |
| --- | --- | --- |
| `python/extract_gherkin.py` | `extract_gherkin.py` | imported by `run_acceptance.py` |
| `python/openspec_effective_spec.py` | `openspec_effective_spec.py` | imported by `run_acceptance.py` |
| `python/run_acceptance.py` | `run_acceptance.py` | documented single entry point |
| `python/behave.ini` | `behave.ini` | behave discovers its config by this name |
| `python/environment.py` | `environment.py` | behave loads hooks from this name at the base dir |
| `../gherkin-lintrc.json` | `.gherkin-lintrc` | gherkin-lint auto-discovers this name and has no built-in defaults |

Dependencies: `behave>=1.2.7`, `behave-html-formatter`, plus an HTTP client and HTML parser for page objects. `requests` and `beautifulsoup4` are the cheerio analogue.

## Required Layout

```text
acceptance-tests/
  run_acceptance.py      environment.py     behave.ini
  extract_gherkin.py     openspec_effective_spec.py
  .gherkin-lintrc
  steps/                 # step definitions
  pages/                 # page objects
  .extracted/  reports/  # generated, gitignored
```

`steps/` and `environment.py` must sit at the `acceptance-tests/` root, not under a `features/` directory.

## Commands

```sh
python run_acceptance.py                    # effective spec
python run_acceptance.py --specs            # source of truth only
python run_acceptance.py --lint             # extract, then gherkin-lint
python run_acceptance.py --print-locations  # show resolved composition
python run_acceptance.py --dry-run          # passthrough to behave
```

Always go through the wrapper. behave parses feature files before `environment.py`'s `before_all` hook fires, so extraction cannot live in a hook. `before_all` refuses to start unless the wrapper set `OPENSPEC_ACCEPTANCE`.

## Exclusion

The contract is that superseded scenarios must not reach the runner and must not appear as skipped. behave's `spec.feature:12` selects at runtime and reports unselected scenarios as skipped, so this pack prunes superseded `Rule:` blocks out of the generated `.extracted/` tree and passes whole files to behave.

Pruning is not a spec edit. `.extracted/` is generated, gitignored, and rebuilt every run. Blanking rather than deleting keeps line fidelity.

## Verification

```sh
python run_acceptance.py --print-locations
python run_acceptance.py --dry-run
```

Locations must show only `.extracted/` entries and nothing under `changes/archive/`. Compare dry-run scenario count and names against the JS stack on the same specs.

## HTML Report

`behave.ini` pairs formatters with outfiles positionally. `html` must be listed first to claim `reports/behave-report.html`; `progress` gets `-` for stdout.

## Page Object Model

Page objects live in `acceptance-tests/pages/`, one per screen or flow. Parse responses with BeautifulSoup; never with regexes over raw HTML. behave's `context` is the World.

```python
@when('they submit a valid email and password')
def step_impl(context):
    context.result = context.signup_page.submit_signup(
        email='user@example.com',
        password='correct-horse-battery-staple',
    )


@then('an error message is shown')
def step_impl(context):
    assert context.signup_page.error_message()
```
