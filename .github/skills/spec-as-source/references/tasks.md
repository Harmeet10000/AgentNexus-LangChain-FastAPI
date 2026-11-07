## 1. First-time setup — skip this whole section if `acceptance-tests/` already exists

- [ ] 1.1 Confirm the acceptance stack: read `stack:` from `openspec/config.yaml` (`javascript` or `python`). If absent, ask and add it — this is a specs-zone edit and must be committed on its own, before any scaffolding
- [ ] 1.2 Set up the application skeleton (entry point, dependencies, run script) so the app can boot
- [ ] 1.3 Create `acceptance-tests/` at the repo root as an independent `<stack>` project that boots the app before the suite and shuts it down after
- [ ] 1.4 Copy the acceptance-test-authoring skill's `<stack>` reference files verbatim into `acceptance-tests/` — see `references/<stack>/SETUP.md` "Files to copy"; destination filenames are load-bearing — so the runner extracts the Gherkin from every `spec.md` under `openspec/` into `acceptance-tests/.extracted/` and discovers the extracted features, excluding `openspec/changes/archive/`
- [ ] 1.5 Make the single test command always generate an HTML report under `acceptance-tests/reports/`
- [ ] 1.6 Add the spec-lint command (extract, then `gherkin-lint .extracted`) and gitignore `acceptance-tests/.extracted/` and `acceptance-tests/reports/`
- [ ] 1.7 Write `acceptance-tests/README.md` with instructions for running the suite and where the HTML report is written

## 2. Step definitions — one task per pending step; each = fails for the right reason → implement → passes → commit

- [ ] 2.1 <!-- feature: step --> — red → green → commit
- [ ] 2.2 <!-- feature: step --> — red → green → commit

## 3. Completion

- [ ] 3.1 Run the full suite: every scenario passes, zero pending/undefined steps, HTML report generated
