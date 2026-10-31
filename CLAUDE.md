# CLAUDE.md

Notes for coding agents (and people) working in this repo. Read this before changing anything.

## What this is

Shelf photo in, structured retail-audit report out: products and facings per shelf, gaps,
share of shelf, planogram compliance. A fine-tuned vision-language model reads the shelf; a
detector helps with counting (ADR 0001). The plan is in `docs/ROADMAP.md`.

## Commands

Always use the make targets, they are exactly what CI runs. Dependencies are managed with uv.

- `make setup` - install the core package and dev tools (once)
- `make lock` - update `uv.lock` after editing dependencies in `pyproject.toml`
- `make test` - pytest
- `make lint` / `make format` - ruff
- `make typecheck` - mypy strict on `src/`
- `make check` - lint + typecheck + test. **Must pass before you commit.**
- `make schema` - the JSON schema the VLM answers in
- `make demo` - rows, gaps and compliance on the bundled example (no data, no GPU)
- `make data` / `make data-stats` - download SKU-110K (13.6 GB) and summarise it
- `make site` - regenerate the case-study page's diagrams, gap picture and chart data

## Layout

```
src/shelf/
  geometry.py       Box, IoU
  schema.py         ShelfReport: the output contract, also the VLM's JSON schema
  rows.py           detector boxes -> shelf rows -> gaps (non-VLM baseline)
  planogram.py      planned layout vs report -> compliance score and issues
  metrics.py        box matching, counting error
  vlm.py            prompt, OpenAI-compatible client, validate + retry
  data/sku110k.py   dataset download and annotation parsing
  cli.py            shelf schema | data | rows | score | analyse
examples/           a planogram and a report used by the demo and tests
tests/unit/         one test file per module: tests/unit/test_<module>.py
tests/fixtures/     tiny annotation file in the SKU-110K format
tests/test_harness.py  checks on the repo itself
docs/adr/           why things are the way they are
site/               public case-study page (Plotly charts, SVG diagrams), deployed by pages.yml
scripts/            site_assets.py builds site/img and site/data.js from the core and examples
train/              optional GPU entry points; RT-DETR training and prediction
```

## Rules

- The core in `src/shelf/` stays plain Python + pydantic. No torch, transformers, vllm or
  gradio imports there (`tests/test_harness.py` checks this). Training code will live in
  `train/`, the demo app in `app/`, both behind optional extras.
- `ShelfReport` is a contract. Changing a field changes the VLM's output format, the training
  data format and the evals. Do it in its own PR, update `examples/report.json`, say why.
- Every pydantic model the VLM fills in keeps `extra="forbid"`: guided decoding needs a
  closed schema, and silently dropped fields hide model mistakes.
- Shelf 0 is the top shelf, position 0 is the leftmost product. Everywhere.
- Tests never call a real model or download data. Use the fake client pattern in
  `tests/unit/test_vlm.py` and the fixtures in `tests/fixtures/`.
- No images of people, and no store photos in the repo. Data goes in `data/` (git-ignored).
- Don't edit `.github/workflows/` or loosen mypy/ruff settings to make a check pass. Fix the code.
- New dependency? Add it to `pyproject.toml`, run `make lock`, commit `uv.lock`, explain why.

## Branches, merges and releases

Nothing is committed to `main` directly. Every change goes through a short-lived branch and
lands only when its checks are green.

1. Branch from an up-to-date `main`: `feature/<topic>` for new work (one milestone or part
   of one), `fix/<topic>` for bugs. Keep it small enough to review in one sitting.
2. Commit in small logical steps; `make check` passes before every commit.
3. Push the branch. CI runs on it (plus any eval workflow the change triggers). Fix until
   every required workflow is green on the branch head.
4. Merge with a merge commit so the branch stays visible in history:
   `git checkout main && git pull && git merge --no-ff <branch> -m "Merge <branch>: <one-line summary>"`,
   then push `main`. The housekeeping workflow deletes merged `feature/` and `fix/` branches.
5. CI runs again on `main`. If it fails, fix it on a `fix/` branch straight away.
6. When a roadmap milestone is complete, end its heading in `docs/ROADMAP.md` with "(done)".
   After the merge, the housekeeping workflow tags `main` as `v0.<n+1>.0` (M0 -> v0.1.0,
   M1 -> v0.2.0, ...). Don't create or push tags by hand.

## Done means

- `make check` passes locally and CI is green on the branch
- merged into `main` through the flow above, never pushed to it directly
- new behaviour has a test; a metric change shows numbers before and after
- if you changed how to work in this repo, this file is updated too
- PR description filled in using `.github/pull_request_template.md`
