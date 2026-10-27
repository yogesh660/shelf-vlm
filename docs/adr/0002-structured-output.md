# 0002 - The VLM answers in a closed JSON schema, validated twice

Status: accepted (2026-10)

## Context

A report is only useful if a downstream system can read it. Free-text answers, or JSON with
invented fields, break that.

## Decision

- `ShelfReport` (pydantic) is the single definition of the output. Its JSON schema is sent to
  the server as `response_format`; vLLM enforces it with guided decoding.
- All models use `extra="forbid"`, so the schema is closed (`additionalProperties: false`).
- The reply is validated again in `vlm.py` regardless of the server, because not every server
  enforces schemas. On failure the model gets one retry with the validation error.
- Cross-field rules (shelf index within `shelf_count`, no two products in one slot) live in
  the model validator, so they apply to training data, model output and tests alike.

## Consequences

- JSON validity rate becomes a reported metric, before and after guided decoding.
- Changing the schema is a deliberate act: it changes training data and evals too.
