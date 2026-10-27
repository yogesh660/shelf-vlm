# 0001 - Detector for counting, VLM for reading and reasoning

Status: accepted (2026-10)

## Context

A retail audit needs two different things from a shelf photo: exact counts and positions
(how many facings, where the gaps are) and reading (which brand, which category, does this
match the plan). Shelves are dense: SKU-110K images have around a hundred or more products
each. VLMs read labels well but are unreliable at counting dozens of small, similar objects.
Detectors count well but don't know brands.

## Decision

Keep both, and measure each alone and combined:

- a detector finds product boxes; `rows.py` turns them into shelves and gaps with simple,
  testable rules;
- the VLM reads brands and categories and fills in a `ShelfReport`, with the detector's boxes
  passed in as hints in the combined setup;
- everything downstream (compliance, metrics) works on `ShelfReport`, whichever path made it.

The core (schema, rows, compliance, metrics) is plain Python so it can be tested without a GPU.

## Consequences

- Two models to train and serve instead of one.
- The rule-based gap finder gives a baseline from day one, before any model is trained.
- The README can answer "why not just ask the VLM?" with numbers.
