# 0003 - RT-DETR from transformers as the detector, not Ultralytics YOLO

Status: accepted (2026-10)

## Context

The first plan used a YOLO detector. The Ultralytics YOLO package is AGPL-3.0, which is a
poor fit for an MIT repo that others may reuse, and a common reason companies avoid it.

## Decision

Use RT-DETR through Hugging Face transformers (Apache-2.0). It trains with the same tooling
as the VLM (transformers, accelerate), so there is one training stack instead of two.

## Consequences

- Possibly a little slower to train than YOLO on the same GPU; acceptable at this size.
- If YOLO is ever benchmarked for comparison, it stays out of the package dependencies.
