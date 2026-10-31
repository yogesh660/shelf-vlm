"""Fine-tune and run RT-DETR on SKU-110K.

Install the optional stack first: ``uv sync --extra train``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from shelf.data.sku110k import ImageAnnotations, read_annotations, split_paths

DEFAULT_MODEL = "PekingU/rtdetr_r50vd"


def format_coco_annotations(image_id: int, ann: ImageAnnotations) -> dict[str, Any]:
    """Convert one SKU-110K record to the COCO format expected by the processor."""
    annotations = []
    for box in ann.boxes:
        width, height = box.width, box.height
        annotations.append(
            {
                "image_id": image_id,
                "category_id": 0,
                "iscrowd": 0,
                "area": width * height,
                "bbox": [box.x1, box.y1, width, height],
            }
        )
    return {"image_id": image_id, "annotations": annotations}


class SKU110KDataset:
    """Lazy image dataset; importing this module does not require the training extras."""

    def __init__(self, annotations: Path, images: Path, processor: Any) -> None:
        self.items = sorted(read_annotations(annotations).values(), key=lambda item: item.image)
        self.images = images
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        ann = self.items[index]
        with Image.open(self.images / ann.image) as source:
            image = source.convert("RGB")
        encoded = self.processor(
            images=image,
            annotations=format_coco_annotations(index, ann),
            return_tensors="pt",
        )
        item = {"pixel_values": encoded["pixel_values"].squeeze(0), "labels": encoded["labels"][0]}
        if "pixel_mask" in encoded:
            item["pixel_mask"] = encoded["pixel_mask"].squeeze(0)
        return item


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    result = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }
    if "pixel_mask" in batch[0]:
        result["pixel_mask"] = torch.stack([item["pixel_mask"] for item in batch])
    return result


def train(args: argparse.Namespace) -> None:
    from transformers import (
        AutoImageProcessor,
        RTDetrForObjectDetection,
        Trainer,
        TrainingArguments,
    )

    paths = split_paths(args.data)
    images = args.data / "SKU110K_fixed" / "images"
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = RTDetrForObjectDetection.from_pretrained(
        args.model,
        id2label={0: "object"},
        label2id={"object": 0},
        ignore_mismatched_sizes=True,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        dataloader_num_workers=args.workers,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SKU110KDataset(paths["train"], images, processor),
        eval_dataset=SKU110KDataset(paths["val"], images, processor),
        processing_class=processor,
        data_collator=collate_fn,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)


def predict(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, RTDetrForObjectDetection

    annotations = read_annotations(args.annotations)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = RTDetrForObjectDetection.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh, torch.no_grad():
        writer = csv.writer(fh)
        for ann in sorted(annotations.values(), key=lambda item: item.image):
            with Image.open(args.images / ann.image) as source:
                image = source.convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            result = processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([[ann.height, ann.width]], device=device),
                threshold=args.score,
            )[0]
            for score, box in zip(result["scores"], result["boxes"], strict=True):
                writer.writerow([ann.image, *box.tolist(), score.item()])


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(required=True)
    command = commands.add_parser("train")
    command.add_argument("--data", type=Path, default=Path("data/sku110k"))
    command.add_argument("--output", type=Path, default=Path("outputs/rtdetr-sku110k"))
    command.add_argument("--model", default=DEFAULT_MODEL)
    command.add_argument("--epochs", type=int, default=10)
    command.add_argument("--batch-size", type=int, default=2)
    command.add_argument("--gradient-accumulation", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=5e-5)
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--precision", choices=("no", "fp16", "bf16"), default="fp16")
    command.add_argument("--resume", action="store_true")
    command.set_defaults(func=train)

    command = commands.add_parser("predict")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--annotations", type=Path, required=True)
    command.add_argument("--images", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--score", type=float, default=0.5)
    command.set_defaults(func=predict)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
