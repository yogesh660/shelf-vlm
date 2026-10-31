from train.rtdetr import format_coco_annotations, parser

from shelf.data.sku110k import ImageAnnotations
from shelf.geometry import Box


def test_formats_sku110k_box_as_coco():
    ann = ImageAnnotations("a.jpg", 100, 80, [Box(x1=10, y1=20, x2=40, y2=60)])
    result = format_coco_annotations(7, ann)
    assert result == {
        "image_id": 7,
        "annotations": [
            {
                "image_id": 7,
                "category_id": 0,
                "iscrowd": 0,
                "area": 1200,
                "bbox": [10.0, 20.0, 30.0, 40.0],
            }
        ],
    }


def test_training_defaults_are_reproducible():
    args = parser().parse_args(["train"])
    assert args.model == "PekingU/rtdetr_r50vd"
    assert args.epochs == 10
    assert args.batch_size == 2
    assert args.gradient_accumulation == 8
