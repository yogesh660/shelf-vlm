from shelf.data.sku110k import split_paths, stats


def test_read_annotations(mini_annotations):
    a = mini_annotations["shelf_a.jpg"]
    assert (a.width, a.height) == (1000, 900)
    assert len(a.boxes) == 25


def test_zero_size_boxes_are_skipped(mini_annotations):
    assert len(mini_annotations["shelf_b.jpg"].boxes) == 3


def test_stats(mini_annotations):
    s = stats(mini_annotations)
    assert (s.images, s.boxes, s.boxes_per_image_max) == (2, 28, 25)
    assert s.boxes_per_image_median == 14.0
    assert stats({}).images == 0


def test_split_paths(tmp_path):
    paths = split_paths(tmp_path)
    assert set(paths) == {"train", "val", "test"}
    assert paths["val"].name == "annotations_val.csv"
