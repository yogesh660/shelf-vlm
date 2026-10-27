from shelf.geometry import Box
from shelf.rows import find_gaps, group_rows


def test_mini_shelf_rows_and_gaps(mini_annotations):
    boxes = mini_annotations["shelf_a.jpg"].boxes
    rows = group_rows(boxes)
    assert [len(r.boxes) for r in rows] == [10, 8, 7]

    gaps = find_gaps(rows)
    # shelf 1: two products missing in the middle; shelf 2: empty right end
    assert [(g.shelf, g.after_position) for g in gaps] == [(1, 3), (2, 6)]
    middle = gaps[0].box
    assert (middle.x1, middle.x2) == (390, 600)


def test_rows_are_left_to_right():
    boxes = [Box(x1=x, y1=0, x2=x + 10, y2=10) for x in (50, 0, 25)]
    (row,) = group_rows(boxes)
    assert [b.x1 for b in row.boxes] == [0, 25, 50]


def test_one_tall_pack_does_not_merge_shelves():
    shelf0 = [Box(x1=x, y1=0, x2=x + 10, y2=40) for x in range(0, 100, 12)]
    shelf1 = [Box(x1=x, y1=50, x2=x + 10, y2=90) for x in range(0, 100, 12)]
    tall = Box(x1=200, y1=0, x2=210, y2=60)  # a big bottle poking into the shelf below
    rows = group_rows([*shelf0, *shelf1, tall])
    assert len(rows) == 2
    assert tall in rows[0].boxes


def test_explicit_shelf_extent_finds_left_gap():
    row = group_rows([Box(x1=x, y1=0, x2=x + 10, y2=10) for x in (100, 112, 124)])
    gaps = find_gaps(row, shelf_x1=0)
    assert [(g.shelf, g.after_position) for g in gaps] == [(0, -1)]


def test_no_boxes():
    assert group_rows([]) == []
    assert find_gaps([]) == []
