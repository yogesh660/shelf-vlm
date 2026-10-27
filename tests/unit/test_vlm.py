import base64
import json
from pathlib import Path

import pytest

from shelf.vlm import extract_report, image_message, parse_report

GOOD = (Path(__file__).parents[2] / "examples" / "report.json").read_text()


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, json_schema):
        self.calls.append((list(messages), json_schema))
        return self.replies.pop(0)


def test_valid_first_time():
    client = FakeClient(GOOD)
    result = extract_report(client, b"jpeg-bytes")
    assert result.ok
    assert result.attempts == 1
    _, schema = client.calls[0]
    assert schema["title"] == "ShelfReport"


def test_retry_shows_the_model_its_error():
    bad = json.dumps({"shelf_count": 1, "products": [{"brand": "X", "shelf": 3, "position": 0}]})
    client = FakeClient(bad, GOOD)
    result = extract_report(client, b"jpeg-bytes")
    assert result.ok
    assert result.attempts == 2
    retry_messages, _ = client.calls[1]
    assert "not valid" in retry_messages[-1]["content"]
    assert retry_messages[-2] == {"role": "assistant", "content": bad}


def test_gives_up_after_max_attempts():
    result = extract_report(FakeClient("no idea", "still no idea"), b"x", max_attempts=2)
    assert not result.ok
    assert len(result.errors) == 2


def test_fenced_json_is_accepted():
    assert parse_report(f"Here you go:\n```json\n{GOOD}\n```").shelf_count == 3


def test_no_json():
    with pytest.raises(ValueError, match="no JSON"):
        parse_report("I can't see the image")


def test_image_message_is_openai_format():
    msg = image_message(b"\xff\xd8abc", "image/png")
    url = msg["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8abc"
