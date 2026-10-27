"""Ask a VLM for a ShelfReport and make sure what comes back is valid.

Works with any OpenAI-compatible server that accepts images (vLLM, Ollama, hosted APIs).
The server is asked for JSON matching ShelfReport's schema; vLLM enforces it with guided
decoding, others may not, so the reply is validated here regardless and the model gets one
retry with the validation error in front of it (ADR 0002).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from shelf.schema import ShelfReport

SYSTEM_PROMPT = """\
You analyse photos of supermarket shelves for retail audits.

Describe what is on the shelves as JSON matching the given schema:
- count shelves from the top (shelf 0 is the top one)
- list products left to right per shelf; identical packs side by side are one entry with
  a facings count
- use the brand name printed on the pack; if you cannot read it, use "unknown"
- report empty spaces where a product is missing as gaps
Answer with the JSON object only.
"""

Message = dict[str, Any]


class ChatClient(Protocol):
    def complete(self, messages: list[Message], json_schema: dict[str, Any] | None) -> str: ...


def image_message(
    image: bytes, mime: str = "image/jpeg", text: str = "Analyse this shelf."
) -> Message:
    data = base64.b64encode(image).decode()
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ],
    }


def parse_report(text: str) -> ShelfReport:
    # tolerate ```json fences and a sentence around the object
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    return ShelfReport.model_validate(json.loads(text[start : end + 1]))


@dataclass
class Extraction:
    report: ShelfReport | None
    attempts: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return self.report is not None


def extract_report(
    client: ChatClient, image: bytes, mime: str = "image/jpeg", max_attempts: int = 2
) -> Extraction:
    schema = ShelfReport.model_json_schema()
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        image_message(image, mime),
    ]
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        reply = client.complete(messages, schema)
        try:
            return Extraction(parse_report(reply), attempt, errors)
        except (ValueError, ValidationError) as exc:
            # json.JSONDecodeError is a ValueError too
            error = str(exc).splitlines()[0] if isinstance(exc, ValidationError) else str(exc)
            errors.append(error)
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That was not valid: {exc}\nReturn corrected JSON."},
            ]
    return Extraction(None, max_attempts, errors)


class OpenAICompatibleClient:
    """Minimal client, standard library only, so the core needs no SDK.

    SHELF_VLM_BASE_URL  e.g. http://localhost:8000/v1 for vLLM
    SHELF_VLM_MODEL     served model name
    SHELF_VLM_API_KEY   optional
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SHELF_VLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("SHELF_VLM_MODEL", "")
        self.api_key = api_key or os.environ.get("SHELF_VLM_API_KEY", "")
        self.timeout = timeout
        if not self.base_url or not self.model:
            raise ValueError("set SHELF_VLM_BASE_URL and SHELF_VLM_MODEL")

    def complete(self, messages: list[Message], json_schema: dict[str, Any] | None) -> str:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "shelf_report", "schema": json_schema},
            }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)
        return str(payload["choices"][0]["message"]["content"])
