from __future__ import annotations

import base64
import io
import math
import os
import re
import threading
import time
from types import TracebackType

from openai import APIConnectionError, APITimeoutError, OpenAI, Timeout
from PIL import Image

from .grounding import GROUNDING_PROMPT, MARKER, parse_blocks, profile
from .postprocess import fix_escaped_dollars

DEFAULT_BASE_URL = "http://127.0.0.1:8010/v1"
DEFAULT_CONCURRENCY = 64


def error_message(error: BaseException, api_key: str | None = None) -> str:
    secret = api_key or os.environ.get("LIGHTONOCR_API_KEY")
    if isinstance(error, APIConnectionError):
        url = error.request.url.copy_with(username=None, password=None, query=None, fragment=None)
        if isinstance(error, APITimeoutError):
            message = f"Model server did not respond in time at {url}. Check whether it is loading or busy."
        else:
            message = f"Cannot connect to the model server at {url}. Start the model server or check the API base URL."
    else:
        message = f"{type(error).__name__}: {error}"
    return message.replace(secret, "[redacted]") if secret else message


def fit(image: Image.Image, longest_edge: int | None = 2048) -> Image.Image:
    if longest_edge is None:
        return image
    width, height = image.size
    scale = longest_edge / max(width, height)
    if scale >= 1:
        return image
    return image.resize(
        (max(1, round(width * scale)),
         max(1, round(height * scale))), Image.Resampling.LANCZOS
    )


def to_data_url(image: Image.Image, longest_edge: int | None = None) -> str:
    with image.convert("RGB") as rgb, io.BytesIO() as buffer:
        resized = fit(rgb, longest_edge)
        try:
            resized.save(buffer, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        finally:
            if resized is not rgb:
                resized.close()


_MARKER_START = re.compile(r"!\[[A-Za-z_][\w+-]*\]\(")


class GroundingError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        raw: str = "",
        usage: dict | None = None,
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.usage = usage or {}
        self.finish_reason = finish_reason


class GroundingClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
        max_tokens: int | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if max_tokens is not None and (
            not isinstance(max_tokens, int) or isinstance(
                max_tokens, bool) or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        self._model = (model if model is not None else os.environ.get(
            "LIGHTONOCR_MODEL")) or None
        if self._model is not None:
            self._validate_model(self._model)
        self.client = OpenAI(
            base_url=base_url or os.environ.get(
                "LIGHTONOCR_BASE_URL") or DEFAULT_BASE_URL,
            api_key=api_key or os.environ.get("LIGHTONOCR_API_KEY") or "EMPTY",
            timeout=Timeout(timeout, connect=10.0),
        )
        self.max_tokens = max_tokens
        self._lock = threading.Lock()

    @staticmethod
    def _validate_model(model: str) -> None:
        if profile(model).generation != 3:
            raise ValueError(
                "A recognizable LightOnOCR-3 checkpoint name is required, but "
                f"received {model!r}. Preserve the checkpoint name in the served alias."
            )

    @property
    def model(self) -> str:
        with self._lock:
            if self._model is None:
                models = (
                    self.client.with_options(timeout=Timeout(
                        10.0, connect=5.0), max_retries=0)
                    .models.list()
                    .data
                )
                candidates = [m.id for m in models if profile(
                    m.id).generation == 3]
                if len(candidates) != 1:
                    raise ValueError(
                        "Model discovery requires exactly one recognizable LightOnOCR-3 "
                        "checkpoint, supply model explicitly when the endpoint serves several."
                    )
                self._model = candidates[0]
            return self._model

    @property
    def longest_edge(self) -> int:
        return profile(self.model).longest_edge

    def recognize(self, image: Image.Image) -> dict:
        started = time.monotonic()
        model = self.model
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": to_data_url(image, self.longest_edge)},
                        },
                        {"type": "text", "text": GROUNDING_PROMPT},
                    ],
                }
            ],
            temperature=0.2,
            top_p=profile(model).top_p,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **({"max_tokens": self.max_tokens} if self.max_tokens is not None else {}),
        )
        usage = response.usage.model_dump() if response.usage is not None else {}
        if not response.choices:
            raise GroundingError(
                "The server returned no completion choices", usage=usage)
        choice = response.choices[0]
        raw = choice.message.content or ""

        def fail(message: str) -> None:
            raise GroundingError(message, raw=raw, usage=usage,
                                 finish_reason=choice.finish_reason)

        if choice.finish_reason != "stop":
            fail(
                f"Grounding did not complete: finish_reason={choice.finish_reason!r}")
        is_blank = not raw.strip()
        blocks = []
        if not is_blank:
            markers = list(MARKER.finditer(raw))
            if not markers or raw[: markers[0].start()].strip():
                fail("The response does not start with a grounding marker")
            valid_starts = {marker.start() for marker in markers}
            if any(m.start() not in valid_starts for m in _MARKER_START.finditer(raw)):
                fail("The response contains a malformed grounding marker")
            blocks = parse_blocks(fix_escaped_dollars(raw))
            for block in blocks:
                x1, y1, x2, y2 = block.bbox
                if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                    fail(f"Invalid normalized grounding bbox: {block.bbox}")
        return {
            "raw": raw,
            "blocks": [block.to_dict() for block in blocks],
            "is_blank": is_blank,
            "usage": usage,
            "elapsed_seconds": time.monotonic() - started,
        }

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> GroundingClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
