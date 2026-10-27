from __future__ import annotations

import unicodedata
from bisect import bisect_right
from difflib import SequenceMatcher

from .line_geometry import _box


def layout_bbox(block: dict) -> list:
    geometry = block.get("table_geometry")
    measured = geometry.get("bbox") if isinstance(geometry, dict) else None
    if _box(measured) is not None:
        return measured
    anchored = block.get("layout_bbox")
    return anchored if _box(anchored) is not None else block["bbox"]


def _canonical(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text.casefold().replace("đ", "d"))
        if char.isalnum()
    )


def normalized_native_spans(page: dict) -> list[dict]:
    region = _box(page.get("bbox"))
    if region is None:
        return []
    left, top, right, bottom = region
    result = []
    spans = page.get("native_spans", [])
    if not isinstance(spans, list):
        return []
    for index, span in enumerate(spans):
        if not isinstance(span, dict) or not isinstance(span.get("text"), str):
            continue
        box = _box(span.get("bbox"))
        if box is None or not (
            left <= box[0] < box[2] <= right and top <= box[1] < box[3] <= bottom
        ):
            continue
        result.append(
            {
                "text": span["text"],
                "bbox": [
                    (box[0] - left) * 1000 / (right - left),
                    (box[1] - top) * 1000 / (bottom - top),
                    (box[2] - left) * 1000 / (right - left),
                    (box[3] - top) * 1000 / (bottom - top),
                ],
                "source_index": index,
            }
        )
    return result


def _unique_interval(text: str, stream: str) -> tuple[int, int] | None:
    if len(text) < 6 or sum(char.isalpha() for char in text) < 3:
        return None
    start = stream.find(text)
    if start >= 0:
        return (start, start + len(text)) if stream.find(text, start + 1) < 0 else None
    if len(text) < 40:
        return None
    matches = [
        match
        for match in SequenceMatcher(None, text, stream, autojunk=False).get_matching_blocks()
        if match.size >= 12
    ]
    if not matches or matches[0].a != 0 or matches[-1].a + matches[-1].size != len(text):
        return None
    left, right = matches[0].b, matches[-1].b + matches[-1].size
    if (
        sum(match.size for match in matches) / len(text) < 0.98
        or not 0.98 <= (right - left) / len(text) <= 1.02
    ):
        return None
    seed = max(matches, key=lambda match: match.size)
    literal = text[seed.a: seed.a + seed.size]
    if stream.count(literal) != 1:
        return None
    return left, right


def _rows(spans: list[dict]) -> list[list[float]]:
    rows = []
    for span in sorted(spans, key=lambda span: (span["bbox"][1], span["bbox"][0])):
        box = span["bbox"]
        matching = [
            row
            for row in rows
            if min(box[3], row[3]) - max(box[1], row[1])
            >= 0.5 * min(box[3] - box[1], row[3] - row[1])
        ]
        if len(matching) > 1:
            return []
        if not matching:
            rows.append(list(box))
            continue
        row = matching[0]
        if max(box[0] - row[2], row[0] - box[2], 0) > 3 * max(box[3] - box[1], row[3] - row[1]):
            return []
        row[:] = [
            min(row[0], box[0]),
            min(row[1], box[1]),
            max(row[2], box[2]),
            max(row[3], box[3]),
        ]
    for before, after in zip(rows, rows[1:]):
        height = max(before[3] - before[1], after[3] - after[1])
        overlap = min(before[2], after[2]) - max(before[0], after[0])
        if after[1] - before[3] > 4 * height or overlap < 0.5 * min(
            before[2] - before[0], after[2] - after[0]
        ):
            return []
    return rows


def anchor_layout(pages: list[dict]) -> list[dict]:
    result = []
    for page in pages:
        spans = normalized_native_spans(page)
        searchable = [(span, _canonical(span["text"])) for span in spans]
        searchable = [(span, text) for span, text in searchable if text]
        offsets, parts = [0], []
        for _, text in searchable:
            parts.append(text)
            offsets.append(offsets[-1] + len(text))
        stream = "".join(parts)
        blocks = []
        for original in page["blocks"]:
            block = dict(original)
            if block["label"] not in {"table", "image", "chart", "header_image", "footer_image"}:
                interval = _unique_interval(_canonical(block["text"]), stream)
                if interval is not None:
                    start, end = interval
                    first, last = bisect_right(
                        offsets, start) - 1, bisect_right(offsets, end - 1)
                    selected = [item[0] for item in searchable[first:last]]
                    rows = _rows(selected)
                    if rows:
                        box = [
                            min(row[0] for row in rows),
                            min(row[1] for row in rows),
                            max(row[2] for row in rows),
                            max(row[3] for row in rows),
                        ]
                        evidence = {
                            "source": "native_pdf_text_alignment",
                            "span_indices": [span["source_index"] for span in selected],
                            "native_text": " ".join(span["text"] for span in selected),
                        }
                        block.update(layout_bbox=box, layout_evidence=evidence)
                        block["layout_lines"] = [
                            {"bbox": row, "source": "native_pdf_text_alignment"} for row in rows
                        ]
            blocks.append(block)
        result.append({**page, "blocks": blocks, "layout_spans": spans})
    return result
