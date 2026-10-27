from __future__ import annotations

import unicodedata
from copy import deepcopy

from bs4 import Tag

from .continuation_table import _declared_header_count, build_table_state_from_html
from .line_geometry import _box, _intersection
from .table_grid import _colspan, _rowspan, full_to_half


def _text(value: str) -> str:
    return "".join(unicodedata.normalize("NFC", full_to_half(value)).split())


def _markup(cell: Tag) -> str:
    clone = deepcopy(cell)
    for node in clone.find_all(string=True):
        node.extract()
    return str(clone)


def _one_combining_mark_difference(previous: str, current: str) -> bool:
    # Digits, dates, URLs, symbols and changes to base letters cannot be reconciled.
    if any(not character.isalpha() for character in previous + current):
        return False
    left, right = unicodedata.normalize(
        "NFD", previous), unicodedata.normalize("NFD", current)
    if len(left) != len(right):
        return False
    differences = [(a, b) for a, b in zip(left, right, strict=True) if a != b]
    return len(differences) == 1 and all(unicodedata.combining(c) for c in differences[0])


def _native_header_cells(spans: list[dict], bbox, labels: list[str]) -> list[list[float]] | None:
    """Match the complete first visual row, retaining native boundaries per cell."""
    bounds = _box(bbox)
    if bounds is None or any(value < 0 or value > 1000 for value in bounds):
        return None
    if not isinstance(spans, list) or not spans:
        return None
    inside = []
    for span in spans:
        if not isinstance(span, dict) or not isinstance(span.get("text"), str):
            return None
        if not (text := _text(span["text"])):
            continue
        rectangle = _box(span.get("bbox"))
        if rectangle is None:
            return None
        if _intersection(rectangle, bounds) == rectangle:
            inside.append((rectangle, text))
    if not inside:
        return None
    first = min(inside, key=lambda item: (item[0][1], item[0][0]))[0]
    first_height = first[3] - first[1]
    row = sorted(
        (
            (rectangle, text)
            for rectangle, text in inside
            if min(first[3], rectangle[3]) - max(first[1], rectangle[1])
            >= 0.5 * min(first_height, rectangle[3] - rectangle[1])
        ),
        key=lambda item: item[0][0],
    )
    cells, cursor = [], 0
    for label in labels:
        gathered, pieces = "", []
        while cursor < len(row) and gathered != label:
            rectangle, text = row[cursor]
            gathered += text
            if not label.startswith(gathered):
                return None
            pieces.append(rectangle)
            cursor += 1
        if gathered != label or not pieces:
            return None
        cells.append(
            [
                min(piece[0] for piece in pieces),
                min(piece[1] for piece in pieces),
                max(piece[2] for piece in pieces),
                max(piece[3] for piece in pieces),
            ]
        )
    # A duplicate/extra native label or overlapping columns makes the anchor ambiguous.
    if cursor != len(row) or any(a[2] >= b[0] for a, b in zip(cells, cells[1:])):
        return None
    return cells


def confirm_repeated_header(
    previous_html: str,
    current_html: str,
    *,
    previous_spans: list[dict],
    current_spans: list[dict],
    previous_bbox,
    current_bbox,
) -> tuple[str, int] | None:
    """Confirm a single mark substitution in an otherwise exact declared schema.

    At least one other header cell must match exactly. Native spans use region
    coordinates 0..1000 and must independently spell the canonical full header
    on BOTH pages, in unambiguous column order at each table's first visual row.
    Multirow/rowspan headers lack a proven row mapping here and safely abstain.
    """
    previous = build_table_state_from_html(previous_html)
    current = build_table_state_from_html(current_html)
    if previous is None or current is None:
        return None
    if _declared_header_count(previous) != 1 or _declared_header_count(current) != 1:
        return None
    before = previous.rows[0].find_all(["th", "td"], recursive=False)
    after = current.rows[0].find_all(["th", "td"], recursive=False)
    if len(before) < 2 or len(before) != len(after) or previous.total_cols != current.total_cols:
        return None
    previous_labels, current_labels, differences = [], [], []
    for column, (left, right) in enumerate(zip(before, after, strict=True)):
        if (
            _rowspan(left) != 1
            or _rowspan(right) != 1
            or _colspan(left) != _colspan(right)
            or _markup(left) != _markup(right)
        ):
            return None
        a, b = _text(left.get_text()), _text(right.get_text())
        if not a or not b:
            return None
        previous_labels.append(a)
        current_labels.append(b)
        if a != b:
            if not _one_combining_mark_difference(a, b):
                return None
            differences.append(column)
    if len(differences) != 1:
        return None
    confirmed = []
    for fragment, labels in (("current", previous_labels), ("previous", current_labels)):
        old_native = _native_header_cells(
            previous_spans, previous_bbox, labels)
        new_native = _native_header_cells(current_spans, current_bbox, labels)
        if old_native is not None and new_native is not None:
            confirmed.append(fragment)
    if len(confirmed) != 1:
        return None
    return confirmed[0], differences[0]
