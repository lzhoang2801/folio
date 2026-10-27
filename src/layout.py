from __future__ import annotations

import re
import statistics
import unicodedata
from copy import deepcopy
from difflib import SequenceMatcher

from .native_layout import layout_bbox
from .table_grid import full_to_half

_MARGINAL = {
    "header",
    "footer",
    "page_header",
    "page_footer",
    "page_number",
    "header_image",
    "footer_image",
    "aside_text",
    "footnote",
    "page_footnote",
}
_HEADINGS = {"title", "section", "caption", "table_caption"}
_TEXT = {"text", "ref_text"}
_CONTINUATION = (
    "(续)",
    "(续表)",
    "(续上表)",
    "(continued)",
    "(cont.)",
    "(cont’d)",
    "(…continued)",
    "continued",
    "续表",
)
_VI_CONTINUATION = re.compile(
    r"\(\s*(?:tiếp\s+theo|còn\s+tiếp)\s*(?:\.{3}|…)?\s*\)", re.I)
_FORM_CODE = re.compile(
    r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+$")


def _visible(text: str) -> str:
    return full_to_half(re.sub(r"[*_`#]", "", text)).strip()


def _signature(text: str) -> str:
    text = unicodedata.normalize(
        "NFKD", _visible(text).casefold()).replace("đ", "d")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[^\W_]+", text))


def is_continuation_caption(text: str) -> bool:
    """Reference word-boundary markers, plus parenthesized Vietnamese continuations."""
    text = _visible(text).casefold()
    if "(continued)" in text or _VI_CONTINUATION.search(text):
        return True
    return any(
        text.endswith(marker)
        and (
            marker != "continued"
            or len(text) == len(marker)
            or not text[-len(marker) - 1].isalpha()
        )
        for marker in _CONTINUATION
    )


def _caption_subject(text: str) -> str:
    text = _VI_CONTINUATION.sub("", _visible(text).casefold())
    for marker in _CONTINUATION:
        if text.endswith(marker) and (marker != "continued" or is_continuation_caption(text)):
            text = text[: -len(marker)].strip()
        elif marker == "(continued)":
            text = text.replace(marker, "")
    return text


def caption_identity(text: str) -> str:
    return _signature(_caption_subject(text))


def _native_caption_identity(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", _caption_subject(text))))


def _overlap(a, b) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def _em(blocks) -> float:
    heights = [
        line["bbox"][3] - line["bbox"][1]
        for b in blocks
        for line in b.get("layout_lines", b.get("lines", []))
        if line["bbox"][3] > line["bbox"][1]
    ]
    return statistics.median(heights) if heights else 10.0


def _outer_lanes(page):
    """Use the inferred content box, as the native OFD auxiliary-role owner does."""
    if page.get("region_role") == "side" and page.get("segmentation", {}).get("accepted") is True:
        side = page.get("reading_stream", "").removeprefix("side-")
        group = [b for b in page["blocks"] if b["label"]
                 in _HEADINGS | _TEXT | {"list"}]
        return [(side, group)] if side in {"left", "right"} and len(group) >= 3 else []
    body_lanes = page.get("layout_lanes", [])
    if not body_lanes:
        return []
    body_left = min(lane["bbox"][0] for lane in body_lanes)
    body_right = max(lane["bbox"][2] for lane in body_lanes)
    gutter = 0.5 * _em(page["blocks"])
    candidates = []
    for block in page["blocks"]:
        box = layout_bbox(block)
        if block["label"] not in _HEADINGS | _TEXT | {"list"}:
            continue
        if box[2] <= body_left - gutter:
            candidates.append(("left", block))
        elif box[0] >= body_right + gutter:
            candidates.append(("right", block))
    groups = []
    for side, block in sorted(candidates, key=lambda item: layout_bbox(item[1])[0]):
        box = layout_bbox(block)
        matching = next(
            (
                g
                for edge, g in groups
                if edge == side
                if any(
                    _overlap(box, layout_bbox(other))
                    >= 0.4 * min(box[2] - box[0], layout_bbox(other)[2] - layout_bbox(other)[0])
                    for other in g
                )
            ),
            None,
        )
        if matching is None:
            groups.append((side, [block]))
        else:
            matching.append(block)
    result = []
    for side, group in groups:
        boxes = [layout_bbox(b) for b in group]
        bounds = [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]
        if len(group) < 3 or bounds[3] - bounds[1] < 3 * max(b[3] - b[1] for b in boxes):
            continue
        body = any(
            b["label"] in _TEXT | {"table"}
            and layout_bbox(b)[2] - layout_bbox(b)[0] >= 3 * (bounds[2] - bounds[0])
            and _overlap(layout_bbox(b), bounds) == 0
            for b in page["blocks"]
            if b not in group
        )
        if body:
            result.append((side, group))
    return result


def _same_repeated_unit(first, second) -> bool:
    a, b = _signature(first["text"]), _signature(second["text"])
    if len(a) < 6 or len(b) < 6:
        return False
    if a == b:
        return True
    return first["label"] == second["label"] == "list" and (a in b or b in a)


def _noisy_repeated_menu(first, second) -> bool:
    if first["label"] != "list" or second["label"] != "list":
        return False
    item_reference = r"(?m)^\s*(?:[-*+•]\s*)?(\d+)\b"
    a_refs = re.findall(item_reference, first["text"])
    b_refs = re.findall(item_reference, second["text"])
    if len(a_refs) < 2 or a_refs != b_refs:
        return False
    if re.findall(r"\d+", first["text"]) != re.findall(r"\d+", second["text"]):
        return False
    a = _signature(first["text"]).replace(" ", "")
    b = _signature(second["text"]).replace(" ", "")
    return min(len(a), len(b)) >= 8 and SequenceMatcher(a=a, b=b, autojunk=False).ratio() >= 0.92


def _navigation(pages):
    lanes = [(p, side, group)
             for p in pages for side, group in _outer_lanes(p)]
    matched = set()
    for page, side, group in lanes:
        for other_page, other_side, other in lanes:
            if page["page"] == other_page["page"]:
                continue
            if side != other_side:
                continue
            pairs = [(a, b)
                     for a in group for b in other if _same_repeated_unit(a, b)]
            left = {a["id"]: a for a, _ in pairs}
            right = {b["id"]: b for _, b in pairs}
            if (
                len(left) < 3
                or len(right) < 3
                or not any(b["label"] == "list" for b in left.values())
                or not any(b["label"] in _HEADINGS for b in left.values())
                or not any(b["label"] == "list" for b in right.values())
                or not any(b["label"] in _HEADINGS for b in right.values())
            ):
                continue
            pairs.extend(
                (a, b)
                for a in group
                if a["id"] not in left
                for b in other
                if b["id"] not in right
                if _noisy_repeated_menu(a, b)
            )
            for a, b in pairs:
                matched.add(a["id"])
    for page in pages:
        for block in page["blocks"]:
            if block["id"] not in matched:
                continue
            block["label"] = "aside_text"


def _body_lanes(page):
    blocks = page["blocks"]
    if page.get("region_role") == "side":
        page["layout_lanes"] = []
        return
    em = _em(blocks)
    seeds = [
        (layout_bbox(b), b.get("table_geometry", {}).get("native_cell_rows", 0))
        for b in blocks
        if b["label"] == "table"
    ]
    text = [
        b
        for b in blocks
        if b["label"] in _TEXT and len(b.get("layout_lines", b.get("lines", []))) >= 3
    ]
    for block in text:
        box = layout_bbox(block)
        if box[2] - box[0] < 100:
            continue
        peers = [
            other
            for other in text
            if (
                abs(layout_bbox(other)[0] - box[0]) <= 0.75 * em
                and abs(layout_bbox(other)[2] - box[2]) <= 0.75 * em
            )
        ]
        if sum(len(b.get("layout_lines", b.get("lines", []))) for b in peers) >= 3:
            seeds.append(
                (
                    box,
                    sum(len(b.get("layout_lines", b.get("lines", [])))
                        for b in peers),
                )
            )
    lanes = []
    for box, _support in sorted(seeds, key=lambda item: (-item[1], item[0][2] - item[0][0])):
        overlaps = [
            lane
            for lane in lanes
            if _overlap(box, lane["bbox"])
            >= 0.25 * min(box[2] - box[0], lane["bbox"][2] - lane["bbox"][0])
        ]
        if len(overlaps) > 1:
            continue
        if overlaps:
            lane = overlaps[0]
            lane["bbox"][0] = min(lane["bbox"][0], box[0])
            lane["bbox"][2] = max(lane["bbox"][2], box[2])
        else:
            lanes.append({"bbox": [box[0], 0, box[2], 1000]})
    lanes.sort(key=lambda lane: lane["bbox"][0])
    if any(b["bbox"][0] - a["bbox"][2] < 0.5 * em for a, b in zip(lanes, lanes[1:])):
        lanes = []
    page["layout_lanes"] = [{**lane, "index": i}
                            for i, lane in enumerate(lanes)]
    for block in blocks:
        if block["label"] in _MARGINAL:
            continue
        box = layout_bbox(block)
        matches = [
            lane
            for lane in page["layout_lanes"]
            if _overlap(box, lane["bbox"])
            >= 0.25 * min(box[2] - box[0], lane["bbox"][2] - lane["bbox"][0])
        ]
        if len(matches) == 1:
            block["layout"]["lane"] = matches[0]["index"]


def _near_horizontal(child, table) -> bool:
    a, b = layout_bbox(child), layout_bbox(table)
    tolerance = max(12, (b[2] - b[0]) * 0.03)
    return not (a[2] < b[0] - tolerance or a[0] > b[2] + tolerance)


def _short(block) -> bool:
    return len(block.get("layout_lines", block.get("lines", []))) <= 1


def _native_field_group(page, block) -> bool:
    evidence = block.get("layout_evidence", {})
    indices = evidence.get("span_indices", [])
    spans = [s for s in page.get("layout_spans", [])
             if s["source_index"] in indices]
    return len(spans) == 2 and any(_FORM_CODE.fullmatch(s["text"].strip()) for s in spans)


def _native_auxiliary(page, block, table, metadata):
    if block.get("layout_evidence"):
        return None
    target = layout_bbox(table)

    def local(signature, target=target):
        return [
            s
            for s in page.get("layout_spans", [])
            if (
                _signature(s["text"]) == signature
                and s["bbox"][3] <= target[1]
                and _overlap(s["bbox"], target) >= 0.8 * (s["bbox"][2] - s["bbox"][0])
            )
        ]

    signature = _signature(block["text"])
    matches = local(signature)
    selected = matches if len(matches) == 1 else []
    if selected:
        positions = {b["id"]: i for i, b in enumerate(page["blocks"])}
        owners = [
            b
            for b in page["blocks"]
            if b["label"] == "table" and len(local(signature, layout_bbox(b))) == 1
        ]
        owner = min(
            owners,
            key=lambda b: (
                abs(positions[b["id"]] - positions[block["id"]]),
                positions[b["id"]] < positions[block["id"]],
            ),
        )
        if owner["id"] != table["id"]:
            return {"table_parent": owner["id"]}
    combined = False
    if not selected:
        fields = set(metadata) | {
            _signature(span["text"]) for span in page.get("layout_spans", []) if ":" in span["text"]
        }
        for repeated in sorted(fields):
            if not signature.endswith(" " + repeated):
                continue
            residual = signature[: -len(repeated)].strip()
            if not any(c.isdigit() for c in residual) or not any(c.isalpha() for c in residual):
                continue
            prefix, suffix = local(residual), local(repeated)
            if len(prefix) == len(suffix) == 1 and _FORM_CODE.fullmatch(prefix[0]["text"].strip()):
                selected = prefix + suffix
                combined = True
                break
    if not selected:
        return None
    boxes = [s["bbox"] for s in selected]
    return {
        "bbox": [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ],
        "source": "native_table_auxiliary_parent",
        "span_indices": [s["source_index"] for s in selected],
        "native_text": " ".join(s["text"] for s in selected),
        "combined_field": combined,
    }


def _table_preambles(page, known_captions, metadata):
    blocks = page["blocks"]
    positions = {b["id"]: i for i, b in enumerate(blocks)}
    for position, table in enumerate(blocks):
        if table["label"] != "table":
            continue
        box = layout_bbox(table)
        candidates = []
        other_parents = set()
        for block in blocks:
            if block["label"] not in _HEADINGS | _TEXT | {"header", "page_header"}:
                continue
            if block["layout"].get("table_parent"):
                continue
            native = _native_auxiliary(page, block, table, metadata)
            if block["label"] in _TEXT and not (
                _short(block)
                or _native_field_group(page, block)
                or (native and native.get("combined_field"))
            ):
                continue
            if native and "bbox" not in native:
                other_parents.add(block["id"])
                continue
            bounds = native["bbox"] if native else layout_bbox(block)
            tolerance = max(12, (box[2] - box[0]) * 0.03)
            if bounds[3] <= box[1] and not (
                bounds[2] < box[0] -
                    tolerance or bounds[0] > box[2] + tolerance
            ):
                candidates.append((block, bounds, native))
        by_signature = {}
        for candidate in candidates:
            by_signature.setdefault(_signature(
                candidate[0]["text"]), []).append(candidate)
        candidates = [
            min(
                group,
                key=lambda item: (
                    abs(positions[item[0]["id"]] - position),
                    positions[item[0]["id"]] > position,
                ),
            )
            for group in by_signature.values()
        ]
        duplicate_ids = {item[0]["id"] for group in by_signature.values() for item in group} - {
            item[0]["id"] for item in candidates
        }
        cluster, next_top, height = [], box[1], 1
        pending = sorted(candidates, key=lambda item: item[1][3], reverse=True)
        while pending:
            for index, (block, bounds, native) in enumerate(pending):
                block_height = bounds[3] - bounds[1]
                gap = next_top - bounds[3]
                within_stack = cluster and min(item[1][1] for item in cluster) <= bounds[
                    1
                ] <= bounds[3] <= max(item[1][3] for item in cluster)
                same_row = cluster and any(
                    min(row[3], peer[3]) - max(row[1], peer[1])
                    >= 0.6 * min(row[3] - row[1], peer[3] - peer[1])
                    or abs((row[1] + row[3] - peer[1] - peer[3]) / 2)
                    <= 0.5 * max(row[3] - row[1], peer[3] - peer[1])
                    for other in cluster
                    for row in [r["bbox"] for r in block.get("layout_lines", [])] or [bounds]
                    for peer in [r["bbox"] for r in other[0].get("layout_lines", [])] or [other[1]]
                )
                if (
                    within_stack
                    or same_row
                    or -0.25 * _em(blocks) <= gap <= max(12, 1.5 * max(height, block_height))
                ):
                    pending.pop(index)
                    cluster.append((block, bounds, native))
                    next_top, height = min(next_top, bounds[1]), max(
                        height, block_height)
                    break
            else:
                break
        seeds = [
            item
            for item in cluster
            for b in [item[0]]
            if (
                b["label"] in _HEADINGS
                or is_continuation_caption(b["text"])
                or caption_identity(b["text"]) in known_captions
            )
        ]
        if seeds:
            seed = max(seeds, key=lambda item: item[1][3])
            members = [item for item in cluster if item[1][3] >= seed[1][1]]
            member_ids = {item[0]["id"] for item in members}
            if any(
                b["id"] not in member_ids
                and b["id"] not in duplicate_ids
                and b["id"] not in other_parents
                and b["label"] not in _MARGINAL
                and b is not table
                and _near_horizontal(b, table)
                and seed[1][3] < (layout_bbox(b)[1] + layout_bbox(b)[3]) / 2 < box[1]
                for b in blocks
            ):
                members = []
            for member, bounds, native in sorted(
                members, key=lambda item: positions[item[0]["id"]]
            ):
                role = "caption" if any(
                    s[0] is member for s in seeds) else "preamble"
                _attach(
                    member,
                    table,
                    role,
                    native=native,
                )
                if role == "caption":
                    known_captions.add(caption_identity(member["text"]))
                else:
                    metadata.setdefault(_signature(
                        member["text"]), set()).add(member["id"])


def _attach(block, table, role, *, native=None):
    if native:
        block.update(layout_bbox=native["bbox"], layout_evidence=native)
    block["layout"].update(table_parent=table["id"], role=role)
    members = table["layout"].setdefault(role, [])
    if block["id"] not in members:
        members.append(block["id"])
    if role == "caption":
        table["layout"]["caption_identity"] = caption_identity(block["text"])
        table["layout"]["continuation_caption"] = is_continuation_caption(
            block["text"])
        native = block.get("layout_evidence", {}).get("native_text")
        if isinstance(native, str) and native.strip():
            table["layout"]["native_caption_identity"] = _native_caption_identity(
                native)
    if block["label"] in {"header", "page_header"}:
        block["label"] = "caption" if role == "caption" else "text"


def _repeated_preambles(page, metadata):
    blocks = page["blocks"]
    for position, table in enumerate(blocks):
        if table["label"] != "table":
            continue
        for block in reversed(blocks[:position]):
            if block["layout"].get("table_parent"):
                break
            if block["label"] in _MARGINAL:
                continue
            peers = metadata.get(_signature(
                block["text"]), set()) - {block["id"]}
            if block["label"] in _TEXT and _short(block) and peers:
                native = _native_auxiliary(page, block, table, metadata)
                if native and "bbox" not in native:
                    break
                _attach(
                    block,
                    table,
                    "preamble",
                    native=native,
                )
            else:
                break


def group_layout(pages: list[dict]) -> list[dict]:
    result = [
        {
            **page,
            "blocks": [
                {**block, "layout": deepcopy(block.get("layout", {}))} for block in page["blocks"]
            ],
        }
        for page in pages
    ]
    for page in result:
        if "reading_stream" in page or "region_role" in page:
            for block in page["blocks"]:
                block["layout"]["region_role"] = page.get(
                    "region_role", "body")
        _body_lanes(page)
    _navigation(result)
    captions, metadata = set(), {}
    for page in result:
        if page.get("region_role") == "side":
            continue
        _body_lanes(page)
        _table_preambles(page, captions, metadata)
    for page in result:
        if page.get("region_role") == "side":
            continue
        _repeated_preambles(page, metadata)
    return result
