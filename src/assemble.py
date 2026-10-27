from __future__ import annotations

import re
import statistics
import unicodedata
from copy import deepcopy

from bs4 import Tag

from .continuation_table import (
    _cell_has_semantic_content,
    build_row_rendered_cell_segments,
    build_table_state_from_html,
    can_merge_by_structure,
    detect_table_headers,
    expand_header_count_by_rowspan,
    merge_table_fragments,
    table_widths_are_compatible,
)
from .continuation_text import (
    LINE_STOP_FLAG,
    TEXT_MERGE_TRANSPARENT_TYPES,
    _starts_list_item,
    decide_text_continuation,
    join_text,
    resolve_text_line_boundary,
    text_continuation_target,
)
from .layout import _FORM_CODE, _signature, group_layout, is_continuation_caption
from .line_geometry import _box
from .native_layout import anchor_layout, layout_bbox
from .table_grid import _colspan, _rowspan, full_to_half
from .table_identity import confirm_repeated_header

DECORATION = {
    "header",
    "footer",
    "page_number",
    "page_header",
    "page_footer",
    "header_image",
    "footer_image",
}
TABLE_TRANSPARENT = DECORATION | {"aside_text", "footnote", "page_footnote"}


def _margin_text(block: dict) -> str:
    text = re.sub(r"[*_`#]", "", block["text"]).strip()
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def _isolated_top_heading(block: dict, blocks: list[dict]) -> bool:
    left, top, right, bottom = layout_bbox(block)
    height = bottom - top
    if not 0 < height <= 60 or (top + bottom) / 2 > 150:
        return False
    followers = [
        layout_bbox(other)
        for other in blocks
        if other is not block
        and other["label"] not in DECORATION
        and min(right, layout_bbox(other)[2]) > max(left, layout_bbox(other)[0])
    ]
    return bool(followers) and min(box[1] for box in followers) - bottom >= 2 * height


def _matching_margin_geometry(first: list, second: list) -> bool:
    first_height, second_height = first[3] - first[1], second[3] - second[1]
    if (
        min(first_height, second_height) <= 0
        or max(first_height, second_height) / min(first_height, second_height) > 1.35
        or abs((first[1] + first[3] - second[1] - second[3]) / 2) > 25
    ):
        return False
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    first_center, second_center = (
        first[0] + first[2]) / 2, (second[0] + second[2]) / 2
    return (
        overlap >= 0.4 * min(first[2] - first[0], second[2] - second[0])
        or abs(first_center - second_center) <= 80
        or abs(first_center + second_center - 1000) <= 120
    )


def _image_margin_label_evidence(block: dict, blocks: list[dict]) -> bool:
    if block["label"] != "text" or block.get("continues"):
        return False
    left, top, right, bottom = layout_bbox(block)
    lines = block.get("lines", [])
    if (
        not 1 <= len(lines) <= 2
        or not 0 < right - left <= 150
        or not 0 < bottom - top <= 60
        or 150 < (left + right) / 2 < 850
    ):
        return False
    wide = [
        other
        for other in blocks
        if other["label"] in {"text", "ref_text"}
        and layout_bbox(other)[2] - layout_bbox(other)[0] >= 500
        and len(other.get("lines", [])) >= 3
    ]
    if len(wide) < 3:
        return False
    em = statistics.median(
        line["bbox"][3] - line["bbox"][1] for other in wide for line in other["lines"]
    )
    if em <= 0:
        return False
    body_left = statistics.median(layout_bbox(other)[0] for other in wide)
    body_right = statistics.median(layout_bbox(other)[2] for other in wide)
    tolerance = max(3, 0.75 * em)
    if any(
        abs(layout_bbox(other)[0] - body_left) > tolerance
        or abs(layout_bbox(other)[2] - body_right) > tolerance
        for other in wide
    ) or not (right <= body_left - 0.5 * em or left >= body_right + 0.5 * em):
        return False
    if not any(layout_bbox(other)[1] <= top < layout_bbox(other)[3] for other in wide):
        return False
    # Another semantic member in this outside corridor may be a real second column.
    if any(
        other is not block
        and other["label"] not in TABLE_TRANSPARENT | {"image", "caption", "table_caption"}
        and min(right, layout_bbox(other)[2]) > max(left, layout_bbox(other)[0])
        for other in blocks
    ):
        return False
    for other in blocks:
        box = layout_bbox(other)
        if (
            other["label"] == "image"
            and 0 < box[2] - box[0] <= right - left
            and 0 < box[3] - box[1] <= 60
            and abs(box[0] - left) <= em
            and box[2] <= right
            and 0 <= top - box[3] <= em
        ):
            return True
    return False


def _reconcile_margin_labels(pages: list[dict]) -> list[dict]:
    anchors = [
        (pi, block)
        for pi, page in enumerate(pages)
        if page.get("region_role") != "side"
        for block in page["blocks"]
        if block["label"] in {"header", "footer", "page_header", "page_footer"}
    ]
    repeated = [
        (pi, block)
        for pi, page in enumerate(pages)
        if page.get("region_role") != "side"
        for block in page["blocks"]
        if block["label"] in {"title", "section"} and _isolated_top_heading(block, page["blocks"])
    ]
    result = []
    for pi, page in enumerate(pages):
        if page.get("region_role") == "side":
            result.append(page)
            continue
        blocks = []
        for block in page["blocks"]:
            replacement = block
            if block["label"] in {"title", "section"} and not block.get("layout", {}).get(
                "table_parent"
            ):
                text = _margin_text(block)
                box = layout_bbox(block)
                for anchor_pi, anchor in anchors:
                    other = _margin_text(anchor)
                    band = layout_bbox(anchor)
                    overlap = max(0, min(box[2], band[2]) - max(box[0], band[0])) * max(
                        0, min(box[3], band[3]) - max(box[1], band[1])
                    )
                    union = (
                        (box[2] - box[0]) * (box[3] - box[1])
                        + (band[2] - band[0]) * (band[3] - band[1])
                        - overlap
                    )
                    lanes = page.get("layout_lanes", [])
                    same_margin = (
                        "header" in anchor["label"] and (
                            band[1] + band[3]) / 2 <= 150
                    ) or ("footer" in anchor["label"] and (band[1] + band[3]) / 2 >= 850)
                    duplicate = bool(
                        pi == anchor_pi
                        and same_margin
                        and lanes
                        and all(
                            box[2] <= lane["bbox"][0] or box[0] >= lane["bbox"][2] for lane in lanes
                        )
                        and union > 0
                        and overlap / union >= 0.8
                    )
                    if (
                        text
                        and text == other
                        and (
                            duplicate
                            or (
                                pi != anchor_pi
                                and abs(box[1] - band[1]) <= 1
                                and abs(box[3] - band[3]) <= 1
                                and table_widths_are_compatible(box, band)
                            )
                        )
                    ):
                        role = "header" if "header" in anchor["label"] else "footer"
                        replacement = {**block, "label": role}
                        break
                if replacement is block and _isolated_top_heading(block, page["blocks"]):
                    for other_pi, other in repeated:
                        if (
                            0 < abs(page["page"] -
                                    pages[other_pi]["page"]) <= 2
                            and text
                            and text == _margin_text(other)
                            and _matching_margin_geometry(box, layout_bbox(other))
                        ):
                            replacement = {
                                **block,
                                "label": "header",
                            }
                            break
            if replacement is block and _image_margin_label_evidence(block, page["blocks"]):
                replacement = {**block, "label": "aside_text"}
            blocks.append(replacement)
        result.append({**page, "blocks": blocks})
    return result


def _adjacent(left: dict, right: dict) -> bool:
    return (left["page"] == right["page"] and right["region"] == left["region"] + 1) or (
        right["page"] == left["page"] + 1 and right["region"] == 1
    )


def _continued_caption(block: dict) -> bool:
    return is_continuation_caption(block.get("text", ""))


def _cells(row: Tag) -> list[Tag]:
    return row.find_all(["td", "th"], recursive=False)


def _declared_headers(state) -> int:
    count = 0
    for row in state.rows:
        cells = _cells(row)
        if not cells or not (
            row.find_parent("thead") is not None or all(
                c.name == "th" for c in cells)
        ):
            break
        count += 1
    return expand_header_count_by_rowspan(state.rows, count)


def _data_header(schema: Tag, candidate: Tag) -> bool:
    left, right = _cells(schema), _cells(candidate)
    if len(left) != len(right) or len(left) < 2:
        return False
    if any(_colspan(c) != 1 or _rowspan(c) != 1 for c in left + right):
        return False
    labels = [c.get_text(" ", strip=True) for c in left]
    values = [c.get_text(" ", strip=True) for c in right]
    if any(not label or len(label) > 80 for label in labels):
        return False
    return (
        sum(
            len(v) >= 40
            and len(v) > 2 * len(k)
            and len(v.split()) >= 6
            and v.endswith((".", "!", "?", "。", "！", "？"))
            for k, v in zip(labels, values, strict=True)
        )
        > len(left) / 2
    )


def _inferred_cell_flags(previous_html: str, current_html: str, header_rows: int):
    previous = build_table_state_from_html(previous_html)
    current = build_table_state_from_html(current_html)
    if previous is None or current is None or header_rows >= len(current.rows):
        return None
    before = build_row_rendered_cell_segments(
        previous.rows, previous.last_data_row_metrics.row_idx)
    after = build_row_rendered_cell_segments(current.rows, header_rows)
    if len(before) != len(after) or len(before) < 2:
        return None
    left, right = [s.text.strip() for s in before], [s.text.strip()
                                                     for s in after]
    current_cells = _cells(current.rows[header_rows])
    if (
        not left[0]
        or right[0]
        or not any(right[1:])
        or not current_cells
        or _cell_has_semantic_content(current_cells[0])
        or any(
            cell.find(["img", "svg", "math", "eq", "figure",
                      "object", "embed", "canvas"])
            is not None
            for cell in current_cells
        )
    ):
        return None
    # Partial cell decisions remain independent, as in MinerU; ambiguity means no transfer.
    two_blank_keys = len(right) >= 3 and not right[1]
    flags = [0] * previous.total_cols
    eligible = []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if not b or not a or _starts_list_item(b) or b[0].isdigit():
            continue
        if index and (
            (two_blank_keys and b[0].isalpha())
            or (b[0].islower() and not a.endswith(LINE_STOP_FLAG))
        ):
            segment = before[index]
            for column in range(segment.start_col, min(segment.end_col, len(flags))):
                flags[column] = 1
            eligible.append(index)
    if not eligible:
        return None
    # If a meaningful first-row cell is ambiguous, keep the entire row separate.
    if any(text and index not in eligible for index, text in enumerate(right)):
        return None
    return flags


def _space_inferred_cells(
    previous_html: str, current_html: str, header_rows: int, flags: list[int]
) -> str:
    previous = build_table_state_from_html(previous_html)
    current = build_table_state_from_html(current_html)
    if previous is None or current is None:
        return current_html
    before = build_row_rendered_cell_segments(
        previous.rows, previous.last_data_row_metrics.row_idx)
    after = build_row_rendered_cell_segments(current.rows, header_rows)
    cells = _cells(current.rows[header_rows])
    if len(before) != len(after) or len(after) != len(cells):
        return current_html
    for old, new, cell in zip(before, after, cells, strict=True):
        if new.text and all(flags[c] == 1 for c in range(old.start_col, old.end_col)):
            processed, separator = resolve_text_line_boundary(
                old.text, next_content=new.text)
            # The table kernel owns markup; do not rewrite a source cell to dehyphenate it.
            if processed == old.text.rstrip() and separator:
                cell.insert(0, separator)
    return str(current.soup.find("table"))


def _merge_tables(
    previous_html: str,
    source: dict,
    previous_source: dict,
    mapping=None,
    *,
    caption_continues=False,
    header_confirmation=None,
):
    retained_changed_header = False
    current_html = source["text"]
    previous = build_table_state_from_html(previous_html)
    current = build_table_state_from_html(current_html)
    if previous is None or current is None:
        return None, "invalid or unsupported HTML table"
    caption = current.soup.find("caption")
    if caption is not None:
        if not _continued_caption({"text": caption.get_text(" ", strip=True)}):
            return None, "current table caption starts a separate table"
        caption_continues = True
        caption.extract()
        current_html = str(current.soup.find("table"))
        current = build_table_state_from_html(current_html)
    detected = detect_table_headers(previous, current)[0]
    left_headers, right_headers = _declared_headers(
        previous), _declared_headers(current)
    # Never discard an identical data row as a header.
    skip = min(detected, left_headers)
    if header_confirmation is not None:
        skip = 1
        fragment, column = header_confirmation
        if fragment == "previous":
            before = previous.rows[0].find_all(["th", "td"], recursive=False)
            after = current.rows[0].find_all(["th", "td"], recursive=False)
            before[column].replace_with(deepcopy(after[column]))
            previous_html = str(previous.soup.find("table"))
            previous = build_table_state_from_html(previous_html)
    if right_headers and skip < right_headers:
        if left_headers == right_headers == 1 and _data_header(previous.rows[0], current.rows[0]):
            for cell in _cells(current.rows[0]):
                cell.name = "td"
            current_html = str(current.soup.find("table"))
            current = build_table_state_from_html(current_html)
            right_headers = 0
        elif source.get("continues"):
            skip = 0
            retained_changed_header = True
        else:
            return None, "declared column headers differ"
    if not can_merge_by_structure(current, previous):
        return None, "rendered boundary structures are incompatible"
    raw_flags = source.get("cell_merge")
    if raw_flags is not None:
        previous_raw_state = build_table_state_from_html(
            previous_source["text"])
        if (
            previous_raw_state is None
            or not isinstance(raw_flags, list)
            or len(raw_flags) != previous_raw_state.total_cols
            or any(type(flag) is not int or flag not in (0, 1) for flag in raw_flags)
        ):
            return None, "invalid per-column cell decisions"
        flags = raw_flags
        if retained_changed_header and any(flags):
            return None, "cell transfer cannot target a changed header"
    else:
        flags = _inferred_cell_flags(
            previous_source["text"], current_html, skip)
    if not (
        skip
        or source.get("continues")
        or caption_continues
        or (left_headers and not right_headers)
        or (flags is not None and any(flags))
    ):
        return None, "no repeated header, explicit continuation or inherited schema"
    if raw_flags is None and flags:
        current_html = _space_inferred_cells(
            previous_source["text"], current_html, skip, flags)
    return merge_table_fragments(
        previous_html,
        current_html,
        cell_merge=flags,
        previous_source_columns=mapping,
        skip_header_rows=skip,
    )


def _previous_text(ordered, page_index: int, explicit: bool, previous_region: int | None):
    for pi, _, candidate in reversed(ordered):
        if pi != page_index and pi != previous_region:
            if previous_region is not None and pi > previous_region:
                continue
            return None
        if candidate["label"] in TEXT_MERGE_TRANSPARENT_TYPES:
            continue
        if candidate.get("layout", {}).get("table_parent"):
            continue
        if text_continuation_target(candidate) is not None:
            return (pi, candidate) if pi != page_index or explicit else None
        return None
    return None


def _boundary_table(blocks, *, from_end: bool, current_table: str | None = None):
    tables = {block["id"] for block in blocks if block["label"] == "table"}
    if current_table is not None:
        tables.add(current_table)
    for candidate in reversed(blocks) if from_end else blocks:
        if candidate["label"] in TABLE_TRANSPARENT:
            continue
        if candidate.get("layout", {}).get("table_parent") in tables:
            continue
        if candidate["label"] in {"caption", "table_caption"} and _continued_caption(candidate):
            continue
        return candidate if candidate["label"] == "table" else None
    return None


def _caption_key(table: dict) -> str:
    context = table.get("layout", {})
    return "".join(
        context.get("native_caption_identity", context.get(
            "caption_identity", "")).split()
    )


def _metadata_conflict(previous: dict, current: dict, blocks: dict[str, dict]) -> bool:
    def symbols(value):
        return tuple(re.findall(r"[^\w\s]", unicodedata.normalize("NFC", full_to_half(value))))

    def field(line):
        key, value = line.split(":", 1)
        prefix = key.split(maxsplit=1)
        if len(prefix) == 2 and _FORM_CODE.fullmatch(prefix[0]):
            key = prefix[1]
        return _signature(key), " ".join(value.split())

    def fields(table):
        result = {}
        for identifier in table.get("layout", {}).get("preamble", []):
            block = blocks[identifier]
            text = re.sub(r"[*_`#]", "", block["text"]).strip().casefold()
            evidence = block.get("layout_evidence", {})
            native_fields = {}
            if (
                block.get("layout", {}).get("role") == "preamble"
                and evidence.get("source")
                in {"native_pdf_text_alignment", "native_table_auxiliary_parent"}
                and evidence.get("span_indices")
            ):
                native = full_to_half(evidence.get(
                    "native_text", "")).casefold()
                for line in native.splitlines():
                    if ":" in line:
                        key, value = field(line)
                        if not value:
                            continue
                        native_fields[key] = (
                            _signature(key + ":" + value),
                            symbols(value),
                        )
            for line in text.splitlines():
                if ":" in line:
                    key, value = field(line)
                    result[("field", key)] = value, native_fields.get(key)
            if re.search(r"\d", text):
                template = re.sub(r"\d+(?:[.,/-]\d+)*", "#", text)
                result[("numeric", "".join(template.split()))] = (
                    tuple(re.findall(r"\d+", text)),
                    None,
                )
        return result

    left, right = fields(previous), fields(current)
    return any(
        left[key][0] != right[key][0]
        and not (
            left[key][1]
            and left[key][1] == right[key][1]
            and symbols(left[key][0]) == symbols(right[key][0])
        )
        for key in left.keys() & right.keys()
    )


def _continued_caption_prefix(previous: dict, current: dict) -> bool:
    def words(table):
        layout = table.get("layout", {})
        return layout.get("native_caption_identity", "").split()

    left, right = words(previous), words(current)
    shorter, longer = sorted((left, right), key=len)
    if not (
        current.get("layout", {}).get("continuation_caption")
        and len(shorter) >= 3
        and len(shorter) < len(longer)
        and longer[: len(shorter)] == shorter
    ):
        return False
    sequences, sections = [], []
    for table in (previous, current):
        state = build_table_state_from_html(table["text"])
        if state is None:
            return False
        keys, ordinals = [], []
        for row in state.rows[_declared_headers(state):]:
            cells = _cells(row)
            if cells and _colspan(cells[0]) == _rowspan(cells[0]) == 1:
                text = cells[0].get_text(strip=True)
                if re.fullmatch(r"\d+", text):
                    try:
                        keys.append(int(text))
                    except ValueError:
                        return False
            elif len(cells) == 1 and _colspan(cells[0]) == state.total_cols:
                text = unicodedata.normalize(
                    "NFKC", cells[0].get_text(" ", strip=True))
                marker = re.match(r"([IVXLCDM]+|\d+)[.)]\s+\S", text)
                if marker:
                    token = marker[1]
                    if token.isdigit():
                        try:
                            ordinals.append(("numeric", int(token)))
                        except ValueError:
                            return False
                    elif re.fullmatch(
                        r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})", token
                    ):
                        values = {"I": 1, "V": 5, "X": 10,
                                  "L": 50, "C": 100, "D": 500, "M": 1000}
                        digits = [values[c] for c in token]
                        value = sum(
                            -number if i +
                            1 < len(digits) and number < digits[i + 1] else number
                            for i, number in enumerate(digits)
                        )
                        ordinals.append(("roman", value))
        if len(keys) < 3 or any(a >= b for a, b in zip(keys, keys[1:])):
            return False
        if any(a[0] != b[0] or a[1] >= b[1] for a, b in zip(ordinals, ordinals[1:])):
            return False
        sequences.append(keys)
        sections.append(ordinals)
    previous_key, current_key = sequences[0][-1], sequences[1][0]
    if previous_key >= current_key:
        return False
    row_continues = previous_key + 1 == current_key
    if (
        sections[0]
        and sections[1]
        and (
            sections[0][-1][0] != sections[1][0][0]
            or not sections[0][-1][1] <= sections[1][0][1] <= sections[0][-1][1] + 1
        )
    ):
        return False
    section_continues = bool(
        sections[0]
        and sections[1]
        and sections[0][-1][0] == sections[1][0][0]
        and sections[0][-1][1] + 1 == sections[1][0][1]
    )
    if not (row_continues or section_continues):
        return False
    return True


def _table_context(previous: dict, current: dict, blocks: dict[str, dict]) -> dict:
    after = current.get("layout", {})
    left, right = _caption_key(previous), _caption_key(current)
    confirmed = bool(left and right and left == right)
    prefix = (
        _continued_caption_prefix(
            previous, current) if left and right and left != right else False
    )
    compatible = not (
        (left and right and left != right and not prefix)
        or (after.get("caption") and not (confirmed or prefix or after.get("continuation_caption")))
        or _metadata_conflict(previous, current, blocks)
    )
    return {
        "compatible": compatible,
        "confirmed": bool(confirmed or prefix),
        "continuation_caption": bool(after.get("continuation_caption")),
    }


def _next_body_lane(previous: dict, current: dict) -> bool:
    before, after = previous.get("layout", {}), current.get("layout", {})
    return (
        type(before.get("lane")) is int
        and type(after.get("lane")) is int
        and after["lane"] == before["lane"] + 1
        and bool(before.get("continuation_caption") or after.get("continuation_caption"))
    )


def _physical_box(page: dict, block: dict) -> list | None:
    crop = _box(page.get("bbox"))
    if crop is None:
        return None
    left, top, right, bottom = crop
    x0, y0, x1, y1 = layout_bbox(block)
    return [
        left + x0 * (right - left) / 1000,
        top + y0 * (bottom - top) / 1000,
        left + x1 * (right - left) / 1000,
        top + y1 * (bottom - top) / 1000,
    ]


def _text_metrics(page: dict, block: dict) -> dict:
    crop = _box(page.get("bbox"))
    if page.get("segmentation", {}).get("accepted") is not True or crop is None:
        return block
    scale = (crop[2] - crop[0]) / (crop[3] - crop[1])

    def box(value):
        return [value[0] * scale, value[1], value[2] * scale, value[3]]

    return {
        **block,
        "bbox": box(layout_bbox(block)),
        "lines": [
            {**line, "bbox": box(line["bbox"])}
            for line in block.get("layout_lines", block.get("lines", []))
        ],
    }


def assemble_document(pages: list[dict]) -> dict:
    pages = anchor_layout(pages)
    pages = group_layout(pages)
    pages = _reconcile_margin_labels(pages)
    raw_blocks = {block["id"]: block for page in pages for block in page["blocks"]}
    blocks, continuations, warnings = [], [], []
    by_source, column_maps, ordered, previous_regions = {}, {}, [], {}
    for pi, page in enumerate(pages):
        stream = page.get("reading_stream", "main")
        previous_region = previous_regions.get(stream)
        position = (page["page"], page["region"])
        if any(type(value) is not int or value < 1 for value in position):
            raise ValueError("page and region must be positive integers")
        if pi and position <= (pages[pi - 1]["page"], pages[pi - 1]["region"]):
            raise ValueError("regions must be unique and in document order")
        for bi, source in enumerate(page["blocks"]):
            identifier = source["id"]
            if identifier in by_source:
                raise ValueError(f"duplicate source block ID: {identifier}")
            current = {
                "id": identifier,
                "label": source["label"],
                "text": source["text"],
                "sources": [identifier],
                "assets": [source["asset"]] if source.get("asset") else [],
                "layout": dict(source.get("layout", {})),
            }
            explicit = bool(source.get("continues"))
            previous_pair = None
            if source["label"] in {"text", "ref_text"} and not source.get("layout", {}).get(
                "table_parent"
            ):
                previous_pair = _previous_text(
                    ordered, pi, explicit, previous_region)
            elif source["label"] == "table":
                candidate = _boundary_table(
                    page["blocks"][:bi], from_end=True, current_table=identifier
                )
                if candidate is not None:
                    if explicit or _next_body_lane(candidate, source):
                        previous_pair = (pi, candidate)
                if (
                    previous_pair is None
                    and previous_region is not None
                    and _boundary_table(page["blocks"][: bi + 1], from_end=False) is source
                ):
                    candidate = _boundary_table(
                        pages[previous_region]["blocks"], from_end=True)
                    if candidate is not None:
                        previous_pair = (previous_region, candidate)
            merged = False
            if previous_pair is not None:
                previous_pi, previous_raw = previous_pair
                adjacent = previous_pi == pi or (
                    previous_pi == previous_region
                    and all(
                        _adjacent(before, after)
                        for before, after in zip(
                            pages[previous_pi:pi], pages[previous_pi + 1: pi + 1]
                        )
                    )
                )
                if adjacent:
                    target = by_source[previous_raw["id"]]
                    kind = "table" if source["label"] == "table" else "text"
                    if kind == "text":
                        merged = decide_text_continuation(
                            _text_metrics(pages[previous_pi], previous_raw),
                            _text_metrics(page, source),
                            explicit=explicit,
                            following=[_text_metrics(page, b)
                                       for b in page["blocks"][bi + 1:]],
                        )
                        if merged:
                            target["text"] = join_text(
                                target["text"], source["text"])
                    else:
                        physical = any(
                            region.get("segmentation", {}).get(
                                "accepted") is True
                            for region in (pages[previous_pi], page)
                        )
                        before_box = (
                            _physical_box(pages[previous_pi], previous_raw)
                            if physical
                            else layout_bbox(previous_raw)
                        )
                        after_box = _physical_box(
                            page, source) if physical else layout_bbox(source)
                        context = _table_context(
                            {**previous_raw,
                                "layout": target.get("layout", {})}, source, raw_blocks
                        )
                        if context["compatible"] and table_widths_are_compatible(
                            before_box, after_box
                        ):
                            confirmation = None
                            if context["confirmed"]:
                                confirmation = confirm_repeated_header(
                                    target["text"],
                                    source["text"],
                                    previous_spans=pages[previous_pi].get(
                                        "layout_spans", []),
                                    current_spans=page.get("layout_spans", []),
                                    previous_bbox=layout_bbox(previous_raw),
                                    current_bbox=layout_bbox(source),
                                )
                            html, state = _merge_tables(
                                target["text"],
                                source,
                                previous_raw,
                                column_maps.get(previous_raw["id"]),
                                header_confirmation=confirmation,
                                caption_continues=context["continuation_caption"]
                                or next(
                                    (
                                        b["label"] in {
                                            "caption", "table_caption"}
                                        and _continued_caption(b)
                                        for b in reversed(page["blocks"][:bi])
                                        if b["label"] not in TABLE_TRANSPARENT
                                    ),
                                    False,
                                ),
                            )
                            merged = html is not None
                            if merged:
                                target["text"] = html
                                column_maps[identifier] = state
                            else:
                                warnings.append(
                                    f"{identifier}: table left separate: {state}")
                    if merged:
                        target["sources"].append(identifier)
                        target["assets"].extend(current["assets"])
                        current = target
                        continuations.append(
                            {"kind": kind,
                                "from": previous_raw["id"], "to": identifier}
                        )
            elif explicit:
                warnings.append(
                    f"{identifier}: continuation marker has no compatible preceding semantic block"
                )
            if not merged:
                blocks.append(current)
            by_source[identifier] = current
            ordered.append((pi, bi, source))
        previous_regions[stream] = pi
    return {"blocks": blocks, "continuations": continuations, "warnings": warnings}


def render_markdown(blocks: list[dict]) -> str:
    parts = []
    table_owners = {
        source: block["id"]
        for block in blocks
        if block["label"] == "table"
        for source in block.get("sources", [block["id"]])
    }
    auxiliaries = {}
    for block in blocks:
        parent = table_owners.get(block.get("layout", {}).get("table_parent"))
        if parent is not None:
            auxiliaries.setdefault(parent, []).append(block)
    for block in blocks:
        if block["label"] in DECORATION and block.get("layout", {}).get("region_role") != "side":
            continue
        if block.get("layout", {}).get("table_parent") in table_owners:
            continue
        if block["label"] == "table":
            caption_seen, seen = False, set()
            for auxiliary in auxiliaries.get(block["id"], []):
                is_caption = auxiliary["layout"].get("role") == "caption"
                if is_caption and caption_seen:
                    continue
                caption_seen |= is_caption
                key = _margin_text(auxiliary)
                if key and key not in seen:
                    parts.append(auxiliary["text"].strip())
                    seen.add(key)
        text = block["text"].strip()
        assets = block.get("assets", [])
        if block["label"] == "image" and assets:
            alt = (
                " ".join(text.split()).replace("\\", "\\\\").replace(
                    "[", "\\[").replace("]", "\\]")
            )
            parts.extend(f"![{alt}]({asset})" for asset in assets)
        else:
            if text:
                parts.append(text)
            if block["label"] == "chart":
                parts.extend(f"![Chart source]({asset})" for asset in assets)
    return "\n\n".join(parts) + "\n"
