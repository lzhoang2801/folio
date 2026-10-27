from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence

TEXT_CONTINUABLE_LABELS = frozenset({"text", "ref_text"})
TEXT_MERGE_TRANSPARENT_TYPES = frozenset(
    {
        "image",
        "table",
        "chart",
        "code",
        "caption",
        "header",
        "footer",
        "page_number",
        "aside_text",
        "footnote",
        "header_image",
        "footer_image",
    }
)
MERGE_TRANSPARENT_BLOCK_TYPES = frozenset(
    {
        "header",
        "footer",
        "page_number",
        "aside_text",
        "footnote",
        "header_image",
        "footer_image",
    }
)
TEXT_BARRIER_LABELS = frozenset({"title", "section", "formula", "list"})
LINE_STOP_FLAG = (".", "!", "?", "。", "！", "？", ")",
                  "）", '"', "”", ":", "：", ";", "；")
VERTICAL_LINE_HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 2
VERTICAL_LINE_IN_BLOCK_THRESHOLD = 0.8
SINGLE_LINE_LOOKAHEAD_LIMIT = 5
SINGLE_LINE_MIN_ALIGNED_LOOKAHEAD = 3
SINGLE_LINE_THICKNESS_RATIO_MAX = 1.5
CalculationBBox = tuple[int, int, int, int]
BlockDict = dict
OrderedBlock = tuple[int, int, dict]


def _label(block: dict) -> str:
    label = block.get("label", block.get("type", ""))
    return {
        "doc_title": "title",
        "paragraph_title": "section",
        "equation": "formula",
        "page_footnote": "footnote",
        "page_header": "header",
        "page_footer": "footer",
    }.get(label, label)


def _bbox_for_calculation(bbox) -> CalculationBBox | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) for value in bbox):
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0 <= value <= 1000 for value in values):
        return None
    left, top, right, bottom = (round(value) for value in values)
    return (left, top, right, bottom) if left < right and top < bottom else None


def _metric_line_bboxes(block: dict) -> list[CalculationBBox]:
    lines = block.get("lines")
    if not isinstance(lines, list) or not lines:
        return []
    result = []
    for line in lines:
        if not isinstance(line, dict) or (bbox := _bbox_for_calculation(line.get("bbox"))) is None:
            return []
        result.append(bbox)
    return result


def _normalized_text_content(block: dict) -> str:
    content = block.get("text")
    return content.strip() if isinstance(content, str) else ""


_UNSPACED_RANGES = (
    (0x3000, 0x303F),
    (0x3040, 0x30FF),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF61, 0xFF9F),
    (0x1AFF0, 0x1AFFF),
    (0x1B000, 0x1B16F),
    (0x20000, 0x2EE5F),
    (0x2F800, 0x2FA1F),
    (0x30000, 0x323AF),
)


_CJK_PUNCTUATION = frozenset("，。！？；：、（）【】《》〈〉「」『』〔〕［］｛｝")


_CLOSING_PUNCTUATION = frozenset(",.;:!?%")


_COMPOUND_PREFIXES = frozenset(
    {"open", "non", "self", "cross", "anti", "pre",
        "post", "co", "semi", "multi", "quasi"}
)


_COMPOUND_CONNECTORS = frozenset(
    {"of", "the", "to", "and", "in", "for", "on", "by", "with"})


_INLINE_STYLE_TAG_RE = re.compile(
    r"</?(?:sup|sub|b|strong|i|em|u|s|del|strike)\b[^>]*>", re.IGNORECASE
)


LINE_END_HYPHEN_CHARS = "-\u00ad\u2010\u2011\u2043"


LINE_END_HYPHEN_RE = re.compile(
    rf"[A-Za-z]+[{re.escape(LINE_END_HYPHEN_CHARS)}]\s*$")


_URL_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:(?:https?|ftp)://|www\.)[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+",
    re.ASCII | re.IGNORECASE,
)


_URL_AT_LINE_START_RE = re.compile(
    r"(?:(?:https?|ftp)://|www\.)",
    re.ASCII | re.IGNORECASE,
)


def remove_invalid_surrogates(text: str) -> str:
    return re.sub("[\ud800-\udfff]", "", text)


def is_hyphen_at_line_end(line: str) -> bool:
    return bool(LINE_END_HYPHEN_RE.search(line))


def _url_spans_line_boundary(previous_content: str, next_content: str) -> bool:
    stripped_previous = previous_content.rstrip()
    stripped_next = next_content.lstrip()
    if not stripped_previous or not stripped_next:
        return False
    candidate = f"{stripped_previous}{stripped_next}"
    boundary = len(stripped_previous)
    return any(
        match.start() < boundary < match.end() for match in _URL_CANDIDATE_RE.finditer(candidate)
    )


def resolve_text_line_boundary(
    previous_content: str,
    *,
    next_content: str,
) -> tuple[str, str]:
    processed_content = remove_invalid_surrogates(previous_content).rstrip()
    if not processed_content:
        return "", ""
    stripped_next = remove_invalid_surrogates(next_content).lstrip()
    if not stripped_next:
        return processed_content, ""
    if _url_spans_line_boundary(processed_content, stripped_next):
        if _URL_AT_LINE_START_RE.match(stripped_next):
            return processed_content, " "
        return processed_content, ""
    if is_hyphen_at_line_end(processed_content):
        previous_word = re.search(
            r"([A-Za-z]+)[-\u00ad\u2010\u2011\u2043]$", processed_content)
        acronym = (
            previous_word is not None and len(
                previous_word[1]) > 1 and previous_word[1].isupper()
        )
        hard_compound = (
            previous_word is not None
            and processed_content.endswith("-")
            and (
                previous_word[1].lower() in _COMPOUND_PREFIXES
                or (
                    previous_word[1].lower() in _COMPOUND_CONNECTORS
                    and previous_word.start() > 0
                    and processed_content[previous_word.start() - 1] == "-"
                )
            )
        )
        if stripped_next[0].islower() and not acronym and not hard_compound:
            return processed_content[:-1], ""
        return processed_content, ""
    if re.search(r"\d[-–]$", processed_content) and stripped_next[0].isdigit():
        return processed_content, ""
    previous_char = _boundary_character(processed_content, from_end=True)
    next_char = _boundary_character(stripped_next, from_end=False)
    if not previous_char or not next_char:
        return processed_content, ""
    if _is_unspaced_character(previous_char) or _is_unspaced_character(next_char):
        return processed_content, ""
    if unicodedata.category(previous_char) in {"Ps", "Pi"}:
        return processed_content, ""
    if unicodedata.category(next_char) in {"Pe", "Pf"} or next_char in _CLOSING_PUNCTUATION:
        return processed_content, ""
    return processed_content, " "


def _boundary_character(content: str, *, from_end: bool) -> str:
    visible = _INLINE_STYLE_TAG_RE.sub("", content).strip()
    characters = reversed(visible) if from_end else iter(visible)
    return next(
        (char for char in characters if unicodedata.category(
            char)[0] not in {"M", "C"}), ""
    )


def _is_unspaced_character(char: str) -> bool:
    return char in _CJK_PUNCTUATION or any(
        start <= ord(char) <= end for start, end in _UNSPACED_RANGES
    )


def merge_text_line_contents(
    line_contents: Sequence[str],
) -> str:
    normalized_lines = [
        cleaned
        for content in line_contents
        if (cleaned := remove_invalid_surrogates(str(content)).strip())
    ]
    if not normalized_lines:
        return ""
    merged_content = normalized_lines[0]
    for current_line in normalized_lines[1:]:
        merged_content, separator = resolve_text_line_boundary(
            merged_content,
            next_content=current_line,
        )
        merged_content = f"{merged_content}{separator}{current_line}"
    return merged_content.strip()


def _starts_list_item(text: str) -> bool:
    return (
        re.match(
            r"^(?:[-*+•●○▪▫◦‣⁃–—](?:\s|$)|[（(]?(?:\d+|[A-Za-z]|[IVXLCDMivxlcdm]+)[.．、)）]|"
            r"[\[［]\d+[\]］]|[一二三四五六七八九十百]+[、.．)）]|[①-⑳])",
            text,
        )
        is not None
    )


def _positive_values_have_max_ratio(first: float, second: float, max_ratio: float) -> bool:
    if first <= 0 or second <= 0:
        return False
    return max(first, second) / min(first, second) <= max_ratio


def _bbox_union(line_bboxes: list[CalculationBBox]) -> CalculationBBox:
    return (
        min(bbox[0] for bbox in line_bboxes),
        min(bbox[1] for bbox in line_bboxes),
        max(bbox[2] for bbox in line_bboxes),
        max(bbox[3] for bbox in line_bboxes),
    )


def _line_height(line_bbox: CalculationBBox) -> int:
    return line_bbox[3] - line_bbox[1]


def _line_width(line_bbox: CalculationBBox) -> int:
    return line_bbox[2] - line_bbox[0]


def _is_vertical_text_block_by_lines(line_bboxes: list[CalculationBBox]) -> bool:
    vertical_line_count = sum(
        _line_height(line_bbox) / _line_width(line_bbox)
        > VERTICAL_LINE_HEIGHT_TO_WIDTH_RATIO_THRESHOLD
        for line_bbox in line_bboxes
    )
    return vertical_line_count / len(line_bboxes) > VERTICAL_LINE_IN_BLOCK_THRESHOLD


def _has_mergeable_text_boundary(
    current_content: str,
    previous_content: str,
    *,
    reject_digit_or_uppercase_start: bool,
) -> bool:
    if previous_content.endswith(LINE_STOP_FLAG):
        return False
    if not reject_digit_or_uppercase_start:
        return True
    first_char = current_content[0]
    return not first_char.isdigit() and not first_char.isupper()


_DOTTED_REFERENCE = r"[0-9]{1,3}(?:\.[0-9]{1,3})+"
_REFERENCE_SEQUENCE_END = re.compile(
    rf"(?<![\w.])(?P<identifier>{_DOTTED_REFERENCE})(?P<period>\.)?"
    r"\s+(?:and|or|&|và|hoặc)\s*$",
    re.IGNORECASE,
)
_REFERENCE_SEQUENCE_START = re.compile(
    rf"^(?P<identifier>{_DOTTED_REFERENCE})(?P<period>\.)?(?=\s|$|[,;:!?])"
)
_REFERENCE_KIND_END = re.compile(
    r"\b(?:sections?|subsections?|articles?|clauses?|paragraphs?|chapters?|items?|points?|"
    r"notes?|figures?|tables?|annex(?:es)?|appendi(?:x|ces)|activit(?:y|ies)|"
    r"mục|điều|khoản|phần|hình|bảng)\s*$",
    re.IGNORECASE,
)
_NUMERIC_UNIT_START = re.compile(
    r"^(?:[%$€£₫]|(?:per\s+cent|percent|thousand|million|billion|trillion|"
    r"usd|eur|vnd|dollars?|euros?|nghìn|ngàn|triệu|tỷ|đồng)\b)",
    re.IGNORECASE,
)


def _continues_numeric_reference(previous_content: str, current_content: str) -> bool:
    previous = _REFERENCE_SEQUENCE_END.search(previous_content)
    current = _REFERENCE_SEQUENCE_START.match(current_content)
    if previous is None or current is None:
        return False
    depth = previous["identifier"].count(".")
    if current["identifier"].count(".") != depth:
        return False
    enumeration_lines = previous_content.splitlines() + \
        current_content.splitlines()[1:]
    if any(
        _starts_list_item(line.lstrip()) or re.match(
            r"^#{1,6}\s", line.lstrip())
        for line in enumeration_lines
    ):
        return False
    if _NUMERIC_UNIT_START.match(current_content[current.end():].lstrip()):
        return False
    return bool(
        depth >= 2
        or (previous["period"] and current["period"])
        or _REFERENCE_KIND_END.search(previous_content[: previous.start()])
    )


def _collect_following_same_orientation_lines(
    ordered_blocks: list[OrderedBlock],
    current_index: int,
    *,
    current_page_idx: int,
    current_type: str,
    is_vertical: bool,
) -> list[CalculationBBox]:
    following_lines: list[CalculationBBox] = []
    transparent_types = (
        TEXT_MERGE_TRANSPARENT_TYPES if current_type == "text" else MERGE_TRANSPARENT_BLOCK_TYPES
    )
    for page_idx, _, block in ordered_blocks[current_index + 1:]:
        if page_idx != current_page_idx:
            break
        block_type = _label(block)
        if block_type != current_type:
            if block_type in transparent_types:
                continue
            break

        block_lines = _metric_line_bboxes(block)
        if not block_lines or _is_vertical_text_block_by_lines(block_lines) != is_vertical:
            break
        for line_bbox in block_lines:
            following_lines.append(line_bbox)
            if len(following_lines) >= SINGLE_LINE_LOOKAHEAD_LIMIT:
                return following_lines
    return following_lines


def _aligned_following_lines(
    current_line: CalculationBBox,
    following_lines: list[CalculationBBox],
    *,
    is_vertical: bool,
) -> list[CalculationBBox]:
    current_start = current_line[1] if is_vertical else current_line[0]
    current_thickness = _line_width(
        current_line) if is_vertical else _line_height(current_line)
    aligned_lines: list[CalculationBBox] = []
    for line_bbox in following_lines:
        line_start = line_bbox[1] if is_vertical else line_bbox[0]
        line_thickness = _line_width(
            line_bbox) if is_vertical else _line_height(line_bbox)
        if not _positive_values_have_max_ratio(
            current_thickness,
            line_thickness,
            SINGLE_LINE_THICKNESS_RATIO_MAX,
        ):
            continue
        if abs(line_start - current_start) <= max(current_thickness, line_thickness):
            aligned_lines.append(line_bbox)
    return aligned_lines


def _virtual_single_line_bbox(
    current_line: CalculationBBox,
    aligned_lines: list[CalculationBBox],
    *,
    is_vertical: bool,
) -> CalculationBBox:
    if is_vertical:
        return (
            current_line[0],
            current_line[1],
            current_line[2],
            max(current_line[3], *(line_bbox[3]
                for line_bbox in aligned_lines)),
        )
    return (
        current_line[0],
        current_line[1],
        max(current_line[2], *(line_bbox[2] for line_bbox in aligned_lines)),
        current_line[3],
    )


def _can_auto_merge_multiline_to_single_line(
    current_block: BlockDict,
    previous_block: BlockDict,
    *,
    ordered_blocks: list[OrderedBlock],
    current_index: int,
    current_page_idx: int,
    current_type: str,
    require_current_leading_edge: bool,
    reject_digit_or_uppercase_start: bool,
) -> bool:
    current_lines = _metric_line_bboxes(current_block)
    previous_lines = _metric_line_bboxes(previous_block)
    if len(current_lines) != 1 or len(previous_lines) <= 1:
        return False

    current_is_vertical = _is_vertical_text_block_by_lines(current_lines)
    if _is_vertical_text_block_by_lines(previous_lines) != current_is_vertical:
        return False
    following_lines = _collect_following_same_orientation_lines(
        ordered_blocks,
        current_index,
        current_page_idx=current_page_idx,
        current_type=current_type,
        is_vertical=current_is_vertical,
    )
    aligned_lines = _aligned_following_lines(
        current_lines[0],
        following_lines,
        is_vertical=current_is_vertical,
    )
    if len(aligned_lines) < SINGLE_LINE_MIN_ALIGNED_LOOKAHEAD:
        return False

    current_bbox = _bbox_for_calculation(current_block.get("bbox"))
    previous_bbox = _bbox_for_calculation(previous_block.get("bbox"))
    current_content = _normalized_text_content(current_block)
    previous_content = _normalized_text_content(previous_block)
    if current_bbox is None or previous_bbox is None or not current_content or not previous_content:
        return False
    if not _has_mergeable_text_boundary(
        current_content,
        previous_content,
        reject_digit_or_uppercase_start=reject_digit_or_uppercase_start,
    ):
        return False

    virtual_current_line = _virtual_single_line_bbox(
        current_lines[0],
        aligned_lines,
        is_vertical=current_is_vertical,
    )
    if current_is_vertical:
        return _can_auto_merge_vertical_text_blocks(
            current_content,
            previous_content,
            current_bbox,
            previous_bbox,
            [virtual_current_line],
            previous_lines,
            require_current_leading_edge=require_current_leading_edge,
        )
    return _can_auto_merge_horizontal_text_blocks(
        current_content,
        previous_content,
        current_bbox,
        previous_bbox,
        [virtual_current_line],
        previous_lines,
        require_current_leading_edge=require_current_leading_edge,
    )


def _can_auto_merge_horizontal_text_blocks(
    current_content: str,
    previous_content: str,
    current_bbox: CalculationBBox,
    previous_bbox: CalculationBBox,
    current_lines: list[CalculationBBox],
    previous_lines: list[CalculationBBox],
    *,
    require_current_leading_edge: bool,
) -> bool:
    first_line = current_lines[0]
    last_line = previous_lines[-1]
    first_line_height = _line_height(first_line)
    last_line_height = _line_height(last_line)
    if first_line_height <= 0 or last_line_height <= 0:
        return False

    current_lines_bbox = _bbox_union(current_lines)
    previous_lines_bbox = _bbox_union(previous_lines)
    if (
        require_current_leading_edge
        and abs(current_lines_bbox[0] - first_line[0]) >= first_line_height / 2
    ):
        return False
    if abs(previous_lines_bbox[2] - last_line[2]) >= last_line_height:
        return False
    current_width = current_lines_bbox[2] - current_lines_bbox[0]
    previous_width = previous_lines_bbox[2] - previous_lines_bbox[0]
    min_width = min(current_width, previous_width)
    if min_width <= 0 or abs(current_width - previous_width) >= min_width:
        return False
    if len(current_lines) <= 1 and len(previous_lines) <= 1:
        return False
    return current_bbox[1] < previous_bbox[3]


def _can_auto_merge_vertical_text_blocks(
    current_content: str,
    previous_content: str,
    current_bbox: CalculationBBox,
    previous_bbox: CalculationBBox,
    current_lines: list[CalculationBBox],
    previous_lines: list[CalculationBBox],
    *,
    require_current_leading_edge: bool,
) -> bool:
    first_line = current_lines[0]
    last_line = previous_lines[-1]
    first_line_width = _line_width(first_line)
    last_line_width = _line_width(last_line)
    if first_line_width <= 0 or last_line_width <= 0:
        return False

    current_lines_bbox = _bbox_union(current_lines)
    previous_lines_bbox = _bbox_union(previous_lines)
    if (
        require_current_leading_edge
        and abs(current_lines_bbox[1] - first_line[1]) >= first_line_width / 2
    ):
        return False
    if abs(previous_lines_bbox[3] - last_line[3]) >= last_line_width:
        return False
    current_height = current_lines_bbox[3] - current_lines_bbox[1]
    previous_height = previous_lines_bbox[3] - previous_lines_bbox[1]
    min_height = min(current_height, previous_height)
    if min_height <= 0 or abs(current_height - previous_height) >= min_height:
        return False
    return current_bbox[2] > previous_bbox[0]


def text_continuation_target(block: dict) -> dict | None:
    label = _label(block)
    if label in TEXT_CONTINUABLE_LABELS:
        return block
    if label != "list" or block.get("sub_type") == "ref_text":
        return None
    children = block.get("children", block.get("content"))
    if isinstance(children, list):
        if (
            block.get("sub_type") != "text"
            or not children
            or not isinstance(children[-1], dict)
            or _label(children[-1]) != "text"
        ):
            return None
        return {**children[-1], "_list_tail": True}
    text = _normalized_text_content(block)
    lines = text.splitlines()
    marker = re.compile(r"^[-*+•●○▪▫◦‣⁃–—](?:\s|$)")
    if not lines or not marker.match(lines[0]):
        return None
    if any(line[:1].isspace() and _starts_list_item(line.lstrip()) for line in lines):
        return None
    starts = [index for index, line in enumerate(lines) if marker.match(line)]
    tail = "\n".join(lines[starts[-1]:]).strip()
    target = {**block, "label": "text", "text": tail, "_list_tail": True}
    if len(starts) > 1:
        target["lines"] = []
    return target


def decide_text_continuation(
    previous: dict,
    current: dict,
    *,
    explicit: bool = False,
    following: Sequence[dict] = (),
) -> bool:
    target = text_continuation_target(previous)
    current_label = _label(current)
    if target is None or current_label not in TEXT_CONTINUABLE_LABELS:
        return False
    if _label(target) != current_label:
        return False
    previous_text = _normalized_text_content(target)
    current_text = _normalized_text_content(current)
    if not previous_text or not current_text:
        return False
    if current.get("_reference_start") is True or current.get("_paragraph_boundary") is True:
        return False
    list_tail = bool(target.get("_list_tail"))
    if list_tail and _starts_list_item(current_text):
        return False
    if explicit:
        return True
    if previous_text[-1] in "*_`~$>" or current_text[0] in "*_`~$[<":
        return False
    if (
        not list_tail
        and current.get("_reference_start") is False
        and target.get("_reference_start") is not None
    ):
        return True
    current_lines = _metric_line_bboxes(current)
    previous_lines = _metric_line_bboxes(target)
    current_bbox = _bbox_for_calculation(current.get("bbox"))
    previous_bbox = _bbox_for_calculation(target.get("bbox"))
    if not current_lines or not previous_lines or current_bbox is None or previous_bbox is None:
        return False
    reference = current_label == "ref_text" and not list_tail
    numeric_reference = (
        not reference
        and not list_tail
        and current_text[0].isdigit()
        and _continues_numeric_reference(previous_text, current_text)
    )
    if not _has_mergeable_text_boundary(
        current_text,
        previous_text,
        reject_digit_or_uppercase_start=not (reference or numeric_reference),
    ):
        return False
    vertical = _is_vertical_text_block_by_lines(current_lines)
    if vertical != _is_vertical_text_block_by_lines(previous_lines):
        return False
    geometry = (
        _can_auto_merge_vertical_text_blocks if vertical else _can_auto_merge_horizontal_text_blocks
    )
    accepted = geometry(
        current_text,
        previous_text,
        current_bbox,
        previous_bbox,
        current_lines,
        previous_lines,
        require_current_leading_edge=not reference,
    )
    if not accepted:
        ordered = [(0, index, block)
                   for index, block in enumerate([current, *following])]
        accepted = _can_auto_merge_multiline_to_single_line(
            current,
            target,
            ordered_blocks=ordered,
            current_index=0,
            current_page_idx=0,
            current_type=current_label,
            require_current_leading_edge=not reference,
            reject_digit_or_uppercase_start=not (
                reference or numeric_reference),
        )
    return accepted


def join_text(previous: str, current: str) -> str:
    return merge_text_line_contents([previous, current])
