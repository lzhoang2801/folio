from __future__ import annotations

import math
import statistics

Box = tuple[float, float, float, float]


def _box(value: object) -> Box | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(type(number) not in (int, float) or not math.isfinite(number) for number in value):
        return None
    left, top, right, bottom = map(float, value)
    return (left, top, right, bottom) if left < right and top < bottom else None


def _intersection(left: Box, right: Box) -> Box | None:
    return _box(
        (
            max(left[0], right[0]),
            max(left[1], right[1]),
            min(left[2], right[2]),
            min(left[3], right[3]),
        )
    )


def displayed_native_box(box: Box, page_bbox: Box, rotation: int) -> Box:
    x0, y0, x1, y1 = page_bbox
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Unsupported native page rotation")

    def displayed(x: float, y: float) -> tuple[float, float]:
        if rotation == 90:
            return y - y0, x - x0
        if rotation == 180:
            return x1 - x, y - y0
        if rotation == 270:
            return y1 - y, x1 - x
        return x - x0, y1 - y

    left, bottom, right, top = box
    points = [
        displayed(x, y) for x, y in ((left, bottom), (left, top), (right, bottom), (right, top))
    ]
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def extract_native_layout(page) -> tuple[list[Box], list[dict], str | None]:
    from pypdfium2 import PdfiumError

    textpage = None
    try:
        width, height = page.get_size()
        x0, y0, x1, y1 = page.get_bbox()
        rotation = page.get_rotation()
        expected = (y1 - y0, x1 - x0) if rotation in (90,
                                                      270) else (x1 - x0, y1 - y0)
        if rotation not in (0, 90, 180, 270) or any(
            not math.isclose(actual, wanted, rel_tol=1e-5, abs_tol=0.01)
            for actual, wanted in zip((width, height), expected, strict=True)
        ):
            return [], [], "Native text geometry has an unsupported page transform"

        textpage = page.get_textpage()
        rectangles, spans = [], []
        for index in range(textpage.count_rects()):
            left, bottom, right, top = textpage.get_rect(index)
            if any(not math.isfinite(value) for value in (left, bottom, right, top)):
                return [], [], "Native text geometry contains non-finite coordinates"
            if left >= right or bottom >= top:
                continue
            rectangle = displayed_native_box(
                (left, bottom, right, top), (x0, y0, x1, y1), rotation)
            clipped = _intersection(rectangle, (0.0, 0.0, width, height))
            if clipped is not None:
                rectangles.append(clipped)
                if clipped == rectangle:
                    text = textpage.get_text_bounded(
                        left, bottom, right, top).strip()
                    if text:
                        spans.append({"bbox": rectangle, "text": text})
        return rectangles, spans, None
    except PdfiumError as error:
        return [], [], f"Native text geometry unavailable: {error}"
    finally:
        if textpage is not None:
            textpage.close()


def extract_native_lines(page) -> tuple[list[Box], str | None]:
    rectangles, _, warning = extract_native_layout(page)
    return rectangles, warning


def region_native_lines(rectangles: list[Box], region_bbox: Box) -> list[Box]:
    return [
        rectangle for rectangle in rectangles if _intersection(rectangle, region_bbox) == rectangle
    ]


def region_native_spans(spans: list[dict], region_bbox: Box) -> list[dict]:
    return [
        span
        for span in spans
        if (rectangle := _box(span.get("bbox"))) is not None
        and _intersection(rectangle, region_bbox) == rectangle
    ]


def _native_lines_for_block(region_record: dict, block: dict) -> list[dict]:
    region = _box(region_record.get("bbox"))
    block_bbox = _box(block.get("page_bbox"))
    if region is None or block_bbox is None:
        return []
    native_lines = region_record.get("native_lines", [])
    if not isinstance(native_lines, list):
        return []
    fragments = []
    for value in native_lines:
        rectangle = _box(value)
        if rectangle is None:
            return []
        if _intersection(rectangle, region) != rectangle:
            continue
        overlap = _intersection(rectangle, block_bbox)
        if overlap is None:
            continue
        area = (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])
        covered = (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])
        if covered / area >= 0.8:
            fragments.append(rectangle)
    if not fragments:
        return []
    vertical_count = sum((r[3] - r[1]) / (r[2] - r[0]) > 2 for r in fragments)
    vertical = vertical_count / len(fragments) > 0.8
    ordered = [(r[1], r[0], r[3], r[2]) if vertical else r for r in fragments]
    rows: list[Box] = []
    for rectangle in sorted(ordered, key=lambda r: (r[1], r[0])):
        matches = []
        for index, row in enumerate(rows):
            overlap = min(row[3], rectangle[3]) - max(row[1], rectangle[1])
            thickness = min(row[3] - row[1], rectangle[3] - rectangle[1])
            if overlap >= thickness * 0.5:
                matches.append(index)
        if len(matches) > 1:
            return []
        if not matches:
            rows.append(rectangle)
            continue
        index = matches[0]
        row = rows[index]
        gap = max(rectangle[0] - row[2], row[0] - rectangle[2], 0)
        if gap > 2 * max(row[3] - row[1], rectangle[3] - rectangle[1]):
            return []
        rows[index] = (
            min(row[0], rectangle[0]),
            min(row[1], rectangle[1]),
            max(row[2], rectangle[2]),
            max(row[3], rectangle[3]),
        )
    if vertical:
        rows = [(r[1], r[0], r[3], r[2]) for r in rows]
        rows.sort(key=lambda r: (-r[0], r[1]))
    else:
        rows.sort(key=lambda r: (r[1], r[0]))
    left, top, right, bottom = region
    return [
        {
            "bbox": [
                (r[0] - left) * 1000 / (right - left),
                (r[1] - top) * 1000 / (bottom - top),
                (r[2] - left) * 1000 / (right - left),
                (r[3] - top) * 1000 / (bottom - top),
            ],
            "source": "native_pdf_text_rectangles",
        }
        for r in rows
    ]


def _raster_lines_for_block(block: dict, image) -> list[dict]:
    bbox = _box(block.get("bbox"))
    if bbox is None or any(value < 0 or value > 1000 for value in bbox):
        return []
    left, top = math.floor(bbox[0] * image.width /
                           1000), math.floor(bbox[1] * image.height / 1000)
    right, bottom = (
        math.ceil(bbox[2] * image.width / 1000),
        math.ceil(bbox[3] * image.height / 1000),
    )
    if right - left < 4 or bottom - top < 3:
        return []
    with image.crop((left, top, right, bottom)) as crop, crop.convert("L") as gray:
        histogram = gray.histogram()
        area = gray.width * gray.height
        if sum(histogram[224:]) / area < 0.75:
            return []
        background = max(range(224, 256), key=lambda value: histogram[value])
        with gray.point(lambda value: 255 if value < min(200, background - 40) else 0) as mask:
            pixels = mask.tobytes()
        width, height = gray.size
    counts = [sum(pixels[y * width: (y + 1) * width]) //
              255 for y in range(height)]
    active_min = max(2, round(width * 0.005))
    bands = []
    start = None
    for y in range(height + 1):
        active = y < height and counts[y] >= active_min
        if active and start is None:
            start = y
        elif not active and start is not None:
            bands.append((start, y))
            start = None
    if not bands:
        return []
    body_height = max(end - start for start, end in bands)
    if body_height > height * 0.6 and body_height > width * 0.3:
        return []
    bodies = [list(band)
              for band in bands if band[1] - band[0] >= body_height * 0.5]
    for start, end in bands:
        if end - start >= body_height * 0.5:
            continue
        distances = [
            (max(body[0] - end, start - body[1], 0), index) for index, body in enumerate(bodies)
        ]
        distance, index = min(distances)
        if distance <= max(1, body_height * 0.35):
            if sum(gap == distance for gap, _ in distances) != 1:
                return []
            bodies[index][0] = min(bodies[index][0], start)
            bodies[index][1] = max(bodies[index][1], end)
        elif sum(counts[start:end]) > max(3, width * 0.01):
            return []
    rows = []
    for start, end in sorted(bodies):
        columns = [
            sum(pixels[y * width + x] for y in range(start, end)) // 255 for x in range(width)
        ]
        occupied = [x for x, count in enumerate(columns) if count]
        if not occupied:
            continue
        x0, x1 = occupied[0], occupied[-1] + 1
        if len(bodies) == 1 and end - start > height * 0.6 and x1 - x0 < 5 * (end - start):
            return []
        gap_start = None
        for x in range(x0, x1):
            if not columns[x] and gap_start is None:
                gap_start = x
            elif columns[x] and gap_start is not None:
                gap = x - gap_start
                if (
                    gap >= max(2 * body_height, width * 0.05)
                    and min(gap_start - x0, x1 - x) >= 2 * body_height
                ):
                    return []
                gap_start = None
        rows.append(
            {
                "bbox": [
                    (left + x0) * 1000 / image.width,
                    (top + start) * 1000 / image.height,
                    (left + x1) * 1000 / image.width,
                    (top + end) * 1000 / image.height,
                ],
                "source": "raster_text_projection",
            }
        )
    if rows and max(r["bbox"][3] - r["bbox"][1] for r in rows) > 2 * min(
        r["bbox"][3] - r["bbox"][1] for r in rows
    ):
        return []
    return rows


def lines_for_block(region_record: dict, block: dict, *, image=None) -> list[dict]:
    native = _native_lines_for_block(region_record, block)
    if native or image is None or block.get("label") not in {"text", "list", "ref_text"}:
        return native
    return _raster_lines_for_block(block, image)


def _horizontal_rule_families(image, top: int, bottom: int, em: float, scale: float):
    minimum_length, tolerance = max(
        40 * scale, 10 * em), max(4 * scale, 2 * em)
    with image.convert("L") as gray, gray.point(lambda value: int(value < 224)) as mask:
        pixels = mask.tobytes()
    strokes = []
    for y in range(top, bottom):
        row, position = pixels[y * image.width: (y + 1) * image.width], 0
        while position < image.width:
            start = row.find(b"\x01", position)
            if start < 0:
                break
            end = row.find(b"\x00", start)
            if end < 0:
                end = image.width
            position = end + 1
            if end - start < minimum_length:
                continue
            strokes.append((start, y, end, y + 1))
    groups = []
    for rectangle in sorted(strokes, key=lambda r: (r[0], r[2], r[1])):
        target = next(
            (
                group
                for group in groups
                if abs(rectangle[0] - group[0][0]) <= tolerance
                and abs(rectangle[2] - group[0][2]) <= tolerance
            ),
            None,
        )
        if target is None:
            groups.append([rectangle])
        else:
            target.append(rectangle)
    families = []
    for group in groups:
        bands = []
        for rectangle in sorted(group, key=lambda r: r[1]):
            if bands and rectangle[1] <= bands[-1][-1][3]:
                bands[-1].append(rectangle)
            else:
                bands.append([rectangle])
        thin = [band for band in bands if band[-1]
                [3] - band[0][1] <= max(2, em / 2)]
        if len(thin) >= 3:
            families.append(
                [
                    (
                        statistics.median(r[0] for r in band),
                        band[0][1],
                        statistics.median(r[2] for r in band),
                        band[-1][3],
                    )
                    for band in thin
                ]
            )
    return families


def table_geometry_for_block(region_record: dict, block: dict, *, image=None) -> dict | None:
    from .continuation_table import build_table_state_from_html

    region, bbox = _box(region_record.get("bbox")), _box(block.get("bbox"))
    if (
        image is None
        or region is None
        or bbox is None
        or block.get("label") != "table"
        or any(value < 0 or value > 1000 for value in bbox)
    ):
        return None
    state = build_table_state_from_html(block.get("text", ""))
    if state is None or state.total_cols < 2:
        return None
    native = region_record.get("native_lines", [])
    if not isinstance(native, list):
        return None
    rectangles = []
    for value in native:
        box = _box(value)
        if box is None:
            return None
        if _intersection(box, region) == box:
            rectangles.append(
                (
                    (box[0] - region[0]) * image.width /
                    (region[2] - region[0]),
                    (box[1] - region[1]) * image.height /
                    (region[3] - region[1]),
                    (box[2] - region[0]) * image.width /
                    (region[2] - region[0]),
                    (box[3] - region[1]) * image.height /
                    (region[3] - region[1]),
                )
            )
    left, top, right, bottom = (
        bbox[0] * image.width / 1000,
        bbox[1] * image.height / 1000,
        bbox[2] * image.width / 1000,
        bbox[3] * image.height / 1000,
    )
    seeds = [r for r in rectangles if left <= (
        r[0] + r[2]) / 2 <= right and top <= r[1] < bottom]
    if len(seeds) < 6:
        return None
    em = statistics.median(r[3] - r[1] for r in seeds)
    if em <= 0:
        return None
    candidates = []
    for rules in _horizontal_rule_families(
        image,
        max(0, math.floor(top - em)),
        min(image.height, math.ceil(bottom + em)),
        em,
        image.width / (region[2] - region[0]),
    ):
        x0, x1 = statistics.median(
            r[0] for r in rules), statistics.median(r[2] for r in rules)
        width = x1 - x0
        if (
            max(0, min(x1, right) - max(x0, left)) / width < 0.8
            or (rules[-1][3] - rules[0][1]) / (bottom - top) < 0.5
        ):
            continue
        rows = []
        for rectangle in sorted(rectangles, key=lambda r: (r[1] + r[3], r[0])):
            center = (rectangle[1] + rectangle[3]) / 2
            if not (x0 <= rectangle[0] < rectangle[2] <= x1 and top <= center <= bottom):
                continue
            target = next((row for row in rows if abs(
                row[0] - center) <= em / 2), None)
            if target is None:
                rows.append([center, [rectangle]])
            else:
                target[1].append(rectangle)
        supported = sum(
            any(
                a[2] - a[0] < width / 2
                and b[2] - b[0] < width / 2
                and (a[0] + a[2]) / 2 < x0 + width * 0.35
                and (b[0] + b[2]) / 2 > x0 + width * 0.65
                and b[0] - a[2] >= em / 2
                for a in cells
                for b in cells
            )
            for _, cells in rows
        )
        stable_counts = []
        for edge in (0, 2):
            columns = []
            positions = sorted(
                (cell[edge], index)
                for index, (_, cells) in enumerate(rows)
                for cell in cells
                if cell[2] - cell[0] < width / 2
            )
            for position, row_index in positions:
                if columns and position - columns[-1][0] <= max(1, em / 2):
                    columns[-1][1].add(row_index)
                else:
                    columns.append((position, {row_index}))
            stable_counts.append(
                sum(len(indices) >= 3 for _, indices in columns))
        if supported >= 3 and max(stable_counts) >= min(state.total_cols, 3):
            candidates.append((x0, x1, rules, supported))
    if len(candidates) != 1:
        return None
    x0, x1, rules, supported = candidates[0]
    return {
        "bbox": [x0 * 1000 / image.width, bbox[1], x1 * 1000 / image.width, bbox[3]],
        "source": "raster_table_rules_with_native_text",
        "rule_count": len(rules),
        "native_cell_rows": supported,
        "rules": [
            [
                r[0] * 1000 / image.width,
                r[1] * 1000 / image.height,
                r[2] * 1000 / image.width,
                r[3] * 1000 / image.height,
            ]
            for r in rules
        ],
    }
