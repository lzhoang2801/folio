from __future__ import annotations

import re
import statistics
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .client import fit
from .line_geometry import (
    _box,
    _intersection,
    displayed_native_box,
    extract_native_layout,
    region_native_lines,
    region_native_spans,
)


@dataclass
class Region:
    page: int
    region: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    image: Image.Image
    split_reason: str = "unsplit"
    native_lines: list[tuple[float, float, float, float]
                       ] = field(default_factory=list)
    geometry_warning: str | None = None
    native_spans: list[dict] = field(default_factory=list)
    reading_stream: str = "main"
    region_role: str = "body"
    segmentation: dict = field(default_factory=dict)


def _gutter(image: Image.Image) -> int | None:
    width, height = image.size
    with (
        image.convert("L") as grayscale,
        grayscale.point(lambda value: 255 if value < 225 else 0) as mask,
    ):
        with mask.crop((0, round(height * 0.02), width, round(height * 0.98))) as cropped:
            height = cropped.height
            pixels = cropped.tobytes()
    counts = [sum(pixels[x::width]) / (255 * height) for x in range(width)]
    start = round(width * 0.43)
    end = round(width * 0.57)
    runs: list[tuple[int, int]] = []
    run = start
    for x in range(start, end + 1):
        if x == end or counts[x] > 0.008:
            if x - run >= max(3, round(width * 0.025)):
                runs.append((run, x))
            run = x + 1
    if not runs:
        return None
    left, right = min(runs, key=lambda pair: abs(sum(pair) / 2 - width / 2))
    cut = round((left + right) / 2)
    if any(sum(counts[a:b]) / max(1, b - a) < 0.01 for a, b in ((0, left), (right, width))):
        return None
    radius = max(2, round(width * 0.03))
    for y in range(height):
        band = pixels[y * width + cut - radius: y * width + cut + radius]
        if sum(band) > 255 * len(band) * 0.6:
            return None
    return cut


def _printed_spread(page, width: float, height: float) -> bool:
    textpage = page.get_textpage()
    try:
        for bottom, top in ((height * 0.88, height), (0, height * 0.12)):
            left = textpage.get_text_bounded(
                0, bottom, width * 0.13, top).strip()
            right = textpage.get_text_bounded(
                width * 0.87, bottom, width, top).strip()
            if re.fullmatch(r"\d{1,5}", left) and re.fullmatch(r"\d{1,5}", right):
                if int(right) == int(left) + 1:
                    return True
        return False
    finally:
        textpage.close()


def _cut(image: Image.Image, split: str, printed_spread: bool = False) -> tuple[int | None, str]:
    if split == "off":
        return None, "disabled"
    if split == "force":
        return image.width // 2, "forced_midpoint"
    if not 1.25 <= image.width / image.height <= 1.8:
        return None, "not_spread_aspect"
    if printed_spread:
        return image.width // 2, "consecutive_printed_pages"
    gutter = _gutter(image)
    return (gutter, "central_gutter") if gutter else (None, "no_spread_evidence")


def _native_panels(
    image: Image.Image,
    size: tuple[float, float],
    rectangles: list,
    *,
    native_spans: list[dict] | None = None,
    protected_images: list | None = None,
) -> tuple[list, dict]:
    width, height = size
    horizontal = [box for box in rectangles if box[2] -
                  box[0] > 2 * (box[3] - box[1])]
    evidence = {
        "method": "native_panels",
        "accepted": False,
        "candidates": [],
        "row_support": "native_prose" if native_spans is not None else "geometry_only",
    }
    if len(horizontal) < 6:
        evidence["reason"] = "insufficient independent native rows"
        return [], evidence
    em = statistics.median(box[3] - box[1] for box in horizontal)
    minimum_width = 12 * em
    intervals = []
    for left, _, right, _ in sorted(rectangles):
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = max(right, intervals[-1][1])
        else:
            intervals.append([max(0, left), min(width, right)])
    gaps = [
        (before[1], after[0])
        for before, after in zip(intervals, intervals[1:])
        if after[0] - before[1] >= max(6, 0.75 * em)
        and minimum_width <= (before[1] + after[0]) / 2 <= width - minimum_width
    ]
    with image.convert("L") as gray:
        pixels = gray.tobytes()
    raster_width, raster_height = image.size
    radius = max(3, round(0.75 * em * raster_width / width))
    accepted = []
    for left, right in gaps:
        center = (left + right) / 2
        x = round(center * raster_width / width)
        crossing_rows = sum(
            all(
                value < 225
                for value in pixels[
                    y * raster_width + x - radius: y * raster_width + x + radius + 1
                ]
            )
            for y in range(round(raster_height * 0.03), round(raster_height * 0.97))
        )
        images = [box for box in protected_images or []
                  if box[0] < center < box[2]]
        candidate = {
            "gap": [left, right],
            "cut": center,
            "visual_crossing_rows": crossing_rows,
            "native_images_crossed": images,
        }
        evidence["candidates"].append(candidate)
        if not crossing_rows and not images:
            accepted.append(candidate)

    prose_boxes = None
    if native_spans is not None:
        prose_boxes = set()
        for span in native_spans:
            letters = sum(char.isalpha() for char in span["text"])
            digits = sum(char.isdigit() for char in span["text"])
            if letters >= max(3, digits):
                prose_boxes.add(tuple(span["bbox"]))

    def supported(left, right):
        rows = [box for box in horizontal if left <= box[0] < box[2] <= right]
        wide = [
            box
            for box in rows
            if box[2] - box[0] >= min(12 * em, 0.4 * (right - left))
            and (prose_boxes is None or tuple(box) in prose_boxes)
        ]
        baselines = []
        for box in sorted(wide, key=lambda box: box[1]):
            center = (box[1] + box[3]) / 2
            if not baselines or center - baselines[-1] >= 0.75 * em:
                baselines.append(center)
        return right - left >= minimum_width and len(baselines) >= 3

    while accepted:
        bounds = [0.0, *(item["cut"] for item in accepted), width]
        unsupported = next(
            (i for i, (a, b) in enumerate(
                zip(bounds, bounds[1:])) if not supported(a, b)), None
        )
        if unsupported is None:
            evidence.update(
                accepted=True, cuts=bounds[1:-1], em=em, minimum_panel_width=minimum_width
            )
            return [(a, 0.0, b, height) for a, b in zip(bounds, bounds[1:])], evidence
        neighbors = [i for i in (
            unsupported - 1, unsupported) if 0 <= i < len(accepted)]
        weakest = min(
            neighbors, key=lambda i: accepted[i]["gap"][1] - accepted[i]["gap"][0])
        accepted[weakest]["rejection"] = "insufficient complete wide reading rows in adjacent panel"
        accepted.pop(weakest)
    evidence["reason"] = "no safe supported panel partition"
    return [], evidence


def _plan_regions(
    image,
    size,
    split,
    *,
    native_lines=None,
    native_spans=None,
    protected_images=None,
    printed_spread=False,
):
    width, height = size
    if split == "auto" and width > height and native_lines and not printed_spread:
        bounds, evidence = _native_panels(
            image, size, native_lines, native_spans=native_spans, protected_images=protected_images
        )
        if bounds:
            widths = [box[2] - box[0] for box in bounds]
            result = []
            for index, box in enumerate(bounds):
                others = [value for i, value in enumerate(
                    widths) if i != index]
                side = index in (0, len(bounds) - 1) and widths[index] < 0.5 * statistics.median(
                    others
                )
                stream = ("side-left" if index ==
                          0 else "side-right") if side else "main"
                result.append((box, stream, "side" if side else "body"))
            return result, "native_panel_separators", evidence
        return [((0.0, 0.0, width, height), "main", "body")], "no_safe_panel_partition", evidence
    cut, reason = _cut(image, split, printed_spread)
    cut_source = cut * width / image.width if cut else None
    bounds = ((0.0, cut_source), (cut_source, width)
              ) if cut_source else ((0.0, width),)
    return [((left, 0.0, right, height), "main", "body") for left, right in bounds], reason, {}


def _native_image_boxes(page) -> list:
    from pypdfium2.raw import FPDF_PAGEOBJ_IMAGE

    width, height = page.get_size()
    page_bbox, rotation = page.get_bbox(), page.get_rotation()
    result = []
    for obj in page.get_objects(filter=[FPDF_PAGEOBJ_IMAGE]):
        bounds = obj.get_bounds()
        container = obj.container
        while container is not None:
            bounds = container.get_matrix().on_rect(*bounds)
            container = container.container
        if _box(bounds) is None:
            continue
        displayed = displayed_native_box(bounds, page_bbox, rotation)
        clipped = _intersection(displayed, (0.0, 0.0, width, height))
        if clipped is not None:
            result.append(clipped)
    return result


def _page_numbers(pages: list[int] | None, count: int) -> list[int]:
    numbers = list(range(1, count + 1)) if pages is None else list(pages)
    if not numbers or any(type(n) is not int or not 1 <= n <= count for n in numbers):
        raise ValueError(f"pages must contain page numbers from 1 to {count}")
    if numbers != sorted(set(numbers)):
        raise ValueError("pages must be unique and in ascending order")
    return numbers


def _render(
    page, bbox: tuple[float, float, float, float], page_size, longest_edge: int
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    width, height = page_size
    scale = longest_edge / max(x1 - x0, y1 - y0)
    bitmap = page.render(scale=scale, crop=(x0, height - y1, width - x1, y0))
    try:
        image = bitmap.to_pil().convert("RGB")
    finally:
        bitmap.close()
    return _fit_owned(image, longest_edge)


def _fit_owned(image: Image.Image, longest_edge: int) -> Image.Image:
    try:
        fitted = fit(image, longest_edge)
    except BaseException:
        image.close()
        raise
    if fitted is not image:
        image.close()
    return fitted


def iter_regions(
    path: Path | str,
    *,
    pages: list[int] | None = None,
    split: str = "auto",
    longest_edge: int = 2048,
) -> Iterator[Region]:
    if split not in {"auto", "off", "force"}:
        raise ValueError("split must be auto, off or force")
    if type(longest_edge) is not int or longest_edge < 2:
        raise ValueError("longest_edge must be at least 2")
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        _page_numbers(pages, 1)
        with Image.open(path) as original, ImageOps.exif_transpose(original) as oriented:
            image = oriented.convert("RGB")
        try:
            width, height = image.size
            preview = fit(image, 1024)
            try:
                planned, reason, segmentation = _plan_regions(
                    preview, (width, height), split)
            finally:
                if preview is not image:
                    preview.close()
            for index, (bounds, stream, role) in enumerate(planned, 1):
                bounds = tuple(round(value) for value in bounds)
                yield Region(
                    1,
                    index,
                    bounds,
                    (width, height),
                    _fit_owned(image.crop(bounds), longest_edge),
                    reason,
                    reading_stream=stream,
                    region_role=role,
                    segmentation=segmentation,
                )
        finally:
            image.close()
        return

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        for number in _page_numbers(pages, len(pdf)):
            page = pdf[number - 1]
            try:
                width, height = page.get_size()
                size = (width, height)
                native_lines, native_spans, geometry_warning = extract_native_layout(
                    page)
                preview = _render(page, (0, 0, width, height), size, 1024)
                try:
                    printed = (
                        split == "auto"
                        and 1.25 <= width / height <= 1.8
                        and _printed_spread(page, width, height)
                    )
                    protected_images = []
                    if split == "auto" and width > height and native_lines and not printed:
                        from pypdfium2 import PdfiumError

                        try:
                            protected_images = _native_image_boxes(page)
                        except PdfiumError as error:
                            geometry_warning = f"Native image geometry unavailable: {error}"
                            protected_images = [(0.0, 0.0, width, height)]
                    planned, reason, segmentation = _plan_regions(
                        preview,
                        size,
                        split,
                        native_lines=native_lines,
                        native_spans=native_spans,
                        protected_images=protected_images,
                        printed_spread=printed,
                    )
                    if geometry_warning and protected_images == [(0.0, 0.0, width, height)]:
                        segmentation["native_image_warning"] = geometry_warning
                finally:
                    preview.close()
                for index, (bbox, stream, role) in enumerate(planned, 1):
                    yield Region(
                        number,
                        index,
                        bbox,
                        size,
                        _render(page, bbox, size, longest_edge),
                        reason,
                        region_native_lines(native_lines, bbox),
                        geometry_warning,
                        region_native_spans(native_spans, bbox),
                        stream,
                        role,
                        segmentation,
                    )
            finally:
                page.close()
    finally:
        pdf.close()
