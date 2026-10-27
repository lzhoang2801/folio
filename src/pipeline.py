from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .assemble import assemble_document, render_markdown
from .client import DEFAULT_CONCURRENCY, GroundingClient, GroundingError, error_message
from .line_geometry import Box, _box, _intersection, lines_for_block, table_geometry_for_block
from .postprocess import cleanup
from .preprocess import Region, iter_regions

logger = logging.getLogger(__name__)


def merge_page_results(results: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for result in sorted(results, key=lambda item: (item["page"], item["region"])):
        grouped.setdefault(result["page"], []).append(result)
    pages = []
    for number, members in grouped.items():
        size = members[0].get("page_size")
        frame = _box([0, 0, *size]) if isinstance(size,
                                                  (list, tuple)) else None
        if frame is None:
            raise ValueError(
                f"Physical page {number} has no valid source page size")
        bounds = []
        for member in members:
            rectangle = _box(member.get("bbox"))
            if (
                member.get("page_size") != size
                or rectangle is None
                or _intersection(rectangle, frame) != rectangle
            ):
                raise ValueError(
                    f"Physical page {number} has inconsistent source coordinates")
            bounds.append(rectangle)
        page = {
            "width": frame[2],
            "height": frame[3],
            "blocks": [],
        }
        for member, rectangle in zip(members, bounds, strict=True):
            for block in member.get("blocks", []):
                copied = {
                    key: block[key] for key in ("id", "label", "text", "asset") if key in block
                }
                copied["bbox"] = _physical_bbox(
                    block.get("bbox"), rectangle, frame)
                if block.get("lines"):
                    copied["lines"] = [
                        {"bbox": _physical_bbox(
                            line.get("bbox"), rectangle, frame)}
                        for line in block["lines"]
                    ]
                page["blocks"].append(copied)
        pages.append(page)
    return pages


def document_content(results: list[dict], assembled: dict | None = None) -> dict:
    pages = merge_page_results(results)
    if assembled is None:
        assembled = {
            "blocks": [
                {
                    "id": block["id"],
                    "label": block["label"],
                    "text": block["text"],
                    "sources": [block["id"]],
                    "assets": [block["asset"]] if block.get("asset") else [],
                }
                for page in pages
                for block in page["blocks"]
            ],
            "continuations": [],
            "warnings": [],
        }
    owners = {source: block["id"] for block in assembled["blocks"]
              for source in block["sources"]}
    for page in pages:
        placements = []
        for block in page["blocks"]:
            placement = {
                "id": block["id"], "block_id": owners[block["id"]], "bbox": block["bbox"]}
            if block.get("lines"):
                placement["lines"] = [{"bbox": line["bbox"]}
                                      for line in block["lines"]]
            placements.append(placement)
        page["blocks"] = placements
    blocks = []
    for block in assembled["blocks"]:
        content = {key: block[key]
                   for key in ("id", "label", "text", "assets")}
        parent = block.get("layout", {}).get("table_parent")
        if parent is not None:
            content["parent"] = owners[parent]
        blocks.append(content)
    return {
        "pages": pages,
        "blocks": blocks,
        "continuations": assembled["continuations"],
        "warnings": assembled["warnings"],
    }


def _physical_bbox(value: object, crop: Box, frame: Box) -> list[float] | None:
    box = _box(value)
    if box is None:
        return None
    left, top, right, bottom = crop
    width, height = frame[2:]
    return [
        (left + box[0] * (right - left) / 1000) * 1000 / width,
        (top + box[1] * (bottom - top) / 1000) * 1000 / height,
        (left + box[2] * (right - left) / 1000) * 1000 / width,
        (top + box[3] * (bottom - top) / 1000) * 1000 / height,
    ]


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(
        value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(4):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.01 * 2**attempt)


def _start_run(source: str | Path, output_dir: str | Path, name: str | None) -> tuple[Path, dict]:
    source = Path(source).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Not a file: {source}")
    name = source.stem if name is None else name
    if not name or name in {".", ".."} or any(c in name for c in "/\\:\x00"):
        raise ValueError("name must be a single directory name")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    out = Path(output_dir).resolve() / name
    out.mkdir(parents=True, exist_ok=False)
    (out / "regions").mkdir()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "source": str(source),
        "source_sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planning_complete": False,
        "regions": [],
    }
    _write_json(out / "manifest.json", manifest)
    return out, manifest


def _region_record(region: Region, out: Path) -> dict:
    stem = f"page-{region.page:04d}-region-{region.region:02d}"
    image = f"regions/{stem}.png"
    region.image.save(out / image)
    return {
        "page": region.page,
        "region": region.region,
        "bbox": list(region.bbox),
        "page_size": list(region.page_size),
        "width": region.image.width,
        "height": region.image.height,
        "image": image,
        "json": f"regions/{stem}.json",
        "raw": f"regions/{stem}.raw.md",
        "split_reason": region.split_reason,
        "native_lines": region.native_lines,
        "geometry_warning": region.geometry_warning,
        "native_spans": region.native_spans,
        "reading_stream": region.reading_stream,
        "region_role": region.region_role,
        "segmentation": region.segmentation,
    }


def _recognize(
    client: GroundingClient, region: Region, record: dict, out: Path, api_key: str | None = None
) -> dict:
    try:
        try:
            response = client.recognize(region.image)
        except GroundingError as error:
            (out / record["raw"]).write_text(error.raw, encoding="utf-8")
            _write_json(
                out / record["json"],
                {
                    **record,
                    "status": "failed",
                    "error": error_message(error, api_key),
                    "usage": error.usage,
                    "finish_reason": error.finish_reason,
                },
            )
            raise
        (out / record["raw"]).write_text(response["raw"], encoding="utf-8")
        blocks = []
        rx1, ry1, rx2, ry2 = region.bbox
        for index, raw_block in enumerate(response["blocks"]):
            block = dict(raw_block)
            block["id"] = f"p{region.page:04d}-r{region.region:02d}-b{index:04d}"
            x1, y1, x2, y2 = block["bbox"]
            block["page_bbox"] = [
                rx1 + x1 * (rx2 - rx1) / 1000,
                ry1 + y1 * (ry2 - ry1) / 1000,
                rx1 + x2 * (rx2 - rx1) / 1000,
                ry1 + y2 * (ry2 - ry1) / 1000,
            ]
            if block["label"] in {"text", "ref_text", "list"}:
                lines = lines_for_block(record, block, image=region.image)
                if lines:
                    block["lines"] = lines
            if block["label"] == "table":
                geometry = table_geometry_for_block(
                    record, block, image=region.image)
                if geometry is not None:
                    block["table_geometry"] = geometry
            if block["label"] in {"image", "chart"}:
                asset = f"assets/{block['id']}.png"
                (out / "assets").mkdir(exist_ok=True)
                crop = region.image.crop(
                    (
                        math.floor(x1 * region.image.width / 1000),
                        math.floor(y1 * region.image.height / 1000),
                        math.ceil(x2 * region.image.width / 1000),
                        math.ceil(y2 * region.image.height / 1000),
                    )
                )
                try:
                    crop.save(out / asset)
                finally:
                    crop.close()
                block["asset"] = asset
            blocks.append(block)
        result = {
            **record,
            "status": "complete",
            "blocks": blocks,
            "usage": response["usage"],
            "elapsed_seconds": response["elapsed_seconds"],
            "is_blank": response.get("is_blank", not blocks),
        }
        _write_json(out / record["json"], result)
        logger.info(
            "Page %s region %s: %s blocks in %.1fs",
            region.page,
            region.region,
            len(blocks),
            response["elapsed_seconds"],
        )
        return result
    finally:
        region.image.close()


def preprocess_document(
    source: str | Path,
    *,
    output_dir: str | Path = "out",
    name: str | None = None,
    pages: list[int] | None = None,
    split: str = "auto",
    longest_edge: int = 2048,
) -> dict:
    out, manifest = _start_run(source, output_dir, name)
    manifest["settings"] = {"split": split, "longest_edge": longest_edge}
    regions = iter_regions(source, pages=pages,
                           split=split, longest_edge=longest_edge)
    try:
        for region in regions:
            try:
                manifest["regions"].append(_region_record(region, out))
                logger.info(
                    "Saved page %s region %s (%s)", region.page, region.region, region.split_reason
                )
            finally:
                region.image.close()
        if not manifest["regions"]:
            raise ValueError("No pages were selected")
        manifest["planning_complete"] = True
        manifest["status"] = "preprocessed"
    except BaseException as error:
        manifest.update(status="failed", error=error_message(error))
        raise
    finally:
        regions.close()
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(out / "manifest.json", manifest)
    return {**manifest, "output_dir": str(out)}


def parse_document(
    source: str | Path,
    *,
    output_dir: str | Path = "out",
    name: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    pages: list[int] | None = None,
    split: str = "auto",
    longest_edge: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = 600,
    max_tokens: int | None = None,
    markdown_profile: str = "model",
) -> dict:
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if markdown_profile not in {"model", "pp3"}:
        raise ValueError("markdown_profile must be model or pp3")
    with GroundingClient(
        base_url=base_url, model=model, api_key=api_key, timeout=timeout, max_tokens=max_tokens
    ) as client:
        edge = longest_edge if longest_edge is not None else client.longest_edge
        if edge > client.longest_edge:
            raise ValueError(
                f"longest_edge exceeds the model's {client.longest_edge}px profile")
        out, manifest = _start_run(source, output_dir, name)
        manifest["model"] = client.model
        manifest["settings"] = {
            "split": split,
            "longest_edge": edge,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "markdown_profile": markdown_profile,
        }
        results = []
        pending: deque[tuple[Future, Region]] = deque()
        regions = iter_regions(source, pages=pages,
                               split=split, longest_edge=edge)
        pool = ThreadPoolExecutor(max_workers=concurrency)
        try:
            for region in regions:
                try:
                    record = _region_record(region, out)
                    manifest["regions"].append(record)
                    logger.info(
                        "Parsing page %s region %s (%s)",
                        region.page,
                        region.region,
                        region.split_reason,
                    )
                    pending.append(
                        (pool.submit(_recognize, client,
                         region, record, out, api_key), region)
                    )
                except BaseException:
                    region.image.close()
                    raise
                if len(pending) >= concurrency:
                    future, _ = pending.popleft()
                    results.append(future.result())
                    _write_json(out / "manifest.json", manifest)
            manifest["planning_complete"] = True
            _write_json(out / "manifest.json", manifest)
            while pending:
                future, _ = pending.popleft()
                results.append(future.result())
            if not results:
                raise ValueError("No pages were selected")
            assembled = assemble_document(results)
            document = {
                "schema_version": 4,
                "source": manifest["source"],
                "source_sha256": manifest["source_sha256"],
                "model": client.model,
                **document_content(results, assembled),
            }
            markdown = render_markdown(assembled["blocks"])
            (out / "document.md").write_text(markdown, encoding="utf-8")
            if markdown_profile == "pp3":
                formatted, _ = cleanup(markdown, profile="pp3")
                (out / "document.pp3.md").write_text(formatted, encoding="utf-8")
            _write_json(out / "document.json", document)
            manifest["status"] = "complete"
        except BaseException as error:
            manifest.update(status="failed",
                            error=error_message(error, api_key))
            raise
        finally:
            regions.close()
            for future, region in pending:
                if future.cancel():
                    region.image.close()
            pool.shutdown(wait=True, cancel_futures=True)
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(out / "manifest.json", manifest)
    return {**document, "output_dir": str(out)}
