from __future__ import annotations

import argparse
import errno
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

from openai import Timeout
from PIL import Image, ImageOps

from .client import DEFAULT_BASE_URL, DEFAULT_CONCURRENCY, GroundingClient, error_message
from .grounding import profile
from .pipeline import document_content, parse_document
from .preprocess import _fit_owned, _render
from .settings import read_settings, write_settings

STATIC = Path(__file__).with_name("static")
DEFAULT_PORT = 8000
MAX_UPLOAD_BYTES = 256 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp",
                  ".tif", ".tiff", ".bmp", ".gif", ".avif"}


def safe_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode(
        "ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("._-")[:120] or "document"


def run_path(out: Path, name: str) -> Path:
    if not name or name.startswith(".") or any(char in name for char in "/\\:\x00"):
        raise ValueError("Invalid run name")
    run = out / name
    if run.is_symlink() or run.resolve().parent != out.resolve():
        raise ValueError("Run must be inside the output directory")
    return run


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def run_summary(run: Path) -> dict | None:
    if not run.is_dir() or run.is_symlink() or run.name.startswith("."):
        return None
    manifest = read_json(run / "manifest.json")
    if not manifest:
        return None
    return {
        "name": run.name,
        "status": manifest.get("status", "complete"),
        "page_count": len({region["page"] for region in manifest["regions"]}),
        "modified": (run / "manifest.json").stat().st_mtime,
    }


def list_runs(out: Path) -> list[dict]:
    if not out.exists():
        return []
    runs = [summary for run in out.iterdir() if (summary := run_summary(run))]
    return sorted(runs, key=lambda run: -run["modified"])


def artifact_path(run: Path, relative: str) -> Path:
    path = (run / relative).resolve()
    if (
        not relative
        or any(char in relative for char in "\\:\x00")
        or not path.is_relative_to(run.resolve())
    ):
        raise ValueError("Artifact must be inside its run")
    if any(part.startswith(".") for part in Path(relative).parts):
        raise ValueError("Hidden artifacts are unavailable")
    return path


def load_run(out: Path, name: str) -> dict | None:
    run = run_path(out, name)
    summary = run_summary(run)
    if summary is None:
        return None
    manifest = read_json(run / "manifest.json")
    document = read_json(run / "document.json")
    base = f"/files/{quote(name, safe='')}/"
    has_document = bool(document)
    if has_document:
        if document.get("schema_version") != 4:
            raise ValueError(
                "Unsupported document schema; document content schema 4 is required")
    else:
        if manifest.get("status") == "complete":
            raise ValueError(
                "Completed run has no physical-page document.json")
        inputs = []
        for region in manifest.get("regions", []):
            saved = read_json(artifact_path(
                run, region["json"])) if region.get("json") else {}
            inputs.append({**saved, **region})
        document = document_content(inputs)
    raw_pages = {}
    for region in manifest["regions"]:
        if region.get("raw"):
            path = artifact_path(run, region["raw"])
            if path.is_file():
                raw_pages.setdefault(region["page"], []).append(
                    path.read_text(encoding="utf-8"))
    pages = [
        {
            **page,
            "raw": "\n\n".join(raw_pages.get(number, [])),
        }
        for number, page in zip(
            sorted({r["page"] for r in manifest["regions"]}), document["pages"], strict=True
        )
    ]
    markdown_name = (
        "document.pp3.md"
        if has_document and manifest["settings"]["markdown_profile"] == "pp3"
        else "document.md"
    )
    markdown_path = artifact_path(run, markdown_name)
    markdown = markdown_path.read_text(
        encoding="utf-8") if markdown_path.is_file() else None
    return {
        **summary,
        "pages": pages,
        "blocks": document["blocks"],
        "continuations": document["continuations"],
        "error": manifest.get("error"),
        "document": {
            "markdown": markdown,
            "base_url": base,
            "download": (
                f"/api/runs/{quote(name, safe='')}/download.zip"
                if has_document and summary["status"] == "complete"
                else None
            ),
        },
    }


def source_page(run: Path, index: int) -> bytes:
    manifest = read_json(run / "manifest.json")
    numbers = sorted({region["page"] for region in manifest["regions"]})
    if not 0 <= index < len(numbers):
        raise ValueError("Invalid page index")
    source = Path(manifest["source"])
    if source.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        with pdfium.PdfDocument(str(source)) as pdf:
            page = pdf[numbers[index] - 1]
            try:
                width, height = page.get_size()
                image = _render(page, (0, 0, width, height),
                                (width, height), 2048)
            finally:
                page.close()
    else:
        with Image.open(source) as original, ImageOps.exif_transpose(original) as oriented:
            image = _fit_owned(oriented.convert("RGB"), 2048)
    with image, io.BytesIO() as stream:
        image.save(stream, format="PNG")
        return stream.getvalue()


def validate_endpoint(base_url: str, model: str | None) -> None:
    url = urlsplit(base_url)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError(
            "base_url must be an HTTP(S) model URL without embedded credentials")
    if url.query or url.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    if model:
        GroundingClient._validate_model(model)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def __init__(
        self,
        address: tuple[str, int],
        out: Path,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        split: str = "auto",
        markdown_profile: str = "model",
        longest_edge: int | None = None,
        timeout: float = 600,
        max_tokens: int | None = None,
    ):
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        saved = read_settings(self.out)
        self.options = {
            "base_url": base_url
            or saved.get("base_url")
            or os.getenv("LIGHTONOCR_BASE_URL")
            or DEFAULT_BASE_URL,
            "model": model or saved.get("model") or os.getenv("LIGHTONOCR_MODEL") or None,
            "api_key": api_key or os.getenv("LIGHTONOCR_API_KEY") or None,
            "concurrency": concurrency,
            "split": split,
            "markdown_profile": markdown_profile,
            "longest_edge": longest_edge,
            "timeout": timeout,
            "max_tokens": max_tokens,
        }
        validate_endpoint(self.options["base_url"], self.options["model"])
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if split not in {"auto", "off", "force"} or markdown_profile not in {"model", "pp3"}:
            raise ValueError("Invalid split or Markdown profile")
        super().__init__(address, Handler)
        self.lock = threading.RLock()
        self.jobs: list[dict] = []
        self.worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="document")

    def server_close(self) -> None:
        super().server_close()
        if worker := getattr(self, "worker", None):
            worker.shutdown(wait=True)

    def describe_endpoint(self) -> dict:
        with self.lock:
            options = dict(self.options)
        result = {
            "base_url": options["base_url"],
            "model": options["model"],
            "serving": None,
            "reachable": False,
            "models": [],
            "error": None,
            "has_api_key": bool(options["api_key"]),
        }
        try:
            with GroundingClient(
                base_url=options["base_url"],
                model=options["model"] or "",
                api_key=options["api_key"] or "EMPTY",
            ) as client:
                models = (
                    client.client.with_options(
                        timeout=Timeout(5, connect=3), max_retries=0)
                    .models.list()
                    .data
                )
                result["models"] = [model.id for model in models]
                candidates = [name for name in result["models"]
                              if profile(name).generation == 3]
                selected = options["model"]
                if selected and selected not in result["models"]:
                    raise ValueError(
                        "The selected model is not listed by the endpoint")
                if not selected and len(candidates) != 1:
                    raise ValueError(
                        "Select exactly one recognizable LightOnOCR-3 model")
                result.update(
                    serving=selected or candidates[0], reachable=True)
        except Exception as error:
            result["error"] = error_message(
                error, options["api_key"] or "EMPTY")
        return result

    def start_job(self, source: Path, title: str, options: dict) -> dict:
        with self.lock:
            name = safe_name(Path(title).stem)
            stem, counter = name, 1
            taken = {job["name"] for job in self.jobs}
            while (self.out / name).exists() or name in taken:
                counter += 1
                name = f"{stem}-{counter}"
            job = {
                "id": uuid.uuid4().hex,
                "name": name,
                "status": "queued",
                "done": 0,
                "total": None,
                "error": None,
            }
            self.jobs.insert(0, job)
            self.worker.submit(self.work, job, source, {
                               **self.options, **options})
            return dict(job)

    def work(self, job: dict, source: Path, options: dict) -> None:
        with self.lock:
            job["status"] = "running"
        try:
            options["api_key"] = options.get("api_key") or "EMPTY"
            options["model"] = options.get("model") or ""
            result = parse_document(
                source, output_dir=self.out, name=job["name"], **options)
            with self.lock:
                job.update(status="done", done=len(
                    result["pages"]), total=len(result["pages"]))
        except Exception as error:
            with self.lock:
                job.update(status="error", error=error_message(
                    error, options.get("api_key")))

    def job_records(self) -> list[dict]:
        with self.lock:
            jobs = [dict(job) for job in self.jobs]
        for job in jobs:
            if job["status"] in {"queued", "running", "error"}:
                run = run_path(self.out, job["name"])
                manifest = read_json(run / "manifest.json")
                regions = manifest.get("regions", [])
                pages = {}
                for region in regions:
                    complete = bool(region.get("json")) and (
                        read_json(artifact_path(run, region["json"])).get(
                            "status") == "complete"
                    )
                    pages.setdefault(region["page"], []).append(complete)
                job["total"] = len(pages) or None
                open_page = (
                    regions[-1]["page"]
                    if regions and not manifest.get("planning_complete")
                    else None
                )
                job["done"] = sum(
                    all(inputs) for number, inputs in pages.items() if number != open_page
                )
        return jobs


class Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        self.dispatch("GET")

    def do_POST(self) -> None:
        self.dispatch("POST")

    def do_PUT(self) -> None:
        self.dispatch("PUT")

    def do_DELETE(self) -> None:
        self.dispatch("DELETE")

    def dispatch(self, method: str) -> None:
        origin = self.headers.get("Origin")
        if method != "GET" and origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self.fail(HTTPStatus.FORBIDDEN, "Cross-origin changes are forbidden")
        url = urlsplit(self.path)
        self.query = {key: values[0]
                      for key, values in parse_qs(url.query).items()}
        path = unquote(url.path)
        try:
            if method == "GET" and path in {"/", "/favicon.ico"}:
                return self.send_file(STATIC / ("index.html" if path == "/" else "favicon.png"))
            if method == "GET" and path.startswith("/static/"):
                return self.send_file(artifact_path(STATIC, path[8:]))
            if method == "GET" and path == "/api/info":
                return self.send_json({"max_upload_bytes": MAX_UPLOAD_BYTES})
            if path == "/api/settings":
                if method == "PUT":
                    return self.update_settings()
                if method == "GET":
                    return self.send_json(self.server.describe_endpoint())
            if path == "/api/runs":
                if method == "GET":
                    return self.send_json(list_runs(self.server.out))
                if method == "POST":
                    return self.create_run()
            if path.startswith("/api/runs/"):
                name = path[len("/api/runs/"):]
                if method == "GET" and (match := re.fullmatch(r"([^/]+)/download\.zip", name)):
                    return self.send_archive(run_path(self.server.out, match[1]))
                if method == "GET" and (match := re.fullmatch(r"([^/]+)/pages/(\d+)\.png", name)):
                    run = run_path(self.server.out, match[1])
                    image = self.server.worker.submit(
                        source_page, run, int(match[2])).result()
                    return self.send_bytes(image, "image/png")
                if method == "GET":
                    run = load_run(self.server.out, name)
                    return (
                        self.send_json(run)
                        if run
                        else self.fail(HTTPStatus.NOT_FOUND, "No such run")
                    )
                if method == "DELETE":
                    return self.delete_run(name)
            if method == "GET" and path.startswith("/files/"):
                name, separator, relative = path[len(
                    "/files/"):].partition("/")
                if not separator:
                    return self.fail(HTTPStatus.NOT_FOUND, "No such artifact")
                run = run_path(self.server.out, name)
                return self.send_file(artifact_path(run, relative))
            if method == "GET" and path == "/api/jobs":
                return self.send_json(self.server.job_records())
            if method == "DELETE" and path.startswith("/api/jobs/"):
                with self.server.lock:
                    job = next(
                        (j for j in self.server.jobs if j["id"] == path.rsplit(
                            "/", 1)[1]), None
                    )
                    if job and job["status"] in {"running", "queued"}:
                        return self.fail(HTTPStatus.CONFLICT, "This job is still active")
                    if job:
                        self.server.jobs.remove(job)
                return self.send_bytes(b"", "text/plain", HTTPStatus.NO_CONTENT)
            return self.fail(HTTPStatus.NOT_FOUND, "Not found")
        except (ValueError, TypeError, KeyError, argparse.ArgumentTypeError) as error:
            return self.fail(HTTPStatus.BAD_REQUEST, str(error))
        except OSError:
            return self.fail(HTTPStatus.NOT_FOUND, "File unavailable")

    def body_length(self, limit: int) -> int:
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= limit:
            raise ValueError(f"Body must contain between 1 and {limit} bytes")
        return length

    def update_settings(self) -> None:
        body = json.loads(self.rfile.read(self.body_length(16 * 1024)))
        if not isinstance(body, dict) or any(not isinstance(value, str) for value in body.values()):
            raise ValueError("Settings must contain string values")
        with self.server.lock:
            base_url = (
                body.get("base_url", self.server.options["base_url"]).strip()
                or os.getenv("LIGHTONOCR_BASE_URL")
                or DEFAULT_BASE_URL
            )
            model = body.get(
                "model", self.server.options["model"] or "").strip() or None
            validate_endpoint(base_url, model)
            write_settings(self.server.out, base_url=base_url, model=model)
            self.server.options.update(base_url=base_url, model=model)
            if "api_key" in body:
                self.server.options["api_key"] = body["api_key"] or None
        self.send_json(self.server.describe_endpoint())

    def create_run(self) -> None:
        from .cli import parse_pages

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            return self.fail(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Upload exceeds 256 MiB")
        self.body_length(MAX_UPLOAD_BYTES)
        self._upload_remaining = length
        filename = (
            unquote(self.headers.get("X-Filename") or "document.pdf")
            .replace("\\", "/")
            .rsplit("/", 1)[-1]
        )
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES | {".pdf"}:
            raise ValueError("Upload a PDF or supported image")
        options = {}
        if self.query.get("pages"):
            options["pages"] = parse_pages(self.query["pages"])
        uploads = self.server.out / ".uploads"
        uploads.mkdir(exist_ok=True)
        if uploads.resolve().parent != self.server.out:
            raise ValueError(
                "Upload directory must be inside the output directory")
        source = uploads / (uuid.uuid4().hex + suffix)
        self.connection.settimeout(30)
        remaining = length
        try:
            with source.open("xb") as stream:
                while remaining:
                    part = self.rfile.read(min(1024 * 1024, remaining))
                    if not part:
                        raise ValueError("Incomplete upload")
                    stream.write(part)
                    remaining -= len(part)
                    self._upload_remaining = remaining
        except (OSError, ValueError):
            source.unlink(missing_ok=True)
            raise
        job = self.server.start_job(source, filename, options)
        self.send_json(job, HTTPStatus.CREATED)

    def delete_run(self, name: str) -> None:
        run = run_path(self.server.out, name)
        with self.server.lock:
            if any(
                job["name"] == name and job["status"] in {"queued", "running"}
                for job in self.server.jobs
            ):
                return self.fail(HTTPStatus.CONFLICT, "This run is still active")
            if not run.is_dir():
                return self.fail(HTTPStatus.NOT_FOUND, "No such run")
            source = read_json(run / "manifest.json").get("source")
            shutil.rmtree(run)
            if source:
                uploaded = Path(source)
                if uploaded.parent == self.server.out / ".uploads" and re.fullmatch(
                    r"[a-f0-9]{32}\.[a-z]+", uploaded.name
                ):
                    uploaded.unlink(missing_ok=True)
            self.server.jobs[:] = [
                job for job in self.server.jobs if job["name"] != name]
        self.send_bytes(b"", "text/plain", HTTPStatus.NO_CONTENT)

    def send_archive(self, run: Path) -> None:
        summary = run_summary(run)
        if summary is None:
            return self.fail(HTTPStatus.NOT_FOUND, "No such run")
        if summary["status"] != "complete" or not all(
            artifact_path(run, name).is_file() for name in ("document.json", "document.md")
        ):
            return self.fail(HTTPStatus.CONFLICT, "Document is not ready for download")
        paths = [
            run / "document.json",
            *sorted(run.glob("document*.md")),
            *sorted((run / "assets").rglob("*")),
        ]
        with io.BytesIO() as stream:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("assets/", b"")
                for path in paths:
                    if path.is_file():
                        relative = path.relative_to(run).as_posix()
                        archive.write(artifact_path(run, relative), relative)
            self.send_bytes(stream.getvalue(), "application/zip", download=f"{run.name}.zip")

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            return self.fail(HTTPStatus.NOT_FOUND, "Not found")
        content_type = mimetypes.guess_type(
            path.name)[0] or "application/octet-stream"
        if (
            content_type.startswith("text/")
            or "javascript" in content_type
            or content_type == "application/json"
        ):
            content_type += "; charset=utf-8"
        self.send_bytes(path.read_bytes(), content_type)

    def send_json(self, value, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        download: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; font-src 'self' https://cdn.jsdelivr.net; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        if download:
            self.send_header(
                "Content-Disposition", f"attachment; filename*=UTF-8''{quote(download)}"
            )
        self.end_headers()
        self.wfile.write(body)

    def fail(self, status: HTTPStatus, message: str) -> None:
        # Finish bounded uploads before replying, so browsers receive validation errors instead of TCP resets.
        remaining = getattr(self, "_upload_remaining", 0)
        if remaining:
            self.connection.settimeout(30)
            try:
                while remaining:
                    part = self.rfile.read(min(1024 * 1024, remaining))
                    if not part:
                        break
                    remaining -= len(part)
            except OSError:
                self.close_connection = True
                return
            self._upload_remaining = 0
        self.send_bytes(message.encode("utf-8"),
                        "text/plain; charset=utf-8", status)

    def log_message(self, format: str, *args) -> None:
        if "/api/jobs" not in self.path:
            super().log_message(format, *args)


def serve(
    output_dir: str | Path = "out", host: str = "127.0.0.1", port: int | None = None, **options
) -> int:
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    for candidate in [port] if port is not None else range(DEFAULT_PORT, DEFAULT_PORT + 20):
        try:
            server = Server((host, candidate), Path(output_dir), **options)
            break
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
    else:
        raise OSError("Viewer port is busy. Select another with --port")
    print(
        f"viewer: http://{host}:{server.server_port}/ | runs in {server.out}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping viewer. Waiting for submitted document jobs to finish.", file=sys.stderr)
    finally:
        server.server_close()
    return 0
