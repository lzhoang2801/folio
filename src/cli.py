from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .client import DEFAULT_CONCURRENCY, error_message
from .pipeline import parse_document, preprocess_document
from .serving import DEFAULT_MODEL, build_llama_command, build_vllm_command, launch
from .settings import read_settings


def parse_pages(value: str) -> list[int]:
    pages = []
    try:
        for part in value.split(","):
            bounds = part.strip().split("-")
            if len(bounds) == 1:
                start = end = int(bounds[0])
            elif len(bounds) == 2:
                start, end = map(int, bounds)
            else:
                raise ValueError
            if start < 1 or end < start:
                raise ValueError
            pages.extend(range(start, end + 1))
        if not pages or pages != sorted(set(pages)):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            "pages must be positive, ordered, unique, e.g. 1,3-5"
        ) from None
    return pages


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="folio", description="Folio document parsing")
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("parse", "preprocess"):
        sub = commands.add_parser(command)
        sub.add_argument("source")
        sub.add_argument("--output-dir", default="out")
        sub.add_argument(
            "--name", help="run directory name"
        )
        sub.add_argument("--pages", type=parse_pages,
                         help="physical PDF pages, e.g. 1,3-5")
        sub.add_argument("--split", choices=("auto",
                         "off", "force"), default="auto")
        sub.add_argument("--longest-edge", type=int,
                         default=None if command == "parse" else 2048)
        if command == "parse":
            sub.add_argument(
                "--base-url", help="server /v1 URL, defaults to LIGHTONOCR_BASE_URL")
            sub.add_argument(
                "--model", help="LightOnOCR-3 checkpoint or discovered served model")
            sub.add_argument("--concurrency", type=int,
                             default=DEFAULT_CONCURRENCY)
            sub.add_argument("--timeout", type=float, default=600)
            sub.add_argument("--max-tokens", type=int)
            sub.add_argument(
                "--markdown-profile",
                choices=("model", "pp3"),
                default="model",
                help="model: upstream dollar repair; pp3: also save document.pp3.md after assembly",
            )
    for command in ("serve-vllm", "serve-llama"):
        sub = commands.add_parser(command)
        sub.add_argument("--host", default="127.0.0.1")
        sub.add_argument("--port", type=int, default=8010)
        if command == "serve-vllm":
            sub.add_argument("--model", default=DEFAULT_MODEL)
            sub.add_argument("--max-model-len", type=int)
            sub.add_argument("--gpu-memory-utilization",
                             type=float, default=0.85)
            sub.add_argument(
                "--dtype", choices=("auto", "half", "bfloat16", "float"), default="auto"
            )
            sub.add_argument("--enforce-eager", action="store_true")
        else:
            sub.add_argument("--binary", default="llama-server")
            sub.add_argument("--model")
            sub.add_argument("--mmproj")
            sub.add_argument(
                "--hf-repo", help="native llama.cpp Hugging Face repository[:quant]")
            sub.add_argument("--gpu-layers", type=int, default=99)
            sub.add_argument("--ctx-size", type=int)
        sub.add_argument(
            "extra_args", nargs=argparse.REMAINDER, help="native backend flags after --"
        )
    viewer = commands.add_parser(
        "viewer", aliases=["web"], help="browse runs and parse uploads in the browser"
    )
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int,
                        help="default: 8000 or the next free port")
    viewer.add_argument("--output-dir", default="out")
    viewer.add_argument(
        "--base-url", help="local or remote OpenAI-compatible model /v1 URL")
    viewer.add_argument("--model")
    viewer.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    viewer.add_argument(
        "--split", choices=("auto", "off", "force"), default="auto")
    viewer.add_argument("--markdown-profile",
                        choices=("model", "pp3"), default="model")
    return result


def main(argv: list[str] | None = None) -> int:
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    package_logger = logging.getLogger("src")
    if not package_logger.handlers:
        package_logger.addHandler(logging.StreamHandler())
    package_logger.setLevel(logging.INFO)
    try:
        if command in {"parse", "viewer", "web"}:
            saved = read_settings(Path(args["output_dir"]))
            for key in ("base_url", "model"):
                args[key] = args.get(key) or saved.get(key)
        if command in {"viewer", "web"}:
            from .server import serve

            return serve(**args)
        if command in {"serve-vllm", "serve-llama"}:
            build = build_vllm_command if command == "serve-vllm" else build_llama_command
            extras = args["extra_args"]
            args["extra_args"] = extras[1:] if extras[:1] == ["--"] else extras
            return launch(build(**args))
        run = parse_document if command == "parse" else preprocess_document
        result = run(**args)
        print(result["output_dir"])
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Inspect manifest.json for retained partial results.", file=sys.stderr)
        return 130
    except Exception as error:
        print(error_message(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
