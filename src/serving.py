from __future__ import annotations

import ipaddress
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .grounding import profile

DEFAULT_MODEL = "lightonai/LightOnOCR-3-0.8B"


def _check_model(model: str) -> None:
    if not re.search(r"(?i)(?:^|[/\\])lightonocr-3-(?:0\.8b|1b|4b)(?:[-_.]|$)", model):
        raise ValueError(
            "A named LightOnOCR-3 0.8B, 1B or 4B checkpoint is required")


def _server_options(host: str, port: int, api_key: str | None) -> list[str]:
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("host must be an IP address or hostname")
    if not 1 <= port <= 65535:
        raise ValueError("port must be 1-65535")
    key = api_key if api_key is not None else os.environ.get(
        "LIGHTONOCR_API_KEY")
    if key is not None and (not key.strip() or any(c.isspace() for c in key) or "\0" in key):
        raise ValueError("API key must be nonempty and contain no whitespace")
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host.lower() == "localhost"
    if not local and (not key or key == "EMPTY"):
        raise ValueError(
            "A non-loopback host requires LIGHTONOCR_API_KEY or api_key")
    options = ["--host", host, "--port", str(port)]
    if key:
        options.extend(["--api-key", key])
    return options


def build_vllm_command(
    *,
    model: str = DEFAULT_MODEL,
    host: str = "127.0.0.1",
    port: int = 8010,
    api_key: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "auto",
    enforce_eager: bool = False,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build a vLLM 0.30.0 command using this Python environment's installation.

    Model weights must already be cached when running offline. vLLM otherwise
    resolves the Hugging Face checkpoint using its standard download behavior.
    """
    _check_model(model)
    context = max_model_len if max_model_len is not None else profile(
        model).context_length
    if context < 1 or not 0 < gpu_memory_utilization < 1:
        raise ValueError(
            "max_model_len must be positive; gpu_memory_utilization must be between 0 and 1"
        )
    if dtype not in {"auto", "half", "bfloat16", "float"}:
        raise ValueError("dtype must be auto, half, bfloat16 or float")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        *_server_options(host, port, api_key),
        "--max-model-len",
        str(context),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--limit-mm-per-prompt",
        '{"image":1}',
        "--mm-processor-cache-gb",
        "0",
        "--no-enable-prefix-caching",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--dtype",
        dtype,
    ]
    if enforce_eager:
        command.append("--enforce-eager")
    return [*command, *extra_args]


def build_llama_command(
    *,
    binary: str = "llama-server",
    model: str | Path | None = None,
    mmproj: str | Path | None = None,
    hf_repo: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8010,
    api_key: str | None = None,
    gpu_layers: int = 99,
    ctx_size: int | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build a native llama.cpp command with local GGUFs or its Hugging Face loader.

    ``gpu_layers=-1`` offloads all layers; zero also keeps the projector on CPU.
    ``ctx_size`` defaults to 24,576 (16,384 for the 1B Pixtral checkpoint).
    """
    if hf_repo:
        if model is not None or mmproj is not None:
            raise ValueError(
                "hf_repo cannot be combined with local model or mmproj")
        _check_model(hf_repo.split(":", 1)[0])
        weights = ["--hf-repo", hf_repo]
        model_name = hf_repo.rsplit("/", 1)[-1].split(":", 1)[0]
    else:
        if model is None or mmproj is None:
            raise ValueError("provide hf_repo or both local model and mmproj")
        model, mmproj = Path(model), Path(mmproj)
        _check_model(model.as_posix())
        if model.suffix.lower() != ".gguf" or mmproj.suffix.lower() != ".gguf":
            raise ValueError("model and mmproj must be GGUF files")
        for path in (model, mmproj):
            if not path.is_file():
                raise FileNotFoundError(f"Not a file: {path}")
        if model.resolve() == mmproj.resolve():
            raise ValueError("model and mmproj must be different files")
        weights = ["--model", str(model.resolve()),
                   "--mmproj", str(mmproj.resolve())]
        model_name = model.stem
    context = ctx_size if ctx_size is not None else profile(
        model_name).context_length
    if context < 1 or gpu_layers < -1:
        raise ValueError(
            "ctx_size must be positive and gpu_layers must be -1 or nonnegative")
    executable = shutil.which(binary)
    if executable is None:
        raise FileNotFoundError(
            "Cannot find llama-server; install a current llama.cpp build or pass binary"
        )
    command = [
        executable,
        *weights,
        "--alias",
        model_name,
        *_server_options(host, port, api_key),
        "--ctx-size",
        str(context),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "all" if gpu_layers == -1 else str(gpu_layers),
        "--jinja",
        "--reasoning",
        "off",
        "--no-cache-prompt",
        "--no-context-shift",
        "--temp",
        "0.2",
        "--top-p",
        "1.0",
        "--top-k",
        "0",
        "--min-p",
        "0",
        "--repeat-penalty",
        "1.0",
    ]
    if gpu_layers == 0:
        command.append("--no-mmproj-offload")
    return [*command, *extra_args]


def launch(command: list[str]) -> int:
    """Run in the foreground without shell expansion or printing credentials.

    Move the API key into the backend's documented environment variable so it
    is absent from the child command line and server argument logs.
    """
    if not command:
        raise ValueError("command cannot be empty")
    arguments = list(command)
    environment = os.environ.copy()
    key = environment.get("LIGHTONOCR_API_KEY")
    while "--api-key" in arguments:
        index = arguments.index("--api-key")
        if index + 1 == len(arguments):
            raise ValueError("--api-key requires a value")
        key = arguments[index + 1]
        del arguments[index : index + 2]
    name = "VLLM_API_KEY" if "vllm.entrypoints.cli.main" in arguments else "LLAMA_API_KEY"
    if key:
        environment[name] = key
    if name == "VLLM_API_KEY" and not any(
        arg.split("=", 1)[0].startswith("--data-parallel-")
        or arg.split("=", 1)[0]
        in {
            "--tensor-parallel-size",
            "--pipeline-parallel-size",
            "-tp",
            "-pp",
            "-dp",
            "-dpl",
            "-dpr",
            "-dpn",
            "-dpa",
            "-dpp",
            "-dpb",
            "-dph",
            "-dpe",
            "-dpm",
        }
        for arg in arguments
    ):
        import torch

        devices = torch.cuda.device_count()
        if not devices:
            raise RuntimeError("No CUDA GPUs are visible to vLLM. Enable a GPU runtime.")
        arguments.extend(["--data-parallel-size", str(devices)])
        print(f"vLLM: {devices} visible CUDA GPU(s), one model replica per GPU.", flush=True)
    try:
        return subprocess.run(arguments, check=False, env=environment).returncode
    except KeyboardInterrupt:
        return 130
