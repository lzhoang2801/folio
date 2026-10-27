# Folio

Parse PDFs and images with LightOnOCR-3. Folio handles layout, reading order,
and paragraph/table continuation across pages.

## Kaggle / Colab

Open [notebooks/folio.ipynb](notebooks/folio.ipynb), enable GPU and Internet,
then select **Run All**. Open the HTTPS URL printed by Cloudflare and upload a document
once vLLM is ready.

## Run locally

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
```

Start a model backend, then launch the web viewer in another terminal:

```bash
uv run folio web --base-url http://127.0.0.1:8010/v1
```

Open `http://127.0.0.1:8000`. Upload a PDF/image, follow its progress in **Runs**,
and review the pages, document, or raw output. **Download ZIP** includes `assets/`,
`document.json`, and all `document*.md` exports.

### vLLM

```bash
uv sync --frozen --extra vllm
uv run --extra vllm folio serve-vllm
```

The default model is `lightonai/LightOnOCR-3-0.8B`. Native backend flags follow `--`.

### llama.cpp

Install a CUDA-enabled `llama-server`, then run:

```bash
uv run folio serve-llama --binary /path/to/llama-server --hf-repo Prathat/LightOnOCR-3-0.8B-GGUF:Q8_0
```

## CLI

```bash
uv run folio parse report.pdf
uv run folio parse report.pdf --pages 1,3-5 --name report-test
uv run folio parse report.pdf --markdown-profile pp3
uv run folio --help
```

Results are saved under `out/<run-name>/`.
