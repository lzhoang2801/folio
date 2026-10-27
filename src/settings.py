from __future__ import annotations

import json
from pathlib import Path

KEYS = ("base_url", "model")


def read_settings(out: Path) -> dict:
    try:
        data = json.loads((out / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: data[key] for key in KEYS if isinstance(data.get(key), str) and data[key]}


def write_settings(out: Path, **values: str | None) -> None:
    data = {**read_settings(out), **{key: value for key,
                                     value in values.items() if key in KEYS}}
    data = {key: value for key, value in data.items() if value}
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "settings.json.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out / "settings.json")
