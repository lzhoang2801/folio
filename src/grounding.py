from __future__ import annotations

import re
from dataclasses import dataclass

GROUNDING_PROMPT = "grounding"
MARKER = re.compile(
    r"!\[([A-Za-z_][\w-]*?)(\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*"
)


@dataclass(frozen=True)
class Profile:
    generation: int | None
    longest_edge: int
    top_p: float = 1.0
    context_length: int = 24576


def profile(model: str) -> Profile:
    name = model.lower()
    pixtral = "lightonocr-3-1b" in name
    return Profile(
        generation=3 if re.search(r"lightonocr-3-", name) else None,
        longest_edge=1540 if pixtral else 2048,
        context_length=16384 if pixtral else 24576,
    )


@dataclass
class Block:
    label: str
    bbox: tuple[int, int, int, int]
    text: str
    continues: bool = False

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bbox": list(self.bbox),
            "text": self.text,
            "continues": self.continues,
        }


def parse_blocks(raw: str) -> list[Block]:
    marks = list(MARKER.finditer(raw))
    blocks = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        blocks.append(
            Block(
                label=mark.group(1),
                continues=mark.group(2) == "+",
                bbox=(
                    int(mark.group(3)),
                    int(mark.group(4)),
                    int(mark.group(5)),
                    int(mark.group(6)),
                ),
                text=raw[mark.end(): end].strip(),
            )
        )
    return blocks
