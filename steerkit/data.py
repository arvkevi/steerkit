from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ContrastPair:
    """A single contrast pair: shared user prompt with a concept-bearing and a neutral assistant response.

    Activations are extracted at the last token of the chat-templated full text
    (prompt + each response), which captures the model's internal state at the
    moment it has just produced (or "decided") the corresponding response style.
    """

    prompt: str
    positive_response: str
    negative_response: str


def load_pairs_jsonl(path: str | Path) -> list[ContrastPair]:
    pairs: list[ContrastPair] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pairs.append(ContrastPair(**d))
    return pairs


def save_pairs_jsonl(pairs: list[ContrastPair], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair)) + "\n")
