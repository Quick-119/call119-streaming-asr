"""The content-CER normalization and known-text alignment used in Gold v3.2."""
from __future__ import annotations

import unicodedata


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    out = []
    for char in value:
        codepoint = ord(char)
        if 0x30A1 <= codepoint <= 0x30F6:
            char = chr(codepoint - 0x60)
        if char.isspace() or unicodedata.category(char).startswith(("P", "Z")):
            continue
        out.append(char)
    return "".join(out)


def masked_cer(segments: list[dict], hypothesis: str, cap: int = 0) -> dict:
    """Exact known-reference case of the original annotation-aware scorer.

    The public synthetic set contains no unknown masks, so cap must be zero.
    State tuple order retains the original tie-breaking of S/D/I counts.
    """
    if cap != 0 or any(segment["type"] != "known" for segment in segments):
        raise ValueError("This public benchmark has known text only")
    reference = "".join(segment["text"] for segment in segments)
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for char in reference:
        current = [(previous[0][0] + 1, previous[0][1], previous[0][2] + 1, previous[0][3])]
        for j, hyp_char in enumerate(hypothesis, 1):
            old = previous[j - 1]
            diagonal = old if char == hyp_char else (old[0] + 1, old[1] + 1, old[2], old[3])
            old = previous[j]
            deletion = (old[0] + 1, old[1], old[2] + 1, old[3])
            old = current[j - 1]
            insertion = (old[0] + 1, old[1], old[2], old[3] + 1)
            current.append(min(diagonal, deletion, insertion))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {"errors": errors, "absorbed_hypothesis_characters": 0,
            "S": substitutions, "D": deletions, "I": insertions,
            "N": len(reference), "cer": errors / max(1, len(reference)),
            "mask_absorptions": [], "mask_cap": 0}
