"""Build per-card text-embedding features for the deck-general agent.

Composes a short natural-language description of every card in the engine
(name, kind, attacks with costs/damage/effect text, ability text, trainer
effect), encodes it with a sentence transformer, and saves an
`(num_cards + 1, dim)` float32 .npy aligned with the engine's global card
index (last row = zeros, the padding id).

This is the sentence-transformer upgrade of `rl/build_text_features.py`
(TF-IDF + SVD, no ML dependencies): same output contract, richer
semantics. Both follow the ygo-agent / Cardsformer idea — identity
embeddings capture what training saw, attribute features capture stats,
but *effect text* is what lets "Flip a coin, if heads do 40 more damage"
on an unseen card mean the same thing it meant on a seen one. Feed the
result to training via `--card-text` (concatenated onto the numeric
attribute table inside `CardEncoder`; checkpoints stay self-contained
because the table is a buffer). Requires `uv pip install
sentence-transformers` and downloads MiniLM (~90 MB) on first run.

Usage:
    uv run python rl/sentence_text_features.py --out runs/card_text_minilm.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np


def card_text(entry: dict) -> str:
    if "Pokemon" in entry:
        c = entry["Pokemon"]
        bits = [
            f"{c['name']}, stage {c['stage']} {c['energy_type']} Pokemon, {c['hp']} HP."
        ]
        if c.get("ability"):
            bits.append(f"Ability {c['ability']['title']}: {c['ability']['effect']}")
        for attack in c.get("attacks", []):
            cost = "+".join(attack.get("energy_required", [])) or "free"
            line = f"Attack {attack['title']} ({cost}) {attack['fixed_damage']} damage."
            if attack.get("effect"):
                line += f" {attack['effect']}"
            bits.append(line)
        if c.get("weakness"):
            bits.append(f"Weak to {c['weakness']}.")
        return " ".join(bits)
    c = entry["Trainer"]
    kind = c.get("trainer_card_type", "Trainer")
    return f"{c['name']}, {kind} card. {c.get('effect', '')}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", default="database.json")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="runs/card_text_minilm.npy")
    args = p.parse_args()

    import deckgym
    from sentence_transformers import SentenceTransformer

    entries = json.load(open(args.database))
    by_id = {}
    for entry in entries:
        card = entry.get("Pokemon") or entry.get("Trainer")
        by_id[card["id"]] = entry

    global_ids = deckgym.global_card_ids()
    texts = []
    missing = 0
    for card_id in global_ids:
        entry = by_id.get(card_id)
        if entry is None:
            texts.append("unknown card")
            missing += 1
        else:
            texts.append(card_text(entry))
    print(f"{len(texts)} cards ({missing} missing from database.json)")
    print("example:", texts[0])

    model = SentenceTransformer(args.model)
    emb = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    table = np.zeros((len(texts) + 1, emb.shape[1]), dtype=np.float32)
    table[:-1] = emb  # last row stays zero: the padding id
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, table)
    print(f"saved {args.out}: {table.shape}")


if __name__ == "__main__":
    main()
