"""Richer card attributes: effect-text keyword flags + evolution-chain stats.

A middle ground between the numeric attribute table (no effect semantics)
and dense text embeddings (measured neutral): hand-picked binary flags for
the effect mechanics that matter tactically (coin flips, healing, status,
energy acceleration/denial, bench damage, search, switching, damage
modifiers), plus evolution-chain features derived by joining the whole
database (does this card evolve, is it a final stage, how deep is its
line). Saved as an `(num_cards + 1, dim)` .npy aligned with the global
card index — feed to training via `--card-text` (the mechanism accepts
any aligned feature table).

Usage:
    uv run --no-sync python rl/keyword_features.py --out runs/card_keywords.npy
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

# (name, regex over the card's combined effect text, lowercase)
KEYWORDS = [
    ("coin_flip", r"flip .*coin|flip a coin"),
    ("draw", r"draw"),
    ("heal", r"heal"),
    ("status_poison", r"poison"),
    ("status_sleep", r"asleep"),
    ("status_paralyze", r"paralyz"),
    ("status_confuse", r"confus"),
    ("status_burn", r"burn"),
    ("energy_attach", r"attach.*energy|take .*energy.*energy zone"),
    ("energy_discard", r"discard.*energy"),
    ("bench_damage", r"damage to .*bench|bench.*damage"),
    ("self_damage", r"this pok.mon also does|damage to itself"),
    ("switch_force", r"switch|to the bench"),
    ("search_deck", r"from your deck|search"),
    ("shuffle", r"shuffle"),
    ("damage_boost", r"more damage"),
    ("damage_reduce", r"less damage|prevent"),
    ("hand_disruption", r"opponent.s hand"),
    ("cant_attack", r"can.t attack|can.t use"),
    ("conditional_dmg", r"if |for each"),
]


def combined_text(entry: dict) -> str:
    card = entry.get("Pokemon") or entry.get("Trainer")
    parts = []
    if "Pokemon" in entry:
        if card.get("ability"):
            parts.append(card["ability"]["effect"])
        for attack in card.get("attacks", []):
            if attack.get("effect"):
                parts.append(attack["effect"])
    else:
        parts.append(card.get("effect", ""))
    return " ".join(p for p in parts if p).lower()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", default="database.json")
    p.add_argument("--out", default="runs/card_keywords.npy")
    args = p.parse_args()

    import deckgym

    entries = json.load(open(args.database))
    by_id, evolves_from_names = {}, {}
    for entry in entries:
        card = entry.get("Pokemon") or entry.get("Trainer")
        by_id[card["id"]] = entry
        if "Pokemon" in entry and card.get("evolves_from"):
            evolves_from_names.setdefault(card["evolves_from"], set()).add(card["name"])

    patterns = [(name, re.compile(rx)) for name, rx in KEYWORDS]
    dim = len(patterns) + 4  # + evolution features
    global_ids = deckgym.global_card_ids()
    table = np.zeros((len(global_ids) + 1, dim), dtype=np.float32)

    for row, card_id in enumerate(global_ids):
        entry = by_id.get(card_id)
        if entry is None:
            continue
        text = combined_text(entry)
        for j, (_, rx) in enumerate(patterns):
            if rx.search(text):
                table[row, j] = 1.0
        if "Pokemon" in entry:
            card = entry["Pokemon"]
            has_evolution = card["name"] in evolves_from_names
            table[row, len(patterns) + 0] = 1.0 if card.get("evolves_from") else 0.0
            table[row, len(patterns) + 1] = 1.0 if has_evolution else 0.0
            # final stage of its line = evolved (stage > 0) with no evolution
            table[row, len(patterns) + 2] = 1.0 if card["stage"] > 0 and not has_evolution else 0.0
            table[row, len(patterns) + 3] = 1.0 if text else 0.0  # has any effect text

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, table)
    coverage = (table[:-1].sum(axis=0) > 0).all()
    print(f"saved {args.out}: {table.shape}; every feature fires somewhere: {coverage}")
    for j, (name, _) in enumerate(patterns):
        print(f"  {name}: {int(table[:, j].sum())} cards")


if __name__ == "__main__":
    main()
