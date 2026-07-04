"""Build text-derived card features from database.json.

Each card becomes a document (name, kind, typing, attack titles/effects,
ability text, trainer effect); documents are TF-IDF vectorized (word
unigrams+bigrams) and reduced with truncated SVD to a dense vector,
L2-normalized, aligned with the engine's global card index (padding row of
zeros appended). Cards with similar *wording* — search, heal, energy
acceleration, status conditions — land close together, giving the
`CardEncoder` semantics that the numeric attribute table cannot express.
This is the offline stand-in for ygo-agent's LLM text embeddings.

Output: rl/card_text_features.npy, shape (num_cards + 1, dim).

Usage:
    python rl/build_text_features.py [--dim 64] [--out rl/card_text_features.npy]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def card_document(entry: dict) -> str:
    parts = []
    if "Pokemon" in entry:
        c = entry["Pokemon"]
        parts += [
            c["name"],
            f"pokemon stage{c['stage']}",
            str(c["energy_type"]),
            f"hp{c['hp'] // 30}",
        ]
        if c.get("evolves_from"):
            parts.append(f"evolves from {c['evolves_from']}")
        if c.get("ability"):
            parts += ["ability", c["ability"]["title"], c["ability"]["effect"]]
        for attack in c.get("attacks", []):
            parts += ["attack", attack["title"]]
            parts.append(f"cost{len(attack['energy_required'])}")
            if attack.get("effect"):
                parts.append(attack["effect"])
    else:
        c = entry["Trainer"]
        parts += [c["name"], "trainer", str(c["trainer_card_type"]), c.get("effect", "")]
    return " ".join(str(p) for p in parts if p)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database", default="database.json")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--out", default="rl/card_text_features.npy")
    args = p.parse_args()

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    import deckgym

    entries = json.load(open(args.database))
    # Align to the engine's global index via card id.
    def entry_id(entry):
        return next(iter(entry.values()))["id"]

    by_id = {entry_id(e): e for e in entries}
    order = deckgym.global_card_ids()
    docs = [card_document(by_id[cid]) for cid in order]

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = tfidf.fit_transform(docs)
    svd = TruncatedSVD(n_components=args.dim, random_state=0)
    dense = svd.fit_transform(matrix)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8

    out = np.zeros((len(order) + 1, args.dim), dtype=np.float32)  # +1 padding row
    out[: len(order)] = dense
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(
        f"wrote {args.out}: {out.shape}, tfidf vocab {len(tfidf.vocabulary_)}, "
        f"svd explained variance {svd.explained_variance_ratio_.sum():.3f}"
    )


if __name__ == "__main__":
    main()
