from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from chase.kg.graph import KG

ENCODER_PATH = "models/all-MiniLM-L6-v2"
CACHE_DIRNAME = "emb_minilm"
SPLITS = ["train", "validation", "test"]
CVT_PATTERN = re.compile(r"^[mg]\.[0-9a-z_]+$")


def load_encoder(path: str = ENCODER_PATH):
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(path, device=device)


def encode(texts: list[str], encoder, batch_size: int = 512) -> np.ndarray:
    """Unit-normalised float16 embeddings, one row per text."""
    if len(texts) == 0:
        return np.zeros((0, encoder.get_sentence_embedding_dimension()), np.float16)
    emb = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=len(texts) > 50_000)
    return emb.astype(np.float16)


def readable_relation(name: str) -> str:
    """'people.person.parents' -> 'people person parents'; 'drug_protein: target' -> 'drug protein: target'."""
    name = name.removesuffix(" (inverse)")
    return re.sub(r"[._]+", " ", name).strip()


def _relation_tables(relation_names: list[str], encoder) -> tuple[np.ndarray, np.ndarray]:
    """Embedding and inverse flag per directed relation id. An inverse shares its base text."""
    inverse = np.array([n.endswith(" (inverse)") for n in relation_names], dtype=np.int8)
    texts = [readable_relation(n) for n in relation_names]
    uniq, row = np.unique(texts, return_inverse=True)
    return encode(list(uniq), encoder)[row], inverse


def build_primekg(kg_dir: str, encoder) -> None:
    out = Path(kg_dir) / CACHE_DIRNAME
    out.mkdir(exist_ok=True)
    kg = KG.load(kg_dir)
    np.save(out / "nodes.npy", encode(kg.nodes["name"].tolist(), encoder))
    rel_emb, inverse = _relation_tables(kg.relation_names, encoder)
    np.save(out / "relations.npy", rel_emb)
    np.save(out / "rel_inverse.npy", inverse)
    print(f"primekg: {kg.num_nodes:,} nodes, {len(kg.relation_names)} relations -> {out}")


def build_freebase(data_dir: str, encoder) -> None:
    """Names are deduplicated across splits; each split gets node_emb_row (-1 for CVT) and question embeddings."""
    d = Path(data_dir)
    out = d / CACHE_DIRNAME
    out.mkdir(exist_ok=True)

    flat = {sp: pc.list_flatten(pq.read_table(d / sp / "node_names.parquet")["names"]).to_numpy(zero_copy_only=False)
            for sp in SPLITS}
    names = pd.unique(np.concatenate(list(flat.values())))
    names = np.array([n for n in names if not CVT_PATTERN.match(n)], dtype=object)
    print(f"encoding {len(names):,} unique entity names")
    np.save(out / "names.npy", encode(list(names), encoder))
    name_index = pd.Index(names)

    for sp in SPLITS:
        (out / sp).mkdir(exist_ok=True)
        np.save(out / sp / "node_emb_row.npy", name_index.get_indexer(flat[sp]).astype(np.int32))
        questions = pd.read_parquet(d / sp / "questions.parquet", columns=["question"])["question"].tolist()
        np.save(out / sp / "questions.npy", encode(questions, encoder))

    relation_names = json.loads((d / "relations.json").read_text())
    rel_emb, inverse = _relation_tables(relation_names, encoder)
    np.save(out / "relations.npy", rel_emb)
    np.save(out / "rel_inverse.npy", inverse)
    print(f"{d.name}: {len(relation_names):,} directed relations -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["primekg", "freebase"])
    ap.add_argument("path", help="data/processed/primekg_full, or data/processed/{webqsp,cwq}")
    args = ap.parse_args()
    t0 = time.time()
    encoder = load_encoder()
    (build_primekg if args.kind == "primekg" else build_freebase)(args.path, encoder)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
