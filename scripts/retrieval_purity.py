"""Free retriever comparison: neighbour-label purity on a labeled-pool sample (no Jev calls).

Run with Jev-Flywheel's interpreter and UNIFIED_FLYWHEEL_CORE pointing at the core src. The
embedding row embeds the sample once with OpenAI text-embedding-3-small (about a cent) and
caches vectors (text-free) in var/; pass --no-embedding to skip it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

CORE = os.environ.get("UNIFIED_FLYWHEEL_CORE")
if CORE:
    sys.path.insert(0, CORE)

from decision_flywheel.embedding_retrieval import EmbeddingRetriever, OpenAIEmbedder, VectorCache  # noqa: E402
from decision_flywheel.lexical_retrieval import LexicalRetriever  # noqa: E402
from decision_flywheel.models import DecisionTask, Item, LabeledItem  # noqa: E402
from decision_flywheel.retrieval import neighbour_label_purity  # noqa: E402

CLONE = Path(__file__).resolve().parents[1] / "var" / "jev-flywheel-cd4a4886"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decision_flywheel_evaluations.unified_splits import load_items  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1500)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--no-embedding", action="store_true")
    parser.add_argument("--out", default="var/unified-flywheel/retrieval-purity.json")
    args = parser.parse_args()
    items = load_items(CLONE / "fixtures")
    pool_ids = sorted(i for i, it in items.items() if it.split == "pool")[: args.sample]
    task = DecisionTask("sentiment", ("positive", "negative"), "Is the text positive or negative?")
    pool = [LabeledItem(Item(i, {"text": items[i].text}), items[i].reference_label, "trusted") for i in pool_ids]
    retrievers = {
        "lexical-v1-equivalent (binary-cosine, no stop words, ascii)": LexicalRetriever(stopwords="none", tokenizer="ascii-v1"),
        "lexical-v2 stop words (binary-cosine)": LexicalRetriever(),
        "lexical-v2 tfidf-cosine": LexicalRetriever(weighting="tfidf-cosine"),
        "lexical-v2 bm25": LexicalRetriever(weighting="bm25"),
    }
    if not args.no_embedding:
        embedder = OpenAIEmbedder.from_environment()
        retrievers["embedding text-embedding-3-small (512d)"] = EmbeddingRetriever.from_embedder(
            embedder, cache=VectorCache("var/unified-flywheel/embeddings.jsonl"))
    out = {}
    for name, retriever in retrievers.items():
        report = neighbour_label_purity(task, pool, retriever, k=args.k)
        out[name] = {"mean": round(report.mean, 4), "per_label": {k: round(v, 4) for k, v in report.per_label.items()},
                     "items": report.items, "k": report.k}
        print(f"{name:62s} purity {report.mean:.4f}")
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
