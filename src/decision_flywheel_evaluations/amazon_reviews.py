"""Amazon review moderation corpus (streaming SME plan, step 1): a partial download, an enriched pool, a manifest.

Source: McAuley Lab "Amazon Reviews 2023" on Hugging Face (``McAuley-Lab/Amazon-Reviews-2023`` at a pinned
revision), the raw review files for three categories. Those files are 7-31 GB each, so ``download`` fetches
only a few fixed byte ranges spread through each file (HTTP Range requests) into the gitignored
``.data/amazon-reviews/``; each range's first and last lines are partial and dropped. Every saved range is
recorded with its URL, byte range, size and sha256. Downloading needs ``confirm=True`` (the owner approved a
~3,000-review research sample); specs pass a fake ``fetch`` and never touch the network.

Filters: review timestamp in 2013-2016 (UTC) and 15-90 words. Exact duplicate texts are kept once.

The candidate pool (default 3,000, fixed seed) is enriched: 60% 1-2 star, 15% 3 star, 25% 4-5 star, and
within each rating band up to ``KEYWORD_SHARE`` of the quota is drawn from reviews that hit a moderation cue
(ship, seller, refund, return, package, $, price, http, "in exchange", discount, profanity). Each item's
stratum is ``<band>-keyword`` or ``<band>-plain``. A band short of eligible reviews is filled from the others.
A separate natural-mix sample (default 300, uniform, disjoint from the pool) measures prevalence.

Text goes only to ``var/amazon-reviews/{pool,natural}.jsonl``. The committed manifest
(``studies/amazon_reviews/manifest.json``) holds IDs, source file, raw-row sha256, normalized-text sha256,
rating, category and stratum, plus the download record; no review text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

DATASET = "McAuley-Lab/Amazon-Reviews-2023"
REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
CATEGORIES = ("Cell_Phones_and_Accessories", "Home_and_Kitchen", "Toys_and_Games")
FILE_SIZES = {"Cell_Phones_and_Accessories": 9342568048, "Home_and_Kitchen": 31408889188,
              "Toys_and_Games": 7316094780}   # bytes at REVISION (Hugging Face file metadata)
CHUNKS, CHUNK_BYTES = 6, 8 * 1024 * 1024
YEARS = (2013, 2016)
WORDS = (15, 90)
POOL_SIZE, NATURAL_SIZE, SEED = 3000, 300, 20261002
BAND_SHARES = (("low", 0.60), ("mid", 0.15), ("high", 0.25))
KEYWORD_SHARE = 0.4

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / ".data" / "amazon-reviews"
VAR_DIR = ROOT / "var" / "amazon-reviews"
MANIFEST_PATH = ROOT / "studies" / "amazon_reviews" / "manifest.json"

_PROFANITY = r"crap|crappy|shit\w*|damn|hell|suck\w*|stupid|idiot\w*|garbage|wtf|f\*+k|fuck\w*|bull\s?shit|ass|scam\w*"
KEYWORDS = re.compile(r"\b(ship\w*|seller\w*|refund\w*|return\w*|package\w*|packaging|price\w*|discount\w*|"
                      r"in exchange|" + _PROFANITY + r")\b|\$|https?://|www\.", re.IGNORECASE)


class DownloadRefused(RuntimeError):
    """The download was not confirmed. Nothing was fetched."""


@dataclass(frozen=True)
class Review:
    review_id: str
    category: str
    rating: float
    text: str
    timestamp_ms: int
    source_file: str
    row_sha256: str

    @property
    def year(self) -> int:
        return datetime.fromtimestamp(self.timestamp_ms / 1000, tz=timezone.utc).year


@dataclass(frozen=True)
class PoolItem:
    review: Review
    stratum: str


def _sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return " ".join(re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE).split())


def text_sha256(text: str) -> str:
    return _sha(normalize(text).lower())


# ---- download -------------------------------------------------------------------------------

def file_url(category: str) -> str:
    return (f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/raw/review_categories/"
            f"{category}.jsonl")


def chunk_plan(size: int, *, chunks: int = CHUNKS, chunk_bytes: int = CHUNK_BYTES) -> List[tuple]:
    """``chunks`` inclusive byte ranges centred on evenly spaced points of the file (never at its very start)."""
    starts = [int(size * (k + 0.5) / chunks) - chunk_bytes // 2 for k in range(chunks)]
    return [(max(s, 1), max(s, 1) + chunk_bytes - 1) for s in starts]


def http_fetch(url: str, start: int, end: int) -> bytes:  # pragma: no cover - network
    import requests

    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
    if response.status_code != 206:
        raise RuntimeError(f"expected a partial response (206), got {response.status_code}")
    return response.content


def download(dest, *, confirm: bool, fetch: Callable[[str, int, int], bytes] = http_fetch,
             sizes: Optional[Dict[str, int]] = None, chunks: int = CHUNKS, chunk_bytes: int = CHUNK_BYTES) -> List[dict]:
    """Fetch the planned byte ranges into ``dest``; returns one text-free record per saved range."""
    if confirm is not True:
        raise DownloadRefused("download needs confirm=True (owner-approved sample); nothing was fetched")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for category in CATEGORIES:
        size = (sizes or FILE_SIZES)[category]
        for k, (start, end) in enumerate(chunk_plan(size, chunks=chunks, chunk_bytes=chunk_bytes)):
            body = fetch(file_url(category), start, end)
            name = f"{category}.part{k}.jsonl"
            (dest / name).write_bytes(body)
            records.append({"file": name, "category": category, "url": file_url(category), "file_size": size,
                            "range": [start, end], "bytes": len(body), "sha256": _sha(body)})
    (dest / "download.json").write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    return records


# ---- parsing and filtering ------------------------------------------------------------------

def parse_chunk(data: bytes, *, category: str, source_file: str) -> List[Review]:
    """Complete JSON lines of a byte range (the first and last lines are partial and dropped)."""
    lines = data.split(b"\n")[1:-1]
    out = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        key = json.dumps([row["user_id"], row["asin"], row["timestamp"]])
        out.append(Review(review_id="amz-" + _sha(key)[:16], category=category, rating=float(row["rating"]),
                          text=normalize(str(row.get("text") or "")), timestamp_ms=int(row["timestamp"]),
                          source_file=source_file, row_sha256=_sha(line)))
    return out


def eligible(review: Review) -> bool:
    return YEARS[0] <= review.year <= YEARS[1] and WORDS[0] <= len(review.text.split()) <= WORDS[1]


def dedupe(reviews: Iterable[Review]) -> List[Review]:
    """First occurrence of each review ID and of each normalized text."""
    seen_ids, seen_texts, out = set(), set(), []
    for r in reviews:
        h = text_sha256(r.text)
        if r.review_id in seen_ids or h in seen_texts:
            continue
        seen_ids.add(r.review_id)
        seen_texts.add(h)
        out.append(r)
    return out


def keyword_hit(text: str) -> bool:
    return bool(KEYWORDS.search(text))


def rating_band(rating: float) -> str:
    return "low" if rating <= 2 else "mid" if rating < 4 else "high"


def load_downloaded(directory=DATA_DIR) -> List[Review]:
    directory = Path(directory)
    records = json.loads((directory / "download.json").read_text(encoding="utf-8"))
    reviews = []
    for record in records:
        reviews += parse_chunk((directory / record["file"]).read_bytes(), category=record["category"],
                               source_file=record["file"])
    return dedupe(reviews)


# ---- sampling -------------------------------------------------------------------------------

def _quotas(size: int) -> Dict[str, int]:
    raw = {band: size * share for band, share in BAND_SHARES}
    quotas = {band: int(v) for band, v in raw.items()}
    for band in sorted(raw, key=lambda b: (-(raw[b] - quotas[b]), b))[:size - sum(quotas.values())]:
        quotas[band] += 1
    return quotas


def build_pool(reviews: Sequence[Review], *, size: int = POOL_SIZE, seed: int = SEED) -> List[PoolItem]:
    """The enriched candidate pool, in a seeded random order."""
    rng = random.Random(seed)
    ordered = sorted(reviews, key=lambda r: r.review_id)
    rng.shuffle(ordered)
    groups: Dict[tuple, List[Review]] = {}
    for r in ordered:
        groups.setdefault((rating_band(r.rating), keyword_hit(r.text)), []).append(r)
    chosen: List[Review] = []
    for band, quota in _quotas(size).items():
        hits, plain = groups.get((band, True), []), groups.get((band, False), [])
        take_kw = min(len(hits), round(quota * KEYWORD_SHARE))
        take_plain = min(len(plain), quota - take_kw)
        take_kw = min(len(hits), quota - take_plain)
        chosen += hits[:take_kw] + plain[:take_plain]
    taken = {r.review_id for r in chosen}
    chosen += [r for r in ordered if r.review_id not in taken][:size - len(chosen)]   # short bands: fill
    rng.shuffle(chosen)
    return [PoolItem(r, f"{rating_band(r.rating)}-{'keyword' if keyword_hit(r.text) else 'plain'}") for r in chosen]


def natural_sample(reviews: Sequence[Review], *, exclude: set, size: int = NATURAL_SIZE, seed: int = SEED) -> List[Review]:
    """A uniform random sample of the eligible reviews outside the pool (prevalence at the natural mix)."""
    rest = sorted((r for r in reviews if r.review_id not in exclude), key=lambda r: r.review_id)
    return random.Random(f"natural-{seed}").sample(rest, min(size, len(rest)))


# ---- outputs --------------------------------------------------------------------------------

def pool_lines(items: Sequence[PoolItem]) -> List[str]:
    """Private JSONL rows (text included) for ``var/``."""
    return [json.dumps({"id": p.review.review_id, "category": p.review.category, "rating": p.review.rating,
                        "stratum": p.stratum, "text": p.review.text}, sort_keys=True) for p in items]


def _entry(review: Review, stratum: str) -> dict:
    return {"id": review.review_id, "category": review.category, "rating": review.rating, "stratum": stratum,
            "source_file": review.source_file, "row_sha256": review.row_sha256, "text_sha256": text_sha256(review.text)}


def _count(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def manifest(pool: Sequence[PoolItem], natural: Sequence[Review], *, downloads: List[dict], seed: int,
             eligible_count: int) -> dict:
    """Text-free: provenance, filters, counts and one entry per item."""
    return {
        "dataset": DATASET, "revision": REVISION, "downloads": downloads,
        "filters": {"years": list(YEARS), "words": list(WORDS), "dedupe": "review id and normalized text"},
        "sampling": {"seed": seed, "pool_size": len(pool), "band_shares": dict(BAND_SHARES),
                     "keyword_share_within_band": KEYWORD_SHARE, "keyword_pattern_sha256": _sha(KEYWORDS.pattern),
                     "natural_size": len(natural)},
        "counts": {"eligible": eligible_count, "pool": len(pool), "natural": len(natural),
                   "pool_by_stratum": _count(p.stratum for p in pool),
                   "pool_by_rating": _count(int(p.review.rating) for p in pool),
                   "pool_by_category": _count(p.review.category for p in pool),
                   "natural_by_rating": _count(int(r.rating) for r in natural),
                   "natural_keyword_hits": sum(keyword_hit(r.text) for r in natural),
                   "natural_by_category": _count(r.category for r in natural)},
        "pool": [_entry(p.review, p.stratum) for p in pool],
        "natural": [_entry(r, "natural") for r in natural],
    }


def dump_manifest(result: dict) -> str:
    """Indented header, then one item per line (diffable, about half the size of fully indented JSON)."""
    head = {k: v for k, v in result.items() if k not in ("pool", "natural")}
    lists = ",\n".join(f' "{k}": [\n' + ",\n".join("  " + json.dumps(e) for e in result[k]) + "\n ]"
                       for k in ("pool", "natural"))
    return json.dumps(head, indent=1)[:-2] + ",\n" + lists + "\n}\n"


def build(data_dir=DATA_DIR, var_dir=VAR_DIR, manifest_path=MANIFEST_PATH, *, seed: int = SEED) -> dict:
    reviews = [r for r in load_downloaded(data_dir) if eligible(r)]
    pool = build_pool(reviews, size=POOL_SIZE, seed=seed)
    natural = natural_sample(reviews, exclude={p.review.review_id for p in pool}, size=NATURAL_SIZE, seed=seed)
    var_dir = Path(var_dir)
    var_dir.mkdir(parents=True, exist_ok=True)
    (var_dir / "pool.jsonl").write_text("\n".join(pool_lines(pool)) + "\n", encoding="utf-8")
    (var_dir / "natural.jsonl").write_text(
        "\n".join(pool_lines([PoolItem(r, "natural") for r in natural])) + "\n", encoding="utf-8")
    downloads = json.loads((Path(data_dir) / "download.json").read_text(encoding="utf-8"))
    result = manifest(pool, natural, downloads=downloads, seed=seed, eligible_count=len(reviews))
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(dump_manifest(result), encoding="utf-8")
    return result["counts"]


def read_pool(path=VAR_DIR / "pool.jsonl") -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="fetch the planned byte ranges (needs --confirm)")
    fetch.add_argument("--confirm", action="store_true")
    commands.add_parser("build", help="filter, sample, write var/ pool files and the text-free manifest")
    args = parser.parse_args(argv)
    if args.command == "download":
        records = download(DATA_DIR, confirm=args.confirm)
        print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records)}))
    else:
        print(json.dumps(build(), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
