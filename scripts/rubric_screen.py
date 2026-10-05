"""R3 live screen: Jev with a one-line rubric (S) vs the published guideline excerpt (F), optionally S plus fixed few-shot examples (S+k).

Items: a seeded sample of the dataset's TRAIN split outside the 1,500 items the R2 local baseline was trained on, so held-out test items stay untouched. Answers are cached in var/rubric-screen/answers.jsonl (keyed by dataset, condition, item), so reruns never pay twice. Outputs: var/rubric-screen/screen.json (text-free).

`--split heldout` (hatecheck, contractnli, swda only) screens a fixed-seed 300-item slice from each dataset's test side instead; results go under "<name>@heldout". `--dry-run` loads and prints counts and request sizes without calling Jev.

`reviews` (Amazon review moderation, streaming SME plan) is heldout-only: the held-out 300 and the stream 600 come from
`unified_reviews` through its frozen text-free split manifest (private pool text + the SME cache, read-only); gold = the SME's labels; S = studies/amazon_reviews/S.txt,
F = the private var/policy/amazon_reviews_F.txt. The label set (merged or not) is the one the SME data decides.
"""
import argparse, collections, concurrent.futures as cf, csv, functools, glob, hashlib, json, random, re, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, PROMPTS, VAR = ROOT/".data/rubric-screen", ROOT/"studies/rubric_screen/prompts", ROOT/"var/rubric-screen"
CAP = {"1": "Macroeconomics", "5": "Labor and Employment", "13": "Social Welfare", "14": "Community Development and Housing Issues", "15": "Banking, Finance, and Domestic Commerce", "18": "Foreign Trade"}

HC_FILE, CNLI_DIR, SWDA_DIR = DATA/"hatecheck/test_suite_cases.csv", DATA/"contractnli", DATA/"swda"
NEW = ("hatecheck", "contractnli", "swda")
REVIEWS = "reviews"

@functools.lru_cache(maxsize=None)
def _reviews():
    """(splits, report) for the reviews corpus; nothing here writes to var/amazon-reviews/."""
    if str(ROOT/"src") not in sys.path: sys.path.insert(0, str(ROOT/"src"))
    from decision_flywheel_evaluations import unified_reviews
    return unified_reviews.load_frozen_reviews_splits()

def _reviews_rows(ids):
    items = _reviews()[0].items
    return [(i, items[i].text, items[i].reference_label) for i in ids]

def _reviews_text(cond):
    from decision_flywheel_evaluations import unified_reviews
    merged = _reviews()[1]["merged"]
    return unified_reviews.starting_rubric(merged) if cond.startswith("S") else unified_reviews.guideline_text(merged)

def toks(text): return round(len(text.split()) * 1.3)  # same rule as candidates.json

def balanced(rows, n, seed):
    """Seeded round-robin over sorted labels: as equal per label as availability allows (same scheme as split())."""
    rows = sorted(rows); random.Random(seed).shuffle(rows)
    by = collections.defaultdict(list)
    for r in rows: by[r[2]].append(r)
    out = []
    while len(out) < n and any(by.values()):
        for l in sorted(by):
            if by[l] and len(out) < n: out.append(by[l].pop(0))
    return out

def _hatecheck(seed=11, n=300):
    """Held-out = 300 cases stratified by functional test (largest-remainder proportional, so the natural label balance is kept),
    drawn as whole templates so no template straddles pool and held-out. Cases from a template split to hit a quota exactly go to neither."""
    rows = list(csv.DictReader(open(HC_FILE, encoding="utf-8")))
    norm = lambda x: re.sub(r"\W", "", re.sub(r"\[IDENTITY_[PSA]\]", "[IDENTITY]", x).lower())
    examples = {norm(e) for e in re.findall(r'"([^"]+)"', (PROMPTS/"hatecheck/F.txt").read_text())}
    in_guide = {r["templ_id"] for r in rows if norm(r["case_templ"]) in examples or norm(r["test_case"]) in examples}
    by_f = collections.defaultdict(lambda: collections.defaultdict(list)); pool = []
    for r in rows:
        row = ("hc-" + r["case_id"], r["test_case"].strip(), r["label_gold"])
        if r["templ_id"] in in_guide: pool.append(row)  # Table 1 examples in F never enter the held-out slice
        else: by_f[r["functionality"]][r["templ_id"]].append(row)
    sizes = {f: sum(map(len, t.values())) for f, t in by_f.items()}
    raw = {f: n * k / sum(sizes.values()) for f, k in sizes.items()}; quota = {f: int(v) for f, v in raw.items()}
    for f in sorted(raw, key=lambda f: (-(raw[f] - quota[f]), f))[: n - sum(quota.values())]: quota[f] += 1
    rng = random.Random(seed); held, dropped = [], []
    for f in sorted(by_f):
        tids = sorted(by_f[f]); rng.shuffle(tids); got = []
        for t in tids:
            cases = sorted(by_f[f][t]); room = quota[f] - len(got)
            if room <= 0: pool += cases
            elif len(cases) <= room: got += cases
            else: got += cases[:room]; dropped += cases[room:]
        held += got
    return pool, held, dropped

def _contractnli(split_name):
    path = next(iter(sorted(glob.glob(str(CNLI_DIR/"**"/f"{split_name}.json"), recursive=True))), None)
    if path is None: raise FileNotFoundError(f"ContractNLI {split_name}.json not found under {CNLI_DIR} (download contract-nli.zip and unzip there)")
    d = json.load(open(path)); hyp = {k: v["hypothesis"] for k, v in d["labels"].items()}; keep, over = [], 0
    for doc in d["documents"]:
        if toks(doc["text"]) > 12000: over += len(doc["annotation_sets"][0]["annotations"]); continue
        for k, a in doc["annotation_sets"][0]["annotations"].items():  # gold evidence spans are never used
            keep.append((f"cnli-{doc['id']}-{k}", f"Hypothesis: {hyp[k]}\n\nContract:\n{doc['text']}", a["choice"]))
    return keep, over

SWDA_KEEP = ("sd", "sv", "b", "aa", "qy", "qy^d", "ba", "%")

def swda_tag(raw):
    """Clustered SWBD-DAMSL tag (coders manual 1c; same rule as cgpotts/swda damsl_act_tag). None = not eligible."""
    if "@" in raw or re.search(r"[,;]", raw): return None  # bad segmentation (manual 1c drops these) or double label
    t = raw
    if t not in ("qy^d", "qw^d", "b^m"):
        t = re.sub(r"[()@*]", "", re.sub(r"(.)\^.*", r"\1", t))
        t = {"qr": "qy", "fe": "ba", "fx": "sv"}.get(t, t)
    return t if t in SWDA_KEEP else None

def _swda():
    """(pool, heldout_candidates): split by conversation; held-out side = the 19 standard test conversations (Stolcke et al. 2000 split file)."""
    test = set(open(SWDA_DIR/"test_split.txt").read().split()); pool, cand = [], []
    for f in sorted(glob.glob(str(SWDA_DIR/"swda/sw*utt/*.utt.csv"))):
        utts = sorted(csv.DictReader(open(f, encoding="utf-8")), key=lambda r: int(r["transcript_index"]))
        for i, u in enumerate(utts):
            lab = swda_tag(u["act_tag"])
            if lab is None: continue
            turns = []
            for p in utts[:i]:
                w = " ".join(p["text"].split())
                if turns and turns[-1][0] == p["caller"]: turns[-1][1].append(w)
                else: turns.append((p["caller"], [w]))
            ctx = []
            for c, ws in turns[-2:]:
                words = " ".join(ws).split(); ctx.append(f"{c}: " + ("... " if len(words) > 80 else "") + " ".join(words[-80:]))
            text = "Previous turns:\n" + ("\n".join(ctx) or "(start of conversation)") + f"\n\nUtterance to label (speaker {u['caller']}):\n>>> {' '.join(u['text'].split())} <<<"
            row = (f"swda-{u['conversation_no']}-{u['transcript_index']}", text, lab)
            (cand if u["conversation_no"] in test else pool).append(row)
    return pool, cand

def heldout(name, n=300, seed=11):
    if name == REVIEWS: return _reviews_rows(_reviews()[0].paper600)
    if name == "hatecheck": return _hatecheck(seed, n)[1]
    if name == "contractnli": return balanced(_contractnli("test")[0], n, seed)
    if name == "swda": return balanced(_swda()[1], n, seed)
    raise SystemExit(f"--split heldout is defined only for {', '.join(NEW)}")

def load(name):
    if name == REVIEWS: return _reviews_rows(_reviews()[0].pool)  # the stream 600: the labeled side, disjoint from held-out
    if name == "hatecheck": return _hatecheck()[0]
    if name == "contractnli": return _contractnli("train")[0]
    if name == "swda": return _swda()[0]
    if name == "edos":
        rows = [r for r in csv.DictReader(open(DATA/"edos/edos_labelled_aggregated.csv")) if r["label_sexist"] == "sexist" and r["split"] == "train"]
        return [(r["rewire_id"], r["text"], r["label_category"]) for r in rows]
    if name == "fomc":
        m = {"0": "dovish", "1": "hawkish", "2": "neutral"}
        return [("fomc-" + r["index"], r["sentence"], m[r["label"]]) for r in csv.DictReader(open(DATA/"fomc/hf_train.csv", encoding="utf-8-sig"))]
    rows = [r for r in csv.DictReader(open(DATA/"cap_nyt/US-Media-NYTimes_front_page_19.3.csv", encoding="latin-1")) if r["majortopic"] in CAP and r["title"].strip()]
    return [("cap-" + r["id"], r["title"], CAP[r["majortopic"]]) for r in rows]

def split(rows, n_screen, seed):
    rows = sorted(rows); rng = random.Random(seed); rng.shuffle(rows)
    pool, rest = rows[:1500], rows[1500:]
    by = collections.defaultdict(list)
    for r in rest: by[r[2]].append(r)
    labels = sorted(by); screen = []
    while len(screen) < n_screen and any(by.values()):
        for l in labels:
            if by[l] and len(screen) < n_screen: screen.append(by[l].pop(0))
    return pool, screen

def instructions(name, cond):
    if name == REVIEWS: return _reviews_text(cond)
    if cond.startswith("S"): return (PROMPTS/name/"S.txt").read_text().strip()
    p = PROMPTS/name/"F.txt"
    return (p if p.exists() else VAR/name/"F.txt").read_text().strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="edos,fomc,cap_nyt"); ap.add_argument("--conditions", default="S,F")
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--shots-per-label", type=int, default=2)
    ap.add_argument("--max-requests", type=int); ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--split", choices=("train", "heldout"), default="train"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run: return dry_run(a)
    if a.max_requests is None: sys.exit("--max-requests is required")
    if not a.confirm: sys.exit("live Jev calls need --confirm")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from decision_flywheel.adapters.jev import JevAdapter, JevConfiguration
    from decision_flywheel.models import DecisionTask, Item, LabeledItem
    adapter = JevAdapter.from_environment(configuration=JevConfiguration(model="jev-1.13.0", max_retries=0))
    VAR.mkdir(parents=True, exist_ok=True); cache_path = VAR/"answers.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            d = json.loads(line); cache[(d["dataset"], d["condition"], d["item"])] = d
    lock, sent, out = threading.Lock(), [0], {}
    for name in a.datasets.split(","):
        rows = load(name)
        if a.split == "heldout":
            pool = sorted(rows); random.Random(11).shuffle(pool); pool = pool[:1500]; screen = heldout(name)
        else: pool, screen = split(rows, a.n, seed=11)
        if name == REVIEWS:
            if a.split != "heldout": sys.exit("reviews is screened on --split heldout only")
            labels = tuple(_reviews()[1]["labels"])
        else: labels = tuple(json.load(open(PROMPTS/name/"answer_format.json"))["options_in_fixed_order"])
        rng = random.Random(5); shots = []
        for l in labels:
            cands = [r for r in pool if r[2] == l]; rng.shuffle(cands); shots += cands[:a.shots_per_label]
        context = [LabeledItem(Item(i, {"text": t}), l) for i, t, l in shots]
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True); X = v.fit_transform([t for _, t, _ in pool])
        clf = LogisticRegression(max_iter=2000, C=4, class_weight="balanced").fit(X, [l for _, _, l in pool])
        gold = [l for _, _, l in screen]; lr = list(clf.predict(v.transform([t for _, t, _ in screen])))
        res = {"n": len(screen), "labels": list(labels), "gold_counts": dict(collections.Counter(gold)),
               "tfidf_lr": {"macro_f1": round(f1_score(gold, lr, average="macro"), 3), "accuracy": round(accuracy_score(gold, lr), 3)}}
        for cond in a.conditions.split(","):
            task = DecisionTask(name, labels, instructions(name, cond))
            ctx = context if cond == "S+shots" else []
            def one(row):
                key = (name, cond, row[0])
                if key in cache: return cache[key]["pred"]
                with lock:
                    if sent[0] >= a.max_requests: raise RuntimeError("request cap reached")
                    sent[0] += 1
                r = adapter.decide_sync(task, Item(row[0], {"text": row[1]}), ctx) if hasattr(adapter, "decide_sync") else __import__("asyncio").run(adapter.decide(task, Item(row[0], {"text": row[1]}), ctx))
                rec = {"dataset": name, "condition": cond, "item": row[0], "pred": r.label, "usage": dict(r.usage or {})}
                with lock:
                    cache[key] = rec
                    with open(cache_path, "a") as f: f.write(json.dumps(rec) + "\n")
                return r.label
            with cf.ThreadPoolExecutor(8) as ex: preds = list(ex.map(lambda r: one_safe(one, r), screen))
            ok = [(g, p) for g, p in zip(gold, preds) if p is not None]
            res[cond] = {"answered": len(ok), "macro_f1": round(f1_score([g for g, _ in ok], [p for _, p in ok], labels=list(labels), average="macro"), 3),
                         "accuracy": round(accuracy_score([g for g, _ in ok], [p for _, p in ok]), 3)}
            print(name, cond, res[cond], flush=True)
        if a.split == "heldout": res["split"] = "heldout"
        out[name if a.split == "train" else name + "@heldout"] = res; print(name, "tfidf_lr", res["tfidf_lr"], flush=True)
    prev = json.load(open(VAR/"screen.json")) if (VAR/"screen.json").exists() else {}
    for k, val in out.items(): prev.setdefault(k, {}).update(val)
    json.dump(prev, open(VAR/"screen.json", "w"), indent=1); print("new Jev requests:", sent[0])

def dry_run(a):
    """No Jev calls: counts and request sizes (instructions + item text, words x 1.3; chars/4 in brackets)."""
    for name in a.datasets.split(","):
        if name == REVIEWS and a.split != "heldout":
            print(name, "is screened on --split heldout only"); continue
        try:
            pool = load(name); screen = heldout(name) if a.split == "heldout" else split(pool, a.n, seed=11)[1]
        except FileNotFoundError as e:
            print(name, "MISSING DATA:", e); continue
        extra = ""
        if name == "hatecheck":
            _, _, dropped = _hatecheck(); extra = f" excluded_template_remainders={len(dropped)}"
        if name == "contractnli": extra = f" dropped_over_12k: train={_contractnli('train')[1]} test={_contractnli('test')[1]}"
        if name == REVIEWS:
            rep = _reviews()[1]
            extra = (f" labels={rep['labels']} merged={rep['merged']} excluded_from_gold={rep['excluded_from_gold']}"
                     f" floor_met={rep['floor_met']} natural_mix={rep['natural_mix']} sme_model={rep['sme_model']}")
        print(f"{name} split={a.split} pool={len(pool)} pool_labels={dict(sorted(collections.Counter(l for _, _, l in pool).items()))}{extra}")
        print(f"{name} screen n={len(screen)} labels={dict(sorted(collections.Counter(l for _, _, l in screen).items()))}")
        longest = max(screen, key=lambda r: len(r[1]))
        for cond in a.conditions.split(","):
            ins = instructions(name, cond)
            print(f"{name} {cond}: instructions={toks(ins)} max_request={max(toks(ins + chr(10) + t) for _, t, _ in screen)} [{(len(ins) + len(longest[1])) // 4}]")
        print(f"{name} jev_requests_upper_bound={len(screen) * len(a.conditions.split(','))} (before the answer cache)")

def one_safe(fn, row):
    try: return fn(row)
    except RuntimeError: raise
    except Exception as e:
        print("failed:", type(e).__name__, file=sys.stderr); return None

if __name__ == "__main__":
    main()
