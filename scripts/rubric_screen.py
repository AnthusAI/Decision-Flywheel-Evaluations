"""R3 live screen: Jev with a one-line rubric (S) vs the published guideline excerpt (F), optionally S plus fixed few-shot examples (S+k).

Items: a seeded sample of the dataset's TRAIN split outside the 1,500 items the R2 local baseline was trained on, so held-out test items stay untouched. Answers are cached in var/rubric-screen/answers.jsonl (keyed by dataset, condition, item), so reruns never pay twice. Outputs: var/rubric-screen/screen.json (text-free).
"""
import argparse, collections, concurrent.futures as cf, csv, hashlib, json, random, sys, threading
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA, PROMPTS, VAR = ROOT/".data/rubric-screen", ROOT/"studies/rubric_screen/prompts", ROOT/"var/rubric-screen"
CAP = {"1": "Macroeconomics", "5": "Labor and Employment", "13": "Social Welfare", "14": "Community Development and Housing Issues", "15": "Banking, Finance, and Domestic Commerce", "18": "Foreign Trade"}

def load(name):
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
    if cond.startswith("S"): return (PROMPTS/name/"S.txt").read_text().strip()
    p = PROMPTS/name/"F.txt"
    return (p if p.exists() else VAR/name/"F.txt").read_text().strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="edos,fomc,cap_nyt"); ap.add_argument("--conditions", default="S,F")
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--shots-per-label", type=int, default=2)
    ap.add_argument("--max-requests", type=int, required=True); ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if not a.confirm: sys.exit("live Jev calls need --confirm")
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
        rows = load(name); pool, screen = split(rows, a.n, seed=11)
        labels = tuple(json.load(open(PROMPTS/name/"answer_format.json"))["options_in_fixed_order"])
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
        out[name] = res; print(name, "tfidf_lr", res["tfidf_lr"], flush=True)
    prev = json.load(open(VAR/"screen.json")) if (VAR/"screen.json").exists() else {}
    for k, val in out.items(): prev.setdefault(k, {}).update(val)
    json.dump(prev, open(VAR/"screen.json", "w"), indent=1); print("new Jev requests:", sent[0])

def one_safe(fn, row):
    try: return fn(row)
    except RuntimeError: raise
    except Exception as e:
        print("failed:", type(e).__name__, file=sys.stderr); return None

if __name__ == "__main__":
    main()
