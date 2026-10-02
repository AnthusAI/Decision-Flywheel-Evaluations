import csv, random, collections, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
B="/Users/home/Projects/Decision-Flywheel-Evaluations/.data/rubric-screen/"
def edos():
    rows=[r for r in csv.DictReader(open(B+"edos/edos_labelled_aggregated.csv")) if r["label_sexist"]=="sexist"]
    tr=[(r["text"],r["label_category"]) for r in rows if r["split"]=="train"]
    te=[(r["text"],r["label_category"]) for r in rows if r["split"]=="test"]
    return tr,te
def fomc():
    m={"0":"dovish","1":"hawkish","2":"neutral"}
    rd=lambda f:[(r["sentence"],m[r["label"]]) for r in csv.DictReader(open(B+"fomc/"+f,encoding="utf-8-sig"))]
    return rd("hf_train.csv"),rd("hf_test.csv")
def cap():
    rows=[(r["title"],r["majortopic"]) for r in csv.DictReader(open(B+"cap_nyt/US-Media-NYTimes_front_page_19.3.csv",encoding="latin-1")) if r["majortopic"] in {"1","5","13","14","15","18"} and r["title"].strip()]
    random.Random(1).shuffle(rows); return rows[:-700],rows[-700:]
out={}
for name,fn in (("edos",edos),("fomc",fomc),("cap_nyt",cap)):
    tr,te=fn(); rng=random.Random(7)
    pool=tr[:]; rng.shuffle(pool); pool=pool[:1500]
    te=te[:]; rng.shuffle(te); te=te[:700]
    v=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True)
    X=v.fit_transform([t for t,_ in pool]); y=[l for _,l in pool]
    clf=LogisticRegression(max_iter=2000,C=4,class_weight="balanced").fit(X,y)
    p=clf.predict(v.transform([t for t,_ in te])); g=[l for _,l in te]
    maj=collections.Counter(y).most_common(1)[0][0]
    out[name]={"pool":len(pool),"test":len(te),"test_labels":dict(collections.Counter(g)),
      "tfidf_lr_macro_f1":round(f1_score(g,p,average="macro"),3),"tfidf_lr_acc":round(accuracy_score(g,p),3),
      "majority_acc":round(sum(x==maj for x in g)/len(g),3)}
    print(name,out[name])
json.dump(out,open("/Users/home/Projects/Decision-Flywheel-Evaluations/var/rubric-screen-r2.json","w"),indent=1)
