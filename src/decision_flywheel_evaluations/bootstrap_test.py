from .bootstrap import paired_macro_f1_bootstrap, paired_order_bootstrap, paired_permutation_family_bootstrap
from .metrics import Observation
from . import bootstrap, metrics
import pytest
import random
def _rows():
 return [Observation(f"{c}{d}{t}",t,c,d,"canonical","yes","yes" if c=="t" else "no","completed") for c in ("b","t") for d in (0,1) for t in ("a","z")]
def test_paired_bootstrap_is_seed_deterministic_and_pairs_zero_shot_baseline_targets():
 a=paired_macro_f1_bootstrap(_rows(),("yes","no"),baseline="b",treatment="t",seed=3,resamples=20)
 assert a==paired_macro_f1_bootstrap(_rows(),("yes","no"),baseline="b",treatment="t",seed=3,resamples=20)
 assert a.effect==.5
def test_paired_bootstrap_rejects_missing_cells_instead_of_complete_case_inference():
 with pytest.raises(ValueError,match="missing cells"):
  paired_macro_f1_bootstrap(_rows()[:-1]+[Observation("x","z","t",1,"canonical","yes",None,"failed")],("yes","no"),baseline="b",treatment="t")
def test_single_draw_zero_shot_baseline_pairs_with_multi_draw_treatment_and_accuracy():
 rows=[Observation(f"b{t}",t,"zero",0,"canonical",label,label,"completed") for t,label in (("a","yes"),("z","no"))]
 rows += [Observation(f"t{d}{t}",t,"few",d,"canonical",label,"yes" if d==0 else label,"completed") for d in (0,1) for t,label in (("a","yes"),("z","no"))]
 result=paired_macro_f1_bootstrap(rows,("yes","no"),baseline="zero",treatment="few",metric="accuracy",seed=2,resamples=9)
 assert result.effect == -.25

def test_multi_draw_baseline_pairs_with_a_single_draw_treatment_without_treating_the_single_draw_as_repeats():
 rows=[]
 for draw in range(5):
  rows.extend(Observation(f"baseline{draw}{target}",target,"random",draw,"canonical",label,
                          label if draw in (1,3) else "yes","completed")
              for target,label in (("a","yes"),("z","no")))
 rows.extend(Observation(f"retrieval{target}",target,"retrieval",0,"canonical",label,label,"completed")
             for target,label in (("a","yes"),("z","no")))
 result=paired_macro_f1_bootstrap(rows,("yes","no"),baseline="random",treatment="retrieval",
                                  metric="accuracy",seed=11,resamples=25,
                                  expected_target_ids=("a","z"))
 assert result.effect == pytest.approx(.3)
 assert result == paired_macro_f1_bootstrap(list(reversed(rows)),("yes","no"),baseline="random",
                                             treatment="retrieval",metric="accuracy",seed=11,resamples=25,
                                             expected_target_ids=("a","z"))
def test_multi_draw_baseline_uses_matched_draw_mean_not_its_first_draw():
 rows=[]
 for condition in ("b","t"):
  for draw in (0,1):
   rows += [Observation(f"{condition}{draw}{target}",target,condition,draw,"canonical","yes", "no" if condition=="b" and draw==0 else "yes","completed") for target in ("a","z")]
 assert paired_macro_f1_bootstrap(rows,("yes","no"),baseline="b",treatment="t",resamples=3).effect == .25

def test_bootstrap_rejects_missing_arm_duplicate_cells_and_target_manifest_gaps():
 rows=_rows()
 with pytest.raises(ValueError,match="missing baseline"):
  paired_macro_f1_bootstrap([r for r in rows if r.condition=="t"],("yes","no"),baseline="b",treatment="t")
 with pytest.raises(ValueError,match="duplicate paired cell"):
  paired_macro_f1_bootstrap(rows+[Observation("another",rows[0].target_id,rows[0].condition,rows[0].draw,rows[0].order,"yes","no","completed")],("yes","no"),baseline="b",treatment="t")
 with pytest.raises(ValueError,match="expected target"):
  paired_macro_f1_bootstrap(rows,("yes","no"),baseline="b",treatment="t",expected_target_ids=("a",))

def test_bootstrap_resamples_treatment_and_multi_draw_baseline_draws_as_matched_pairs():
 rows=[]
 for condition in ("b","t"):
  for draw in (0,1):
   rows.extend(Observation(f"{condition}{draw}{target}",target,condition,draw,"canonical",label,
                           label if (condition,draw)!=("t",0) else "yes","completed")
               for target,label in (("a","yes"),("z","no")))
 first=paired_macro_f1_bootstrap(rows,("yes","no"),baseline="b",treatment="t",seed=19,resamples=12)
 assert first == paired_macro_f1_bootstrap(list(reversed(rows)),("yes","no"),baseline="b",treatment="t",seed=19,resamples=12)

def test_joint_target_and_draw_pairing_keeps_identical_arms_at_zero_for_every_resample():
 rows=[]
 for condition in ("b","t"):
  for draw in (0,1):
   rows.extend(Observation(f"{condition}{draw}{target}",target,condition,draw,"canonical",label,
                           label if draw==0 else ("no" if label=="yes" else "yes"),"completed")
               for target,label in (("a","yes"),("z","no")))
 result=paired_macro_f1_bootstrap(rows,("yes","no"),baseline="b",treatment="t",seed=7,resamples=30)
 assert result.effect==0 and set(result.replicates)=={0}

def test_order_bootstrap_matches_target_draw_and_order_cells_without_flattening_rows():
 rows=[]
 for condition in ("c",):
  for draw in (0,1):
   for order in ("canonical","reversed"):
    rows.extend(Observation(f"{draw}{order}{target}",target,condition,draw,order,label,
                            label if order=="canonical" else ("no" if label=="yes" else "yes"),"completed")
                for target,label in (("a","yes"),("z","no")))
 result=paired_order_bootstrap(rows,("yes","no"),condition="c",baseline_order="canonical",treatment_order="reversed",seed=1,resamples=9,metric="accuracy")
 assert result.effect==-1 and all(value==-1 for value in result.replicates)

def test_order_bootstrap_rejects_inconsistent_truth_and_wrong_target_manifest():
 rows=[Observation(f"{order}{target}",target,"c",0,order,label,label,"completed")
       for order in ("canonical","reversed") for target,label in (("a","yes"),("z","no"))]
 bad=rows[:-1]+[Observation("bad","z","c",0,"reversed","yes","yes","completed")]
 with pytest.raises(ValueError,match="identical true"):
  paired_order_bootstrap(bad,("yes","no"),condition="c",baseline_order="canonical",treatment_order="reversed")
 with pytest.raises(ValueError,match="expected target"):
  paired_order_bootstrap(rows,("yes","no"),condition="c",baseline_order="canonical",treatment_order="reversed",expected_target_ids=("a",))

def test_permutation_family_keeps_identical_order_arms_zero_under_resampling():
 rows=[]
 for draw in (0,1):
  for order in ("canonical","shuffle-1","shuffle-2"):
   rows.extend(Observation(f"{draw}{order}{target}",target,"c",draw,order,label,
                           label if draw==0 else ("no" if label=="yes" else "yes"),"completed")
               for target,label in (("a","yes"),("z","no")))
 result=paired_permutation_family_bootstrap(rows,("yes","no"),condition="c",reference_order="canonical",exchangeable_orders=("shuffle-1","shuffle-2"),seed=2,resamples=12)
 assert result.effect==0 and set(result.replicates)=={0}

def test_permutation_family_uses_one_shared_target_draw_and_order_vector_per_replicate():
 rows=[]
 predictions={
  (0,"canonical","a"):"yes",(0,"canonical","z"):"no",(1,"canonical","a"):"no",(1,"canonical","z"):"no",
  (0,"shuffle-1","a"):"yes",(0,"shuffle-1","z"):"yes",(1,"shuffle-1","a"):"yes",(1,"shuffle-1","z"):"no",
  (0,"shuffle-2","a"):"no",(0,"shuffle-2","z"):"no",(1,"shuffle-2","a"):"yes",(1,"shuffle-2","z"):"yes"}
 for draw in (0,1):
  for order in ("canonical","shuffle-1","shuffle-2"):
   for target,label in (("a","yes"),("z","no")):
    rows.append(Observation(f"{draw}{order}{target}",target,"c",draw,order,label,predictions[(draw,order,target)],"completed"))
 result=paired_permutation_family_bootstrap(rows,("yes","no"),condition="c",reference_order="canonical",exchangeable_orders=("shuffle-1","shuffle-2"),seed=4,resamples=5)
 rng=random.Random(4); expected=[]
 for _ in range(5):
  targets=[rng.choice(("a","z")) for _ in range(2)]; draws=[rng.choice((0,1)) for _ in range(2)]; orders=[rng.choice(("shuffle-1","shuffle-2")) for _ in range(2)]
  def score(order): return sum(sum(predictions[(draw,order,target)]==label for target,label in ((target,"yes" if target=="a" else "no") for target in targets))/len(targets) for draw in draws)/len(draws)
  expected.append(sum(score(order)-score("canonical") for order in orders)/len(orders))
 assert result.replicates==tuple(sorted(expected))
 with pytest.raises(ValueError,match="shuffled"):
  paired_permutation_family_bootstrap(rows,("yes","no"),condition="c",reference_order="canonical",exchangeable_orders=("reversed",))


@pytest.mark.parametrize("metric", ("accuracy", "macro_f1"))
@pytest.mark.parametrize("kind", ("conditions", "orders", "permutations"))
def test_bootstrap_validates_probability_fields_once_not_again_for_each_label_only_resample(monkeypatch, metric, kind):
 rows = [Observation(f"{condition}{draw}{order}{target}", target, condition, draw, order, label,
                     label if order == "canonical" else "yes", "completed",
                     probabilities={"yes": .6, "no": .4})
         for condition in ("b", "t") for draw in (0, 1)
         for order in ("canonical", "reversed", "shuffled-1", "shuffled-2")
         for target, label in (("a", "yes"), ("z", "no"))]
 if kind == "conditions":
  rows = [row for row in rows if row.order == "canonical"]
  run = lambda: paired_macro_f1_bootstrap(rows, ("yes", "no"), baseline="b", treatment="t", metric=metric, resamples=7)
 elif kind == "orders":
  run = lambda: paired_order_bootstrap(rows, ("yes", "no"), condition="b", baseline_order="canonical", treatment_order="reversed", metric=metric, resamples=7)
 else:
  run = lambda: paired_permutation_family_bootstrap(rows, ("yes", "no"), condition="b", reference_order="canonical", exchangeable_orders=("shuffled-1", "shuffled-2"), metric=metric, resamples=7)
 original = metrics._valid_probs
 checked = []
 def count(probabilities, labels):
  checked.append(1)
  return original(probabilities, labels)
 monkeypatch.setattr(metrics, "_valid_probs", count)
 result = run()
 assert len(result.replicates) == 7
 assert len(checked) == len(rows)


def test_label_only_bootstrap_scoring_matches_full_metrics_for_repeated_multiclass_targets():
 labels = ("yes", "no", "other")
 rows = [Observation(str(index), str(index), "c", 0, "canonical", actual, predicted, "completed")
         for index, (actual, predicted) in enumerate((("yes", "no"), ("yes", "yes"), ("no", "other"), ("other", "other")))]
 rng = random.Random(4)
 for _ in range(20):
  sample = [rng.choice(rows) for _ in range(9)]
  summary = metrics.summarize(sample, labels)
  assert bootstrap._label_score(sample, labels, "accuracy") == summary.accuracy
  assert bootstrap._label_score(sample, labels, "macro_f1") == summary.macro_f1


@pytest.mark.parametrize("expected", (("a", "z", "a"), "az", 5))
def test_order_bootstrap_requires_a_unique_typed_target_manifest(expected):
 rows = [Observation(f"{order}{target}", target, "c", 0, order, label, label, "completed")
         for order in ("canonical", "reversed") for target, label in (("a", "yes"), ("z", "no"))]
 with pytest.raises(ValueError, match="expected target"):
  paired_order_bootstrap(rows, ("yes", "no"), condition="c", baseline_order="canonical",
                         treatment_order="reversed", expected_target_ids=expected)
