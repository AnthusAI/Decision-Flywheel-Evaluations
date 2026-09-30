from .metrics import Observation, summarize, reliability_bins
import pytest
def test_metrics_use_declared_labels_and_mark_incomplete_probability_coverage_unavailable():
 rows=[Observation("r1","a","x",0,"canonical","yes","yes","completed",{"yes":.8,"no":.2}),Observation("r2","b","x",0,"canonical","no","yes","completed")]
 m=summarize(rows,("yes","no"))
 assert m.accuracy==.5 and m.macro_f1==1/3 and m.log_loss is None and m.probability_coverage==1
def test_ece_uses_probability_argmax_not_returned_choice():
 m=summarize([Observation("r","a","x",0,"o","no","no","completed",{"yes":.9,"no":.1})],("yes","no"))
 assert m.ece==.9 and [(b.index,b.lower,b.upper,b.mean_confidence,b.accuracy,b.count) for b in m.reliability]==[(9,.9,1.,.9,0.,1)]
def test_observations_reject_invalid_status_usage_and_reliability_keeps_empty_bin_bounds():
 with pytest.raises(ValueError): Observation("r","t","c",0,"o","yes","yes","unknown")
 with pytest.raises(ValueError): Observation("r","t","c",0,"o","yes","yes","completed",usage={"tokens":float("nan")})
 bins=reliability_bins([Observation("r","t","c",0,"o","yes","yes","completed",{"yes":.5,"no":.5})],("yes","no"),bins=2)
 assert [(b.index,b.lower,b.upper,b.count) for b in bins]==[(0,0,.5,0),(1,.5,1.,1)]

def test_explicit_provider_confidence_is_optional_and_never_substitutes_probability_confidence():
 row=Observation("r","t","c",0,"o","yes","yes","completed",{"yes":.9,"no":.1},confidence=.37)
 assert row.confidence==.37 and row.probabilities=={"yes":.9,"no":.1}
 for invalid in (True,-.01,1.01,float("nan"),".37"):
  with pytest.raises(ValueError,match="confidence"):
   Observation("r","t","c",0,"o","yes","yes","completed",confidence=invalid)

def test_unknown_or_missing_completed_labels_are_errors_not_silently_dropped():
 rows=[Observation("r","t","c",0,"o","yes","maybe","completed")]
 with pytest.raises(ValueError,match="predicted label"):
  summarize(rows,("yes","no"))
 with pytest.raises(ValueError,match="true label"):
  summarize([Observation("r","t","c",0,"o","maybe","yes","failed")],("yes","no"))
 with pytest.raises(ValueError,match="completed observation"):
  summarize([Observation("r","t","c",0,"o","yes",None,"completed")],("yes","no"))

def test_metric_summary_retains_failed_malformed_and_missing_accounting():
 rows=[Observation("ok","a","c",0,"o","yes","yes","completed"),
       Observation("bad","b","c",0,"o","no",None,"failed"),
       Observation("shape","c","c",0,"o","yes",None,"malformed"),
       Observation("gone","d","c",0,"o","no",None,"missing")]
 m=summarize(rows,("yes","no"))
 assert (m.total,m.completed,m.failed,m.malformed,m.missing,m.label_coverage)==(4,1,1,1,1,1)
