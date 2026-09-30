from .metrics import Observation
from .reporting import report_cells,mean_draw_accuracy,report_pooled,report_study,holm_correction
from .reporting import percentage_effect
def test_reports_reconcile_cache_failures_missing_attempts_and_numeric_usage_by_cell():
 rows=[Observation("a","t","c",0,"o","yes","yes","completed",usage={"input_tokens":2},attempt_count=1,cache_hit=True),Observation("b","u","c",0,"o","no",None,"failed",attempt_count=2)]
 c=report_cells(rows,("yes","no"))[0]
 assert (c.requests,c.cache_hits,c.failures,c.missing,c.attempts,c.numeric_usage)==(2,1,1,0,3,{"input_tokens":2.0})
 assert mean_draw_accuracy((c,))=={("c","o"):1.0}
 assert percentage_effect(.6,.5)=={"percentage_points":9.999999999999998,"relative_percent":19.999999999999996}

def test_reports_reject_duplicate_logical_cells_and_require_exact_manifest_ids():
 row=Observation("logical-a","target","c",0,"canonical","yes","yes","completed")
 duplicate=Observation("logical-b","target","c",0,"canonical","yes","yes","completed")
 with __import__("pytest").raises(ValueError,match="duplicate logical"):
  report_cells((row,duplicate),("yes","no"))
 with __import__("pytest").raises(ValueError,match="missing expected"):
  report_cells((row,),("yes","no"),expected_request_ids=("logical-a","logical-b"))
 with __import__("pytest").raises(ValueError,match="outside expected"):
  report_cells((row,),("yes","no"),expected_request_ids=("other",))
 assert report_cells((row,),("yes","no"),expected_matrix_ids=(("c",0,"canonical","target"),))[0].logical_cells==1

def test_shared_physical_request_is_counted_once_and_pooled_f1_is_not_mean_draw_f1():
 rows=[
  Observation("a0","a","c",0,"canonical","yes","yes","completed",physical_request_id="shared",usage={"tokens":7},attempt_count=2,latency_ms=11),
  Observation("b0","b","c",0,"canonical","no","yes","completed",physical_request_id="first",usage={"tokens":7},attempt_count=2,latency_ms=11),
  Observation("a1","a","c",1,"canonical","yes","yes","completed",physical_request_id="second",usage={"tokens":3},attempt_count=1,latency_ms=5),
  Observation("b1","b","c",1,"canonical","no","no","completed",physical_request_id="third",usage={"tokens":4},attempt_count=1,latency_ms=7),
  Observation("other-a0","a","other",0,"canonical","yes","yes","completed",physical_request_id="shared",usage={"tokens":7},attempt_count=2,latency_ms=11),
 ]
 cells=report_cells(rows,("yes","no"))
 assert cells[0].physical.requests==2 and cells[0].physical.numeric_usage=={"tokens":14.0}
 pooled=report_pooled(rows,("yes","no"))[0]
 assert pooled.physical.requests==4 and pooled.physical.attempts==6
 assert pooled.physical.numeric_usage=={"tokens":21.0} and pooled.physical.latency_ms_mean==34/4
 assert pooled.metrics.macro_f1 != pooled.mean_draw_metrics.macro_f1
 assert report_study(rows,("yes","no")).physical.requests==4

def test_shared_physical_request_rejects_conflicting_explicit_provider_confidence():
 rows=[
  Observation("first","a","c",0,"canonical","yes","yes","completed",{"yes":.9,"no":.1},confidence=.37,physical_request_id="shared"),
  Observation("second","a","other",0,"canonical","yes","yes","completed",{"yes":.9,"no":.1},confidence=.9,physical_request_id="shared"),
 ]
 with __import__("pytest").raises(ValueError,match="physical request"):
  report_study(rows,("yes","no"))

def test_failed_physical_attempts_are_counted_while_unknown_usage_stays_unavailable():
 rows=[Observation("cell","target","c",0,"canonical","yes",None,"failed",physical_request_id="crashed",attempt_count=3)]
 physical=report_study(rows,("yes","no")).physical
 assert (physical.requests,physical.failures,physical.attempts,physical.usage_coverage,physical.usage_complete)==(1,1,3,0,False)
 assert physical.numeric_usage=={} and physical.physical_request_ids==("crashed",)

def test_holm_correction_computes_only_valid_declared_p_values():
 corrected=holm_correction({"selection":.01,"size":.04})
 assert corrected.available and corrected.adjusted_p_values=={"selection":.02,"size":.04}
 unavailable=holm_correction({"selection":None,"size":.04})
 assert not unavailable.available and unavailable.adjusted_p_values=={}

def test_descriptive_primary_contrasts_never_acquire_holm_adjustment_without_p_values():
 unavailable=holm_correction({"selection":None,"size":None})
 assert not unavailable.available and unavailable.reason=="valid p-values are unavailable for every declared contrast"
