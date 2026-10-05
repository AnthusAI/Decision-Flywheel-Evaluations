from pathlib import Path


def test_the_reviews_freeze_script_is_only_a_thin_one_shot_entry_point():
    source = (Path(__file__).parent / "reviews_freeze.py").read_text(encoding="utf-8")
    assert "unified_reviews_manifest import freeze_main" in source
    assert "freeze_main()" in source
