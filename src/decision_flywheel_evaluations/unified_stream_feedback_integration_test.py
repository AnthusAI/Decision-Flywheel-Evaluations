from types import SimpleNamespace

from .unified_stream import StreamConfig, StreamDriver


class _Arm:
    labels = ("a", "b")
    def __init__(self): self.events = []
    def begin_batch(self, batch): self.events.append(("batch", batch))
    def predict(self, item): self.events.append(("predict", item)); return "a", {"a": .8, "b": .2}, "ok"
    def prepare_review(self, item, gold, predicted, mode):
        self.events.append(("reveal", item, gold, predicted, mode))
        return SimpleNamespace(label=gold, status="accepted", reason_sha256="a" * 64)
    def record_label(self, item, label, **kwargs): self.events.append(("record", item, label))


def test_feedback_is_revealed_once_only_after_the_prequential_row_and_before_recording():
    arm = _Arm()
    result = StreamDriver(StreamConfig(review_probability=1, batch_size=1, checkpoint_at=9)).run(
        "E", arm, [("one", "b")], selection_plan=[("one", True, 1.0)])
    assert [event[0] for event in arm.events] == ["batch", "predict", "reveal", "record"]
    assert result.served[0].reviewed_count_at_prediction == 0
    assert result.reviews[0].reason_sha256 == "a" * 64


def test_driver_appends_the_served_row_before_invoking_the_feedback_provider(monkeypatch):
    from . import unified_stream as stream_module
    seen = {"appended": False}

    class SpyRows(list):
        def append(self, value):
            seen["appended"] = True
            super().append(value)
    class SpyResult:
        def __init__(self, arm, seed, synthetic, **_kwargs):
            self.arm, self.seed, self.is_synthetic = arm, seed, synthetic
            self.served, self.reviews, self.checkpoints = SpyRows(), [], []
            self.counters, self.usage = {}, {}
    class CheckingArm(_Arm):
        def prepare_review(self, *args):
            assert seen["appended"]
            return super().prepare_review(*args)
    monkeypatch.setattr(stream_module, "StreamResult", SpyResult)
    StreamDriver(StreamConfig(review_probability=1, batch_size=1, checkpoint_at=9)).run(
        "E", CheckingArm(), [("one", "b")], selection_plan=[("one", True, 1.0)])


def test_a_valid_label_only_reveal_is_recorded_but_an_unavailable_reveal_is_not():
    class LabelOnly(_Arm):
        def prepare_review(self, item, gold, predicted, mode):
            return SimpleNamespace(label=gold, status="label_only", reason_sha256=None)
    arm = LabelOnly()
    StreamDriver(StreamConfig(review_probability=1, batch_size=1, checkpoint_at=9)).run(
        "L", arm, [("one", "b")], selection_plan=[("one", True, 1.0)])
    assert ("record", "one", "b") in arm.events

    class Unavailable(_Arm):
        def prepare_review(self, *_args): return False
        def comment_for(self, *_args): return "must not become a digest"
    unavailable = Unavailable()
    result = StreamDriver(StreamConfig(review_probability=1, batch_size=1, checkpoint_at=9)).run(
        "L", unavailable, [("one", "b")], selection_plan=[("one", True, 1.0)])
    assert not any(event[0] == "record" for event in unavailable.events)
    assert result.reviews[0].status == "unavailable" and result.reviews[0].reason_sha256 is None
