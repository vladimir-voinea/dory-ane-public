"""The placement gate keeps three situations apart.

An op on the CPU or the GPU is a misplaced model and is refused.  An attribution
that names no device at all is the toolchain being unable to answer, and it is a
distinct outcome -- UNMEASURED -- that stops the conversion unless the caller asks
for it, and that never renders as a pass in the meta or the report.  A mixed
report is judged on the ops that WERE attributed, with the unattributed count kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anemll.ane_converter import dory_converter as dcv
from anemll.models.dory_ane_model import DoryANEConfig


def report(*counts, name="infer"):
    """A compute-plan report as placement_report would return it."""
    ane, gpu, cpu, none = counts
    return {name: {"ane": ane, "gpu": gpu, "cpu": cpu, "none": none,
                   "ops": ane + gpu + cpu + none,
                   "fallbacks": [f"op{i}" for i in range(gpu + cpu)]}}


def meta_with(placement):
    config = DoryANEConfig(context_length=1024, batch_size=64)
    specs = dcv.component_specs(config)
    files = {spec.name: {fn: f"{spec.name}_{fn}.mlpackage"
                         for fn in ("infer", "prefill")} for spec in specs}
    return dcv.build_meta(config=config, specs=specs, files=files,
                          lut_bits={"embeddings": None, "blocks": 6, "lm_head": 6},
                          lm_head_chunk_sizes=[config.vocab_size // 2] * 2,
                          chat_template=None, placement=placement)


def converter(**overrides):
    instance = object.__new__(dcv.DoryConverter)
    instance.placement = {}
    instance.unmeasured = []
    instance.check_placement_gate = overrides.pop("check_placement_gate", True)
    instance.accept_unmeasurable_placement = overrides.pop("accept_unmeasurable", False)
    return instance


def spec(name="input"):
    return [s for s in dcv.component_specs(DoryANEConfig(context_length=1024,
                                                         batch_size=64))
            if s.name == name][0]


# -- (a) an op off the ANE is a hard failure, as it always was --------------

def test_cpu_attribution_fails_and_is_not_the_unmeasurable_type():
    with pytest.raises(dcv.PlacementError) as caught:
        dcv.assert_ane_only(report(0, 0, 4, 0), "input/infer")
    assert not isinstance(caught.value, dcv.PlacementUnmeasurable)
    assert "off the Neural Engine" in str(caught.value)


def test_gpu_attribution_fails_too():
    with pytest.raises(dcv.PlacementError):
        dcv.assert_ane_only(report(3, 1, 0, 0), "input/infer")


# -- (b) an empty attribution is incapacity, not a pass ---------------------

def test_an_empty_attribution_is_unmeasurable():
    status = dcv.classify_placement(report(0, 0, 0, 9))
    assert status["outcome"] == "unmeasurable"
    assert status["attributed_ops"] == 0
    assert status["unattributed_ops"] == 9
    with pytest.raises(dcv.PlacementUnmeasurable) as caught:
        dcv.assert_ane_only(report(0, 0, 0, 9), "input/infer")
    assert "not a pass" in str(caught.value)


def test_no_operations_at_all_is_unmeasurable():
    assert dcv.classify_placement({"infer": {"ane": 0, "gpu": 0, "cpu": 0,
                                             "none": 0, "ops": 0}})["outcome"] == "unmeasurable"


def test_unmeasurable_never_renders_as_a_pass():
    status = dcv.classify_placement(report(0, 0, 0, 9))
    summary = dcv.placement_summary({"input": {"infer": {**status, "report": report(0, 0, 0, 9)}}})
    assert summary["gate"] == "UNMEASURED"
    assert "not a pass" in summary["reason"]
    meta = meta_with(summary)
    assert meta["placement"]["gate"] == "UNMEASURED"
    assert dcv.validate_meta(meta) == []
    # no component may claim a measured outcome, and no gate value may read as a pass
    assert all(outcome == "unmeasurable"
               for outcomes in meta["placement"]["components"].values()
               for outcome in outcomes)
    assert meta["placement"]["gate"] not in ("MEASURED", "PASS")
    assert json.dumps(meta).count('"MEASURED"') == 0


# -- (c) a mixed report is judged on what was attributed --------------------

def test_a_mixed_report_passes_and_counts_the_unattributed():
    mixed = report(7, 0, 0, 2)
    status = dcv.classify_placement(mixed)
    assert status["outcome"] == "measured"
    assert status["attributed_ops"] == 7
    assert status["unattributed_ops"] == 2
    assert dcv.assert_ane_only(mixed, "input/infer") is mixed


def test_a_mixed_report_with_one_cpu_op_fails_and_says_how_many_were_unattributed():
    with pytest.raises(dcv.PlacementError) as caught:
        dcv.assert_ane_only(report(6, 0, 1, 2), "input/infer")
    message = str(caught.value)
    assert "CPU" in message and "unattributed" in message


# -- the converter's own decision ------------------------------------------

def test_the_default_converter_refuses_unmeasurable_placement(monkeypatch):
    monkeypatch.setattr(dcv, "placement_report", lambda path: report(0, 0, 0, 9))
    with pytest.raises(dcv.PlacementUnmeasurable):
        converter()._gate_placement(spec(), Path("/tmp/pkg.mlpackage"), "infer")


def test_the_flag_accepts_it_and_records_unmeasured(monkeypatch):
    monkeypatch.setattr(dcv, "placement_report", lambda path: report(0, 0, 0, 9))
    instance = converter(accept_unmeasurable=True)
    instance._gate_placement(spec(), Path("/tmp/pkg.mlpackage"), "infer")
    assert instance.placement["input"]["infer"]["outcome"] == "unmeasurable"
    assert instance.unmeasured == ["input/infer"]
    summary = dcv.placement_summary(instance.placement)
    assert summary["gate"] == "UNMEASURED"
    assert "not a pass" in summary["reason"]


def test_a_skipped_gate_records_skipped_not_measured(monkeypatch):
    monkeypatch.setattr(dcv, "placement_report",
                        lambda path: pytest.fail("the gate must not run when skipped"))
    instance = converter(check_placement_gate=False)
    instance._gate_placement(spec(), Path("/tmp/pkg.mlpackage"), "infer")
    assert instance.placement["input"]["infer"]["outcome"] == "skipped"
    assert dcv.placement_summary(instance.placement)["gate"] == "SKIPPED"


def test_measured_placement_renders_as_measured():
    status = dcv.classify_placement(report(9, 0, 0, 0))
    meta = meta_with(dcv.placement_summary(
        {"input": {"infer": {**status, "report": report(9, 0, 0, 0)}}}))
    assert meta["placement"]["gate"] == "MEASURED"
    assert dcv.validate_meta(meta) == []


def test_a_meta_with_an_unknown_placement_gate_is_refused():
    meta = meta_with({"gate": "PASS", "reason": "fine", "unattributed_ops": 0,
                      "components": {}})
    assert any("placement.gate" in problem for problem in dcv.validate_meta(meta))


def test_a_meta_recording_no_placement_check_says_skipped():
    meta = meta_with(None)
    assert meta["placement"]["gate"] == "SKIPPED"
    assert dcv.validate_meta(meta) == []
