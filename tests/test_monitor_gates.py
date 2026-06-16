from app.monitor.gates import check_hard_gates
from app.schemas import DiscussionSummary, FinalScenario, RunMetrics, YearBlock


def _good_final_dict():
    return FinalScenario(
        run_id="t1",
        seed="Taiwan election scenario",
        scenario_mode="base_case",
        scenario_title="Test Scenario",
        scenario_summary="A plausible five-year path.",
        timeline=[
            YearBlock(year=y, headline="Event in %d" % y, events=[])
            for y in (2026, 2027, 2028, 2029, 2030, 2031)
        ],
        main_disagreements=["Pace of decoupling"],
        discussion_summary=[
            DiscussionSummary(
                round_number=1,
                areas_of_disagreement=["Trade speed"],
            )
        ],
        run_metrics=RunMetrics(
            synthesis_validation_passed=True,
            synthesis_used_fallback=False,
            discussion_rounds_completed=2,
        ),
    ).model_dump()


def test_gates_pass_on_healthy_run():
    report = check_hard_gates(_good_final_dict())
    assert report.passed is True
    assert report.blockers == []
    assert len(report.checks) == 6
    assert all(c.status == "pass" for c in report.checks)


def test_gates_block_on_fallback():
    data = _good_final_dict()
    data["run_metrics"]["synthesis_used_fallback"] = True
    report = check_hard_gates(data)
    assert report.passed is False
    assert any("G2" in b for b in report.blockers)


def test_gates_warn_on_no_disagreements():
    data = _good_final_dict()
    data["main_disagreements"] = []
    data["discussion_summary"] = [{"round_number": 1, "areas_of_disagreement": []}]
    report = check_hard_gates(data)
    assert any("G5" in w for w in report.warnings)


def test_gates_block_missing_timeline():
    data = _good_final_dict()
    data["timeline"] = []
    report = check_hard_gates(data)
    assert any("G1" in b for b in report.blockers)
    g1 = next(c for c in report.checks if c.id == "G1")
    assert g1.status == "fail"


def test_gate_checks_cover_all_ids():
    report = check_hard_gates(_good_final_dict())
    assert [c.id for c in report.checks] == [
        "G1", "G2", "G3", "G4", "G5", "G6"
    ]
