from app.monitor.timeline_gates import check_timeline_gates
from app.schemas import YearBlock, YearSimulationRecord, YearJudgeVerdict


def test_timeline_gates_pass_on_full_timeline():
    timeline = [
        YearBlock(year=y, headline="Event in %d" % y, events=[])
        for y in (2026, 2027, 2028, 2029, 2030, 2031)
    ]
    records = [
        YearSimulationRecord(
            year=y,
            year_judge=YearJudgeVerdict(pass_quality_bar=True, overall_score=4.0),
        )
        for y in (2026, 2027, 2028, 2029, 2030, 2031)
    ]
    report = check_timeline_gates(timeline, records)
    assert report.passed is True
    assert len(report.checks) == 5


def test_timeline_gates_fail_on_wrong_count():
    timeline = [YearBlock(year=2026, headline="Only one year", events=[])]
    report = check_timeline_gates(timeline, [])
    assert report.passed is False
    assert any("T1" in b for b in report.blockers)
