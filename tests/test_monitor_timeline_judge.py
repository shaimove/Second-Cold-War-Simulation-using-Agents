from app.llm import LLMClient
from app.monitor.timeline_judge import evaluate_locked_timeline
from app.schemas import TIMELINE_JUDGE_DIMENSION_NAMES, YearBlock, YearSimulationRecord


def test_timeline_judge_mock():
    timeline = [
        YearBlock(year=y, headline="Headline %d" % y, events=[])
        for y in (2026, 2027, 2028, 2029, 2030, 2031)
    ]
    records = [YearSimulationRecord(year=y) for y in (2026, 2027, 2028, 2029, 2030, 2031)]
    result = evaluate_locked_timeline(
        "Taiwan election shock",
        "base_case",
        timeline,
        records,
        llm=LLMClient(),
    )
    assert result.gates.passed is True
    assert result.judge is not None
    assert result.judge.pass_quality_bar is True
    assert {d.name for d in result.judge.dimensions} == set(TIMELINE_JUDGE_DIMENSION_NAMES)
