from app.monitor.year_gates import check_year_gates
from app.schemas import (
    AgentOutput,
    DiscussionSummary,
    YearBlock,
)


def _outputs(year: int = 2026):
    return {
        name: AgentOutput(
            agent_name=name,
            round_number=1,
            target_year=year,
            main_assessment="Assessment for %d." % year,
        )
        for name in (
            "geo_strategy",
            "economy_technology",
            "domestic_ideology",
            "security_taiwan",
            "historical_analogy",
        )
    }


def test_year_gates_pass_on_healthy_year():
    block = YearBlock(year=2026, headline="Tech controls tighten", events=[])
    rounds = [
        DiscussionSummary(
            round_number=1,
            target_year=2026,
            areas_of_disagreement=["Pace of decoupling"],
            key_uncertainties=["Domestic politics"],
        )
    ]
    report = check_year_gates(2026, block, rounds, _outputs(2026), [])
    assert report.passed is True
    assert report.blockers == []


def test_year_gates_block_on_wrong_year_field():
    block = YearBlock(year=2027, headline="Wrong year", events=[])
    report = check_year_gates(
        2026,
        block,
        [DiscussionSummary(round_number=1, target_year=2026)],
        _outputs(2026),
        [],
    )
    assert report.passed is False
    assert any("Y1" in b for b in report.blockers)


def test_year_gates_pass_with_five_checks():
    block = YearBlock(year=2026, headline="Quiet year", events=[])
    rounds = [
        DiscussionSummary(
            round_number=1,
            target_year=2026,
            areas_of_disagreement=[],
            key_uncertainties=[],
        )
    ]
    report = check_year_gates(2026, block, rounds, _outputs(2026), [])
    assert report.passed is True
    assert len(report.checks) == 5
