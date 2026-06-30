from app.monitor.year_judge import run_year_quality_judge
from app.schemas import AgentOutput, DiscussionSummary, YearBlock


def test_year_judge_mock_returns_verdict():
    block = YearBlock(year=2026, headline="Mock lock", events=[])
    outputs = {
        "geo_strategy": AgentOutput(
            agent_name="geo_strategy",
            round_number=1,
            target_year=2026,
            main_assessment="Alliances shift.",
        ),
        "economy_technology": AgentOutput(
            agent_name="economy_technology",
            round_number=1,
            target_year=2026,
            main_assessment="Chips matter.",
        ),
        "domestic_ideology": AgentOutput(
            agent_name="domestic_ideology",
            round_number=1,
            target_year=2026,
            main_assessment="Nationalism rises.",
        ),
        "security_taiwan": AgentOutput(
            agent_name="security_taiwan",
            round_number=1,
            target_year=2026,
            main_assessment="Gray-zone pressure.",
        ),
        "historical_analogy": AgentOutput(
            agent_name="historical_analogy",
            round_number=1,
            target_year=2026,
            main_assessment="Cold War partial fit.",
        ),
    }
    rounds = [
        DiscussionSummary(
            round_number=1,
            target_year=2026,
            areas_of_disagreement=["Trade speed"],
        )
    ]
    verdict = run_year_quality_judge(
        2026,
        "Taiwan election shock",
        "base_case",
        block,
        rounds,
        outputs,
        [],
    )
    assert verdict.overall_score >= 3.5
    assert verdict.pass_quality_bar is True
    assert len(verdict.dimensions) == 5
