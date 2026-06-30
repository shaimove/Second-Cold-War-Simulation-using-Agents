from app import agents as agent_mod
from app import prompts as prompt_mod
from app.llm import LLMClient
from app.schemas import EvidenceSummary, DOMAIN_AGENTS


def test_evidence_agent_labels_hypothetical_in_mock_mode():
    llm = LLMClient()
    summary, n, _ = agent_mod.run_evidence_agent(
        llm,
        seed="If China enters a major financial crisis next year",
        scenario_mode="base_case",
    )
    assert isinstance(summary, EvidenceSummary)
    assert summary.hypothetical_assumptions
    assert n == 0  # empty KB in tests


def test_each_domain_agent_returns_required_fields():
    llm = LLMClient()
    for name in DOMAIN_AGENTS:
        out = agent_mod.run_domain_agent(
            llm,
            agent_name=name,
            seed="seed",
            scenario_mode="base_case",
            target_year=2026,
            evidence_blob="evidence",
            round_number=1,
            previous_summary=None,
            previous_self_position=None,
        )
        assert out.agent_name == name
        assert out.main_assessment
        assert out.timeline_contributions
        assert out.timeline_contributions[0].year == 2026


def test_security_agent_prompt_contains_safety_constraint():
    text = prompt_mod.AGENT_SYSTEM_PROMPTS["security_taiwan"]
    assert "operational tactics" in text.lower()


def test_classify_event_status_marks_future_as_hypothetical():
    assert agent_mod.classify_event_status("If Taiwan elects ...") == "hypothetical"
    assert (
        agent_mod.classify_event_status("Suppose the U.S. announces export controls")
        == "hypothetical"
    )


def test_red_team_agent_returns_findings():
    llm = LLMClient()
    out, findings = agent_mod.run_red_team_agent(
        llm,
        seed="x",
        scenario_mode="base_case",
        evidence_blob="ev",
        final_round_summary=None,
    )
    assert out.agent_name == "red_team"
    assert isinstance(findings, list)
    assert len(findings) >= 1


def test_build_final_timeline_uses_locked_years():
    from app.schemas import TimelineEvent, YearBlock

    locked = [
        YearBlock(year=2026, headline="Event A", events=[TimelineEvent(event="A")]),
        YearBlock(year=2027, headline="Event B", events=[TimelineEvent(event="B")]),
    ]
    timeline = agent_mod.build_final_timeline(locked)
    assert [yb.year for yb in timeline] == [2026, 2027]
    assert timeline[0].headline == "Event A"
