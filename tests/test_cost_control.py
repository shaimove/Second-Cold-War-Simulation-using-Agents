"""Tests for cost-control compaction helpers."""
from app.cost_control import compact_agent_position
from app.schemas import AgentOutput, AgentTimelineContribution


def test_compact_agent_position_includes_key_fields():
    output = AgentOutput(
        agent_name="geo_strategy",
        round_number=2,
        main_assessment="Alliance cohesion weakens under fiscal stress.",
        key_drivers=["trade tension", "alliance burden-sharing", "Taiwan signaling"],
        timeline_contributions=[
            AgentTimelineContribution(year=2026, event="NATO consults on Indo-Pacific"),
            AgentTimelineContribution(year=2028, event="US-Japan defense budget rises"),
        ],
        uncertainties=["pace of decoupling", "EU alignment"],
        disagreements=["severity of Taiwan risk"],
        risks=["alliance fracture"],
        position_changed_from_previous_round=True,
    )

    text = compact_agent_position(output, max_chars=1000)

    assert "Assessment:" in text
    assert "Drivers:" in text
    assert "Timeline:" in text
    assert "2026:" in text
    assert "Uncertainties:" in text
    assert "Disagrees on:" in text
    assert "Risks:" in text
    assert "Position changed: yes" in text
    assert len(text) <= 1000


def test_compact_agent_position_respects_max_chars():
    output = AgentOutput(
        agent_name="economy_technology",
        main_assessment="x" * 2000,
        key_drivers=["a" * 200, "b" * 200],
    )

    text = compact_agent_position(output, max_chars=500)

    assert len(text) <= 500
    assert text.endswith("...")
