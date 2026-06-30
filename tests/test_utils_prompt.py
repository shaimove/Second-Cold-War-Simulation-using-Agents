from app.utils import assemble_user_prompt, budget_prompt_sections


def test_assemble_user_prompt_preserves_fixed_suffix():
    fixed = "FIXED_SCHEMA_TAIL"
    context = "X" * 5000
    out = assemble_user_prompt(context, fixed, max_chars=8000)
    assert out.endswith(fixed)
    assert len(out) <= 8000


def test_budget_prompt_sections_trims_high_priority_first():
    out = budget_prompt_sections(
        [
            ("KEEP", 1),
            ("TRIMME", 5),
        ],
        max_chars=20,
    )
    assert "KEEP" in out
    assert len(out) <= 20
