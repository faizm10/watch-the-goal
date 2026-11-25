"""The agent skills in .agents/skills (Codex) and .claude/skills (Claude Code)
must stay identical, carry a name and trigger description, and only point at
scripts that exist."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ["soccer-vision", "ffmpeg-essentials", "video-qa", "webapp-testing"]


@pytest.mark.parametrize("name", SKILLS)
def test_skill_copies_match_and_are_well_formed(name):
    codex = (ROOT / ".agents" / "skills" / name / "SKILL.md").read_text()
    claude = (ROOT / ".claude" / "skills" / name / "SKILL.md").read_text()
    assert codex == claude, f"{name}: .agents and .claude copies differ; copy one over the other"
    front = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", codex)
    assert front, f"{name}: SKILL.md needs name and description frontmatter"
    assert front.group(1) == name
    assert "Use when" in front.group(2) or "Use after" in front.group(2), "description should say when to use it"
    for script in set(re.findall(r"scripts/[\w./-]+\.py", codex)):
        assert (ROOT / script).exists(), f"{name} refers to missing {script}"


def test_no_stray_skills():
    for base in (".agents", ".claude"):
        found = sorted(p.name for p in (ROOT / base / "skills").iterdir() if p.is_dir())
        assert found == sorted(SKILLS)
