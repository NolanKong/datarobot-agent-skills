"""
Validate CODEOWNERS rules for the skills/ directory:
- No skill directory should be owned by @datarobot-oss/datarobot-agent-skills
  (that team owns repo scaffolding only, not individual skills)
"""

import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
CODEOWNERS_FILE = REPO_ROOT / ".github" / "CODEOWNERS"
SKILLS_DIR = REPO_ROOT / "skills"
DEFAULT_OWNER = "@datarobot-oss/datarobot-agent-skills"


def parse_codeowners() -> list[tuple[str, list[str]]]:
    """Parse CODEOWNERS into a list of (pattern, owners) tuples, in file order."""
    rules: list[tuple[str, list[str]]] = []
    with open(CODEOWNERS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pattern, owners = parts[0], parts[1:]
            if owners:
                rules.append((pattern, owners))
    return rules


def effective_owners(path: str, rules: list[tuple[str, list[str]]]) -> list[str]:
    """Return the owners for *path* by applying CODEOWNERS last-match-wins semantics."""
    matched_owners: list[str] = []
    for pattern, owners in rules:
        # Strip leading slash for matching — CODEOWNERS anchors patterns with /
        pat = pattern.lstrip("/")
        # A pattern ending in / applies to everything under that directory
        if pat.endswith("/"):
            pat = pat + "**"
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path, pat + "/**"):
            matched_owners = owners
    return matched_owners


def skill_dirs() -> list[Path]:
    return [
        item
        for item in SKILLS_DIR.iterdir()
        if item.is_dir() and (item / "SKILL.md").exists()
    ]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "skill_dir" in metafunc.fixturenames:
        dirs = skill_dirs()
        metafunc.parametrize("skill_dir", dirs, ids=[d.name for d in dirs])


def test_codeowners_file_exists() -> None:
    assert CODEOWNERS_FILE.exists(), ".github/CODEOWNERS not found"


def test_skill_not_owned_by_default_team(skill_dir: Path) -> None:
    """Each skill must have an explicit owner that is not the default repo team."""
    rules = parse_codeowners()
    # Use the skill dir path relative to repo root, with trailing slash
    rel_path = str(skill_dir.relative_to(REPO_ROOT)) + "/"
    owners = effective_owners(rel_path, rules)
    assert owners, (
        f"Skill '{skill_dir.name}' has no CODEOWNERS entry — add an explicit rule"
    )
    assert DEFAULT_OWNER not in owners, (
        f"Skill '{skill_dir.name}' is owned by {DEFAULT_OWNER}, "
        "which should only own repo scaffolding. "
        "Add a specific team or user to CODEOWNERS for this skill."
    )
