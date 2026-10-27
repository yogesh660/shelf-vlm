"""Checks on the repo itself, so a coding agent can work in it without breaking conventions."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "shelf"


def makefile_targets() -> set[str]:
    return set(re.findall(r"^([a-z-]+):", (ROOT / "Makefile").read_text(), flags=re.MULTILINE))


def test_make_targets_in_docs_exist():
    for doc in ("CLAUDE.md", "README.md"):
        mentioned = set(re.findall(r"`make ([a-z-]+)", (ROOT / doc).read_text()))
        missing = mentioned - makefile_targets()
        assert not missing, f"{doc} mentions targets the Makefile does not have: {missing}"


def test_agents_md_points_to_claude_md():
    assert "CLAUDE.md" in (ROOT / "AGENTS.md").read_text()


MODULES = sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.stem)
def test_every_module_has_a_test_file(module):
    test_file = ROOT / "tests" / "unit" / f"test_{module.stem}.py"
    assert test_file.exists(), f"add {test_file.relative_to(ROOT)}"


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.stem)
def test_core_does_not_import_ml_frameworks(module):
    # the core must stay testable on a laptop without a GPU (ADR 0001); framework code goes
    # in train/ and app/, which are optional extras
    text = module.read_text()
    for heavy in ("torch", "transformers", "vllm", "gradio", "peft", "trl"):
        assert not re.search(rf"^\s*(import|from) {heavy}\b", text, re.MULTILINE), (
            f"{module.relative_to(ROOT)} imports {heavy}"
        )
