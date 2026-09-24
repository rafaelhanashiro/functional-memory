"""Canonical relocation must preserve recipes and one experiment document."""

from scaffolds.state_rank.artifacts import (
    CONFIG_DIRECTORY, resolved_config_path, retained_path, write_report_section,
)


def test_historical_recipe_resolves_relocated_baseline_without_rewriting(tmp_path):
    recipe = CONFIG_DIRECTORY / "random_vs_pretrained.json"
    original = recipe.read_bytes()
    old_path = "results/llama-3.2-1b-l8-kvall-t512"
    assert retained_path(tmp_path, old_path) == (
        tmp_path / "results/state_rank/runs/llama-3.2-1b-l8-kvall-t512"
    )
    assert recipe.read_bytes() == original
    assert retained_path(tmp_path, "results/custom") == tmp_path / "results/custom"


def test_generators_share_one_readme_and_preserve_other_sections(tmp_path):
    family = tmp_path / "results/state_rank"
    comparison = family / "random_vs_pretrained"
    unified = family / "unified_comparison"
    comparison.mkdir(parents=True)
    unified.mkdir()
    (family / "README.md").write_text("# State rank\n\nRetained methods and findings.\n")
    write_report_section(comparison, "comparison", "# Comparison\n\n[Figure](plot.png)\n")
    write_report_section(unified, "unified", "# Unified\n\n[Figure](llama/heads.png)\n")
    write_report_section(comparison, "comparison", "# Comparison\n\nUpdated statistics.\n")
    text = (family / "README.md").read_text()
    assert "Retained methods and findings." in text
    assert "Updated statistics." in text
    assert "unified_comparison/llama/heads.png" in text
    assert text.count("<!-- state-rank:comparison:start -->") == 1
    assert list(family.rglob("README.md")) == [family / "README.md"]
    assert not list(family.rglob("REPORT.md"))


def test_resolved_config_lives_with_family_recipes(tmp_path):
    output = tmp_path / "results/state_rank/unified_comparison"
    expected = tmp_path / "results/state_rank/configs/unified_comparison.resolved.json"
    assert resolved_config_path(output, writing=True) == expected
    assert expected.parent.is_dir()


def test_independent_export_keeps_its_own_document(tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    path = write_report_section(output, "analysis", "# Analysis\n\n[Figure](plot.png)\n")
    assert path == output / "README.md"
    assert "[Figure](plot.png)" in path.read_text()
