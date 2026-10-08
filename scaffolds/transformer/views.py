"""Saved W&B workspace views for an experiment's runs (requires the `views` dependency group).

An experiment's `adapter/analysis/views.py` builds its view from these defaults plus
any experiment-specific sections, then saves it:

    from scaffolds.transformer.views import experiment_workspace, line_section
    workspace = experiment_workspace("my_experiment", extra_sections=[line_section(...)])
    print(workspace.save().url)
"""

from __future__ import annotations

from collections.abc import Sequence

ENTITY = "rafah-massachusetts-institute-of-technology"
PROJECT = "functional-memory"


def line_section(name: str, metrics: Sequence[str], *, log_y: bool = False, is_open: bool = True):
    """One section with a line plot per metric against the trainer's `step` axis."""
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    return ws.Section(name=name, is_open=is_open, panels=[
        wr.LinePlot(title=metric, x="step", y=[metric], log_y=log_y or None) for metric in metrics
    ])


def regex_section(name: str, regex: str, *, is_open: bool = False):
    """One line plot of every metric matching `regex`, for example per-layer attention statistics."""
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    return ws.Section(name=name, is_open=is_open, panels=[wr.LinePlot(title=regex, x="step", metric_regex=regex)])


def experiment_workspace(experiment: str, *, name: str | None = None, extra_sections: Sequence = (),
                         entity: str = ENTITY, project: str = PROJECT):
    """A view restricted to one experiment's W&B group, with the shared training panels first."""
    import wandb_workspaces.workspaces as ws

    sections = [
        line_section("Validation", ["val/ppl", "val/nll"]),
        line_section("Training", ["train/nll", "train/ppl", "train/learning_rate", "train/tokens_per_second"]),
        line_section("Gradient norms", [
            "train/gradient_norm_pre_clip", "train/gradient_norm_post_clip",
            "train/gradient_norm_pre_clip_window_mean", "train/gradient_norm_pre_clip_window_max",
        ], log_y=True),
        line_section("Gradient clipping", ["train/gradient_clip_scale", "train/gradient_clipped_fraction"]),
        *extra_sections,
        regex_section("Attention diagnostics", r"attention/.*"),
    ]
    return ws.Workspace(
        entity=entity, project=project, name=name or experiment, sections=sections,
        runset_settings=ws.RunsetSettings(filters=[ws.Metric("Group") == experiment]),
    )
