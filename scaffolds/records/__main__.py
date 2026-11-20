"""Allocate runs, inspect canonical records, and re-pin result selections without a model runtime."""
import argparse
import json
from pathlib import Path

import yaml

from . import (
    check_receipts, create_run, load_study, recipe_receipt, regenerate_index, rename_study, run_directory,
    update_record, validate_record,
)
from .selections import refresh_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="allocate a fresh run from a complete recipe")
    create.add_argument("--study", required=True, type=Path)
    create.add_argument("--config", required=True, type=Path)
    create.add_argument("--label")
    create.add_argument(
        "--allow-plain-qk", action="store_true",
        help="explicitly opt a fresh dot-product experiment out of canonical QK normalization",
    )
    create.add_argument(
        "--allow-legacy-qk-normalization", action="store_true",
        help="explicitly preserve a fresh comparison's older QK normalization instead of RMSNorm with per-head scaling",
    )
    create.add_argument(
        "--allow-flax-default-init", action="store_true",
        help="explicitly opt a fresh experiment out of its mechanism's reference initialization",
    )
    create.add_argument(
        "--allow-triton-attention", action="store_true",
        help="explicitly keep the fused Triton kernels for a fresh recipe that a Mosaic GPU kernel implements",
    )
    create.add_argument(
        "--allow-scan-blocked-attention", action="store_true",
        help="explicitly keep the historical blocked scan for a fresh dot-product recipe on the blocked "
             "(or auto) backend instead of causal strips",
    )
    create.add_argument(
        "--allow-unshifted-exp", action="store_true",
        help="explicitly keep the unshifted exponential (model.attention_exp_shift: none) for a fresh unnormalized "
             "exponential recipe with head RMSNorm, instead of its row-maximum shift",
    )
    create.add_argument(
        "--allow-plain-head-rms", action="store_true",
        help="explicitly keep the plain head RMSNorm arithmetic (model.attention_head_rms_prescale: none) for a "
             "fresh recipe with head RMSNorm output scaling, instead of its max-abs prescale",
    )
    receipts = commands.add_parser(
        "receipts", help="recompute runs' recipe receipts with this checkout and compare them with their records "
                         "(or, for runs that predate receipts, the scaffold's baseline)")
    receipts.add_argument("study", type=Path, nargs="+")
    receipts.add_argument("--jobs", type=int, default=12)
    index = commands.add_parser("index")
    index.add_argument("study", type=Path)
    validate = commands.add_parser(
        "validate", help="check run identities, recipes, and artifact hashes; checkpoints by names and sizes")
    validate.add_argument("study", type=Path)
    validate.add_argument("runs", nargs="*", metavar="run", help="IDs or labels to check (default: every run)")
    validate.add_argument(
        "--checkpoint-hashes", action="store_true",
        help="also re-hash retained checkpoints, which reads every byte of them; use after copying checkpoints",
    )
    refresh = commands.add_parser("refresh", help="re-pin a result's selections to its runs' scientific files")
    refresh.add_argument("result", type=Path, nargs="+")
    rename = commands.add_parser(
        "rename-study",
        help="after `git mv experiments/<old> experiments/<new>`, point run records, result selections, and "
             "sweep.yaml at the new folder; frozen recipes and execution history keep their paths")
    rename.add_argument("study", type=Path, help="the renamed study folder, experiments/<new>")
    rename.add_argument("--from", dest="old", required=True, metavar="OLD", help="the previous folder name")
    rename.add_argument("--to", dest="new", required=True, metavar="NEW", help="the new folder name")
    rename.add_argument(
        "--artifact-root", action="append", default=[], metavar="ROOT",
        help="another root whose experiments/<old>/ tree was moved as well, such as a cluster's artifact root")
    args = parser.parse_args()
    if args.command == "create":
        recipe = yaml.safe_load(args.config.read_text())
        if not isinstance(recipe, dict):
            parser.error("--config must contain a YAML mapping")
        if isinstance(recipe.get("model"), dict):
            from scaffolds.transformer.configs import Config, config_to_dict, load_config, validate_config
            from scaffolds.transformer.configs.base import (
                fresh_recipe_uses_blocked_strips, fresh_recipe_uses_exp_shift, fresh_recipe_uses_head_rms_prescale,
            )
            from scaffolds.transformer.models.initialization import REFERENCE
            supplied_model = recipe["model"]
            unset = "initialization" not in supplied_model
            canonical_qk = {"qk_norm_type": "rms_norm", "qk_norm_sharing": "per_layer",
                            "qk_norm_initial_scale": None, "qk_norm_head_scale": True}
            use_canonical_qk = (
                supplied_model.get("attention_mechanism", "dot_product") == "dot_product"
                and supplied_model.get("attention_degree") != 0
                and supplied_model.get("qk_norm") is not False
                and not args.allow_legacy_qk_normalization)
            # Dot-product attention with RoPE or no positional encoding that can train on the
            # blocked backend (blocked or auto) uses its causal strips. PaTH, GLA, and Gated
            # DeltaNet have their own blocked code; other backends keep the inert scan default.
            use_strips = fresh_recipe_uses_blocked_strips(supplied_model)
            if use_canonical_qk:
                incompatible = [key for key, value in canonical_qk.items()
                                if key in supplied_model and supplied_model[key] != value]
                if incompatible:
                    parser.error(
                        "new dot-product runs use QK RMSNorm with per-head scaling; this recipe has "
                        f"noncanonical values for {', '.join(incompatible)}. Use the canonical settings "
                        "or --allow-legacy-qk-normalization only for an explicitly requested comparison."
                    )

            def fresh_config():
                # Apply allocation defaults before validating partial input recipes.
                # Ordinary load_config and historical checkpoint settings are unchanged.
                config = Config()
                if use_canonical_qk:
                    for key, value in canonical_qk.items():
                        setattr(config.model, key, value)
                if use_strips:
                    config.model.attention_blocked_implementation = "strips"
                return config

            config = load_config(args.config, config_factory=fresh_config)
            # The unnormalized exponential with head RMSNorm evaluates its row-maximum shift wherever the shift
            # applies (configs.base.fresh_recipe_uses_exp_shift), unless the recipe names its setting.
            use_exp_shift = fresh_recipe_uses_exp_shift(config.model)
            if use_exp_shift and "attention_exp_shift" not in supplied_model:
                config.model.attention_exp_shift = "row_max"
                validate_config(config)
            # Head RMSNorm output scaling is evaluated with its max-abs prescale wherever it runs
            # (configs.base.fresh_recipe_uses_head_rms_prescale), unless the recipe names its setting.
            use_head_rms_prescale = fresh_recipe_uses_head_rms_prescale(config.model)
            if use_head_rms_prescale and "attention_head_rms_prescale" not in supplied_model:
                config.model.attention_head_rms_prescale = "max_abs"
                validate_config(config)
            recipe = config_to_dict(config)
            model = recipe["model"]
            if unset:
                # Fresh runs use their mechanism's reference initialization. The config
                # default stays Flax so that recipes predating the field keep their meaning.
                model["initialization"] = REFERENCE[model["attention_mechanism"]]
            if (model["attention_mechanism"] == "dot_product"
                    and model["attention_degree"] != 0
                    and not model["qk_norm"] and not args.allow_plain_qk):
                parser.error(
                    "new dot-product runs require model.qk_norm: true; this recipe disables it. "
                    "Use --allow-plain-qk only when plain QK scores were explicitly requested."
                )
            if model["initialization"] == "flax_default" and not args.allow_flax_default_init:
                parser.error(
                    "new runs use their mechanism's reference initialization; this recipe sets "
                    "model.initialization: flax_default. Use --allow-flax-default-init only when "
                    "Flax-default initialization was explicitly requested."
                )
            if model["attention_score_bias"] != "none" and model["attention_backend"] not in ("mosaic", "blocked"):
                # Configuration validation rejects the other backends first; this states the rule.
                parser.error(
                    "per-degree score biases train on H100's Mosaic score-bias kernels or the blocked backend's "
                    f"causal strips; this recipe sets model.attention_backend: {model['attention_backend']}."
                )
            if (model["attention_score_bias"] != "none" and model["attention_backend"] == "blocked"
                    and model["attention_blocked_implementation"] != "strips"):
                # Also with --allow-scan-blocked-attention: acceptance refuses the scan for score biases.
                parser.error(
                    "per-degree score biases on the blocked backend train on its causal strips, the only blocked "
                    "computation with a GPU qualification for them; this recipe sets "
                    f"model.attention_blocked_implementation: {model['attention_blocked_implementation']}."
                )
            if (model["attention_score_transform"] != "identity" and model["attention_backend"] == "blocked"
                    and model["attention_blocked_implementation"] != "strips"):
                # Validation refuses every backend but blocked and mosaic. Also with --allow-scan-blocked-attention:
                # acceptance refuses the scan for a transform.
                parser.error(
                    "a score transform on the blocked backend trains on its causal strips, the only blocked "
                    "computation with a GPU qualification for it; this recipe sets "
                    f"model.attention_blocked_implementation: {model['attention_blocked_implementation']}."
                )
            if (model["attention_backend"] == "fused" and config.model.mosaic_supported()
                    and not args.allow_triton_attention):
                parser.error(
                    "new runs of this mechanism use the Mosaic GPU kernels; this recipe sets "
                    "model.attention_backend: fused. Use mosaic (or auto), or --allow-triton-attention "
                    "only when the Triton kernels were explicitly requested."
                )
            if (use_strips and model["attention_blocked_implementation"] == "scan"
                    and not args.allow_scan_blocked_attention):
                parser.error(
                    "new dot-product runs on the blocked backend train on its causal strips; this recipe sets "
                    "model.attention_blocked_implementation: scan. Use strips (or leave it unset), or "
                    "--allow-scan-blocked-attention only when the scan was explicitly requested."
                )
            if use_exp_shift and model["attention_exp_shift"] == "none" and not args.allow_unshifted_exp:
                parser.error(
                    "new unnormalized exponential runs with head RMSNorm shift the exponential by its row maximum; "
                    "this recipe sets model.attention_exp_shift: none. Use row_max (or leave it unset), or "
                    "--allow-unshifted-exp only when the unshifted exponential was explicitly requested."
                )
            if (use_head_rms_prescale and model["attention_head_rms_prescale"] == "none"
                    and not args.allow_plain_head_rms):
                parser.error(
                    "new runs with head RMSNorm output scaling evaluate it with the max-abs prescale; this recipe "
                    "sets model.attention_head_rms_prescale: none. Use max_abs (or leave it unset), or "
                    "--allow-plain-head-rms only when the plain arithmetic was explicitly requested."
                )
            recipe["train"]["source_commit"] = "auto"
            study = args.study.parent if args.study.name == "sweep.yaml" else args.study
            recipe["train"]["wandb_group"] = study.name
            recipe["train"]["wandb_run_name"] = args.label
            recipe["train"]["wandb_dir"] = None
        run = create_run(args.study, recipe, args.label)
        if isinstance(recipe.get("model"), dict):
            # What this checkout's code computes for the recipe; merges and refactors compare against it.
            update_record(run, {"receipt": recipe_receipt(run / "config.yaml")})
        print(run)
    elif args.command == "refresh":
        for result in args.result:
            changes = refresh_selection(result)
            print("\n".join(f"{result}/{change}" for change in changes) or f"{result}: selections already current")
    elif args.command == "receipts":
        runs = [study / "runs" / run["run_id"] for study in args.study for run in load_study(study)["runs"]]
        results = check_receipts(runs, jobs=args.jobs)
        for result in results:
            if result["status"] not in ("same", "same rejection"):
                print(json.dumps(result))
        counts = {status: sum(r["status"] == status for r in results) for status in sorted({r["status"] for r in results})}
        print(json.dumps({"runs": len(results), **counts}))
        if any(r["status"] == "differs" for r in results):
            raise SystemExit(1)
    elif args.command == "index":
        print(regenerate_index(args.study))
    elif args.command == "rename-study":
        changes = rename_study(args.study, args.old, args.new, tuple(args.artifact_root))
        print("\n".join(changes) or f"{args.study}: nothing referenced experiments/{args.old}")
    else:
        study = args.study.parent if args.study.name == "sweep.yaml" else args.study
        runs = ([run_directory(study, run) for run in args.runs] if args.runs
                else [study / "runs" / run["run_id"] for run in load_study(study)["runs"]])
        results = [validate_record(run, checkpoint_hashes=args.checkpoint_hashes) for run in runs]
        print(json.dumps({"study": str(study), "runs": results}, indent=2))


if __name__ == "__main__":
    main()
