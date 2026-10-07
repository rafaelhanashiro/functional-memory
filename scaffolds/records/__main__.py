"""Allocate runs, inspect canonical records, and re-pin result selections without a model runtime."""
import argparse
import json
from pathlib import Path

import yaml

from . import create_run, load_study, regenerate_index, run_directory, validate_record
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
        "--allow-flax-default-init", action="store_true",
        help="explicitly opt a fresh experiment out of its mechanism's reference initialization",
    )
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
    args = parser.parse_args()
    if args.command == "create":
        recipe = yaml.safe_load(args.config.read_text())
        if not isinstance(recipe, dict):
            parser.error("--config must contain a YAML mapping")
        if isinstance(recipe.get("model"), dict):
            from scaffolds.transformer.configs import config_to_dict, load_config
            from scaffolds.transformer.models.initialization import REFERENCE
            unset = "initialization" not in recipe["model"]
            recipe = config_to_dict(load_config(args.config))
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
            recipe["train"]["source_commit"] = "auto"
            study = args.study.parent if args.study.name == "sweep.yaml" else args.study
            recipe["train"]["wandb_group"] = study.name
            recipe["train"]["wandb_run_name"] = args.label
            recipe["train"]["wandb_dir"] = None
        print(create_run(args.study, recipe, args.label))
    elif args.command == "refresh":
        for result in args.result:
            changes = refresh_selection(result)
            print("\n".join(f"{result}/{change}" for change in changes) or f"{result}: selections already current")
    elif args.command == "index":
        print(regenerate_index(args.study))
    else:
        study = args.study.parent if args.study.name == "sweep.yaml" else args.study
        runs = ([run_directory(study, run) for run in args.runs] if args.runs
                else [study / "runs" / run["run_id"] for run in load_study(study)["runs"]])
        results = [validate_record(run, checkpoint_hashes=args.checkpoint_hashes) for run in runs]
        print(json.dumps({"study": str(study), "runs": results}, indent=2))


if __name__ == "__main__":
    main()
