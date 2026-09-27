"""Allocate runs and inspect canonical records without importing a model runtime."""
import argparse
import json
from pathlib import Path

import yaml

from . import create_run, load_study, regenerate_index, validate_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="allocate a fresh run from a complete recipe")
    create.add_argument("--study", required=True, type=Path)
    create.add_argument("--config", required=True, type=Path)
    create.add_argument("--label")
    for name in ("index", "validate"):
        command = commands.add_parser(name)
        command.add_argument("study", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        recipe = yaml.safe_load(args.config.read_text())
        if not isinstance(recipe, dict):
            parser.error("--config must contain a YAML mapping")
        if isinstance(recipe.get("model"), dict) and isinstance(recipe.get("train"), dict):
            from scaffolds.transformer.configs import config_to_dict, load_config
            recipe = config_to_dict(load_config(args.config))
            recipe["train"]["source_commit"] = "auto"
        print(create_run(args.study, recipe, args.label))
    elif args.command == "index":
        print(regenerate_index(args.study))
    else:
        study = args.study.parent if args.study.name == "sweep.yaml" else args.study
        results = [validate_record(study / "runs" / run["run_id"]) for run in load_study(study)["runs"]]
        print(json.dumps({"study": str(study), "runs": results}, indent=2))


if __name__ == "__main__":
    main()
