"""Legacy entry point using the shared Transformer training CLI."""

from scaffolds.transformer.studies.key_dimension_taylor.degree_sweep.configs import Config
from scaffolds.transformer.train import (
    attention_degree_argument,
    main as _main,
    should_resume,
)


def main(argv=None) -> None:
    _main(argv, config_factory=Config, source_commit_env="POLYNOMIAL_ATTENTION_SOURCE_COMMIT")


if __name__ == "__main__":
    main()
