"""Legacy entry point using the shared Transformer training CLI."""

from experiments.key_dimension.configs import Config
from experiments.shared_transformer.train import (
    attention_degree_argument,
    main as _main,
    should_resume,
)


def main(argv=None) -> None:
    _main(argv, config_factory=Config, source_commit_env="KEY_DIMENSION_SOURCE_COMMIT")


if __name__ == "__main__":
    main()
