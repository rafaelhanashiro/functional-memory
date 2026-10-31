"""Qualify the study's Transformer recipes through the shared runtime checks."""
from functools import partial

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance


def main(argv=None) -> None:
    run_acceptance(partial(attention_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
