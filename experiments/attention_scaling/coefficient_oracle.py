"""Study entry point for shared fixed-input coefficient fitting."""
from scaffolds.transformer.training import coefficient_fit as _implementation
from scaffolds.transformer.training.coefficient_fit import (
    main, make_functions, plateau_action, restore_state, save_state,
)


def __getattr__(name):
    # Preserve the study module's helper API for existing diagnostic callers.
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
