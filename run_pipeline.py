# =============================================================================
# run_pipeline.py
# Runs the same pipeline as main.py but for a different country's config.
#
# Every pipeline module (data_acquisition.py, feature_engineering.py,
# models.py, prediction_engine.py, spatial_predictor.py, main.py, ...) does
# `from config import X` at its own top level, so the standard way to target
# a different region without editing any of those files is to make the name
# "config" resolve to a different module before they're imported for the
# first time in this process. That's what this script does: it swaps
# sys.modules["config"] for the requested country's config module, then
# imports main.py (which triggers every downstream `from config import ...`
# to resolve against the swapped module) and hands off to its CLI.
#
# Usage:
#   python run_pipeline.py japan --mode full        (targets main.py, default)
#   python run_pipeline.py myanmar --mode train --refresh
#   python run_pipeline.py japan generate           (targets generate.py instead)
# =============================================================================

import sys
import importlib

VALID_COUNTRIES = {"myanmar", "japan"}
VALID_TARGETS   = {"main", "generate"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in VALID_COUNTRIES:
        print(f"Usage: python run_pipeline.py <{'|'.join(sorted(VALID_COUNTRIES))}> "
              f"[generate | --mode ...]")
        sys.exit(1)

    country = sys.argv[1]
    remaining_args = sys.argv[2:]

    # Second arg selects which script to drive. Anything starting with "-"
    # is a main.py flag, not a target name, so main stays the default -
    # this keeps the existing `run_pipeline.py <country> --mode ...` calls
    # working unchanged.
    target = "main"
    if remaining_args and not remaining_args[0].startswith("-"):
        target = remaining_args[0]
        remaining_args = remaining_args[1:]
        if target not in VALID_TARGETS:
            print(f"Unknown target '{target}', expected one of {sorted(VALID_TARGETS)}")
            sys.exit(1)

    if country != "myanmar":
        cfg_module = importlib.import_module(f"config_{country}")
        sys.modules["config"] = cfg_module
        print(f"[PIPELINE] Using config_{country}.py for this run "
              f"(swapped into sys.modules['config']).")
    else:
        print("[PIPELINE] Using default config.py (Myanmar).")

    # The target module's own `from config import (...)` resolves against
    # whatever sys.modules["config"] currently points to, so this import
    # must happen after the swap above, and it must not already be
    # imported elsewhere in this process.
    sys.argv = [sys.argv[0]] + remaining_args
    if target == "generate":
        import generate
        generate.main()
    else:
        import main
        main.main()


if __name__ == "__main__":
    main()
