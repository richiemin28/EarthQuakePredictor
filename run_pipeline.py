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
#   python run_pipeline.py japan --mode full
#   python run_pipeline.py myanmar --mode train --refresh
# =============================================================================

import sys
import importlib

VALID_COUNTRIES = {"myanmar", "japan"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in VALID_COUNTRIES:
        print(f"Usage: python run_pipeline.py <{'|'.join(sorted(VALID_COUNTRIES))}> [main.py args...]")
        sys.exit(1)

    country = sys.argv[1]
    remaining_args = sys.argv[2:]

    if country != "myanmar":
        cfg_module = importlib.import_module(f"config_{country}")
        sys.modules["config"] = cfg_module
        print(f"[PIPELINE] Using config_{country}.py for this run "
              f"(swapped into sys.modules['config']).")
    else:
        print("[PIPELINE] Using default config.py (Myanmar).")

    # main.py's own `from config import (...)` resolves against whatever
    # sys.modules["config"] currently points to, so this import must happen
    # after the swap above, and main must not already be imported elsewhere
    # in this process.
    sys.argv = [sys.argv[0]] + remaining_args
    import main
    main.main()


if __name__ == "__main__":
    main()
