"""Launcher for openpi scripts: usage run.py <script.py> [args...]   (run from the openpi repo root)

1. Preloads LeRobot's dataset module before openpi.training.config; the other import order segfaults
   (native-lib clash).
2. Registers the RoboColosseum configs (configs/openpi/yam.py) so the unchanged repo scripts can use them.
3. normstats mode: env RC_NORMSTATS_WORKERS=<n> raises the data-loader workers of compute_norm_stats.py
   (it decodes 3 videos per sample; the config's 2 workers take ~45 min).
"""
import dataclasses
import os
import runpy
import sys

import lerobot.common.datasets.lerobot_dataset  # noqa: F401  (must be first)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Guard: openpi data loaders use the "spawn" start method, which re-imports this module in workers.
if __name__ == "__main__":
    import openpi.training.config as _config
    import yam

    yam.register()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    if os.environ.get("RC_NORMSTATS_WORKERS"):
        # Import (not runpy) so classes defined in the script are picklable by spawn workers.
        _orig = _config.get_config
        _config.get_config = lambda name: dataclasses.replace(_orig(name), num_workers=int(os.environ["RC_NORMSTATS_WORKERS"]))
        sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
        import compute_norm_stats
        import tyro

        tyro.cli(compute_norm_stats.main)
    else:
        runpy.run_path(script, run_name="__main__")
