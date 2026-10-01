"""Launcher for MolmoAct2 launch_scripts/train_lerobot.py (run from molmoact2/experiments).

Registers the RoboColosseum LeRobot mixture(s) into MolmoAct2's mixture table at runtime (the repo's
data_mixtures.py is not edited), then runs the unchanged training script. Mixture "yam":
$LEROBOT_DATA_ROOT/<EMBODIMENT>/<TASK>_joint, i.e. LEROBOT_DATA_ROOT = $RC_DATA.
"""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
from launch_scripts import data_mixtures as dm  # noqa: E402

EMBODIMENT, TASK = os.environ.get("EMBODIMENT", "BimanualYAM"), os.environ.get("TASK", "Dustpan")


def build_yam():
    return dm.build_single_lerobot_mixture(
        name="yam",
        tag="yam",
        repo_ids=[f"{EMBODIMENT}/{TASK}_joint"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=["observation.images.top", "observation.images.left", "observation.images.right"],
        normalize_gripper=False,
        setup_type="bimanual yam robotic arms",
        control_mode="absolute joint pose",
        action_horizon=30,
        n_action_steps=30,
    )


dm.MOLMOACT2_LEROBOT_MIXTURES["yam"] = build_yam

if __name__ == "__main__":
    runpy.run_path("launch_scripts/train_lerobot.py", run_name="__main__")
