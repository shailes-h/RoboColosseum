# BimanualYAM

Two 6-DoF YAM arms with parallel grippers. Source: [`RoboColosseum/BimanualYAM-datasets`](https://huggingface.co/datasets/RoboColosseum/BimanualYAM-datasets) (LeRobot v3.0).

| Task | Episodes | Frames | FPS | Instruction |
|---|---|---|---|---|
| Dustpan | 101 | 55,046 | 30 | "Clean the table." |

**Raw features**
- Cameras: `observation.images.top` (360x640), `.left` and `.right` wrist (480x640).
- State/action in two spaces: joint (`*_joint_angles`, 14-D) and end-effector (`*_eef_absolute`, `action_eef_delta`).
- 14-D joint layout: `[left arm 6, left gripper 1, right arm 6, right gripper 1]`, absolute.

All VLAs are trained in **joint space**.

## Get the data

```bash
bash datasets/BimanualYAM/download.sh          # -> $RC_DATA/BimanualYAM/Dustpan          (~0.8 GB)
bash scripts/setup/install_env.sh tools        # one-time: conversion env
bash datasets/BimanualYAM/convert.sh           # -> the four copies below (CPU only; ~2.5 GB total)
```

Pass a task name to either script to use another task (default `$TASK`, i.e. `Dustpan`).

## Conversions (`convert.sh`)

Each VLA's loader expects a different layout. Rather than patch the loaders, we write one dataset copy per layout:

| Copy | Format | Used by | What changes |
|---|---|---|---|
| `Dustpan_joint` | v3.0 | MolmoAct2, LingBot-VLA v2 | `observation.state` / `action` = the 14-D joint columns; EEF columns dropped; videos symlinked |
| `Dustpan_galaxea` | v3.0 | G0.5 | state/action split into `left_arm` / `left_gripper` / `right_arm` / `right_gripper`; cameras renamed to R1Lite keys (`head_rgb`, `left_wrist_rgb`, `right_wrist_rgb`); `coarse_task_index` added; videos symlinked |
| `Dustpan_joint_v21` | v2.1 | pi0.5 (openpi) | `Dustpan_joint` converted with GR00T's `scripts/lerobot_conversion/convert_v3_to_v2.py` |
| `Dustpan_joint_v21_gr00t` | v2.1 | GR00T N1.7 | top camera letterboxed 360x640 → 480x640 (GR00T stacks views, so they must share a size; AV1 re-encode with LeRobot settings), plus `meta/modality.json` from `configs/gr00t/yam_modality.json` |

The converters live in `scripts/data/` (`make_joint_dataset.py`, `make_galaxea_dataset.py`, `pad_video.py`). `convert.sh` first runs
`check_videos.py`, which verifies that each video file holds exactly the frames its episodes claim. It caught a mislabeled
`file_index` in Microwave (top camera, episodes 22-75), fixed on the Hub on 2026-10-01.
Symlinks are relative, so the whole `BimanualYAM/` tree can be copied with `cp -a`.
