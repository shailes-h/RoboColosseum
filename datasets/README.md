# Datasets

One folder per embodiment: `datasets/<Embodiment>/<task>/`. Every raw dataset is **LeRobot v3.0**.
Only READMEs and scripts are tracked; the data itself is downloaded and converted locally (gitignored).

Each embodiment folder has:
- `download.sh`: fetches the raw v3.0 task(s) from the Hugging Face Hub.
- `convert.sh`: builds the per-VLA copies (`<task>_joint`, `_galaxea`, `_joint_v21`, `_joint_v21_gr00t`) next to the raw data.
- `README.md`: the embodiment's state/action/camera layout and what each conversion does.

| Embodiment | Tasks | README |
|---|---|---|
| BimanualYAM | Dustpan | [BimanualYAM/README.md](BimanualYAM/README.md) |

`RC_DATA` (default `datasets/`, see `env.sh`) is the root that all training scripts read from. To train from a fast
local disk, copy the tree with `bash scripts/data/stage.sh <src> <dst>` and set `RC_DATA=<dst>` in `env.local.sh`.
