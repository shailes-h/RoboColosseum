"""Galaxea R1Lite-style copy of Dustpan_joint (LeRobot v3.0).

Splits 14-D joint state/action into per-part columns and renames cameras to
R1Lite keys so GalaxeaVLA's stock GalaxeaLerobotDataset + r1lite-style config work.
usage: make_galaxea_dataset.py <src Dustpan_joint> <dst>
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(sys.argv[1])  # Dustpan_joint
DST = Path(sys.argv[2])
PARTS = {"left_arm": (0, 6), "left_gripper": (6, 7), "right_arm": (7, 13), "right_gripper": (13, 14)}
CAMS = {"observation.images.top": "observation.images.head_rgb",
        "observation.images.left": "observation.images.left_wrist_rgb",
        "observation.images.right": "observation.images.right_wrist_rgb"}


def split_names(base):
    return {f"{base}.{p}": (a, b) for p, (a, b) in PARTS.items()}


def rename_meta_col(name):
    for old, new in CAMS.items():
        name = name.replace(old, new)
    return name


if DST.exists():
    shutil.rmtree(DST)
(DST / "meta").mkdir(parents=True)

for p in (SRC / "data").rglob("*.parquet"):
    t = pq.read_table(p)
    cols = {n: t[n] for n in t.column_names if n not in ("observation.state", "action")}
    for base in ("observation.state", "action"):
        arr = np.stack(t[base].to_numpy(zero_copy_only=False)).astype(np.float32)
        for name, (a, b) in split_names(base).items():
            if b - a == 1:  # LeRobot treats shape [1] features as scalar float columns
                cols[name] = pa.array(arr[:, a], type=pa.float32())
            else:
                cols[name] = pa.array(list(arr[:, a:b]), type=pa.list_(pa.float32(), b - a))
    cols["coarse_task_index"] = t["task_index"]  # GalaxeaLerobotDataset reads a high-level task; single task here
    out = DST / p.relative_to(SRC)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out)

# Episode metadata: rename camera-keyed columns; drop whole-vector stats (loader recomputes its own).
for p in (SRC / "meta" / "episodes").rglob("*.parquet"):
    t = pq.read_table(p)
    keep = [(rename_meta_col(n), c) for n, c in zip(t.column_names, t.columns)
            if not (n.startswith("stats/observation.state/") or n.startswith("stats/action/"))]
    out = DST / p.relative_to(SRC)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table([c for _, c in keep], names=[n for n, _ in keep]), out)
shutil.copy(SRC / "meta" / "tasks.parquet", DST / "meta" / "tasks.parquet")

info = json.loads((SRC / "meta" / "info.json").read_text())
feats = {}
for k, v in info["features"].items():
    if k in ("observation.state", "action"):
        for name, (a, b) in split_names(k).items():
            feats[name] = {"dtype": "float32", "shape": [b - a], "names": v["names"][a:b]}
    else:
        feats[CAMS.get(k, k)] = v
feats["coarse_task_index"] = dict(info["features"]["task_index"])
info["features"] = feats
(DST / "meta" / "info.json").write_text(json.dumps(info, indent=4))

stats = json.loads((SRC / "meta" / "stats.json").read_text())
new_stats = {}
for k, v in stats.items():
    if k in ("observation.state", "action"):
        for name, (a, b) in split_names(k).items():
            new_stats[name] = {s: (x[a:b] if isinstance(x, list) and len(x) == 14 else x) for s, x in v.items()}
    else:
        new_stats[CAMS.get(k, k)] = v
(DST / "meta" / "stats.json").write_text(json.dumps(new_stats, indent=4))

(DST / "videos").mkdir()
src_videos = (SRC / "videos").resolve()
for old, new in CAMS.items():
    (DST / "videos" / new).symlink_to(os.path.relpath(src_videos / old, (DST / "videos").resolve()))
print("features:", list(feats))
