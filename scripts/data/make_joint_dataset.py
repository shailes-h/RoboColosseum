"""Build a joint-space-only copy of a BimanualYAM LeRobot v3.0 dataset.

observation.state <- observation.state_joint_angles (14)
action            <- action_joint_angles (14)
EEF columns are dropped; videos are symlinked (relative link, not copied).
usage: make_joint_dataset.py <src> <dst>
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
RENAME = {"observation.state_joint_angles": "observation.state", "action_joint_angles": "action"}
DROP = {"observation.state_eef_absolute", "action_eef_absolute", "action_eef_delta"}


def fix_name(name: str) -> str | None:
    # Handles plain column names and episode-stats names like "stats/action_joint_angles/mean".
    for d in DROP:
        if name == d or f"/{d}/" in name or name.startswith(f"{d}/"):
            return None
    for old, new in RENAME.items():
        if name == old:
            return new
        name = name.replace(f"/{old}/", f"/{new}/")
        if name.startswith(f"{old}/"):
            name = new + name[len(old):]
    return name


def rewrite_parquet(src: Path, dst: Path):
    t = pq.read_table(src)
    cols, names = [], []
    for n, c in zip(t.column_names, t.columns):
        nn = fix_name(n)
        if nn is not None:
            cols.append(c)
            names.append(nn)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols, names=names), dst)


if DST.exists():
    shutil.rmtree(DST)
(DST / "meta").mkdir(parents=True)

for p in (SRC / "data").rglob("*.parquet"):
    rewrite_parquet(p, DST / p.relative_to(SRC))
for p in (SRC / "meta" / "episodes").rglob("*.parquet"):
    rewrite_parquet(p, DST / p.relative_to(SRC))
shutil.copy(SRC / "meta" / "tasks.parquet", DST / "meta" / "tasks.parquet")

info = json.loads((SRC / "meta" / "info.json").read_text())
info["features"] = {fix_name(k): v for k, v in info["features"].items() if fix_name(k)}
(DST / "meta" / "info.json").write_text(json.dumps(info, indent=4))

stats = json.loads((SRC / "meta" / "stats.json").read_text())
stats = {fix_name(k): v for k, v in stats.items() if fix_name(k)}
(DST / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

(DST / "videos").symlink_to(os.path.relpath((SRC / "videos").resolve(), DST.resolve()))
print("features:", list(info["features"]))
print("episode meta cols:", pq.read_table(next((DST / "meta" / "episodes").rglob("*.parquet"))).column_names[:12], "...")
