"""Sanity-check a LeRobot v3.0 dataset's video metadata against the actual video files.

For every camera and video file: the episodes the metadata assigns to it must account for all of its frames, and no
episode may extend past the end of its file. Catches mislabeled file_index (e.g. BimanualYAM Microwave top camera).
usage: check_videos.py <dataset_root>        exit code 1 on any mismatch
"""
import glob
import sys

import av
import pandas as pd
import pyarrow.parquet as pq

root = sys.argv[1]
ep = pd.concat(pq.read_table(p).to_pandas() for p in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")))
cams = sorted({c.split("/")[1] for c in ep.columns if c.startswith("videos/") and c.endswith("/file_index")})
ok = True
for cam in cams:
    k = f"videos/{cam}"
    for (ch, fi), g in ep.groupby([f"{k}/chunk_index", f"{k}/file_index"]):
        f = f"{root}/videos/{cam}/chunk-{int(ch):03d}/file-{int(fi):03d}.mp4"
        with av.open(f) as c:
            s = c.streams.video[0]
            n, dur = s.frames, float(s.duration * s.time_base)
        frames, end = int(g.length.sum()), float(g[f"{k}/to_timestamp"].max())
        good = abs(n - frames) <= 1 and end <= dur + 0.1
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {cam} file-{int(fi):03d}: episodes {g.episode_index.min()}-{g.episode_index.max()} "
              f"claim {frames} frames / end {end:.1f}s; file has {n} frames / {dur:.1f}s")
sys.exit(0 if ok else 1)
