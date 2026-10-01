"""Letterbox AV1 videos to 480x640 in place (black bars), re-encoded with LeRobot's AV1 settings.
usage: pad_video.py <dir with *.mp4>"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np


def pad(path: Path):
    tmp = path.with_suffix(".tmp.mp4")
    with av.open(str(path)) as inp:
        s = inp.streams.video[0]
        rate = s.average_rate
        with av.open(str(tmp), "w") as out:
            o = out.add_stream("libsvtav1", rate=rate)
            o.width, o.height, o.pix_fmt = 640, 480, "yuv420p"
            o.options = {"g": "2", "crf": "30", "preset": "12"}
            for f in inp.decode(s):
                img = f.to_ndarray(format="rgb24")
                h = img.shape[0]
                top = (480 - h) // 2
                canvas = np.zeros((480, 640, 3), np.uint8)
                canvas[top:top + h] = img
                for p in o.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                    out.mux(p)
            for p in o.encode():
                out.mux(p)
    tmp.replace(path)
    return path.name


if __name__ == "__main__":
    files = sorted(Path(sys.argv[1]).glob("*.mp4"))
    with Pool(min(32, os.cpu_count())) as pool:
        for n in pool.imap_unordered(pad, files):
            pass
    print("padded", len(files))
