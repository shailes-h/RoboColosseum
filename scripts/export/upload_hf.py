"""Upload the final checkpoint of one run (inference files only) to the HF model repo under <model>/<task>/.

usage: python scripts/export/upload_hf.py <model> <run_dir> <task> <model_card.md> [--repo RoboColosseum/BimanualYAM-models]
  model: gr00t | pi05 | g05 | molmoact2 | lingbot
  MolmoAct2: convert first (scripts/export/molmoact2_to_hf.sh <run_dir>/<last step> <run_dir>/hf_final).
"""
import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi

HF_FOLDER = {"gr00t": "gr00t", "pi05": "pi05", "g05": "g05", "molmoact2": "molmoact2", "lingbot": "lingbot-vla-v2"}


def last(paths, key):
    paths = sorted(paths, key=key)
    if not paths:
        raise SystemExit("no checkpoint found")
    return paths[-1]


def num(p):
    return int(re.findall(r"\d+", p.name)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=HF_FOLDER)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("task")
    ap.add_argument("card", type=Path)
    ap.add_argument("--repo", default="RoboColosseum/BimanualYAM-models")
    a = ap.parse_args()
    api, run, dst = HfApi(), a.run_dir.resolve(), f"{HF_FOLDER[a.model]}/{a.task.lower()}"
    up = lambda **kw: api.upload_folder(repo_id=a.repo, path_in_repo=dst, **kw)  # noqa: E731
    put = lambda src, name: api.upload_file(path_or_fileobj=str(src), path_in_repo=f"{dst}/{name}", repo_id=a.repo)  # noqa: E731

    if a.model == "gr00t":  # final model is saved at the run root; skip per-epoch checkpoint-* and optimizer state
        root = run / run.name if (run / run.name).is_dir() else run
        up(folder_path=str(root), ignore_patterns=["checkpoint-*", "training_args.bin", "wandb*"])
    elif a.model == "pi05":  # orbax params (EMA) + norm stats; no train_state
        up(folder_path=str(last([p for p in run.iterdir() if p.name.isdigit()], num)), allow_patterns=["params/**", "assets/**"])
    elif a.model == "g05":
        put(last(list((run / "checkpoints").glob("step_*.pt")), num), "model.pt")
        put(run / "dataset_stats.json", "dataset_stats.json")
        put(run / "action_tokenizer.pt", "action_tokenizer.pt")
    elif a.model == "molmoact2":
        up(folder_path=str(run / "hf_final"))
    elif a.model == "lingbot":
        ck = last(list((run / "checkpoints").glob("global_step_*")), num)
        up(folder_path=str(ck / "hf_ckpt"))
        put(Path(os.environ["RC_OUTPUTS"]) / "norm_stats" / "lingbot" / f"yam_{a.task}.json", "norm_stats.json")
    put(a.card, "README.md")
    print(f"UPLOAD_OK {a.repo}/{dst}")


if __name__ == "__main__":
    main()
