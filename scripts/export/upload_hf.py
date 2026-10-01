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


ROOT = Path(__file__).resolve().parents[2]


def add_eval_files(model, run, task, put):
    """Small files the eval side needs to load the checkpoint without this repo's runtime plumbing."""
    t = task.lower()
    if model == "gr00t":
        put(ROOT / "configs/gr00t/yam_config.py", "yam_config.py")
        put(ROOT / "configs/gr00t/yam_modality.json", "yam_modality.json")
    elif model == "g05":  # fully composed hydra config of the run
        put(run / ".hydra/config.yaml", ".hydra/config.yaml")
        put(run / ".hydra/overrides.yaml", ".hydra/overrides.yaml")
    elif model == "lingbot":  # 14-D YAM -> 55-D LingBot mapping, norm stats pointing at the file next to it
        txt = (ROOT / "configs/lingbot/robot_configs/yam.yaml").read_text()
        txt = re.sub(r"(?m)^norm_stats: .*$", "norm_stats: norm_stats.json  # the norm_stats.json in this folder", txt)
        put(txt.encode(), f"robot_config_yam_{t}.yaml")
        put(run / "lingbotvla_cli.yaml", "lingbotvla_cli.yaml")  # full training CLI config (the loader reads it)
    elif model == "pi05":  # standalone TrainConfig (configs/openpi/yam.py without the runtime registration)
        txt = (ROOT / "configs/openpi/yam.py").read_text()
        txt = txt.replace('TASK = os.environ.get("TASK", "Dustpan")', f'TASK = "{task}"')
        txt = txt.replace('name="pi05_yam"', f'name="pi05_yam_{t}"')
        txt = re.sub(r'(?m)^    assets_base_dir=.*\n', "", txt)  # norm stats come from the checkpoint's assets/
        txt = txt.split("\n\ndef register():")[0] + "\n\nCONFIG = PI05_YAM  # create_trained_policy(CONFIG, <this folder>)\n"
        put(txt.encode(), "train_config.py")


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
    put = lambda src, name: api.upload_file(  # noqa: E731
        path_or_fileobj=src if isinstance(src, bytes) else str(src), path_in_repo=f"{dst}/{name}", repo_id=a.repo)

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
    add_eval_files(a.model, run, a.task, put)
    put(a.card, "README.md")
    print(f"UPLOAD_OK {a.repo}/{dst}")


if __name__ == "__main__":
    main()
