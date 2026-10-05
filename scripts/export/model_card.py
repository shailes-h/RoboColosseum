"""Write the HF model card (README.md) for one run. Epoch-level info only (no step counts).
usage: python scripts/export/model_card.py <model> <task> <wall_time> <out.md>
env: CARD_NGPU=4 CARD_GPU="H100" (hardware the run was trained on; per-GPU batch = GBS / CARD_NGPU)
"""
import json
import os
import re
import sys

model, task, wall, out = sys.argv[1:5]
emb = os.environ.get("EMBODIMENT", "BimanualYAM")
info = json.load(open(f"{os.environ['RC_DATA']}/{emb}/{task}/meta/info.json"))
tasks_hint = {"Dustpan": "Clean the table.", "Cups": "Stack the cups.", "Drawer": "Put the cup into the drawer.",
              "Microwave": "Open the microwave and take out the bowl."}
R = {
    "gr00t": ("GR00T N1.7", "`nvidia/GR00T-N1.7-3B` (Isaac-GR00T)",
              "Full fine-tune: VLM backbone (LLM + vision) unfrozen + action head",
              "AdamW, LR 1e-4 (repo default for GBS 64), cosine, 5% warmup; GBS 64 (16/GPU x 4)",
              "Final model in HF format (`model-*.safetensors`, `config.json`, `processor/`, `experiment_cfg/`). Optimizer state not included.",
              "`gr00t.policy.Gr00tPolicy(model_path=<this folder>, embodiment_tag=NEW_EMBODIMENT)` with the BimanualYAM modality config (`configs/gr00t/yam_config.py`)"),
    "pi05": ("π0.5", "`gs://openpi-assets/checkpoints/pi05_base` (openpi, JAX)",
             "Full fine-tune, EMA 0.99 (config `pi05_yam`, recipe of `pi05_aloha_pen_uncap`, `adapt_to_pi=False`)",
             "AdamW, cosine, peak LR 2.5e-5 → 2.5e-6, 1k warmup; GBS 64 (16/GPU x 4)",
             "Final checkpoint in openpi format: `params/` (EMA weights) and `assets/<dataset>/norm_stats.json`. Optimizer state not included.",
             "`openpi.policies.policy_config.create_trained_policy(get_config(\"pi05_yam\"), <this folder>)` (config: `configs/openpi/yam.py`)"),
    "g05": ("Galaxea G0.5", "`OpenGalaxea/G05` g05-base (GalaxeaVLA)",
            "Full fine-tune, vision LR x0.1, FSDP (R1Lite post-training recipe; data in R1Lite layout)",
            "LR 4e-5, cosine, warmup 200, wd 0.03; GBS 32 (8/GPU x 4)",
            "`model.pt` (final epoch), `dataset_stats.json`, `action_tokenizer.pt`.",
            "GalaxeaVLA with task config `configs/galaxea/task/yam.yaml`"),
    "molmoact2": ("MolmoAct2", "`allenai/MolmoAct2` (base)",
                  "Full fine-tune (VLM + action expert)",
                  "LLM LR 2e-5, ViT/connector 1e-5, action expert 1e-4 (recipe x 128/64), warmup, decay to 0.1x; GBS 128 (32/GPU x 4)",
                  "Final checkpoint converted to HF format (`model-*.safetensors`, processor, `norm_stats.json`).",
                  "HF `trust_remote_code` model; see `inference.py` in this folder"),
    "lingbot": ("LingBot-VLA v2 6B", "`robbyant/lingbot-vla-v2-6b`",
                "Full post-train (recipe `configs/vla/real_robot/real_robot.yaml`)",
                "Muon, constant LR 1.25e-5 (5e-5 x 64/256); GBS 64 (16/GPU x 4)",
                "Final checkpoint in HF format plus `norm_stats.json` (meanstd).",
                "use as `--model_path` with the lingbot-vla-v2 inference code, together with `--norm_stats_file norm_stats.json`"),
}
name, base, trained, optim, files, load = R[model]
ngpu, gpu = int(os.environ.get("CARD_NGPU", 4)), os.environ.get("CARD_GPU", "H100")
optim = re.sub(r"GBS (\d+) \(\d+/GPU x 4\)", lambda m: f"GBS {m[1]} ({int(m[1]) // ngpu}/GPU x {ngpu})", optim)
open(out, "w").write(f"""# {name} — {emb} {task}

- **Task:** "{tasks_hint.get(task, task)}" — {info['total_episodes']} episodes, {info['total_frames']:,} frames @ {info['fps']} fps (`RoboColosseum/BimanualYAM-datasets/{task}`)
- **Inputs/outputs:** cameras top / left wrist / right wrist + instruction; 14-D absolute joint state and action
- **Base:** {base}
- **Trained:** {trained}
- **Optim:** {optim}; 5 epochs; fp32 master weights, bf16 compute; {ngpu}x {gpu}; wall time {wall}
- **Files:** {files}
- **Load:** {load}
- **Code:** RoboColosseum harness (`scripts/train/{model}.sh`); logs in W&B project `RoboColosseum`
""")
print("card ->", out)
