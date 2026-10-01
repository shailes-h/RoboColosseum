"""RoboColosseum openpi configs, registered at runtime (openpi's config.py is not edited).

pi05_yam: BimanualYAM (14-D joints), modeled on pi05_aloha_pen_uncap. adapt_to_pi=False: YAM is not
Aloha, so skip the Aloha-specific joint/gripper conversions. Data: $HF_LEROBOT_HOME/<TASK>_joint_v21.
Norm stats are written to / read from $RC_OUTPUTS/norm_stats/openpi/pi05_yam/<repo_id>/.
"""
import os

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

TASK = os.environ.get("TASK", "Dustpan")

PI05_YAM = _config.TrainConfig(
    name="pi05_yam",
    model=pi0_config.Pi0Config(pi05=True),
    data=_config.LeRobotAlohaDataConfig(
        repo_id=f"{TASK}_joint_v21",
        adapt_to_pi=False,
        base_config=_config.DataConfig(prompt_from_task=True),
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.top",
                            "cam_left_wrist": "observation.images.left",
                            "cam_right_wrist": "observation.images.right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    assets_base_dir=os.path.join(os.environ["RC_OUTPUTS"], "norm_stats", "openpi"),
    num_train_steps=20_000,
    batch_size=64,
)


def register():
    for cfg in (PI05_YAM,):
        if cfg.name not in _config._CONFIGS_DICT:
            _config._CONFIGS.append(cfg)
            _config._CONFIGS_DICT[cfg.name] = cfg
