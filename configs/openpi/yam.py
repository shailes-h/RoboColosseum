"""RoboColosseum openpi configs, registered at runtime (openpi's config.py is not edited).

pi05_yam: BimanualYAM (14-D joints), modeled on pi05_aloha_pen_uncap. adapt_to_pi=False: YAM is not
Aloha, so skip the Aloha-specific joint/gripper conversions. Data: $HF_LEROBOT_HOME/<TASK>_joint_v21.
pi05_droid_joint: DROID (Franka, 8-D joints + gripper), cameras left_image + wrist, via openpi's DroidInputs.
Norm stats are written to / read from $RC_OUTPUTS/norm_stats/openpi/<config>/<repo_id>/.
"""
import dataclasses
import os
import pathlib

import numpy as np
from typing_extensions import override

import openpi.models.pi0_config as pi0_config
import openpi.policies.droid_policy as droid_policy
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


@dataclasses.dataclass(frozen=True)
class _SplitDroidState(_transforms.DataTransformFn):
    """8-D DROID state (7 joints + gripper) -> the joint/gripper keys DroidInputs expects."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data.pop("state"))
        data["observation/joint_position"], data["observation/gripper_position"] = state[:7], state[7:8]
        return data


@dataclasses.dataclass(frozen=True)
class LeRobotDroidJointDataConfig(_config.DataConfigFactory):
    """DROID (Franka, 7 joints + gripper) in LeRobot v2.1: exterior left_image + wrist (right_image unused).
    Like pi05_yam: joints as deltas to the current state, gripper absolute."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/exterior_image_1_left": "observation.images.left_image",
            "observation/wrist_image_left": "observation.images.wrist",
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })])
        mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[_SplitDroidState(), droid_policy.DroidInputs(model_type=model_config.model_type), _transforms.DeltaActions(mask)],
            outputs=[_transforms.AbsoluteActions(mask), droid_policy.DroidOutputs()],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


PI05_DROID = _config.TrainConfig(
    name="pi05_droid_joint",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotDroidJointDataConfig(repo_id=f"{TASK}_joint_v21", base_config=_config.DataConfig(prompt_from_task=True, action_sequence_keys=("action",))),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    assets_base_dir=os.path.join(os.environ["RC_OUTPUTS"], "norm_stats", "openpi"),
    num_train_steps=20_000,
    batch_size=32,
)


def register():
    for cfg in (PI05_YAM, PI05_DROID):
        if cfg.name not in _config._CONFIGS_DICT:
            _config._CONFIGS.append(cfg)
            _config._CONFIGS_DICT[cfg.name] = cfg
