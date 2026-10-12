# GR00T N1.7 modality config for DROID (Franka, joint space), same recipe as yam_config.py: arm relative, gripper absolute.
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ARM = ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
GRIPPER = ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
PARTS = ["arm", "gripper"]

rc_droid_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["left", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=PARTS),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=PARTS,
        action_configs=[ARM, GRIPPER],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(rc_droid_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
