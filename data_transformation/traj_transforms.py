import tensorflow as tf
from data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_bridge_actions,
)
from typing import Dict, Any


def fractal_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def libero_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    
    # find the max gripper state in the trajectory and keep dim
    max_gripper_state = tf.reduce_max(trajectory["observation"]["state"][:, -2:-1], axis=0, keepdims=True)
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:-1] / max_gripper_state # 1D gripper state (1 for fully open, 0 for fully closed)
    return trajectory


STANDARDIZATION_TRANSFORMS = {
    "fractal20220817_data": fractal_transform,
    "libero_10_no_noops": libero_transform,
    "libero_goal_no_noops": libero_transform,
    "libero_object_no_noops": libero_transform,
    "libero_spatial_no_noops": libero_transform,
}