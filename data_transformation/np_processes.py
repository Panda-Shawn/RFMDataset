import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R


def fractal_tf2np(traj):
    # convert pos+quat to pos+euler
    eef_state = traj["observation"]["base_pose_tool_reached"].numpy()
    eef_euler = R.from_quat(eef_state[:, -4:]).as_euler("xyz")
    proprio_state = {
        "eef_state": np.concatenate([eef_state[:, :3], eef_euler], axis=1),
        "gripper_state": traj["observation"]["gripper_closed"].numpy(),
        "joint_value": None,
    }
    thrid_view_img = np.stack([tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8) for img in traj["observation"]["image"]])
    wrist_img = None
    lang_instruction = traj["language_instruction"].numpy()
    action = traj["action"].numpy()
    return proprio_state, thrid_view_img, wrist_img, lang_instruction, action


def libero_tf2np(traj):
    eef_state = traj["observation"]["EEF_state"].numpy()
    gripper_state = traj["observation"]["gripper_state"].numpy()
    proprio_state = {
        "eef_state": eef_state,
        "gripper_state": gripper_state,
        "joint_value": traj["observation"]["joint_state"].numpy(),
    }
    thrid_view_img = np.stack([tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8) for img in traj["observation"]["image"]])
    wrist_img = np.stack([tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8) for img in traj["observation"]["wrist_image"]])
    lang_instruction = traj["language_instruction"].numpy()
    action = traj["action"].numpy()
    return proprio_state, thrid_view_img, wrist_img, lang_instruction, action


STANDARDIZATION_PROCESSES = {
    "fractal20220817_data": fractal_tf2np,
    "libero_10_no_noops": libero_tf2np,
    "libero_goal_no_noops": libero_tf2np,
    "libero_object_no_noops": libero_tf2np,
    "libero_spatial_no_noops": libero_tf2np,
}