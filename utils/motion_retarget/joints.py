from __future__ import annotations

import numpy as np

from .bvh import BVHMotion


SMPLX_BODY_JOINTS = (
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)

BODY_ALIASES = {
    "left_hip": ("LeftUpLeg", "LeftHip"), "right_hip": ("RightUpLeg", "RightHip"),
    "spine1": ("Spine", "Chest"), "spine2": ("Spine2", "Chest2"),
    "spine3": ("Spine4", "Chest4", "Spine3", "Chest3"),
    "left_knee": ("LeftLeg", "LeftKnee"), "right_knee": ("RightLeg", "RightKnee"),
    "left_ankle": ("LeftFoot", "LeftAnkle"), "right_ankle": ("RightFoot", "RightAnkle"),
    "left_foot": ("LeftToeBase", "LeftToe"), "right_foot": ("RightToeBase", "RightToe"),
    "neck": ("Neck", "Neck1"), "head": ("Head",),
    "left_collar": ("LeftShoulder", "LeftCollar"), "right_collar": ("RightShoulder", "RightCollar"),
    "left_shoulder": ("LeftArm", "LeftShoulder"), "right_shoulder": ("RightArm", "RightShoulder"),
    "left_elbow": ("LeftForeArm", "LeftElbow"), "right_elbow": ("RightForeArm", "RightElbow"),
    "left_wrist": ("LeftHand", "LeftWrist"), "right_wrist": ("RightHand", "RightWrist"),
}

MANO_FINGERS = ("Index", "Middle", "Pinky", "Ring", "Thumb")


def _lookup(motion: BVHMotion, aliases: tuple[str, ...]) -> int | None:
    names = {joint.name.lower(): i for i, joint in enumerate(motion.joints)}
    return next((names[name.lower()] for name in aliases if name.lower() in names), None)


def bvh_to_smplx_pose(motion: BVHMotion) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local = motion.local_rotvecs()
    root = local[:, 0].copy()
    body = np.zeros((motion.frame_count, 21, 3), dtype=np.float32)
    for target_index, target_name in enumerate(SMPLX_BODY_JOINTS):
        source_index = _lookup(motion, BODY_ALIASES[target_name])
        if source_index is not None:
            body[:, target_index] = local[:, source_index]
    hands: list[np.ndarray] = []
    for side in ("Left", "Right"):
        hand = np.zeros((motion.frame_count, 15, 3), dtype=np.float32)
        target = 0
        for finger in MANO_FINGERS:
            for segment in range(1, 4):
                source_index = _lookup(motion, (f"{side}Hand{finger}{segment}",))
                if source_index is not None:
                    hand[:, target] = local[:, source_index]
                target += 1
        hands.append(hand.reshape(motion.frame_count, 45))
    return root, body.reshape(motion.frame_count, 63), hands[0], hands[1]

