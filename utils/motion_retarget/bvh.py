from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .rotations import euler_degrees_to_rotvec, rotvec_to_matrix


@dataclass(frozen=True)
class BVHJoint:
    name: str
    parent: int
    offset: np.ndarray
    channels: tuple[str, ...]
    channel_start: int
    end_sites: tuple[np.ndarray, ...] = field(default_factory=tuple)


@dataclass
class BVHMotion:
    joints: list[BVHJoint]
    frames: np.ndarray
    fps: float

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    def joint_index(self, name: str) -> int:
        lookup = {joint.name.lower(): i for i, joint in enumerate(self.joints)}
        return lookup[name.lower()]

    def local_rotvecs(self, dtype=np.float32) -> np.ndarray:
        result = np.zeros((self.frame_count, len(self.joints), 3), dtype=dtype)
        for i, joint in enumerate(self.joints):
            rotation_channels = [(k, c) for k, c in enumerate(joint.channels) if c.lower().endswith("rotation")]
            if not rotation_channels:
                continue
            order = "".join(channel[0] for _, channel in rotation_channels)
            columns = [joint.channel_start + local_index for local_index, _ in rotation_channels]
            result[:, i] = euler_degrees_to_rotvec(self.frames[:, columns], order)
        return result

    def local_positions(self, include_nonroot_translation: bool = True,
                        translation_mode: str = 'offset_plus_channel') -> np.ndarray:
        """Explicit dialect: legacy additive or animated absolute local channels.

        Blender's BVH armature importer subtracts rest_head_local from animated
        locations. Its resulting local translation is the channel value; OFFSET
        supplies only axes without translation channels. CoLLaboratE uses that
        contract. The legacy default is retained for reproducibility.
        """
        if translation_mode not in ('offset_plus_channel','absolute_channels'):
            raise ValueError('Unknown BVH translation mode: '+translation_mode)
        result = np.broadcast_to(
            np.stack([joint.offset for joint in self.joints]),
            (self.frame_count, len(self.joints), 3),
        ).astype(np.float64).copy()
        axes = {"xposition": 0, "yposition": 1, "zposition": 2}
        for i, joint in enumerate(self.joints):
            if i != 0 and not include_nonroot_translation:
                continue
            for local_index, channel in enumerate(joint.channels):
                axis = axes.get(channel.lower())
                if axis is not None:
                    value = self.frames[:, joint.channel_start + local_index]
                    if translation_mode == 'absolute_channels':result[:, i, axis] = value
                    else:result[:, i, axis] += value
        return result

    def forward_kinematics(self, include_nonroot_translation: bool = True,
                           translation_mode: str = 'offset_plus_channel', dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
        local_r = rotvec_to_matrix(self.local_rotvecs(dtype=dtype))
        local_p = self.local_positions(include_nonroot_translation=include_nonroot_translation,translation_mode=translation_mode)
        global_r = np.zeros_like(local_r)
        global_p = np.zeros_like(local_p)
        for i, joint in enumerate(self.joints):
            if joint.parent < 0:
                global_r[:, i] = local_r[:, i]
                global_p[:, i] = local_p[:, i]
            else:
                parent = joint.parent
                global_r[:, i] = global_r[:, parent] @ local_r[:, i]
                global_p[:, i] = global_p[:, parent] + np.einsum(
                    "tij,tj->ti", global_r[:, parent], local_p[:, i]
                )
        return global_p.astype(dtype), global_r.astype(dtype)

    def end_site_positions(
        self, joint_index: int, include_nonroot_translation: bool = False,
    ) -> np.ndarray:
        """Return every End Site attached to ``joint_index`` in world space."""
        joint = self.joints[int(joint_index)]
        if not joint.end_sites:
            raise ValueError(f"BVH joint {joint.name} has no End Site")
        positions, rotations = self.forward_kinematics(
            include_nonroot_translation=include_nonroot_translation,
        )
        offsets = np.stack(joint.end_sites).astype(np.float32)
        return positions[:, joint_index, None] + np.einsum(
            "tij,sj->tsi", rotations[:, joint_index], offsets,
        )


_TOKEN = re.compile(r"[{}]|[^\s{}]+")


def parse_bvh(payload: bytes | str) -> BVHMotion:
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
    hierarchy_text, marker, motion_text = text.partition("MOTION")
    if not marker:
        raise ValueError("BVH payload has no MOTION section")
    tokens = _TOKEN.findall(hierarchy_text)
    joints: list[BVHJoint] = []
    stack: list[int] = []
    channel_cursor = 0
    i = 0
    pending_joint: tuple[str, int] | None = None
    pending_end_parent: int | None = None
    while i < len(tokens):
        token = tokens[i]
        upper = token.upper()
        if upper in {"ROOT", "JOINT"}:
            pending_joint = (tokens[i + 1], stack[-1] if stack else -1)
            i += 2
            continue
        if upper == "END":
            pending_joint = None
            pending_end_parent = stack[-1] if stack and stack[-1] >= 0 else None
            i += 2  # End Site
            continue
        if token == "{":
            if pending_joint is not None:
                name, parent = pending_joint
                joints.append(BVHJoint(name, parent, np.zeros(3), (), channel_cursor))
                stack.append(len(joints) - 1)
                pending_joint = None
            elif pending_end_parent is not None:
                # Encode an End Site stack frame without creating a kinematic
                # joint.  ``-2`` represents parent zero, ``-3`` parent one,
                # and so on; ``-1`` remains the legacy anonymous sentinel.
                stack.append(-pending_end_parent - 2)
                pending_end_parent = None
            else:
                stack.append(-1)  # End Site body
            i += 1
            continue
        if token == "}":
            if stack:
                stack.pop()
            i += 1
            continue
        if upper == "OFFSET":
            if stack and stack[-1] >= 0:
                idx = stack[-1]
                joint = joints[idx]
                joints[idx] = BVHJoint(
                    joint.name, joint.parent,
                    np.asarray(tokens[i + 1 : i + 4], dtype=np.float64),
                    joint.channels, joint.channel_start, joint.end_sites,
                )
            elif stack and stack[-1] <= -2:
                parent = -stack[-1] - 2
                joint = joints[parent]
                offset = np.asarray(tokens[i + 1 : i + 4], dtype=np.float64)
                joints[parent] = BVHJoint(
                    joint.name, joint.parent, joint.offset, joint.channels,
                    joint.channel_start, joint.end_sites + (offset,),
                )
            i += 4
            continue
        if upper == "CHANNELS":
            count = int(tokens[i + 1])
            channels = tuple(tokens[i + 2 : i + 2 + count])
            idx = stack[-1]
            joint = joints[idx]
            joints[idx] = BVHJoint(
                joint.name, joint.parent, joint.offset, channels,
                channel_cursor, joint.end_sites,
            )
            channel_cursor += count
            i += 2 + count
            continue
        i += 1

    lines = [line.strip() for line in motion_text.splitlines() if line.strip()]
    frame_count = int(next(line for line in lines if line.lower().startswith("frames:" )).split(":", 1)[1])
    frame_time = float(next(line for line in lines if line.lower().startswith("frame time:" )).split(":", 1)[1])
    data_start = next(i for i, line in enumerate(lines) if line.lower().startswith("frame time:")) + 1
    values = np.fromstring(" ".join(lines[data_start:]), sep=" ", dtype=np.float64)
    expected = frame_count * channel_cursor
    if values.size != expected:
        raise ValueError(f"BVH motion has {values.size} values; expected {expected}")
    return BVHMotion(joints, values.reshape(frame_count, channel_cursor), 1.0 / frame_time)

