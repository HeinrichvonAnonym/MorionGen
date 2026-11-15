from __future__ import annotations

import hashlib
from pathlib import Path


HAND_POSE_CONVENTION = "smplx_axis_angle_flat_hand_mean"
FLAT_HAND_MEAN = True
HAND_FINGERS = ("index", "middle", "pinky", "ring", "thumb")
# Official SMPL-X fingertip vertex regressors in the same anatomical order as
# the 15 articulated hand joints above.  Using an extrapolated DIP direction
# ignores the third MANO joint rotation and produced visibly wrong hands.
HAND_TIP_VERTEX_IDS = {
    "left": (4933, 5058, 5286, 5169, 5361),
    "right": (7669, 7794, 8022, 7905, 8079),
}


def smplx_asset_path(model_dir: str | Path, gender: str) -> Path:
    path = Path(model_dir) / f"SMPLX_{gender.upper()}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def create_smplx_model(
    model_dir: str | Path, gender: str, num_betas: int,
    batch_size: int = 1, device=None,
):
    """Construct the one canonical SMPL-X decoder used by this pipeline.

    HiPHI BVH rotations and the native-hand IK are expressed relative to the
    flat MANO rest hand.  Allowing a caller to silently select the learned
    MANO mean changes the geometry by centimetres while leaving the 45-D pose
    array unchanged, so the convention is deliberately not an argument.
    """
    import smplx

    model = smplx.SMPLX(
        str(smplx_asset_path(model_dir, gender)), gender=gender,
        num_betas=int(num_betas), use_pca=False,
        flat_hand_mean=FLAT_HAND_MEAN, batch_size=int(batch_size),
    )
    return model.to(device) if device is not None else model


def smplx_hand_landmarks(joints, vertices):
    """Return wrist, palm, 15 joints and five true fingertip vertices.

    The result is ``[T, 2, 22, 3]`` with each finger represented by MCP, PIP,
    DIP and fingertip in ``HAND_FINGERS`` order.
    """
    import torch

    sides = []
    for side, wrist_index, start in (("left", 20, 25), ("right", 21, 40)):
        wrist = joints[:, wrist_index]
        fingers = joints[:, start:start + 15].reshape(len(joints), 5, 3, 3)
        bases = fingers[:, :, 0]
        palm = 0.45 * wrist + 0.55 * bases.mean(dim=1)
        tip_ids = torch.as_tensor(HAND_TIP_VERTEX_IDS[side], dtype=torch.long, device=vertices.device)
        tips = vertices[:, tip_ids]
        packed = [wrist, palm]
        for finger in range(5):
            packed.extend((fingers[:, finger, 0], fingers[:, finger, 1],
                           fingers[:, finger, 2], tips[:, finger]))
        sides.append(torch.stack(packed, dim=1))
    return torch.stack(sides, dim=1)


def smplx_model_signature(model_dir: str | Path, gender: str, num_betas: int) -> str:
    digest = hashlib.sha256()
    path = smplx_asset_path(model_dir, gender)
    digest.update(path.read_bytes())
    digest.update(b"\0")
    digest.update(HAND_POSE_CONVENTION.encode())
    digest.update(b"\0")
    digest.update(str(int(num_betas)).encode())
    digest.update(b"\0")
    digest.update(repr(HAND_TIP_VERTEX_IDS).encode())
    return digest.hexdigest()
