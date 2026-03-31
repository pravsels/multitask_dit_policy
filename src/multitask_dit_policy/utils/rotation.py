"""Rotation utilities for converting between RPY Euler angles and 6D rotation.

6D rotation representation from Zhou et al. 2019 "On the Continuity of Rotation
Representations in Neural Networks". Uses the first two columns of the rotation
matrix, which provides a continuous representation suitable for learning.
"""

import torch
from torch import Tensor


def _rpy_to_matrix(rpy: Tensor) -> Tensor:
    """Convert roll-pitch-yaw Euler angles (ZYX convention) to rotation matrices.

    Args:
        rpy: (..., 3) tensor of [roll, pitch, yaw] in radians.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    # ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = torch.stack([
        torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=-1),
        torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=-1),
        torch.stack([-sp, cp * sr, cp * cr], dim=-1),
    ], dim=-2)
    return R


def _matrix_to_rpy(R: Tensor) -> Tensor:
    """Extract roll-pitch-yaw Euler angles (ZYX convention) from rotation matrices.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) tensor of [roll, pitch, yaw] in radians.
    """
    pitch = torch.atan2(-R[..., 2, 0], torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    return torch.stack([roll, pitch, yaw], dim=-1)


def _gram_schmidt(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Gram-Schmidt orthogonalization to recover a valid rotation matrix from two vectors."""
    e1 = torch.nn.functional.normalize(a, dim=-1)
    b_proj = (b * e1).sum(dim=-1, keepdim=True) * e1
    e2 = torch.nn.functional.normalize(b - b_proj, dim=-1)
    e3 = torch.linalg.cross(e1, e2)
    return e1, e2, e3


def rpy_to_rot6d(rpy: Tensor) -> Tensor:
    """Convert RPY Euler angles to 6D rotation representation.

    Args:
        rpy: (..., 3) tensor of [roll, pitch, yaw].

    Returns:
        (..., 6) tensor -- first two columns of the rotation matrix, flattened.
    """
    R = _rpy_to_matrix(rpy)
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_rpy(rot6d: Tensor) -> Tensor:
    """Convert 6D rotation representation back to RPY Euler angles.

    Uses Gram-Schmidt orthogonalization to recover a valid rotation matrix.

    Args:
        rot6d: (..., 6) tensor.

    Returns:
        (..., 3) tensor of [roll, pitch, yaw].
    """
    a = rot6d[..., :3]
    b = rot6d[..., 3:]
    e1, e2, e3 = _gram_schmidt(a, b)
    R = torch.stack([e1, e2, e3], dim=-1)  # (..., 3, 3)
    return _matrix_to_rpy(R)


def convert_eef_pose(eef: Tensor) -> Tensor:
    """Convert eef_pose from [x,y,z,roll,pitch,yaw,gripper] (7D) to [x,y,z,rot6d,gripper] (10D).

    Args:
        eef: (..., 7) tensor.

    Returns:
        (..., 10) tensor.
    """
    xyz = eef[..., :3]
    rpy = eef[..., 3:6]
    grip = eef[..., 6:]
    rot6d = rpy_to_rot6d(rpy)
    return torch.cat([xyz, rot6d, grip], dim=-1)


def unconvert_eef_pose(eef: Tensor) -> Tensor:
    """Convert eef_pose from [x,y,z,rot6d,gripper] (10D) back to [x,y,z,r,p,y,gripper] (7D).

    Args:
        eef: (..., 10) tensor.

    Returns:
        (..., 7) tensor.
    """
    xyz = eef[..., :3]
    rot6d = eef[..., 3:9]
    grip = eef[..., 9:]
    rpy = rot6d_to_rpy(rot6d)
    return torch.cat([xyz, rpy, grip], dim=-1)
