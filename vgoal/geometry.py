"""Camera geometry and 2D-to-3D back-projection for Visual Goal WAM.

Converts 2D bounding boxes + monocular depth maps into body-frame 3D relative
goal vectors: goal_rel = [d_fwd, d_left, d_up, remaining_dist] matching the
Aerial-WAM policy input contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, fov_deg: float, width: int, height: int) -> "CameraIntrinsics":
        """Construct intrinsics from horizontal field-of-view in degrees."""
        fov_rad = np.deg2rad(fov_deg)
        fx = (width / 2.0) / np.tan(fov_rad / 2.0)
        fy = (height / 2.0) / np.tan(fov_rad / 2.0)  # square sensor aspect
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=int(width), height=int(height))


def extract_target_depth(
    depth_map: np.ndarray,
    bbox: Sequence[float],
    *,
    src_shape: Optional[Tuple[int, int]] = None,
    core_frac: float = 0.5,
) -> float:
    """Extract robust target depth from the central core of the bounding box.

    Args:
        depth_map: [H, W] float array in meters.
        bbox: [u_min, v_min, u_max, v_max] in pixel coordinates.
        src_shape: (src_w, src_h) of the detection bounding box coordinate system (if different from depth_map).
        core_frac: Fraction of the central box to sample (defaults to 0.5 to avoid background bleed).

    Returns:
        Median depth in meters, or float("nan") if invalid/empty.
    """
    h, w = depth_map.shape[:2]
    u0, v0, u1, v1 = [float(x) for x in bbox]

    # Scale bbox if source detection resolution differs from depth_map resolution
    if src_shape is not None:
        src_w, src_h = src_shape
        if src_w > 0 and src_h > 0:
            scale_x = float(w) / float(src_w)
            scale_y = float(h) / float(src_h)
            u0 *= scale_x
            u1 *= scale_x
            v0 *= scale_y
            v1 *= scale_y

    u0, u1 = max(0, min(w - 1, u0)), max(0, min(w - 1, u1))
    v0, v1 = max(0, min(h - 1, v0)), max(0, min(h - 1, v1))

    if u1 <= u0:
        u1 = min(w, u0 + 2)
    if v1 <= v0:
        v1 = min(h, v0 + 2)

    bw = u1 - u0
    bh = v1 - v0
    cf = max(0.1, min(1.0, float(core_frac)))

    # Central core box
    cu0 = int(u0 + bw * (1.0 - cf) * 0.5)
    cu1 = max(cu0 + 1, int(u1 - bw * (1.0 - cf) * 0.5))
    cv0 = int(v0 + bh * (1.0 - cf) * 0.5)
    cv1 = max(cv0 + 1, int(v1 - bh * (1.0 - cf) * 0.5))

    patch = depth_map[cv0:cv1, cu0:cu1]
    valid = np.isfinite(patch) & (patch > 0.0)
    if not np.any(valid):
        # Fallback to whole bbox if core has holes
        patch_full = depth_map[int(v0):int(v1), int(u0):int(u1)]
        valid_full = np.isfinite(patch_full) & (patch_full > 0.0)
        if not np.any(valid_full):
            return float("nan")
        return float(np.median(patch_full[valid_full]))

    return float(np.median(patch[valid]))


def bbox_to_goal_rel(
    bbox: Sequence[float],
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    src_shape: Optional[Tuple[int, int]] = None,
    core_frac: float = 0.5,
) -> Optional[np.ndarray]:
    """Back-project 2D bbox + depth map to 4D body-frame goal_rel.

    Coordinate system conventions:
    - Camera frame: X right, Y down, Z forward
    - Body frame (WAM): X forward (Z_cam), Y left (-X_cam), Z up (-Y_cam)

    Returns:
        np.ndarray [d_fwd, d_left, d_up, remaining_dist] in float32, or None if invalid.
    """
    d_target = extract_target_depth(depth_map, bbox, src_shape=src_shape, core_frac=core_frac)
    if not np.isfinite(d_target) or d_target <= 0.0:
        return None

    u0, v0, u1, v1 = bbox
    u_c = (u0 + u1) * 0.5
    v_c = (v0 + v1) * 0.5

    # If bbox was given in different resolution than intrinsics, scale center
    if src_shape is not None:
        src_w, src_h = src_shape
        u_c = (u_c / float(src_w)) * float(intrinsics.width)
        v_c = (v_c / float(src_h)) * float(intrinsics.height)

    # Pinhole back-projection in camera frame
    x_cam = (u_c - intrinsics.cx) * d_target / intrinsics.fx
    y_cam = (v_c - intrinsics.cy) * d_target / intrinsics.fy
    z_cam = d_target

    # Transform to drone body frame: (forward, left, up)
    d_fwd = float(z_cam)
    d_left = float(-x_cam)
    d_up = float(-y_cam)
    dist = float(np.sqrt(d_fwd**2 + d_left**2 + d_up**2))

    return np.array([d_fwd, d_left, d_up, dist], dtype=np.float32)


def project_3d_to_pixel(
    p_body: Sequence[float],
    intrinsics: CameraIntrinsics,
) -> Tuple[float, float, float]:
    """Project 3D body-frame point [d_fwd, d_left, d_up] to pixel (u, v) and depth."""
    d_fwd, d_left, d_up = p_body[:3]
    z_cam = d_fwd
    x_cam = -d_left
    y_cam = -d_up

    if z_cam <= 0:
        return float("nan"), float("nan"), float(z_cam)

    u = intrinsics.fx * (x_cam / z_cam) + intrinsics.cx
    v = intrinsics.fy * (y_cam / z_cam) + intrinsics.cy
    return float(u), float(v), float(z_cam)
