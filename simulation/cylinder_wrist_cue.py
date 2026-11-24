"""Localize the existing color marks on a cylinder resting on the calibrated tray."""
import hashlib

import cv2
import mujoco
import numpy as np

from cylinder_reader import components
from inspect_cylinder import CAPACITY_ML, SCALE_SPAN, SCALE_ZERO


WRIST_CAMERA = 'right_wrist_view'
WRIST_RESOLUTION = 1024
MAX_MARKER_GAP_M = .004
MARKER_OFFSET_XY_M = np.array([-.006, -.0126])


def pixel_on_plane(uv, shape, origin, axes, focal, elevation):
    height, width = shape
    x, y = uv
    ray = axes @ np.array([(x+.5-width*.5)/focal, -(y+.5-height*.5)/focal, -1.])
    if abs(ray[2]) < 1e-5:
        return None
    distance = (elevation - origin[2]) / ray[2]
    return (origin + distance * ray)[:2] if 0 < distance < 2.5 else None


def estimate_wrist_location(rgb, origin, axes, fovy, scene):
    """Estimate initial world XY from RGB and camera/tray calibration."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.moveaxis(hsv, 2, 0)
    pink, _ = components((hue >= 140) & (hue <= 175) & (saturation > 90) & (value > 90))
    orange, _ = components((hue >= 5) & (hue <= 22) & (saturation > 120) & (value > 100))
    focal = (rgb.shape[0] / 2) / np.tan(np.deg2rad(fovy) / 2)
    tray = np.asarray(scene['target_tray']['center_xy'])
    half_size = np.asarray(scene['target_tray']['size_xy']) / 2
    base = scene['table']['height'] + .016 + SCALE_ZERO
    candidates = []
    for _, pink_stats, pink_uv in pink:
        if pink_stats[2] < 2:
            continue
        low = pixel_on_plane(pink_uv, hue.shape, origin, axes, focal, base + 20/CAPACITY_ML*SCALE_SPAN)
        if low is None:
            continue
        for _, amber_stats, amber_uv in orange:
            if amber_stats[2] < 2 or np.linalg.norm(pink_uv - amber_uv) < 7:
                continue
            high = pixel_on_plane(amber_uv, hue.shape, origin, axes, focal, base + 35/CAPACITY_ML*SCALE_SPAN)
            if high is None:
                continue
            gap = float(np.linalg.norm(low - high))
            center = (low + high) / 2 - MARKER_OFFSET_XY_M
            if gap <= MAX_MARKER_GAP_M and np.all(abs(center - tray) <= half_size):
                candidates.append((gap, center))
    return min(candidates, key=lambda item: item[0]) if candidates else None


def cue_from_wrist_rgb(env, rgb):
    """Apply the estimator using only the simulated robot camera calibration."""
    if rgb.shape != (WRIST_RESOLUTION, WRIST_RESOLUTION, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected 1024px uint8 wrist RGB')
    index = env.model.camera(WRIST_CAMERA).id
    location = estimate_wrist_location(rgb, env.data.cam_xpos[index],
                                      env.data.cam_xmat[index].reshape(3, 3),
                                      float(env.model.cam_fovy[index]), env.config)
    return dict(camera=WRIST_CAMERA, resolution=WRIST_RESOLUTION,
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                marker_pair_gap_m=None if location is None else location[0],
                world_xy_m=None if location is None else location[1].tolist())


def capture_initial_wrist_cue(env):
    """Take one RGB measurement at reset while preserving normal image rendering."""
    view = env.model.vis.global_
    width, height = view.offwidth, view.offheight
    view.offwidth = view.offheight = WRIST_RESOLUTION
    try:
        with mujoco.Renderer(env.model, width=WRIST_RESOLUTION, height=WRIST_RESOLUTION) as renderer:
            renderer.update_scene(env.data, camera=WRIST_CAMERA)
            rgb = renderer.render()
    finally:
        view.offwidth, view.offheight = width, height
    return rgb, cue_from_wrist_rgb(env, rgb)
