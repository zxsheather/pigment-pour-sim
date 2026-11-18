"""Estimate the upright cylinder's position from calibrated robot RGB."""

import itertools

import cv2
import numpy as np

from cylinder_joint_actions import require, with_joint_wrist_cue_state
from cylinder_reader import components
from cylinder_wrist_cue import MARKER_OFFSET_XY_M, capture_initial_wrist_cue, cue_from_wrist_rgb, pixel_on_plane
from inspect_cylinder import CAPACITY_ML, SCALE_SPAN, SCALE_ZERO


TRAY_VISION_CAMERAS = ('face_view', 'right_wrist_view')
TRAY_VISION_IMAGE_SIZE = 768
TRAY_CUE_PROTOCOL = 'blue_bearings_and_scale_at_reset_v1'
TRAY_CENTER_CUE_PROTOCOL = 'blue_bearings_and_scale_from_tray_center_v1'
TRAY_CUE_PROTOCOLS = (TRAY_CUE_PROTOCOL, TRAY_CENTER_CUE_PROTOCOL)
TRAY_CENTER_CUE_NAMES = (
    'initial_cylinder_from_tray_center_forward_m',
    'initial_cylinder_from_tray_center_left_m',
    'initial_rgb_detection_valid',
)
TRAY_CENTER_CUE_RADIUS_M = .04
TRAY_FALLBACK_VIEW = 'right_wrist_view_1024'


def with_tray_cue_state(servo_state, estimate, tray_center_xy, protocol):
    require(protocol in TRAY_CUE_PROTOCOLS, 'Unknown tray cue protocol')
    origin = tray_center_xy if protocol == TRAY_CENTER_CUE_PROTOCOL else None
    require(protocol != TRAY_CENTER_CUE_PROTOCOL or origin is not None, 'Tray center calibration required')
    xy = None if estimate is None else estimate['world_xy_m']
    return with_joint_wrist_cue_state(servo_state, xy, origin_world_xy=origin)


def liquid_direction(mask, origin, axes, focal, bottom_z, on_tray):
    blobs, labeled = components(mask)
    height, width = mask.shape
    for label, stats, _ in sorted(blobs, key=lambda blob: blob[1][4], reverse=True):
        x, y, w, h, area = stats
        if w < 3 or h < 4 or area < 12 or x <= 1 or x + w >= width - 1:
            continue
        rough = pixel_on_plane((x + (w - 1) / 2, y + h - 1), mask.shape,
                               origin, axes, focal, bottom_z)
        if rough is None or not on_tray(rough):
            continue
        # In the horizontal plane, rays through the vertical liquid axis have
        # the same direction at every height. Center the blue span on each row
        # before computing that direction, so perspective and height can vary.
        yy, xx = np.nonzero(labeled[y:y + h, x:x + w] == label)
        left, right = np.full(h, w), np.full(h, -1)
        np.minimum.at(left, yy, xx)
        np.maximum.at(right, yy, xx)
        valid = np.flatnonzero(right - left >= 2)
        if len(valid) < 3:
            continue
        uv = np.stack((x + (left[valid] + right[valid]) / 2, y + valid), axis=-1)
        rays = np.column_stack(((uv[:, 0] + .5 - width / 2) / focal,
                                -(uv[:, 1] + .5 - height / 2) / focal,
                                -np.ones(len(uv)))) @ axes.T
        horizontal = rays[:, :2]
        length = np.linalg.norm(horizontal, axis=-1)
        if np.count_nonzero(length > 1e-8) < 3:
            continue
        horizontal = horizontal[length > 1e-8] / length[length > 1e-8, None]
        direction = np.median(horizontal, axis=0)
        if np.linalg.norm(direction) > 1e-8:
            return direction / np.linalg.norm(direction)
    return None


def colored_anchors(hsv, origin, axes, focal, bottom_z, on_tray):
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    magenta = (hue >= 140) & (hue <= 175) & (saturation > 90) & (value > 90)
    orange = (hue >= 5) & (hue <= 22) & (saturation > 120) & (value > 100)
    for mark, mask in ((20, magenta), (35, orange)):
        _, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for stat, uv in zip(stats[1:], centers[1:], strict=True):
            if not 2 <= stat[4] <= 100:
                continue
            xy = pixel_on_plane(uv, hue.shape, origin, axes, focal, bottom_z + mark / CAPACITY_ML * SCALE_SPAN)
            if xy is not None:
                xy = xy - MARKER_OFFSET_XY_M
                if on_tray(xy):
                    yield mark, xy


def distance_to_bearing(origin, direction, xy):
    delta = xy - origin
    return abs(delta[0] * direction[1] - delta[1] * direction[0]) if delta @ direction > 0 else np.inf


def estimate_tray_cylinder(images, calibrations, scene):
    """Use blue liquid bearings and ruler marks of known height, without fill volume."""
    tray = np.asarray(scene['target_tray']['center_xy'])
    half = np.asarray(scene['target_tray']['size_xy']) / 2
    bottom_z = scene['table']['height'] + .016 + SCALE_ZERO

    def on_tray(xy):
        return np.all(np.abs(xy - tray) <= half)

    bearings, anchors = {}, []
    for camera in TRAY_VISION_CAMERAS:
        rgb = images[camera]
        if rgb.dtype != np.uint8 or rgb.shape != (TRAY_VISION_IMAGE_SIZE, TRAY_VISION_IMAGE_SIZE, 3):
            raise ValueError('Expected 768px uint8 RGB from the head and right wrist')
        origin, axes, fovy = calibrations[camera]
        focal = rgb.shape[0] / (2 * np.tan(np.deg2rad(fovy) / 2))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = np.moveaxis(hsv, -1, 0)
        blue = (hue >= 95) & (hue <= 135) & (saturation > 90) & (value > 60)
        direction = liquid_direction(blue, origin, axes, focal, bottom_z, on_tray)
        if direction is not None:
            bearings[camera] = (origin[:2], direction)
        anchors.extend(colored_anchors(hsv, origin, axes, focal, bottom_z, on_tray))
    if not bearings:
        return None

    triangulated = None
    if len(bearings) == 2:
        (front, a), (wrist, b) = bearings.values()
        pair = np.column_stack((a, -b))
        if abs(np.linalg.det(pair)) >= .10:
            alpha, beta = np.linalg.solve(pair, wrist - front)
            xy = front + alpha * a
            if min(alpha, beta) > 0 and on_tray(xy):
                triangulated = xy

    compatible = [(mark, xy) for mark, xy in anchors
                  if all(distance_to_bearing(origin, direction, xy) <= .004
                         for origin, direction in bearings.values())]
    pairs = []
    for (level, first), (other_level, second) in itertools.combinations(compatible, 2):
        gap = np.linalg.norm(first - second)
        center = (first + second) / 2
        if level != other_level and gap <= .005:
            if triangulated is None or np.linalg.norm(center - triangulated) <= .008:
                pairs.append((gap, center))
    if not pairs:
        return None if triangulated is None else dict(world_xy_m=triangulated.tolist(), method='blue_bearings')

    _, estimate = min(pairs, key=lambda item: item[0])
    head_origin, head_direction = bearings.get('face_view', next(iter(bearings.values())))
    estimate = head_origin + head_direction * ((estimate - head_origin) @ head_direction)
    if not on_tray(estimate):
        return None if triangulated is None else dict(world_xy_m=triangulated.tolist(), method='blue_bearings')
    if triangulated is None:
        return dict(world_xy_m=estimate.tolist(), method='single_blue_bearing_and_scale_pair')
    return dict(world_xy_m=((estimate + triangulated) / 2).tolist(), method='blue_bearings_and_scale_pair')


def estimate_from_robot_views(env, images):
    """Read robot-mounted camera extrinsics and the static tray calibration."""
    calibrations = {}
    for camera in TRAY_VISION_CAMERAS:
        index = env.model.camera(camera).id
        calibrations[camera] = (env.data.cam_xpos[index], env.data.cam_xmat[index].reshape(3, 3),
                                env.model.cam_fovy[index])
    estimate = estimate_tray_cylinder(images, calibrations, env.config)
    if estimate is None and TRAY_FALLBACK_VIEW in images:
        previous = cue_from_wrist_rgb(env, images[TRAY_FALLBACK_VIEW])
        if previous['world_xy_m'] is not None:
            return dict(world_xy_m=previous['world_xy_m'], method='wrist_scale_pair_fallback')
    return estimate


def capture_initial_tray_cue(env):
    """Capture reset RGB and use the high-resolution wrist marks if bearings are ambiguous."""
    images = {camera: env.render(camera, width=TRAY_VISION_IMAGE_SIZE) for camera in TRAY_VISION_CAMERAS}
    estimate = estimate_from_robot_views(env, images)
    if estimate is None:
        images[TRAY_FALLBACK_VIEW], _ = capture_initial_wrist_cue(env)
        estimate = estimate_from_robot_views(env, images)
    return images, estimate
