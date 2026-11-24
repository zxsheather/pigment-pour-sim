"""Explicit SE(3) mappings; metres, radians, quaternion xyzw."""
import numpy as np
from scipy.spatial.transform import Rotation


def finite(value, shape):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'Expected finite array with shape {shape}')
    return value


def transform(value):
    value = finite(value, (4, 4))
    r = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-6)):
        raise ValueError('Expected a proper rigid transform')
    return value


def from_xyz_quat(value):
    value = finite(value, (7,))
    norm = np.linalg.norm(value[3:])
    if abs(norm - 1) > .01:
        raise ValueError('Invalid SDK quaternion; no identity-rotation fallback')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(value[3:] / norm).as_matrix()
    result[:3, 3] = value[:3]
    return result


def from_model_pose(value):
    value = finite(value, (7,))
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', value[3:6]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def to_model_pose(matrix, gripper):
    matrix = transform(matrix)
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz'), gripper]


def model_from_sdk(sdk_matrix, arm):
    return (transform(arm['T_model_reference_from_sdk_reference']) @ transform(sdk_matrix)
            @ transform(arm['T_sdk_tool_from_model_tcp']))


def sdk_from_model(model_matrix, arm):
    return (np.linalg.inv(transform(arm['T_model_reference_from_sdk_reference']))
            @ transform(model_matrix) @ np.linalg.inv(transform(arm['T_sdk_tool_from_model_tcp'])))


def grip_from_sdk(value, arm):
    scale, offset = finite([arm['gripper_model_per_sdk'], arm['gripper_model_offset']], (2,))
    if abs(scale) < 1e-8:
        raise ValueError('Gripper scale must be nonzero')
    return float(scale * value + offset)


def grip_to_sdk(value, arm):
    scale, offset = finite([arm['gripper_model_per_sdk'], arm['gripper_model_offset']], (2,))
    if abs(scale) < 1e-8:
        raise ValueError('Gripper scale must be nonzero')
    return float((value - offset) / scale)


def pose_distance(a, b):
    a, b = transform(a), transform(b)
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))
