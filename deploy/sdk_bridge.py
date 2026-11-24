"""Direct official SDK reads and right-arm writes, without SDK sample startup actions.

API reference: sdk_robot 63610199bbd70eafd1fc087a015ec5c74a0c906c.
No mode, navigation, base, lift, head or left-arm commands in this adapter.
"""
from io import BytesIO
import time

import numpy as np
from PIL import Image

from deploy.geometry import finite, from_xyz_quat, transform


def sdk_pose(value):
    if hasattr(value, 'pose'):
        value = value.pose
    if hasattr(value, 'position'):
        p, q = value.position, value.orientation
        value = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    return from_xyz_quat(value)


def scalar(value):
    value = float(value.position if hasattr(value, 'position') else value)
    if not np.isfinite(value):
        raise ValueError('Nonfinite SDK scalar')
    return value


def image_rgb(value, array_order):
    if isinstance(value, np.ndarray):
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError('Unsupported SDK image array')
        if array_order not in ('RGB', 'BGR'):
            raise ValueError('camera_array_order must be RGB or BGR')
        return value.copy() if array_order == 'RGB' else value[..., ::-1].copy()
    payload = value.data if hasattr(value, 'data') else value
    with Image.open(BytesIO(bytes(payload))) as img:
        return np.asarray(img.convert('RGB')).copy()


class SDKBridge:
    def __init__(self, robot, array_order='RGB', rpc_timeout=1.0):
        self.robot = robot
        self.array_order = array_order
        self.rpc_timeout = rpc_timeout

    @classmethod
    def connect(cls, endpoint, array_order='RGB'):
        if not endpoint.startswith('x2://') or 'ROBOT_IP' in endpoint:
            raise ValueError('Fill a real x2:// endpoint in your local configuration')
        from x2robot import connect
        return cls(connect(endpoint), array_order)

    def read(self, cameras=True):
        started = time.monotonic()
        robot = self.robot
        raw = {}
        for side in ['left', 'right']:
            raw[side] = {'T_sdk_reference_from_sdk_tool': sdk_pose(getattr(robot, side + '_arm').get_end_pose(timeout=self.rpc_timeout)),
                         'gripper_sdk': scalar(getattr(robot, side + '_gripper').get_position(timeout=self.rpc_timeout))}
        head = robot.head.get_pose(timeout=self.rpc_timeout)
        raw['head_pitch_yaw'] = finite([head.pitch, head.yaw], (2,))
        raw['lift_sdk'] = scalar(robot.lift.get_lift_position(timeout=self.rpc_timeout))
        chassis = robot.chassis.get_relative_position(timeout=self.rpc_timeout)
        raw['chassis_xy_yaw'] = finite([chassis.x, chassis.y, chassis.yaw], (3,))
        twist = robot.chassis.get_odometry(timeout=self.rpc_timeout).twist.twist
        raw['chassis_velocity'] = finite([twist.linear.x, twist.linear.y, twist.angular.z], (3,))
        if cameras:
            raw['views'] = {'face_view': image_rgb(robot.head_camera.get_rgb_image(timeout=self.rpc_timeout), self.array_order),
                            'right_wrist_view': image_rgb(robot.right_arm_camera.get_raw_image(timeout=self.rpc_timeout), self.array_order)}
        raw['read_started_monotonic'] = started
        raw['read_finished_monotonic'] = time.monotonic()
        return raw

    def write_right(self, matrix, gripper_sdk=None):
        from scipy.spatial.transform import Rotation
        from x2robot.geometry_msgs import Point, Pose, Quaternion
        from x2robot.sdk import GripperPosition
        matrix = transform(matrix)
        x, y, z = (float(v) for v in matrix[:3, 3])
        qx, qy, qz, qw = (float(v) for v in Rotation.from_matrix(matrix[:3, :3]).as_quat())
        result = self.robot.right_arm.set_end_pose(Pose(position=Point(x=x, y=y, z=z),
                                               orientation=Quaternion(x=qx, y=qy, z=qz, w=qw)), timeout=self.rpc_timeout)
        if getattr(result, 'is_success', None) is not True:
            raise RuntimeError(f'Right arm command rejected: {result}')
        if gripper_sdk is not None:
            result = self.robot.right_gripper.set_position(GripperPosition(position=float(gripper_sdk)), timeout=self.rpc_timeout)
            if getattr(result, 'is_success', None) is not True:
                raise RuntimeError(f'Right gripper command rejected after arm target was accepted: {result}')
