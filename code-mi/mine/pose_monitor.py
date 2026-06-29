#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class PoseMonitor(Node):
    def __init__(self):
        super().__init__('pose_monitor')

        self.declare_parameter('robot_name', 'robot')
        self.declare_parameter('world_frame', 'vodom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('p3d_topic', '/gazebo/p3d/odom')
        self.declare_parameter('prefer_source', 'p3d_odom')  # p3d_odom | model_states | tf

        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.rate_hz = max(1.0, float(self.get_parameter('rate_hz').value))
        self.p3d_topic = self.get_parameter('p3d_topic').get_parameter_value().string_value
        self.prefer_source = self.get_parameter('prefer_source').get_parameter_value().string_value

        self.last_source: str = 'none'
        self.last_pose: Optional[Tuple[float, float, float, float, float, float, float]] = None
        self.last_update_sec: Optional[float] = None

        self.create_subscription(ModelStates, '/gazebo/model_states', self._cb_model_states, 10)
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Odometry, self.p3d_topic, self._cb_odom, odom_qos)
        self.create_subscription(TFMessage, '/tf', self._cb_tf, 50)
        self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f"monitor started: robot_name='{self.robot_name}', tf={self.world_frame}->{self.base_frame}, "
            f"p3d_topic='{self.p3d_topic}', prefer_source='{self.prefer_source}', rate={self.rate_hz:.1f}Hz"
        )

    @staticmethod
    def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def wrap_deg(deg: float) -> float:
        return (deg + 180.0) % 360.0 - 180.0

    def _set_pose(self, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float, source: str) -> None:
        self.last_pose = (x, y, z, qx, qy, qz, qw)
        self.last_source = source
        self.last_update_sec = self.get_clock().now().nanoseconds / 1e9

    def _cb_model_states(self, msg: ModelStates) -> None:
        if self.prefer_source not in ('model_states', 'auto'):
            return
        try:
            i = msg.name.index(self.robot_name)
        except ValueError:
            return
        p = msg.pose[i].position
        q = msg.pose[i].orientation
        self._set_pose(p.x, p.y, p.z, q.x, q.y, q.z, q.w, 'model_states')

    def _cb_tf(self, msg: TFMessage) -> None:
        if self.prefer_source not in ('tf', 'auto'):
            return
        if self.last_source in ('model_states', 'p3d_odom'):
            return
        for t in msg.transforms:
            if t.header.frame_id == self.world_frame and t.child_frame_id == self.base_frame:
                tr = t.transform.translation
                q = t.transform.rotation
                self._set_pose(tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w, 'tf')
                return

    def _cb_odom(self, msg: Odometry) -> None:
        if self.prefer_source not in ('p3d_odom', 'auto'):
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._set_pose(p.x, p.y, p.z, q.x, q.y, q.z, q.w, 'p3d_odom')

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_pose is None or self.last_update_sec is None:
            print("\r[waiting] no pose yet", end='', flush=True)
            return

        x, y, z, qx, qy, qz, qw = self.last_pose
        yaw_deg = self.wrap_deg(math.degrees(self.quat_to_yaw(qx, qy, qz, qw)))
        age = now - self.last_update_sec
        stale = 'stale' if age > 0.5 else 'fresh'

        line = (
            f"[{self.last_source}|{stale}|age={age:.2f}s] "
            f"xyz=({x:.3f}, {y:.3f}, {z:.3f}) yaw={yaw_deg:.1f} deg"
        )
        print(f"\r{line}", end='', flush=True)


def main() -> None:
    rclpy.init()
    node = PoseMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
