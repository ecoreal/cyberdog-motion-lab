#!/usr/bin/env python3
"""
Move forward while keeping centered on a gray-black raised strip.
"""

import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import lcm
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT / "line_follow_hl_example"))
sys.path.append(str(_ROOT / "code_bieren" / "lcm"))
from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from simulator_lcmt import simulator_lcmt


@dataclass
class MotionConfig:
    forward_speed_mps: float = 0.12
    min_speed_mps: float = 0.05
    max_yaw_radps: float = 0.70
    kp: float = 1.05
    kd: float = 0.18
    search_yaw_radps: float = 0.0
    strong_turn_yaw_radps: float = 0.66
    medium_turn_yaw_radps: float = 0.44
    weak_turn_yaw_radps: float = 0.20
    deadband_error: float = 0.06
    medium_error: float = 0.14
    strong_error: float = 0.24
    good_quality_ratio: float = 0.10
    poor_quality_ratio: float = 0.05
    good_pixels: int = 1200
    caution_speed_mps: float = 0.08
    freeze_on_lost_s: float = 0.28
    x_guard_enable: bool = True
    x_guard_deadband_m: float = 0.02
    x_guard_medium_m: float = 0.05
    x_guard_strong_m: float = 0.08
    x_guard_hold_vx_mps: float = 0.03
    x_guard_vy_mps: float = 0.0
    x_guard_wz_radps: float = 0.26
    warmup_straight_s: float = 2.0
    yaw_lock_kp: float = 1.2
    yaw_lock_max_wz: float = 0.45
    x_lock_kp: float = 1.2
    x_lock_max_vy: float = 0.14
    x_to_yaw_k: float = 0.55
    use_vy_for_x_lock: bool = True
    x_lock_deadband_m: float = 0.010
    x_correction_trigger_m: float = 0.020
    x_correction_release_m: float = 0.010
    x_correction_medium_m: float = 0.045
    x_correction_strong_m: float = 0.070
    correction_speed_mps: float = 0.08
    x_vy_sign: float = 1.0
    x_ref_calib_s: float = 0.8
    x_trigger_consecutive: int = 5
    lock_axis: str = "y"
    settle_time_s: float = 0.12
    locomotion_gait_id: int = 26
    goal_x_m: float = 3.05
    goal_y_m: float = 0.622
    goal_yaw_deg: float = 90.0
    x_tol_m: float = 0.020
    y_tol_m: float = 0.020
    yaw_tol_deg: float = 2.0
    phase_vx_mps: float = 0.10
    phase_vy_mps: float = 0.10
    phase_wz_radps: float = 0.45
    y_hold_kp: float = 1.4
    y_hold_max_wz: float = 0.35
    y_ref_m: float = 0.0
    y_lock_kp: float = 1.2
    y_lock_max_vy: float = 0.12


@dataclass
class DetectConfig:
    image_topic: str = "/rgb_camera/image_raw"
    image_topic_alt: str = "/image_raw"
    roi_top_ratio: float = 0.74
    roi_bottom_ratio: float = 1.0
    dark_ratio_threshold: float = 0.08
    min_dark_pixels: int = 220
    dark_s_high: int = 90
    dark_v_high: int = 95
    sensor_wait_timeout_s: float = 15.0


def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class RobotCtrl:
    def __init__(self):
        self.lc_s = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd_msg = robot_control_cmd_lcmt()
        self.send_lock = threading.Lock()
        self.delay_cnt = 0
        self.running = True
        self.life_count = 0
        self.send_thread = threading.Thread(target=self._send_publish, daemon=True)

    def run(self):
        self.send_thread.start()

    def _send_publish(self):
        while self.running:
            with self.send_lock:
                if self.delay_cnt > 20:
                    self.lc_s.publish("robot_control_cmd", self.cmd_msg.encode())
                    self.delay_cnt = 0
                self.delay_cnt += 1
            time.sleep(0.005)

    def _next_life_count(self):
        self.life_count = -128 if self.life_count >= 127 else self.life_count + 1
        return self.life_count

    def send_cmd(self, msg: robot_control_cmd_lcmt):
        with self.send_lock:
            msg.life_count = self._next_life_count()
            self.cmd_msg = msg
            self.delay_cnt = 50

    def to_recovery_stand(self):
        msg = robot_control_cmd_lcmt()
        msg.mode = 12
        msg.gait_id = 0
        msg.duration = 0
        self.send_cmd(msg)

    def to_locomotion(self, gait_id=26):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = gait_id
        msg.duration = 0
        msg.step_height = [0.16, 0.16]
        self.send_cmd(msg)

    def send_locomotion(self, vx: float, vy: float, wz: float, gait_id=26):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = gait_id
        msg.vel_des = [float(vx), float(vy), float(wz)]
        msg.duration = 0
        msg.step_height = [0.16, 0.16]
        self.send_cmd(msg)

    def stop(self):
        self.send_locomotion(0.0, 0.0, 0.0)

    def to_pure_damper(self):
        msg = robot_control_cmd_lcmt()
        msg.mode = 7
        msg.gait_id = 0
        msg.duration = 0
        self.send_cmd(msg)

    def quit(self):
        self.running = False
        self.send_thread.join(timeout=1.0)


class PosMsg:
    def __init__(self, data_lock: threading.Lock):
        self.lc_r = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.rec_msg = simulator_lcmt()
        self.lc_r.subscribe("simulator_state", self.msg_handler)
        self.data_lock = data_lock
        self.position = [0.0, 0.0, 0.0]
        self.rpy_deg = [0.0, 0.0, 0.0]

    def run(self):
        while True:
            self.lc_r.handle()

    def msg_handler(self, _channel, data):
        msg = simulator_lcmt.decode(data)
        with self.data_lock:
            self.position = [float(msg.p[0]), float(msg.p[1]), float(msg.p[2])]
            rpy = [float(v) * 180.0 / np.pi for v in msg.rpy]
            flag = abs(abs(rpy[0]) - 0.0) < abs(abs(rpy[0]) - 180.0)
            if not flag:
                rpy[2] += 180.0
            self.rpy_deg = rpy


class RidgeFollowerNode(Node):
    def __init__(self, cfg: DetectConfig, mcfg: MotionConfig):
        super().__init__("first_ridge_follow")
        self.cfg = cfg
        self.mcfg = mcfg
        self.bridge = CvBridge()
        self.latest_image_bgr: Optional[np.ndarray] = None
        self.last_image_time: float = 0.0
        self.image_subs = []
        self.active_image_topics = []
        self.detected_image_topics = []
        self.detect_ok = False
        self.lost_count = 0
        self.last_error = 0.0
        self.last_ts = time.time()
        self.target_speed = 0.0
        self.target_yaw = 0.0
        self.last_turn_sign = 1.0
        self.last_ratio = 0.0
        self.last_pixels = 0
        self.left_ratio = 0.0
        self.right_ratio = 0.0
        self.left_seen = False
        self.right_seen = False
        self.lost_since = 0.0
        self.camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._subscribe_topics([cfg.image_topic, cfg.image_topic_alt])

    def _subscribe_topics(self, image_topics):
        for sub in self.image_subs:
            self.destroy_subscription(sub)
        self.image_subs = []

        uniq = []
        for topic in image_topics:
            if topic and topic not in uniq:
                uniq.append(topic)

        for topic in uniq:
            self.image_subs.append(
                self.create_subscription(Image, topic, self._image_cb, self.camera_qos)
            )
        self.active_image_topics = uniq

    def autodetect_topics(self):
        graph = self.get_topic_names_and_types()
        image_topics = sorted([name for name, types in graph if "sensor_msgs/msg/Image" in types])
        self.detected_image_topics = image_topics
        if not image_topics:
            return

        preferred = [
            self.cfg.image_topic,
            self.cfg.image_topic_alt,
            "/rgb_camera/image_raw",
            "/image_raw",
            "/camera/image_raw",
        ]
        chosen = []
        for topic in preferred:
            if topic in image_topics and topic not in chosen:
                chosen.append(topic)
        if not chosen:
            chosen.append(image_topics[0])
        if chosen != self.active_image_topics:
            self._subscribe_topics(chosen)
            self.get_logger().warn(f"[AUTO] remap image topics: {chosen}")

    def _image_cb(self, msg: Image):
        try:
            self.latest_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_image_time = time.time()
            self._update_tracking(self.latest_image_bgr)
        except Exception as exc:
            self.get_logger().error(f"image convert error: {exc}")

    def sensor_ready(self) -> bool:
        return self.latest_image_bgr is not None

    def sensor_status_summary(self) -> str:
        return (
            f"image_ok={int(self.latest_image_bgr is not None)} "
            f"image_topics={self.active_image_topics} "
            f"detected_image_topics={self.detected_image_topics}"
        )

    def _dark_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_dark = np.array([0, 0, 0], dtype=np.uint8)
        upper_dark = np.array([180, self.cfg.dark_s_high, self.cfg.dark_v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_dark, upper_dark)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def ridge_underfoot_state(self):
        if self.latest_image_bgr is None:
            return False, 0.0, 0

        img = self.latest_image_bgr
        h, w = img.shape[:2]
        top = int(h * self.cfg.roi_top_ratio)
        bottom = int(h * self.cfg.roi_bottom_ratio)
        if bottom <= top:
            bottom = h
        roi = img[top:bottom, :]
        mask = self._dark_mask(roi)
        dark_pixels = int(np.count_nonzero(mask))
        ratio = float(dark_pixels) / float(mask.size)
        seen = dark_pixels >= self.cfg.min_dark_pixels and ratio >= self.cfg.dark_ratio_threshold
        return seen, ratio, dark_pixels

    def _weighted_center_x(self, binary_img):
        cols = np.sum(binary_img > 0, axis=0).astype(np.float32)
        mass = float(np.sum(cols))
        if mass < float(self.cfg.min_dark_pixels):
            return None
        idx = np.arange(cols.shape[0], dtype=np.float32)
        return float(np.sum(idx * cols) / mass)

    def _update_tracking(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        top = int(h * self.cfg.roi_top_ratio)
        bottom = int(h * self.cfg.roi_bottom_ratio)
        if bottom <= top:
            bottom = h

        roi = frame[top:bottom, :]
        mask = self._dark_mask(roi)
        dark_pixels = int(np.count_nonzero(mask))
        ratio = float(dark_pixels) / float(mask.size) if mask.size > 0 else 0.0
        self.last_pixels = dark_pixels
        self.last_ratio = ratio

        left_mask = mask[:, : w // 2]
        right_mask = mask[:, w // 2 :]
        left_pixels = int(np.count_nonzero(left_mask))
        right_pixels = int(np.count_nonzero(right_mask))
        half_size = float(left_mask.size) if left_mask.size > 0 else 1.0
        self.left_ratio = left_pixels / half_size
        self.right_ratio = right_pixels / half_size
        self.left_seen = left_pixels >= max(60, self.cfg.min_dark_pixels // 5)
        self.right_seen = right_pixels >= max(60, self.cfg.min_dark_pixels // 5)
        left_x = self._weighted_center_x(left_mask)
        right_x = self._weighted_center_x(right_mask)
        if right_x is not None:
            right_x += (w // 2)

        lane_center = None
        lane_width = None
        if left_x is not None and right_x is not None and right_x > left_x:
            lane_center = 0.5 * (left_x + right_x)
            lane_width = right_x - left_x
        elif left_x is not None:
            nominal_width = 0.55 * w
            lane_center = left_x + 0.5 * nominal_width
            lane_width = nominal_width
        elif right_x is not None:
            nominal_width = 0.55 * w
            lane_center = right_x - 0.5 * nominal_width
            lane_width = nominal_width

        if lane_center is None:
            self.detect_ok = False
            self.lost_count += 1
            if self.lost_since <= 0.0:
                self.lost_since = time.time()
            return

        self.detect_ok = True
        self.lost_count = 0
        self.lost_since = 0.0
        error = (w * 0.5 - lane_center) / (w * 0.5)
        now = time.time()
        dt = max(now - self.last_ts, 1e-3)
        d_err = (error - self.last_error) / dt
        pd_yaw = self.mcfg.kp * error + self.mcfg.kd * d_err
        abs_err = abs(error)

        # 学 bieren 的“阈值分段纠偏”：误差越大，给固定更强的转向而非纯线性 PD
        if abs_err >= self.mcfg.strong_error:
            turn_mag = self.mcfg.strong_turn_yaw_radps
        elif abs_err >= self.mcfg.medium_error:
            turn_mag = self.mcfg.medium_turn_yaw_radps
        elif abs_err >= self.mcfg.deadband_error:
            turn_mag = self.mcfg.weak_turn_yaw_radps
        else:
            turn_mag = 0.0

        if error > self.mcfg.deadband_error:
            turn_sign = 1.0
        elif error < -self.mcfg.deadband_error:
            turn_sign = -1.0
        else:
            turn_sign = self.last_turn_sign

        raw_yaw = turn_sign * turn_mag + 0.30 * pd_yaw
        raw_yaw = float(np.clip(raw_yaw, -self.mcfg.max_yaw_radps, self.mcfg.max_yaw_radps))
        self.target_yaw = 0.6 * self.target_yaw + 0.4 * raw_yaw
        self.last_turn_sign = 1.0 if self.target_yaw >= 0.0 else -1.0

        # 按检测质量和误差分档限速，防止石板段高速抖动偏航
        if (
            ratio >= self.mcfg.good_quality_ratio
            and dark_pixels >= self.mcfg.good_pixels
            and abs_err < self.mcfg.medium_error
        ):
            speed_scale = 1.0
        elif ratio < self.mcfg.poor_quality_ratio or abs_err >= self.mcfg.strong_error:
            speed_scale = 0.55
        else:
            speed_scale = 0.75
        self.target_speed = max(self.mcfg.min_speed_mps, self.mcfg.forward_speed_mps * speed_scale)

        # 单边边缘丢失时，执行更强的“回拉”策略，避免继续外飘
        if self.left_seen and not self.right_seen:
            self.target_yaw = max(self.target_yaw, self.mcfg.medium_turn_yaw_radps)
            self.target_speed = min(self.target_speed, self.mcfg.caution_speed_mps)
        elif self.right_seen and not self.left_seen:
            self.target_yaw = min(self.target_yaw, -self.mcfg.medium_turn_yaw_radps)
            self.target_speed = min(self.target_speed, self.mcfg.caution_speed_mps)
        self.last_error = error
        self.last_ts = now

    def current_cmd(self):
        if self.detect_ok:
            return self.target_speed, self.target_yaw
        if self.lost_count < 8:
            # 短时丢线：低速直行
            return self.mcfg.min_speed_mps, 0.0
        # 长时低置信度时短暂停走，避免“看不清还前冲”导致越偏越远
        if self.lost_since > 0.0 and (time.time() - self.lost_since) < self.mcfg.freeze_on_lost_s:
            return 0.0, 0.0
        # 长时丢线：继续低速直行，避免原地打转
        return self.mcfg.min_speed_mps, 0.0


def main():
    rclpy.init()

    mcfg = MotionConfig()
    dcfg = DetectConfig()

    node = RidgeFollowerNode(dcfg, mcfg)
    spin_thread_running = {"run": True}

    def _spin():
        while rclpy.ok() and spin_thread_running["run"]:
            rclpy.spin_once(node, timeout_sec=0.05)

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    ctrl = RobotCtrl()
    ctrl.run()
    state_lock = threading.Lock()
    pos_msg = PosMsg(state_lock)
    pos_thread = threading.Thread(target=pos_msg.run, daemon=True)
    pos_thread.start()

    stop_flag = {"stop": False}

    def _handle_sigint(_sig, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        t0 = time.time()
        while not node.sensor_ready() and (time.time() - t0) < 1.5:
            node.autodetect_topics()
            time.sleep(0.1)

        ctrl.to_recovery_stand()
        time.sleep(1.5)
        ctrl.to_locomotion(gait_id=mcfg.locomotion_gait_id)
        time.sleep(0.6)

        node.get_logger().info("[MISSION] phase1: x->3.05, phase2: yaw->90deg, phase3: y->0.622")
        phase = "GO_X"
        last_abs_x_err = None
        last_abs_yaw_err = None
        last_abs_y_err = None
        try:
            motion_t0 = time.time()
            while not stop_flag["stop"]:
                vx = 0.0
                vy = 0.0
                wz = 0.0
                if (time.time() - motion_t0) < mcfg.warmup_straight_s:
                    ctrl.send_locomotion(mcfg.min_speed_mps, 0.0, 0.0, gait_id=mcfg.locomotion_gait_id)
                    time.sleep(0.05)
                    continue
                with state_lock:
                    x_now = float(pos_msg.position[0])
                    y_now = float(pos_msg.position[1])
                    yaw_now = float(pos_msg.rpy_deg[2])
                yaw_now = wrap_to_180(yaw_now)
                z_now = float(pos_msg.position[2])

                if phase == "GO_X":
                    x_err = mcfg.goal_x_m - x_now
                    abs_x_err = abs(x_err)
                    if abs_x_err <= mcfg.x_tol_m:
                        phase = "TURN_YAW"
                        node.get_logger().info(f"[PHASE] GO_X done at x={x_now:.3f}, switch TURN_YAW")
                        ctrl.stop()
                        time.sleep(0.15)
                        last_abs_x_err = None
                        continue
                    x_sign = 1.0 if x_err >= 0.0 else -1.0
                    if last_abs_x_err is not None and abs_x_err > last_abs_x_err + 0.003:
                        x_sign = -x_sign
                    vx = x_sign * mcfg.phase_vx_mps
                    # While heading ~ +x, keep world y near 0 by body-frame right/left shift.
                    y_err = mcfg.y_ref_m - y_now
                    vy = float(np.clip(mcfg.y_lock_kp * y_err, -mcfg.y_lock_max_vy, mcfg.y_lock_max_vy))
                    last_abs_x_err = abs_x_err
                    print(
                        f"pianli={y_err:.4f} phase={phase} "
                        f"dog_xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                        f"cmd(vx,vy,wz)=({vx:.3f},{vy:.3f},{wz:.3f})"
                    )

                elif phase == "TURN_YAW":
                    yaw_err_deg = wrap_to_180(mcfg.goal_yaw_deg - yaw_now)
                    abs_yaw_err = abs(yaw_err_deg)
                    if abs_yaw_err <= mcfg.yaw_tol_deg:
                        phase = "GO_Y"
                        node.get_logger().info(f"[PHASE] TURN_YAW done at yaw={yaw_now:.2f}, switch GO_Y")
                        ctrl.stop()
                        time.sleep(0.15)
                        last_abs_yaw_err = None
                        continue
                    yaw_sign = 1.0 if yaw_err_deg >= 0.0 else -1.0
                    if last_abs_yaw_err is not None and abs_yaw_err > last_abs_yaw_err + 0.5:
                        yaw_sign = -yaw_sign
                    wz = yaw_sign * mcfg.phase_wz_radps
                    last_abs_yaw_err = abs_yaw_err
                    y_err = mcfg.y_ref_m - y_now
                    print(
                        f"pianli={y_err:.4f} phase={phase} "
                        f"dog_xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                        f"cmd(vx,vy,wz)=({vx:.3f},{vy:.3f},{wz:.3f})"
                    )

                elif phase == "GO_Y":
                    y_err = mcfg.goal_y_m - y_now
                    abs_y_err = abs(y_err)
                    if abs_y_err <= mcfg.y_tol_m:
                        node.get_logger().info(
                            f"[DONE] reached target: x={x_now:.3f}, y={y_now:.3f}, yaw={yaw_now:.2f}"
                        )
                        ctrl.stop()
                        break
                    y_sign = 1.0 if y_err >= 0.0 else -1.0
                    if last_abs_y_err is not None and abs_y_err > last_abs_y_err + 0.003:
                        y_sign = -y_sign
                    # After turning to +y, use forward vx to increase/decrease world y.
                    vx = y_sign * mcfg.phase_vx_mps
                    vy = 0.0
                    yaw_err_deg = wrap_to_180(mcfg.goal_yaw_deg - yaw_now)
                    wz = float(
                        np.clip(
                            np.deg2rad(yaw_err_deg) * mcfg.y_hold_kp,
                            -mcfg.y_hold_max_wz,
                            mcfg.y_hold_max_wz,
                        )
                    )
                    last_abs_y_err = abs_y_err
                    print(
                        f"pianli={y_err:.4f} phase={phase} "
                        f"dog_xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                        f"cmd(vx,vy,wz)=({vx:.3f},{vy:.3f},{wz:.3f})"
                    )
                ctrl.send_locomotion(vx, vy, wz, gait_id=mcfg.locomotion_gait_id)
                time.sleep(0.05)
        finally:
            ctrl.stop()
            time.sleep(mcfg.settle_time_s)

    finally:
        ctrl.stop()
        time.sleep(0.3)
        ctrl.to_recovery_stand()
        time.sleep(0.6)
        ctrl.to_pure_damper()
        ctrl.quit()

        spin_thread_running["run"] = False
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
