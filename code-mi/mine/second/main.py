#!/usr/bin/env python3
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import cv2
import lcm
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT / "code_bieren" / "lcm"))
from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from simulator_lcmt import simulator_lcmt


@dataclass
class MotionConfig:
    goal_y1_m: float = 0.81
    goal_x2_m: float = 2.655
    goal_y3_m: float = 1.346
    goal_yaw_scan_deg: float = 0.0
    goal_yaw3_pre_deg: float = -0.3
    goal_yaw4_deg: float = -179.4
    scan_y_targets_m: tuple = (2.160, 3.042, 3.920)
    scan_y_targets_back_m: tuple = (3.920, 3.042, 2.160, 1.345)
    scan3_y_targets_m: tuple = (1.346, 2.160, 3.042, 3.920)
    scan3_y_targets_back_m: tuple = (3.920, 3.042, 2.160, 1.346)
    goal_y_left_after_back_m: float = 0.900
    goal_x_final_m: float = 0.196
    final_x_tol_m: float = 0.005
    goal_y_right_final_m: float = 4.350
    goal_x_after_right_m: float = -0.302
    goal_yaw_end_deg: float = 89.8

    y_tol_m: float = 0.020
    x_tol_m: float = 0.020
    yaw_tol_deg: float = 2.0

    phase_vx_mps: float = 0.15
    phase_vy_mps: float = 0.15
    phase_wz_radps: float = 0.45

    yaw_hold_kp: float = 1.4
    yaw_hold_max_wz: float = 0.35
    x_hold_kp: float = 1.0
    x_hold_max_vx: float = 0.05

    warmup_straight_s: float = 1.0
    settle_time_s: float = 0.12
    locomotion_gait_id: int = 26

    hit_forward_m: float = 0.432
    hit_retreat_m: float = 0.432
    hit_speed_mps: float = 0.20
    hit_back_forward_m: float = 0.60
    hit_back_retreat_m: float = 0.60

    orange_h1_low: int = 5
    orange_h1_high: int = 24
    orange_h2_low: int = 170
    orange_h2_high: int = 179
    orange_s_low: int = 90
    orange_v_low: int = 80
    orange_ratio_threshold: float = 0.04
    center_crop_ratio: float = 0.60
    scan_confirm_time_s: float = 0.8
    scan_confirm_dt_s: float = 0.08


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


class OrangeVision(Node):
    def __init__(self):
        super().__init__("second_orange_vision")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.bridge = CvBridge()
        self.latest_image_bgr: Optional[np.ndarray] = None
        self.create_subscription(Image, "/rgb_camera/image_raw", self._image_cb, qos)

    def _image_cb(self, msg: Image):
        try:
            self.latest_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.latest_image_bgr = None

    def is_orange_front(self, cfg: MotionConfig) -> tuple:
        if self.latest_image_bgr is None:
            return False, 0.0
        img = self.latest_image_bgr
        h, w = img.shape[:2]
        cr = min(max(cfg.center_crop_ratio, 0.1), 1.0)
        cw = int(w * cr)
        ch = int(h * cr)
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        roi = img[y0:y0 + ch, x0:x0 + cw]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        low1 = np.array([cfg.orange_h1_low, cfg.orange_s_low, cfg.orange_v_low], dtype=np.uint8)
        high1 = np.array([cfg.orange_h1_high, 255, 255], dtype=np.uint8)
        low2 = np.array([cfg.orange_h2_low, cfg.orange_s_low, cfg.orange_v_low], dtype=np.uint8)
        high2 = np.array([cfg.orange_h2_high, 255, 255], dtype=np.uint8)
        m1 = cv2.inRange(hsv, low1, high1)
        m2 = cv2.inRange(hsv, low2, high2)
        mask = cv2.bitwise_or(m1, m2)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return ratio >= cfg.orange_ratio_threshold, ratio


def move_distance(ctrl: RobotCtrl, vx: float, vy: float, wz: float, dist_m: float, speed_mps: float, gait_id: int):
    t = max(0.0, dist_m / max(1e-6, speed_mps))
    t0 = time.time()
    while time.time() - t0 < t:
        ctrl.send_locomotion(vx, vy, wz, gait_id=gait_id)
        time.sleep(0.05)
    ctrl.stop()
    time.sleep(0.12)


def confirm_orange(vision: OrangeVision, cfg: MotionConfig) -> tuple:
    t0 = time.time()
    best_ratio = 0.0
    while time.time() - t0 < cfg.scan_confirm_time_s:
        is_orange, ratio = vision.is_orange_front(cfg)
        best_ratio = max(best_ratio, ratio)
        if is_orange:
            return True, ratio, best_ratio
        time.sleep(cfg.scan_confirm_dt_s)
    return False, best_ratio, best_ratio


def main():
    mcfg = MotionConfig()
    rclpy.init()
    vision = OrangeVision()
    vision_thread_running = {"ok": True}

    def _vision_spin():
        while vision_thread_running["ok"]:
            rclpy.spin_once(vision, timeout_sec=0.05)

    vision_thread = threading.Thread(target=_vision_spin, daemon=True)
    vision_thread.start()

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
        ctrl.to_recovery_stand()
        time.sleep(1.2)
        ctrl.to_locomotion(gait_id=mcfg.locomotion_gait_id)
        time.sleep(0.6)

        phase = "GO_Y1"
        last_abs_err = None
        motion_t0 = time.time()
        scan_index = 0
        back_scan_index = 0
        scan3_index = 0
        scan3_back_index = 0

        while not stop_flag["stop"]:
            vx = 0.0
            vy = 0.0
            wz = 0.0

            if (time.time() - motion_t0) < mcfg.warmup_straight_s:
                ctrl.send_locomotion(0.05, 0.0, 0.0, gait_id=mcfg.locomotion_gait_id)
                time.sleep(0.05)
                continue

            with state_lock:
                x_now = float(pos_msg.position[0])
                y_now = float(pos_msg.position[1])
                z_now = float(pos_msg.position[2])
                yaw_now = wrap_to_180(float(pos_msg.rpy_deg[2]))

            if phase == "GO_Y1":
                err = mcfg.goal_y1_m - y_now
                abs_err = abs(err)
                if abs_err <= mcfg.y_tol_m:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = "GO_X2"
                    last_abs_err = None
                    print(f"[PHASE] GO_Y1 done at y={y_now:.3f} -> GO_X2")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.003:
                    sign = -sign
                # Facing +y (yaw~90): forward axis is body vx, which changes world y.
                vx = sign * mcfg.phase_vx_mps
                yaw_err = wrap_to_180(90.0 - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * mcfg.yaw_hold_kp, -mcfg.yaw_hold_max_wz, mcfg.yaw_hold_max_wz))
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "GO_X2":
                err = mcfg.goal_x2_m - x_now
                abs_err = abs(err)
                if abs_err <= mcfg.x_tol_m:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = "GO_Y3"
                    last_abs_err = None
                    print(f"[PHASE] GO_X2 done at x={x_now:.3f} -> GO_Y3")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.003:
                    sign = -sign
                # Facing +y (yaw~90): lateral axis is body vy.
                # Need x decrease when moving left, so invert sign mapping here.
                vy = -sign * mcfg.phase_vy_mps
                yaw_err = wrap_to_180(90.0 - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * mcfg.yaw_hold_kp, -mcfg.yaw_hold_max_wz, mcfg.yaw_hold_max_wz))
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "GO_Y3":
                err = mcfg.goal_y3_m - y_now
                abs_err = abs(err)
                if abs_err <= mcfg.y_tol_m:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = "TURN_YAW0_PRE"
                    last_abs_err = None
                    print(f"[PHASE] GO_Y3 done at y={y_now:.3f} -> TURN_YAW0_PRE")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.003:
                    sign = -sign
                vx = sign * mcfg.phase_vx_mps
                yaw_err = wrap_to_180(90.0 - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * mcfg.yaw_hold_kp, -mcfg.yaw_hold_max_wz, mcfg.yaw_hold_max_wz))
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "TURN_YAW0_PRE":
                err = wrap_to_180(mcfg.goal_yaw_scan_deg - yaw_now)
                abs_err = abs(err)
                if abs_err <= mcfg.yaw_tol_deg:
                    ctrl.stop()
                    phase = "SCAN_1"
                    last_abs_err = None
                    print(f"[PHASE] TURN_YAW0_PRE done at yaw={yaw_now:.2f} -> SCAN_1")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.5:
                    sign = -sign
                wz = sign * mcfg.phase_wz_radps
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase.startswith("SCAN_") and (not phase.startswith("SCAN_BACK_")):
                is_orange, ratio = vision.is_orange_front(mcfg)
                print(f"[CHECK] {phase} y={y_now:.3f} orange={is_orange} ratio={ratio:.4f}")
                if is_orange:
                    print(f"[HIT] {phase} orange confirmed, forward+retreat")
                    move_distance(
                        ctrl, mcfg.hit_speed_mps, 0.0, 0.0,
                        mcfg.hit_forward_m, mcfg.hit_speed_mps, mcfg.locomotion_gait_id
                    )
                    move_distance(
                        ctrl, -mcfg.hit_speed_mps, 0.0, 0.0,
                        mcfg.hit_retreat_m, mcfg.hit_speed_mps, mcfg.locomotion_gait_id
                    )
                if scan_index < len(mcfg.scan_y_targets_m):
                    phase = "GO_SCAN_Y"
                    last_abs_err = None
                else:
                    phase = "TURN_YAW_FINAL"
                    last_abs_err = None
                continue

            elif phase == "GO_SCAN_Y":
                target_y = mcfg.scan_y_targets_m[scan_index]
                err = target_y - y_now
                abs_err = abs(err)
                if abs_err <= mcfg.y_tol_m:
                    ctrl.stop()
                    time.sleep(0.15)
                    scan_index += 1
                    phase = f"SCAN_{scan_index + 1}"
                    print(f"[PHASE] GO_SCAN_Y done at y={y_now:.3f} -> {phase}")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.003:
                    sign = -sign
                # After turning to yaw~0, moving left/right along world y is body vy.
                vy = sign * mcfg.phase_vy_mps
                # Keep x near 2.655 while moving left.
                x_err = mcfg.goal_x2_m - x_now
                vx = float(np.clip(x_err * mcfg.x_hold_kp, -mcfg.x_hold_max_vx, mcfg.x_hold_max_vx))
                yaw_err = wrap_to_180(mcfg.goal_yaw_scan_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * mcfg.yaw_hold_kp, -mcfg.yaw_hold_max_wz, mcfg.yaw_hold_max_wz))
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} x_err={x_err:.4f} phase={phase} target_y={target_y:.3f} "
                    f"xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "TURN_YAW_FINAL":
                err = wrap_to_180(mcfg.goal_yaw4_deg - yaw_now)
                abs_err = abs(err)
                if abs_err <= mcfg.yaw_tol_deg:
                    ctrl.stop()
                    phase = "SCAN_BACK_1"
                    last_abs_err = None
                    print(f"[PHASE] TURN_YAW_FINAL done at yaw={yaw_now:.2f} -> SCAN_BACK_1")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.5:
                    sign = -sign
                wz = sign * mcfg.phase_wz_radps
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase.startswith("SCAN_BACK_"):
                ctrl.stop()
                time.sleep(0.10)
                is_orange, ratio, best_ratio = confirm_orange(vision, mcfg)
                print(
                    f"[CHECK] {phase} y={y_now:.3f} orange={is_orange} "
                    f"ratio={ratio:.4f} best_ratio={best_ratio:.4f}"
                )
                if is_orange:
                    print(
                        f"[HIT] {phase} orange confirmed, "
                        f"forward={mcfg.hit_back_forward_m:.2f}m retreat={mcfg.hit_back_retreat_m:.2f}m"
                    )
                    move_distance(
                        ctrl, mcfg.hit_speed_mps, 0.0, 0.0,
                        mcfg.hit_back_forward_m, mcfg.hit_speed_mps, mcfg.locomotion_gait_id
                    )
                    move_distance(
                        ctrl, -mcfg.hit_speed_mps, 0.0, 0.0,
                        mcfg.hit_back_retreat_m, mcfg.hit_speed_mps, mcfg.locomotion_gait_id
                    )
                if back_scan_index < len(mcfg.scan_y_targets_back_m) - 1:
                    phase = "GO_SCAN_Y_BACK"
                    last_abs_err = None
                else:
                    phase = "GO_Y_LEFT_AFTER_BACK"
                    last_abs_err = None
                continue

            elif phase == "GO_SCAN_Y_BACK":
                target_y = mcfg.scan_y_targets_back_m[back_scan_index + 1]
                err = target_y - y_now
                abs_err = abs(err)
                if abs_err <= mcfg.y_tol_m or y_now <= target_y:
                    ctrl.stop()
                    time.sleep(0.15)
                    back_scan_index += 1
                    phase = f"SCAN_BACK_{back_scan_index + 1}"
                    print(f"[PHASE] GO_SCAN_Y_BACK done at y={y_now:.3f} -> {phase}")
                    continue
                # Back-row scan must keep moving left (towards smaller y) monotonically.
                vy = mcfg.phase_vy_mps
                x_err = mcfg.goal_x2_m - x_now
                vx = -float(np.clip(x_err * mcfg.x_hold_kp, -mcfg.x_hold_max_vx, mcfg.x_hold_max_vx))
                wz = 0.0
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} x_err={x_err:.4f} phase={phase} target_y={target_y:.3f} "
                    f"xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "GO_Y_LEFT_AFTER_BACK":
                err = mcfg.goal_y_left_after_back_m - y_now
                abs_err = abs(err)
                if abs_err <= mcfg.y_tol_m or y_now <= mcfg.goal_y_left_after_back_m:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = "GO_X_FINAL"
                    last_abs_err = None
                    print(f"[PHASE] GO_Y_LEFT_AFTER_BACK done at y={y_now:.3f} -> GO_X_FINAL")
                    continue
                # Keep moving left to y=0.9 without oscillation.
                vy = mcfg.phase_vy_mps
                x_err = mcfg.goal_x2_m - x_now
                vx = -float(np.clip(x_err * mcfg.x_hold_kp, -mcfg.x_hold_max_vx, mcfg.x_hold_max_vx))
                wz = 0.0
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} x_err={x_err:.4f} phase={phase} target_y={mcfg.goal_y_left_after_back_m:.3f} "
                    f"xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "GO_X_FINAL":
                err = mcfg.goal_x_final_m - x_now
                abs_err = abs(err)
                if abs_err <= mcfg.final_x_tol_m:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = "TURN_YAW3_PRE"
                    last_abs_err = None
                    print(f"[PHASE] GO_X_FINAL done at x={x_now:.3f} -> TURN_YAW3_PRE")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.003:
                    sign = -sign
                # Facing yaw~-179.4: world x decreases when body vx increases.
                vx = -sign * mcfg.phase_vx_mps
                wz = 0.0
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} target_x={mcfg.goal_x_final_m:.3f} "
                    f"xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "TURN_YAW3_PRE":
                err = wrap_to_180(mcfg.goal_yaw3_pre_deg - yaw_now)
                abs_err = abs(err)
                if abs_err <= mcfg.yaw_tol_deg:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = "GO_Y3_SCAN"
                    last_abs_err = None
                    print(f"[PHASE] TURN_YAW3_PRE done at yaw={yaw_now:.2f} -> GO_Y3_SCAN")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.5:
                    sign = -sign
                wz = sign * mcfg.phase_wz_radps
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "GO_Y3_SCAN":
                target_y = mcfg.scan3_y_targets_m[scan3_index]
                err = target_y - y_now
                abs_err = abs(err)
                if abs_err <= mcfg.y_tol_m:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = f"SCAN3_{scan3_index + 1}"
                    print(f"[PHASE] GO_Y3_SCAN done at y={y_now:.3f} -> {phase}")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.003:
                    sign = -sign
                # Facing yaw~-0.3: world y changes mainly by body vy.
                vy = sign * mcfg.phase_vy_mps
                yaw_err = wrap_to_180(mcfg.goal_yaw3_pre_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * mcfg.yaw_hold_kp, -mcfg.yaw_hold_max_wz, mcfg.yaw_hold_max_wz))
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} target_y={target_y:.3f} "
                    f"xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase.startswith("SCAN3_") and (not phase.startswith("SCAN3_BACK_")):
                ctrl.stop()
                time.sleep(0.10)
                is_orange, ratio, best_ratio = confirm_orange(vision, mcfg)
                print(
                    f"[CHECK] {phase} y={y_now:.3f} orange={is_orange} "
                    f"ratio={ratio:.4f} best_ratio={best_ratio:.4f}"
                )
                if is_orange:
                    print(f"[HIT] {phase} orange confirmed, forward+retreat")
                    move_distance(
                        ctrl, mcfg.hit_speed_mps, 0.0, 0.0,
                        mcfg.hit_forward_m, mcfg.hit_speed_mps, mcfg.locomotion_gait_id
                    )
                    move_distance(
                        ctrl, -mcfg.hit_speed_mps, 0.0, 0.0,
                        mcfg.hit_retreat_m, mcfg.hit_speed_mps, mcfg.locomotion_gait_id
                    )
                if scan3_index < len(mcfg.scan3_y_targets_m) - 1:
                    scan3_index += 1
                    phase = "GO_Y3_SCAN"
                    last_abs_err = None
                else:
                    phase = "TURN_YAW3_FINAL"
                    last_abs_err = None
                continue

            elif phase == "TURN_YAW3_FINAL":
                err = wrap_to_180(mcfg.goal_yaw4_deg - yaw_now)
                abs_err = abs(err)
                if abs_err <= mcfg.yaw_tol_deg:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = "SCAN3_BACK_1"
                    last_abs_err = None
                    print(f"[PHASE] TURN_YAW3_FINAL done at yaw={yaw_now:.2f} -> SCAN3_BACK_1")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.5:
                    sign = -sign
                wz = sign * mcfg.phase_wz_radps
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase.startswith("SCAN3_BACK_"):
                ctrl.stop()
                time.sleep(0.10)
                is_orange, ratio, best_ratio = confirm_orange(vision, mcfg)
                print(
                    f"[CHECK] {phase} y={y_now:.3f} orange={is_orange} "
                    f"ratio={ratio:.4f} best_ratio={best_ratio:.4f}"
                )
                if is_orange:
                    print(
                        f"[HIT] {phase} orange confirmed, "
                        f"forward={mcfg.hit_back_forward_m:.2f}m retreat={mcfg.hit_back_retreat_m:.2f}m"
                    )
                    move_distance(
                        ctrl, mcfg.hit_speed_mps, 0.0, 0.0,
                        mcfg.hit_back_forward_m, mcfg.hit_speed_mps, mcfg.locomotion_gait_id
                    )
                    move_distance(
                        ctrl, -mcfg.hit_speed_mps, 0.0, 0.0,
                        mcfg.hit_back_retreat_m, mcfg.hit_speed_mps, mcfg.locomotion_gait_id
                    )
                if scan3_back_index < len(mcfg.scan3_y_targets_back_m) - 1:
                    phase = "GO_Y3_SCAN_BACK"
                    last_abs_err = None
                else:
                    phase = "GO_Y_RIGHT_FINAL"
                    last_abs_err = None
                    print(f"[PHASE] {phase} start, target_y={mcfg.goal_y_right_final_m:.3f}")
                continue

            elif phase == "GO_Y3_SCAN_BACK":
                target_y = mcfg.scan3_y_targets_back_m[scan3_back_index + 1]
                err = target_y - y_now
                abs_err = abs(err)
                if abs_err <= mcfg.y_tol_m or y_now <= target_y:
                    ctrl.stop()
                    time.sleep(0.15)
                    scan3_back_index += 1
                    phase = f"SCAN3_BACK_{scan3_back_index + 1}"
                    print(f"[PHASE] GO_Y3_SCAN_BACK done at y={y_now:.3f} -> {phase}")
                    continue
                # Third back-scan: keep moving left; keep x near 0.196.
                vy = mcfg.phase_vy_mps
                x_err = mcfg.goal_x_final_m - x_now
                vx = -float(np.clip(x_err * mcfg.x_hold_kp, -mcfg.x_hold_max_vx, mcfg.x_hold_max_vx))
                wz = 0.0
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} x_err={x_err:.4f} phase={phase} target_y={target_y:.3f} "
                    f"xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "GO_Y_RIGHT_FINAL":
                err = mcfg.goal_y_right_final_m - y_now
                abs_err = abs(err)
                if abs_err <= mcfg.y_tol_m or y_now >= mcfg.goal_y_right_final_m:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = "GO_X_AFTER_RIGHT"
                    last_abs_err = None
                    print(f"[PHASE] GO_Y_RIGHT_FINAL done at y={y_now:.3f} -> GO_X_AFTER_RIGHT")
                    continue
                # Facing yaw~-179.4: move right (towards larger world y) with body vy<0.
                vy = -mcfg.phase_vy_mps
                # Keep x near 0.196 during right move, same x-hold strategy as before.
                x_err = mcfg.goal_x_final_m - x_now
                vx = -float(np.clip(x_err * mcfg.x_hold_kp, -mcfg.x_hold_max_vx, mcfg.x_hold_max_vx))
                wz = 0.0
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} x_err={x_err:.4f} phase={phase} target_y={mcfg.goal_y_right_final_m:.3f} "
                    f"xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "GO_X_AFTER_RIGHT":
                err = mcfg.goal_x_after_right_m - x_now
                abs_err = abs(err)
                if abs_err <= mcfg.final_x_tol_m:
                    ctrl.stop()
                    time.sleep(0.15)
                    phase = "TURN_YAW_END"
                    last_abs_err = None
                    print(f"[PHASE] GO_X_AFTER_RIGHT done at x={x_now:.3f} -> TURN_YAW_END")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.003:
                    sign = -sign
                # Facing yaw~-179.4: world x decreases when body vx increases.
                vx = -sign * mcfg.phase_vx_mps
                wz = 0.0
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} target_x={mcfg.goal_x_after_right_m:.3f} "
                    f"xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            elif phase == "TURN_YAW_END":
                err = wrap_to_180(mcfg.goal_yaw_end_deg - yaw_now)
                abs_err = abs(err)
                if abs_err <= mcfg.yaw_tol_deg:
                    ctrl.stop()
                    print(f"[DONE] x={x_now:.3f} y={y_now:.3f} yaw={yaw_now:.2f}")
                    break
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.5:
                    sign = -sign
                wz = sign * mcfg.phase_wz_radps
                last_abs_err = abs_err
                print(
                    f"pianli={err:.4f} phase={phase} target_yaw={mcfg.goal_yaw_end_deg:.1f} "
                    f"xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )

            ctrl.send_locomotion(vx, vy, wz, gait_id=mcfg.locomotion_gait_id)
            time.sleep(0.05)

    finally:
        vision_thread_running["ok"] = False
        vision_thread.join(timeout=1.0)
        vision.destroy_node()
        rclpy.shutdown()
        ctrl.stop()
        time.sleep(mcfg.settle_time_s)
        ctrl.to_recovery_stand()
        time.sleep(0.6)
        ctrl.to_pure_damper()
        ctrl.quit()


if __name__ == "__main__":
    main()
