#!/usr/bin/env python3
import math
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import lcm
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT / "code_bieren" / "lcm"))
from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from simulator_lcmt import simulator_lcmt


def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class Config:
    goal_x_1_m: float = 0.969
    goal_yaw_1_deg: float = 90.1

    bar_left_min_rad: float = 0.0793
    bar_left_max_rad: float = 0.7136
    bar_right_min_rad: float = -0.7136
    bar_right_max_rad: float = -0.0793
    bar_detect_dist_m: float = 0.50
    bar_probe_interval_s: float = 8.0
    bar_probe_hold_s: float = 6.0

    goal_x_2_m: float = 0.38
    goal_yaw_2_deg: float = 0.0
    goal_y_2_m: float = 8.90
    goal_x_3_m: float = 0.95
    goal_y_3_m: float = 9.2

    goal_x_track_m: float = 0.95
    goal_yaw_3_deg: float = 90.0
    ball_front_min_rad: float = -0.12
    ball_front_max_rad: float = 0.12
    ball_detect_dist_m: float = 0.30

    hit_forward_m: float = 0.55
    goal_yaw_back_route_deg: float = 180.0
    goal_y_back1_m: float = 8.85
    goal_x_back1_m: float = 0.28
    goal_y_back2_m: float = 7.241
    goal_yaw_back2_deg: float = 0.0
    goal_x_back2_m: float = 2.02

    x_tol_m: float = 0.020
    y_tol_m: float = 0.020
    yaw_tol_deg: float = 2.0
    return_pos_tol_m: float = 0.03

    phase_vx_mps: float = 0.16
    phase_vy_mps: float = 0.14
    phase_wz_radps: float = 0.45

    yaw_hold_kp: float = 1.4
    yaw_hold_max_wz: float = 0.35
    x_hold_kp: float = 1.0
    x_hold_max_vy: float = 0.07

    locomotion_gait_id: int = 26
    settle_time_s: float = 0.12
    step_height_m: float = 0.16
    body_height_m: float = 0.20
    body_pitch_rad: float = 0.0
    contact_mask: int = 15

    lidar_valid_min_m: float = 0.10
    lidar_valid_max_m: float = 8.0
    scan_topic: str = "/scan"


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

    def to_locomotion(self, gait_id: int, step_height: float, body_height: float, body_pitch: float, contact_mask: int):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = gait_id
        msg.contact = int(contact_mask)
        msg.duration = 0
        msg.step_height = [float(step_height), float(step_height)]
        msg.rpy_des = [0.0, float(body_pitch), 0.0]
        msg.pos_des = [0.0, 0.0, float(body_height)]
        self.send_cmd(msg)

    def send_locomotion(self, vx: float, vy: float, wz: float, cfg: Config):
        self.send_locomotion_custom(
            vx=vx,
            vy=vy,
            wz=wz,
            gait_id=cfg.locomotion_gait_id,
            step_height=cfg.step_height_m,
            body_height=cfg.body_height_m,
            body_pitch=cfg.body_pitch_rad,
            contact_mask=cfg.contact_mask,
        )

    def send_locomotion_custom(
        self,
        vx: float,
        vy: float,
        wz: float,
        gait_id: int,
        step_height: float,
        body_height: float,
        body_pitch: float,
        contact_mask: int,
    ):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = int(gait_id)
        msg.contact = int(contact_mask)
        msg.vel_des = [float(vx), float(vy), float(wz)]
        msg.duration = 0
        msg.step_height = [float(step_height), float(step_height)]
        msg.rpy_des = [0.0, float(body_pitch), 0.0]
        msg.pos_des = [0.0, 0.0, float(body_height)]
        self.send_cmd(msg)

    def stop(self, cfg: Config):
        self.send_locomotion(0.0, 0.0, 0.0, cfg)

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


class LidarNode(Node):
    def __init__(self, scan_topic: str):
        super().__init__("fourth1_lidar")
        self.latest_scan: Optional[LaserScan] = None
        self.active_scan_topic = scan_topic
        self.last_scan_time = 0.0
        self.subs = {}
        self._ensure_subscriptions([scan_topic])

    def _scan_cb(self, topic: str, msg: LaserScan):
        self.latest_scan = msg
        self.last_scan_time = time.time()
        self.active_scan_topic = topic

    def _ensure_subscriptions(self, topics):
        for t in topics:
            if t in self.subs:
                continue
            self.subs[t] = self.create_subscription(
                LaserScan,
                t,
                lambda msg, topic=t: self._scan_cb(topic, msg),
                qos_profile_sensor_data,
            )

    def min_dist_in_sector(self, ang_min: float, ang_max: float, cfg: Config) -> Optional[float]:
        if self.latest_scan is None:
            return None
        scan = self.latest_scan
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return None
        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * scan.angle_increment
        valid = np.isfinite(ranges)
        valid &= ranges >= cfg.lidar_valid_min_m
        valid &= ranges <= min(cfg.lidar_valid_max_m, float(scan.range_max))
        valid &= angles >= ang_min
        valid &= angles <= ang_max
        if not np.any(valid):
            return None
        return float(np.min(ranges[valid]))


def speak(text: str):
    print(f"[VOICE] {text}")
    exe = shutil.which("spd-say") or shutil.which("espeak")
    if exe is None:
        return
    try:
        subprocess.Popen([exe, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def move_distance(ctrl: RobotCtrl, vx: float, vy: float, wz: float, dist_m: float, speed_mps: float, cfg: Config):
    t = max(0.0, dist_m / max(1e-6, speed_mps))
    t0 = time.time()
    while time.time() - t0 < t:
        ctrl.send_locomotion(vx, vy, wz, cfg)
        time.sleep(0.05)
    ctrl.stop(cfg)
    time.sleep(0.12)


def lateral_cmd_for_target_x(x_now: float, target_x: float, cfg: Config) -> float:
    x_err = target_x - x_now
    mag = float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
    # Rule required by user: x smaller -> move right; x bigger -> move left.
    return -mag if x_now < target_x else mag


def main():
    cfg = Config()
    rclpy.init()
    lidar = LidarNode(cfg.scan_topic)
    lidar_running = {"ok": True}

    def _spin():
        while lidar_running["ok"]:
            rclpy.spin_once(lidar, timeout_sec=0.05)

    lidar_thread = threading.Thread(target=_spin, daemon=True)
    lidar_thread.start()

    ctrl = RobotCtrl()
    ctrl.run()
    state_lock = threading.Lock()
    pos = PosMsg(state_lock)
    pos_thread = threading.Thread(target=pos.run, daemon=True)
    pos_thread.start()

    stop_flag = {"stop": False}

    def _handle_sigint(_sig, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    hit_start_pose: Optional[Tuple[float, float]] = None

    try:
        ctrl.to_recovery_stand()
        time.sleep(1.2)
        ctrl.to_locomotion(cfg.locomotion_gait_id, cfg.step_height_m, cfg.body_height_m, cfg.body_pitch_rad, cfg.contact_mask)
        time.sleep(0.6)

        phase = "GO_X_1"
        last_abs_err = None
        last_log_t = 0.0
        last_lidar_log_t = 0.0
        bar_probe_active = False
        bar_probe_end_t = 0.0
        # Run first crouch-scan immediately after entering GO_FORWARD_BAR.
        last_bar_probe_t = time.time() - cfg.bar_probe_interval_s

        while not stop_flag["stop"]:
            vx = 0.0
            vy = 0.0
            wz = 0.0

            with state_lock:
                x_now = float(pos.position[0])
                y_now = float(pos.position[1])
                z_now = float(pos.position[2])
                yaw_now = wrap_to_180(float(pos.rpy_deg[2]))

            if phase == "GO_X_1":
                err = cfg.goal_x_1_m - x_now
                abs_err = abs(err)
                if abs_err <= cfg.x_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "TURN_YAW_1"
                    last_abs_err = None
                    print(f"[PHASE] GO_X_1 done x={x_now:.3f} -> TURN_YAW_1")
                    continue
                # User requested: at startup, move LEFT in robot frame.
                # Field verification shows robot-left corresponds to vy>0 here.
                vy = cfg.phase_vy_mps
                last_abs_err = abs_err

            elif phase == "TURN_YAW_1":
                err = wrap_to_180(cfg.goal_yaw_1_deg - yaw_now)
                abs_err = abs(err)
                if abs_err <= cfg.yaw_tol_deg:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "GO_FORWARD_BAR"
                    last_abs_err = None
                    print(f"[PHASE] TURN_YAW_1 done yaw={yaw_now:.2f} -> GO_FORWARD_BAR")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.5:
                    sign = -sign
                wz = sign * cfg.phase_wz_radps
                last_abs_err = abs_err

            elif phase == "GO_FORWARD_BAR":
                left_d = lidar.min_dist_in_sector(cfg.bar_left_min_rad, cfg.bar_left_max_rad, cfg)
                right_d = lidar.min_dist_in_sector(cfg.bar_right_min_rad, cfg.bar_right_max_rad, cfg)
                now_t = time.time()
                if now_t - last_lidar_log_t > 0.35:
                    left_dbg = left_d if left_d is not None else float("inf")
                    right_dbg = right_d if right_d is not None else float("inf")
                    print(
                        "[LIDAR-BAR] "
                        f"left({cfg.bar_left_min_rad:.4f},{cfg.bar_left_max_rad:.4f})={left_dbg:.3f} "
                        f"right({cfg.bar_right_min_rad:.4f},{cfg.bar_right_max_rad:.4f})={right_dbg:.3f} "
                        f"th={cfg.bar_detect_dist_m:.3f} topic={lidar.active_scan_topic}"
                    )
                    last_lidar_log_t = now_t
                if (left_d is not None) and (right_d is not None) and left_d < cfg.bar_detect_dist_m and right_d < cfg.bar_detect_dist_m:
                    ctrl.stop(cfg)
                    speak("检测到前方双侧障碍")
                    time.sleep(0.15)
                    phase = "GO_X_2"
                    bar_probe_active = False
                    # Leave pure_damper reliably before moving on.
                    ctrl.to_recovery_stand()
                    time.sleep(0.35)
                    ctrl.to_locomotion(cfg.locomotion_gait_id, cfg.step_height_m, cfg.body_height_m, cfg.body_pitch_rad, cfg.contact_mask)
                    time.sleep(0.15)
                    print(f"[PHASE] GO_FORWARD_BAR detected left={left_d:.3f}, right={right_d:.3f} -> GO_X_2")
                    continue

                # Every 3s: crouch and scan briefly; if not detected, stand and continue walking.
                if bar_probe_active:
                    vx = 0.0
                    vy = 0.0
                    wz = 0.0
                    if now_t >= bar_probe_end_t:
                        bar_probe_active = False
                        last_bar_probe_t = now_t
                        print("[PROBE] exit crouch-scan, stand and continue")
                        ctrl.to_recovery_stand()
                        time.sleep(0.35)
                        ctrl.to_locomotion(cfg.locomotion_gait_id, cfg.step_height_m, cfg.body_height_m, cfg.body_pitch_rad, cfg.contact_mask)
                        time.sleep(0.08)
                    # keep pure damper state during probe window
                    time.sleep(0.05)
                    continue

                if (now_t - last_bar_probe_t) >= cfg.bar_probe_interval_s:
                    bar_probe_active = True
                    bar_probe_end_t = now_t + cfg.bar_probe_hold_s
                    print(f"[PROBE] enter crouch-scan hold={cfg.bar_probe_hold_s:.1f}s")
                    ctrl.stop(cfg)
                    time.sleep(0.05)
                    ctrl.to_pure_damper()
                    time.sleep(0.10)
                    time.sleep(0.05)
                    continue

                vx = cfg.phase_vx_mps
                vy = lateral_cmd_for_target_x(x_now, cfg.goal_x_1_m, cfg)
                yaw_err = wrap_to_180(cfg.goal_yaw_1_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "GO_X_2":
                err = cfg.goal_x_2_m - x_now
                if abs(err) <= cfg.x_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "TURN_YAW_2"
                    print(f"[PHASE] GO_X_2 done x={x_now:.3f} -> TURN_YAW_2")
                    continue
                vy = lateral_cmd_for_target_x(x_now, cfg.goal_x_2_m, cfg)
                yaw_err = wrap_to_180(cfg.goal_yaw_1_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "TURN_YAW_2":
                err = wrap_to_180(cfg.goal_yaw_2_deg - yaw_now)
                if abs(err) <= cfg.yaw_tol_deg:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "GO_Y_2"
                    print(f"[PHASE] TURN_YAW_2 done yaw={yaw_now:.2f} -> GO_Y_2")
                    continue
                wz = (1.0 if err >= 0.0 else -1.0) * cfg.phase_wz_radps

            elif phase == "GO_Y_2":
                y_err = cfg.goal_y_2_m - y_now
                if abs(y_err) <= cfg.y_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "GO_X_3"
                    print(f"[PHASE] GO_Y_2 done y={y_now:.3f} -> GO_X_3")
                    continue
                # yaw~0: robot-left is +y direction
                vy = (1.0 if y_err >= 0.0 else -1.0) * cfg.phase_vy_mps
                yaw_err = wrap_to_180(cfg.goal_yaw_2_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "GO_X_3":
                x_err = cfg.goal_x_3_m - x_now
                if abs(x_err) <= cfg.x_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "GO_Y_3"
                    print(f"[PHASE] GO_X_3 done x={x_now:.3f} -> GO_Y_3")
                    continue
                # yaw~0: move straight to adjust world x
                vx = (1.0 if x_err >= 0.0 else -1.0) * cfg.phase_vx_mps
                yaw_err = wrap_to_180(cfg.goal_yaw_2_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "GO_Y_3":
                y_err = cfg.goal_y_3_m - y_now
                if abs(y_err) <= cfg.y_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "TURN_YAW_3"
                    print(f"[PHASE] GO_Y_3 done y={y_now:.3f} -> TURN_YAW_3")
                    continue
                # yaw~0: robot-left is +y direction
                vy = (1.0 if y_err >= 0.0 else -1.0) * cfg.phase_vy_mps
                yaw_err = wrap_to_180(cfg.goal_yaw_2_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "TURN_YAW_3":
                err = wrap_to_180(cfg.goal_yaw_3_deg - yaw_now)
                if abs(err) <= cfg.yaw_tol_deg:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "FORWARD_TRACK_BALL"
                    print(f"[PHASE] TURN_YAW_3 done yaw={yaw_now:.2f} -> FORWARD_TRACK_BALL")
                    continue
                wz = (1.0 if err >= 0.0 else -1.0) * cfg.phase_wz_radps

            elif phase == "FORWARD_TRACK_BALL":
                front_d = lidar.min_dist_in_sector(cfg.ball_front_min_rad, cfg.ball_front_max_rad, cfg)
                now_t = time.time()
                if now_t - last_lidar_log_t > 0.35:
                    front_dbg = front_d if front_d is not None else float("inf")
                    print(
                        "[LIDAR-BALL] "
                        f"front({cfg.ball_front_min_rad:.4f},{cfg.ball_front_max_rad:.4f})={front_dbg:.3f} "
                        f"th={cfg.ball_detect_dist_m:.3f} topic={lidar.active_scan_topic}"
                    )
                    last_lidar_log_t = now_t
                if (front_d is not None) and (front_d < cfg.ball_detect_dist_m):
                    ctrl.stop(cfg)
                    hit_start_pose = (x_now, y_now)
                    speak("检测到小球，开始撞击")
                    time.sleep(0.12)
                    phase = "HIT_BALL"
                    print(f"[PHASE] ball detected front={front_d:.3f} at ({x_now:.3f},{y_now:.3f}) -> HIT_BALL")
                    continue
                vx = cfg.phase_vx_mps
                vy = lateral_cmd_for_target_x(x_now, cfg.goal_x_track_m, cfg)
                yaw_err = wrap_to_180(cfg.goal_yaw_3_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "HIT_BALL":
                move_distance(ctrl, cfg.phase_vx_mps, 0.0, 0.0, cfg.hit_forward_m, cfg.phase_vx_mps, cfg)
                phase = "BACK_TO_PRE_HIT"
                print("[PHASE] HIT_BALL done -> BACK_TO_PRE_HIT")
                continue

            elif phase == "BACK_TO_PRE_HIT":
                if hit_start_pose is None:
                    print("[WARN] no hit_start_pose, mission stop")
                    break
                dx = hit_start_pose[0] - x_now
                dy = hit_start_pose[1] - y_now
                dist = math.hypot(dx, dy)
                if dist <= cfg.return_pos_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "TURN_YAW_BACK_ROUTE"
                    print(f"[PHASE] returned pre-hit pos x={x_now:.3f} y={y_now:.3f} -> TURN_YAW_BACK_ROUTE")
                    continue
                vx = -cfg.phase_vx_mps
                vy = lateral_cmd_for_target_x(x_now, cfg.goal_x_track_m, cfg)
                yaw_err = wrap_to_180(cfg.goal_yaw_3_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "TURN_YAW_BACK_ROUTE":
                err = wrap_to_180(cfg.goal_yaw_back_route_deg - yaw_now)
                if abs(err) <= cfg.yaw_tol_deg:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "GO_Y_BACK_1"
                    print(f"[PHASE] TURN_YAW_BACK_ROUTE done yaw={yaw_now:.2f} -> GO_Y_BACK_1")
                    continue
                wz = (1.0 if err >= 0.0 else -1.0) * cfg.phase_wz_radps

            elif phase == "GO_Y_BACK_1":
                y_err = cfg.goal_y_back1_m - y_now
                if abs(y_err) <= cfg.y_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "GO_X_BACK_1"
                    print(f"[PHASE] GO_Y_BACK_1 done y={y_now:.3f} -> GO_X_BACK_1")
                    continue
                # Field-verified for this route: invert sign so robot moves left in body frame.
                vy = (1.0 if y_err < 0.0 else -1.0) * cfg.phase_vy_mps
                yaw_err = wrap_to_180(cfg.goal_yaw_back_route_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "GO_X_BACK_1":
                x_err = cfg.goal_x_back1_m - x_now
                if abs(x_err) <= cfg.x_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "GO_Y_BACK_2"
                    print(f"[PHASE] GO_X_BACK_1 done x={x_now:.3f} -> GO_Y_BACK_2")
                    continue
                # yaw~180: body forward maps to world -x.
                vx = (-1.0 if x_err > 0.0 else 1.0) * cfg.phase_vx_mps
                yaw_err = wrap_to_180(cfg.goal_yaw_back_route_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "GO_Y_BACK_2":
                y_err = cfg.goal_y_back2_m - y_now
                if abs(y_err) <= cfg.y_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "TURN_YAW_BACK_2"
                    print(f"[PHASE] GO_Y_BACK_2 done y={y_now:.3f} -> TURN_YAW_BACK_2")
                    continue
                # Field-verified for this route: invert sign so robot moves left in body frame.
                vy = (1.0 if y_err < 0.0 else -1.0) * cfg.phase_vy_mps
                yaw_err = wrap_to_180(cfg.goal_yaw_back_route_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "TURN_YAW_BACK_2":
                err = wrap_to_180(cfg.goal_yaw_back2_deg - yaw_now)
                if abs(err) <= cfg.yaw_tol_deg:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "GO_X_BACK_2"
                    print(f"[PHASE] TURN_YAW_BACK_2 done yaw={yaw_now:.2f} -> GO_X_BACK_2")
                    continue
                wz = (1.0 if err >= 0.0 else -1.0) * cfg.phase_wz_radps

            elif phase == "GO_X_BACK_2":
                x_err = cfg.goal_x_back2_m - x_now
                if abs(x_err) <= cfg.x_tol_m:
                    ctrl.stop(cfg)
                    print(f"[DONE] final reached x={x_now:.3f} y={y_now:.3f} yaw={yaw_now:.2f}")
                    break
                # yaw~0: body forward maps to world +x.
                vx = (1.0 if x_err >= 0.0 else -1.0) * cfg.phase_vx_mps
                yaw_err = wrap_to_180(cfg.goal_yaw_back2_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            now = time.time()
            if now - last_log_t > 0.25:
                lr = "L" if vy > 0.0 else ("R" if vy < 0.0 else "-")
                print(
                    f"phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f}) lateral={lr}"
                )
                last_log_t = now

            ctrl.send_locomotion(vx, vy, wz, cfg)
            time.sleep(0.05)

    finally:
        lidar_running["ok"] = False
        lidar_thread.join(timeout=1.0)
        lidar.destroy_node()
        rclpy.shutdown()
        ctrl.stop(cfg)
        time.sleep(cfg.settle_time_s)
        ctrl.to_recovery_stand()
        time.sleep(0.6)
        ctrl.to_pure_damper()
        ctrl.quit()


if __name__ == "__main__":
    main()
