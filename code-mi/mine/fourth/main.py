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
from typing import Optional

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
from file_send_lcmt import file_send_lcmt


def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class Config:
    goal_x_left_m: float = -0.105
    goal_yaw_deg: float = 90.4
    goal_yaw_back_deg: float = -90.0
    coke_stop_dist_m: float = 0.30
    bar_pair_dist_m: float = 0.80
    hit_forward_m: float = 1.3
    goal_y_back_m: float = 7.108

    x_tol_m: float = 0.020
    y_tol_m: float = 0.020
    yaw_tol_deg: float = 2.0
    x_align_tol_m: float = 0.0025
    align_interval_s: float = 12.0
    align_timeout_s: float = 10.0

    phase_vx_mps: float = 0.15
    phase_vy_mps: float = 0.15
    phase_wz_radps: float = 0.45

    yaw_hold_kp: float = 1.4
    yaw_hold_max_wz: float = 0.35
    x_hold_kp: float = 1.0
    x_hold_max_vy: float = 0.06

    locomotion_gait_id: int = 26
    settle_time_s: float = 0.12
    step_height_normal_m: float = 0.16
    step_height_low_m: float = 0.04
    body_height_normal_m: float = 0.20
    body_height_low_m: float = 0.03
    body_pitch_normal_rad: float = 0.0
    body_pitch_low_rad: float = -0.30
    contact_mask_normal: int = 15
    contact_mask_rear_only: int = 12
    use_usergait_low_posture: bool = True
    low_usergait_id: int = 110
    usergait_forward_vx_mps: float = 0.22

    lidar_valid_min_m: float = 0.10
    lidar_valid_max_m: float = 8.0
    bar_left_min_rad: float = 0.25
    bar_left_max_rad: float = 1.20
    bar_right_min_rad: float = -1.20
    bar_right_max_rad: float = -0.25
    coke_front_min_rad: float = -0.30
    coke_front_max_rad: float = 0.30
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

    def to_locomotion(
        self,
        gait_id=26,
        step_height: float = 0.16,
        body_height: float = 0.23,
        body_pitch: float = 0.0,
        contact_mask: int = 15,
    ):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = gait_id
        msg.contact = int(contact_mask)
        msg.duration = 0
        msg.step_height = [float(step_height), float(step_height)]
        msg.rpy_des = [0.0, float(body_pitch), 0.0]
        msg.pos_des = [0.0, 0.0, float(body_height)]
        self.send_cmd(msg)

    def send_locomotion(
        self,
        vx: float,
        vy: float,
        wz: float,
        gait_id=26,
        step_height: float = 0.16,
        body_height: float = 0.23,
        body_pitch: float = 0.0,
        contact_mask: int = 15,
    ):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = gait_id
        msg.contact = int(contact_mask)
        msg.vel_des = [float(vx), float(vy), float(wz)]
        msg.duration = 0
        msg.step_height = [float(step_height), float(step_height)]
        msg.rpy_des = [0.0, float(body_pitch), 0.0]
        msg.pos_des = [0.0, 0.0, float(body_height)]
        self.send_cmd(msg)

    def stop(
        self,
        step_height: float = 0.16,
        body_height: float = 0.23,
        body_pitch: float = 0.0,
        contact_mask: int = 15,
    ):
        self.send_locomotion(
            0.0,
            0.0,
            0.0,
            step_height=step_height,
            body_height=body_height,
            body_pitch=body_pitch,
            contact_mask=contact_mask,
        )

    def send_usergait(
        self,
        vx: float,
        vy: float,
        wz: float,
        gait_id: int,
        body_height: float,
        body_pitch: float,
        contact_mask: int = 15,
    ):
        msg = robot_control_cmd_lcmt()
        msg.mode = 62
        msg.gait_id = int(gait_id)
        msg.contact = int(contact_mask)
        msg.vel_des = [float(vx), float(vy), float(wz)]
        msg.rpy_des = [0.0, float(body_pitch), 0.0]
        msg.pos_des = [0.0, 0.0, float(body_height)]
        msg.duration = 0
        self.send_cmd(msg)

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
        super().__init__("fourth_lidar")
        self.latest_scan: Optional[LaserScan] = None
        self.scan_topic = scan_topic
        self.active_scan_topic = scan_topic
        self.last_scan_time = 0.0
        self.detected_scan_topics = []
        self.topic_last_time = {}
        self.subs = {}
        self._ensure_subscriptions([self.scan_topic])

    def _scan_cb(self, topic: str, msg: LaserScan):
        self.latest_scan = msg
        self.last_scan_time = time.time()
        self.topic_last_time[topic] = self.last_scan_time
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
        for t in list(self.subs.keys()):
            if t not in topics:
                self.destroy_subscription(self.subs[t])
                del self.subs[t]
                self.topic_last_time.pop(t, None)

    def autodetect_topics(self):
        graph = self.get_topic_names_and_types()
        scan_topics_all = sorted([name for name, types in graph if "sensor_msgs/msg/LaserScan" in types])
        scan_topics = []
        pub_counts = {}
        for name in scan_topics_all:
            infos = self.get_publishers_info_by_topic(name)
            pub_counts[name] = len(infos)
            if len(infos) > 0:
                scan_topics.append(name)
        self.detected_scan_topics = scan_topics
        if not scan_topics:
            return
        # Subscribe all scan topics, pick active by actual message arrival.
        self._ensure_subscriptions(scan_topics)
        if self.active_scan_topic not in self.subs:
            self.active_scan_topic = scan_topics[0]
        # one-line debug for topic/publisher truth
        print(f"[LIDAR] scan_topics_all={scan_topics_all}, pub_counts={pub_counts}, usable={scan_topics}")

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


def publish_low_posture_usergait() -> None:
    # Reuse bieren's usergait definition so mode=62/gait_id=110 is available.
    base = Path("/home/cyberdog_sim/code_bieren/toml")
    gait_def_path = base / "usergait_def.toml"
    gait_param_path = base / "usergait_param.toml"
    if not gait_def_path.exists() or not gait_param_path.exists():
        print("[USERGAIT] skip: usergait toml not found")
        return

    import copy
    import math
    import toml

    robot_cmd = {
        "mode": 0, "gait_id": 0, "contact": 0, "life_count": 0,
        "vel_des": [0.0, 0.0, 0.0], "rpy_des": [0.0, 0.0, 0.0], "pos_des": [0.0, 0.0, 0.0],
        "acc_des": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "ctrl_point": [0.0, 0.0, 0.0],
        "foot_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "step_height": [0.0, 0.0],
        "value": 0, "duration": 0,
    }

    steps = toml.load(str(gait_param_path))
    full_steps = {"step": []}
    for i in steps.get("step", []):
        cmd = copy.deepcopy(robot_cmd)
        cmd["duration"] = i.get("duration", 0)
        if i.get("type") == "usergait":
            cmd["mode"] = 11
            cmd["gait_id"] = 110
            cmd["vel_des"] = i["body_vel_des"]
            cmd["rpy_des"] = i["body_pos_des"][0:3]
            cmd["pos_des"] = i["body_pos_des"][3:6]
            cmd["foot_pose"][0:2] = i["landing_pos_des"][0:2]
            cmd["foot_pose"][2:4] = i["landing_pos_des"][3:5]
            cmd["foot_pose"][4:6] = i["landing_pos_des"][6:8]
            cmd["ctrl_point"][0:2] = i["landing_pos_des"][9:11]
            cmd["step_height"][0] = math.ceil(i["step_height"][0] * 1e3) + math.ceil(i["step_height"][1] * 1e3) * 1e3
            cmd["step_height"][1] = math.ceil(i["step_height"][2] * 1e3) + math.ceil(i["step_height"][3] * 1e3) * 1e3
            cmd["acc_des"] = i["weight"]
            cmd["value"] = i["use_mpc_traj"]
            cmd["contact"] = math.floor(i["landing_gain"] * 1e1)
            cmd["ctrl_point"][2] = i["mu"]
        full_steps["step"].append(cmd)

    full_path = Path("/home/cyberdog_sim/fourth/usergait_param_full.toml")
    full_path.write_text("# Gait Params\n" + toml.dumps(full_steps))

    lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
    msg = file_send_lcmt()
    msg.data = gait_def_path.read_text()
    lc.publish("user_gait_file", msg.encode())
    time.sleep(0.5)
    msg.data = full_path.read_text()
    lc.publish("user_gait_file", msg.encode())
    time.sleep(0.1)
    print("[USERGAIT] published mode62 gait_id=110 definition")


def move_distance(
    ctrl: RobotCtrl,
    vx: float,
    vy: float,
    wz: float,
    dist_m: float,
    speed_mps: float,
    gait_id: int,
    step_height: float,
    body_height: float,
    body_pitch: float,
    contact_mask: int,
):
    t = max(0.0, dist_m / max(1e-6, speed_mps))
    t0 = time.time()
    while time.time() - t0 < t:
        ctrl.send_locomotion(
            vx,
            vy,
            wz,
            gait_id=gait_id,
            step_height=step_height,
            body_height=body_height,
            body_pitch=body_pitch,
            contact_mask=contact_mask,
        )
        time.sleep(0.05)
    ctrl.stop(
        step_height=step_height,
        body_height=body_height,
        body_pitch=body_pitch,
        contact_mask=contact_mask,
    )
    time.sleep(0.12)


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

    try:
        publish_low_posture_usergait()
        ctrl.to_recovery_stand()
        time.sleep(1.2)
        ctrl.to_locomotion(
            gait_id=cfg.locomotion_gait_id,
            step_height=cfg.step_height_normal_m,
            body_height=cfg.body_height_normal_m,
            body_pitch=cfg.body_pitch_normal_rad,
            contact_mask=cfg.contact_mask_normal,
        )
        time.sleep(0.6)

        phase = "GO_X_LEFT"
        bar_done = False
        low_posture = False
        back_usergait_enabled = False
        align_return_phase: Optional[str] = None
        align_switched_to_default = False
        align_phase_t0 = 0.0
        last_usergait_align_t = time.time()
        last_lidar_log_t = 0.0
        last_topic_detect_t = 0.0
        last_phase_log_t = 0.0
        last_abs_err = None

        while not stop_flag["stop"]:
            vx = 0.0
            vy = 0.0
            wz = 0.0

            with state_lock:
                x_now = float(pos.position[0])
                y_now = float(pos.position[1])
                z_now = float(pos.position[2])
                yaw_now = wrap_to_180(float(pos.rpy_deg[2]))
            now_wall = time.time()
            if now_wall - last_topic_detect_t > 1.0:
                lidar.autodetect_topics()
                last_topic_detect_t = now_wall
            if lidar.last_scan_time <= 0.0 and (now_wall - last_lidar_log_t > 2.0):
                stale_parts = []
                for t in lidar.detected_scan_topics:
                    lt = lidar.topic_last_time.get(t, 0.0)
                    age = (now_wall - lt) if lt > 0.0 else -1.0
                    stale_parts.append(f"{t}:{'no_msg' if age < 0 else f'{age:.2f}s'}")
                print(
                    f"[LIDAR] waiting any scan, active={lidar.active_scan_topic}, "
                    f"detected={lidar.detected_scan_topics}, status={stale_parts}"
                )
                last_lidar_log_t = now_wall

            if phase == "GO_X_LEFT":
                err = cfg.goal_x_left_m - x_now
                abs_err = abs(err)
                if abs_err <= cfg.x_tol_m:
                    ctrl.stop(
                        step_height=cfg.step_height_normal_m,
                        body_height=cfg.body_height_normal_m,
                        body_pitch=cfg.body_pitch_normal_rad,
                        contact_mask=cfg.contact_mask_normal,
                    )
                    time.sleep(0.15)
                    phase = "TURN_YAW"
                    last_abs_err = None
                    print(f"[PHASE] GO_X_LEFT done at x={x_now:.3f} -> TURN_YAW")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.003:
                    sign = -sign
                vy = -sign * cfg.phase_vy_mps
                last_abs_err = abs_err

            elif phase == "TURN_YAW":
                err = wrap_to_180(cfg.goal_yaw_deg - yaw_now)
                abs_err = abs(err)
                if abs_err <= cfg.yaw_tol_deg:
                    ctrl.stop(
                        step_height=cfg.step_height_normal_m,
                        body_height=cfg.body_height_normal_m,
                        body_pitch=cfg.body_pitch_normal_rad,
                        contact_mask=cfg.contact_mask_normal,
                    )
                    time.sleep(0.15)
                    phase = "FORWARD_DETECT"
                    low_posture = True
                    last_abs_err = None
                    print(f"[PHASE] TURN_YAW done at yaw={yaw_now:.2f} -> FORWARD_DETECT")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.5:
                    sign = -sign
                wz = sign * cfg.phase_wz_radps
                last_abs_err = abs_err

            elif phase == "FORWARD_DETECT":
                left_d = lidar.min_dist_in_sector(cfg.bar_left_min_rad, cfg.bar_left_max_rad, cfg)
                right_d = lidar.min_dist_in_sector(cfg.bar_right_min_rad, cfg.bar_right_max_rad, cfg)
                front_d = lidar.min_dist_in_sector(cfg.coke_front_min_rad, cfg.coke_front_max_rad, cfg)

                now_t = time.time()
                if now_t - last_lidar_log_t > 0.50:
                    left_dbg = left_d if left_d is not None else cfg.lidar_valid_max_m
                    right_dbg = right_d if right_d is not None else cfg.lidar_valid_max_m
                    front_dbg = front_d if front_d is not None else cfg.lidar_valid_max_m
                    print(
                        "[LIDAR] "
                        f"active_topic={lidar.active_scan_topic} "
                        f"left({cfg.bar_left_min_rad:.2f},{cfg.bar_left_max_rad:.2f})={left_dbg:.3f} "
                        f"right({cfg.bar_right_min_rad:.2f},{cfg.bar_right_max_rad:.2f})={right_dbg:.3f} "
                        f"front({cfg.coke_front_min_rad:.2f},{cfg.coke_front_max_rad:.2f})={front_dbg:.3f} "
                        f"th_bar<={cfg.bar_pair_dist_m:.2f} th_coke<={cfg.coke_stop_dist_m:.2f}"
                    )
                    last_lidar_log_t = now_t

                if (not bar_done) and (left_d is not None) and (right_d is not None):
                    if left_d <= cfg.bar_pair_dist_m and right_d <= cfg.bar_pair_dist_m:
                        speak("识别到限高杆")
                        bar_done = True
                        for i in range(20):
                            print(
                                f"[DETECTED {i + 1:02d}/20] bar "
                                f"left={left_d:.3f} right={right_d:.3f} switch to coke mode"
                            )

                if bar_done and (front_d is not None) and front_d <= cfg.coke_stop_dist_m:
                    speak("识别到可乐瓶")
                    ctrl.stop(
                        step_height=cfg.step_height_low_m,
                        body_height=cfg.body_height_low_m,
                        body_pitch=cfg.body_pitch_low_rad,
                        contact_mask=cfg.contact_mask_rear_only,
                    )
                    time.sleep(0.15)
                    phase = "HIT_COKE"
                    print(f"[PHASE] coke front={front_d:.3f} -> HIT_COKE")
                    continue

                if (
                    low_posture
                    and cfg.use_usergait_low_posture
                    and (time.time() - last_usergait_align_t) >= cfg.align_interval_s
                ):
                    align_return_phase = "FORWARD_DETECT"
                    align_switched_to_default = False
                    align_phase_t0 = time.time()
                    phase = "ALIGN_X_PERIODIC"
                    print("[PHASE] FORWARD_DETECT periodic x-align -> ALIGN_X_PERIODIC")
                    continue

                # Facing +y: use vx for straight; use vy to hold x near -0.143.
                vx = cfg.usergait_forward_vx_mps if cfg.use_usergait_low_posture else cfg.phase_vx_mps
                x_err = cfg.goal_x_left_m - x_now
                # Rule required by user:
                # if x becomes smaller than target -> move right; otherwise move left.
                if x_now < cfg.goal_x_left_m:
                    vy = -float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                else:
                    vy = float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                yaw_err = wrap_to_180(cfg.goal_yaw_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "HIT_COKE":
                move_distance(
                    ctrl,
                    cfg.phase_vx_mps,
                    0.0,
                    0.0,
                    cfg.hit_forward_m,
                    cfg.phase_vx_mps,
                    cfg.locomotion_gait_id,
                    cfg.step_height_low_m,
                    cfg.body_height_low_m,
                    cfg.body_pitch_low_rad,
                    cfg.contact_mask_rear_only,
                )
                phase = "BACK_SWITCH_DEFAULT"
                last_abs_err = None
                print(f"[PHASE] HIT_COKE done -> BACK_SWITCH_DEFAULT")
                continue

            elif phase == "BACK_SWITCH_DEFAULT":
                # First switch back to default locomotion as requested.
                ctrl.to_locomotion(
                    gait_id=cfg.locomotion_gait_id,
                    step_height=cfg.step_height_normal_m,
                    body_height=cfg.body_height_normal_m,
                    body_pitch=cfg.body_pitch_normal_rad,
                    contact_mask=cfg.contact_mask_normal,
                )
                time.sleep(0.30)
                phase = "TURN_YAW_BACK"
                last_abs_err = None
                print(f"[PHASE] BACK_SWITCH_DEFAULT done -> TURN_YAW_BACK")
                continue

            elif phase == "TURN_YAW_BACK":
                err = wrap_to_180(cfg.goal_yaw_back_deg - yaw_now)
                abs_err = abs(err)
                if abs_err <= cfg.yaw_tol_deg:
                    ctrl.stop(
                        step_height=cfg.step_height_normal_m,
                        body_height=cfg.body_height_normal_m,
                        body_pitch=cfg.body_pitch_normal_rad,
                        contact_mask=cfg.contact_mask_normal,
                    )
                    time.sleep(0.15)
                    low_posture = True
                    back_usergait_enabled = False
                    phase = "BACK_ALIGN_X"
                    last_abs_err = None
                    print(f"[PHASE] TURN_YAW_BACK done at yaw={yaw_now:.2f} -> BACK_ALIGN_X")
                    continue
                sign = 1.0 if err >= 0.0 else -1.0
                if last_abs_err is not None and abs_err > last_abs_err + 0.5:
                    sign = -sign
                wz = sign * cfg.phase_wz_radps
                last_abs_err = abs_err

            elif phase == "BACK_ALIGN_X":
                x_err = cfg.goal_x_left_m - x_now
                if abs(x_err) <= cfg.x_align_tol_m:
                    phase = "BACK_TO_Y"
                    back_usergait_enabled = True
                    last_abs_err = None
                    print(
                        f"[PHASE] BACK_ALIGN_X done at x={x_now:.4f} "
                        f"(tol={cfg.x_align_tol_m:.4f}) -> BACK_TO_Y(usergait)"
                    )
                    continue
                if x_now > cfg.goal_x_left_m:
                    vy = -float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                else:
                    vy = float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                yaw_err = wrap_to_180(cfg.goal_yaw_back_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "BACK_TO_Y":
                err = cfg.goal_y_back_m - y_now
                abs_err = abs(err)
                if abs_err <= cfg.y_tol_m:
                    ctrl.stop(
                        step_height=cfg.step_height_low_m,
                        body_height=cfg.body_height_low_m,
                        body_pitch=cfg.body_pitch_low_rad,
                        contact_mask=cfg.contact_mask_rear_only,
                    )
                    low_posture = False
                    print(f"[DONE] x={x_now:.3f} y={y_now:.3f} yaw={yaw_now:.2f}")
                    break
                # Facing yaw~-90: move along y while keeping x close to goal_x_left_m.
                # Always move toward target y; this phase is intended as retreat in current map.
                sign = 1.0 if err >= 0.0 else -1.0
                vx = -sign * cfg.phase_vx_mps
                x_err = cfg.goal_x_left_m - x_now
                # User rule for BACK phase:
                # if x increases (x_now > target), move right.
                if x_now > cfg.goal_x_left_m:
                    vy = -float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                else:
                    vy = float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                yaw_err = wrap_to_180(cfg.goal_yaw_back_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))
                last_abs_err = abs_err

                if (
                    low_posture
                    and cfg.use_usergait_low_posture
                    and (time.time() - last_usergait_align_t) >= cfg.align_interval_s
                ):
                    align_return_phase = "BACK_TO_Y"
                    align_switched_to_default = False
                    align_phase_t0 = time.time()
                    phase = "ALIGN_X_PERIODIC"
                    print("[PHASE] BACK_TO_Y periodic x-align -> ALIGN_X_PERIODIC")
                    continue

            elif phase == "ALIGN_X_PERIODIC":
                if (time.time() - align_phase_t0) >= cfg.align_timeout_s:
                    last_usergait_align_t = time.time()
                    phase = align_return_phase if align_return_phase is not None else "FORWARD_DETECT"
                    print(
                        f"[PHASE] ALIGN_X_PERIODIC timeout {cfg.align_timeout_s:.1f}s "
                        f"at x={x_now:.4f} -> {phase} (resume usergait)"
                    )
                    continue
                if not align_switched_to_default:
                    ctrl.to_locomotion(
                        gait_id=cfg.locomotion_gait_id,
                        step_height=cfg.step_height_normal_m,
                        body_height=cfg.body_height_normal_m,
                        body_pitch=cfg.body_pitch_normal_rad,
                        contact_mask=cfg.contact_mask_normal,
                    )
                    time.sleep(0.20)
                    align_switched_to_default = True
                x_err = cfg.goal_x_left_m - x_now
                if abs(x_err) <= cfg.x_align_tol_m:
                    last_usergait_align_t = time.time()
                    phase = align_return_phase if align_return_phase is not None else "FORWARD_DETECT"
                    print(
                        f"[PHASE] ALIGN_X_PERIODIC done at x={x_now:.4f} "
                        f"-> {phase} (resume usergait)"
                    )
                    continue
                # Phase-dependent x trim direction:
                # FORWARD: x smaller -> move right
                # BACK:    x smaller -> move left
                if align_return_phase == "BACK_TO_Y":
                    if x_now < cfg.goal_x_left_m:
                        vy = float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                    else:
                        vy = -float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                    yaw_target = cfg.goal_yaw_back_deg
                else:
                    if x_now < cfg.goal_x_left_m:
                        vy = -float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                    else:
                        vy = float(np.clip(abs(x_err) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
                    yaw_target = cfg.goal_yaw_deg
                yaw_err = wrap_to_180(yaw_target - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            now_t = time.time()
            if phase != "FORWARD_DETECT" or (now_t - last_phase_log_t > 0.50):
                print(
                    f"phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )
                last_phase_log_t = now_t
            step_height = cfg.step_height_low_m if low_posture else cfg.step_height_normal_m
            body_height = cfg.body_height_low_m if low_posture else cfg.body_height_normal_m
            body_pitch = cfg.body_pitch_low_rad if low_posture else cfg.body_pitch_normal_rad
            contact_mask = cfg.contact_mask_rear_only if low_posture else cfg.contact_mask_normal
            use_usergait_now = (
                low_posture
                and cfg.use_usergait_low_posture
                and phase not in ("TURN_YAW_BACK", "BACK_SWITCH_DEFAULT", "BACK_ALIGN_X", "ALIGN_X_PERIODIC")
                and (phase != "BACK_TO_Y" or back_usergait_enabled)
            )
            if use_usergait_now:
                ctrl.send_usergait(
                    vx=vx,
                    vy=vy,
                    wz=wz,
                    gait_id=cfg.low_usergait_id,
                    body_height=body_height,
                    body_pitch=body_pitch,
                    contact_mask=contact_mask,
                )
            else:
                ctrl.send_locomotion(
                    vx,
                    vy,
                    wz,
                    gait_id=cfg.locomotion_gait_id,
                    step_height=step_height,
                    body_height=body_height,
                    body_pitch=body_pitch,
                    contact_mask=contact_mask,
                )
            time.sleep(0.05)

    finally:
        lidar_running["ok"] = False
        lidar_thread.join(timeout=1.0)
        lidar.destroy_node()
        rclpy.shutdown()
        ctrl.stop(
            step_height=cfg.step_height_normal_m,
            body_height=cfg.body_height_normal_m,
            body_pitch=cfg.body_pitch_normal_rad,
            contact_mask=cfg.contact_mask_normal,
        )
        time.sleep(cfg.settle_time_s)
        ctrl.to_recovery_stand()
        time.sleep(0.6)
        ctrl.to_pure_damper()
        ctrl.quit()


if __name__ == "__main__":
    main()
