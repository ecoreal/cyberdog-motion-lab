#!/usr/bin/env python3
import copy
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
import toml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT / "code_bieren" / "lcm"))
from file_send_lcmt import file_send_lcmt
from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from simulator_lcmt import simulator_lcmt


def wrap_to_180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class Config:
    target_x_m: float = 2.095

    yaw_first_deg: float = 90.0
    yaw_second_deg: float = -90.0

    y_after_first_default_m: float = 10.985
    y_after_first_usergait_m: float = 11.0
    y_final_m: float = 7.164

    bar_left_min_rad: float = 0.25
    bar_left_max_rad: float = 1.20
    bar_right_min_rad: float = -1.20
    bar_right_max_rad: float = -0.25
    bar_detect_dist_m: float = 0.65

    usergait_forward_1_s: float = 15.0
    usergait_forward_2_s: float = 15.0

    x_tol_m: float = 0.02
    y_tol_m: float = 0.02
    yaw_tol_deg: float = 2.0

    phase_vx_mps: float = 0.16
    phase_vy_mps: float = 0.12
    phase_wz_radps: float = 0.45

    yaw_hold_kp: float = 1.4
    yaw_hold_max_wz: float = 0.35
    x_hold_kp: float = 1.0
    x_hold_max_vy: float = 0.07

    locomotion_gait_id: int = 26
    usergait_id: int = 110
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

    def to_locomotion(self, cfg: Config):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = cfg.locomotion_gait_id
        msg.contact = int(cfg.contact_mask)
        msg.duration = 0
        msg.step_height = [float(cfg.step_height_m), float(cfg.step_height_m)]
        msg.rpy_des = [0.0, float(cfg.body_pitch_rad), 0.0]
        msg.pos_des = [0.0, 0.0, float(cfg.body_height_m)]
        self.send_cmd(msg)

    def send_locomotion(self, vx: float, vy: float, wz: float, cfg: Config):
        msg = robot_control_cmd_lcmt()
        msg.mode = 11
        msg.gait_id = cfg.locomotion_gait_id
        msg.contact = int(cfg.contact_mask)
        msg.vel_des = [float(vx), float(vy), float(wz)]
        msg.duration = 0
        msg.step_height = [float(cfg.step_height_m), float(cfg.step_height_m)]
        msg.rpy_des = [0.0, float(cfg.body_pitch_rad), 0.0]
        msg.pos_des = [0.0, 0.0, float(cfg.body_height_m)]
        self.send_cmd(msg)

    def send_usergait(self, vx: float, vy: float, wz: float, cfg: Config):
        msg = robot_control_cmd_lcmt()
        msg.mode = 62
        msg.gait_id = int(cfg.usergait_id)
        msg.contact = int(cfg.contact_mask)
        msg.vel_des = [float(vx), float(vy), float(wz)]
        msg.rpy_des = [0.0, float(cfg.body_pitch_rad), 0.0]
        msg.pos_des = [0.0, 0.0, float(cfg.body_height_m)]
        msg.duration = 0
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
        super().__init__("fourth2_lidar")
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
        self._ensure_subscriptions(scan_topics)
        if self.active_scan_topic not in self.subs:
            self.active_scan_topic = scan_topics[0]
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


def publish_usergait_110() -> None:
    base = Path("/home/cyberdog_sim/code_bieren/toml")
    gait_def_path = base / "usergait_def.toml"
    gait_param_path = base / "usergait_param.toml"
    if not gait_def_path.exists() or not gait_param_path.exists():
        print("[USERGAIT] skip: toml not found")
        return

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

    full_path = Path("/home/cyberdog_sim/fourth2/usergait_param_full.toml")
    full_path.write_text("# Gait Params\n" + toml.dumps(full_steps))

    lc = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
    msg = file_send_lcmt()
    msg.data = gait_def_path.read_text()
    lc.publish("user_gait_file", msg.encode())
    time.sleep(0.5)
    msg.data = full_path.read_text()
    lc.publish("user_gait_file", msg.encode())
    time.sleep(0.1)
    print("[USERGAIT] published gait_id=110")


def bar_detected(lidar: LidarNode, cfg: Config) -> bool:
    left_d = lidar.min_dist_in_sector(cfg.bar_left_min_rad, cfg.bar_left_max_rad, cfg)
    right_d = lidar.min_dist_in_sector(cfg.bar_right_min_rad, cfg.bar_right_max_rad, cfg)
    left_dbg = left_d if left_d is not None else float("inf")
    right_dbg = right_d if right_d is not None else float("inf")
    print(
        "[LIDAR-BAR] "
        f"left({cfg.bar_left_min_rad:.2f},{cfg.bar_left_max_rad:.2f})={left_dbg:.3f} "
        f"right({cfg.bar_right_min_rad:.2f},{cfg.bar_right_max_rad:.2f})={right_dbg:.3f} "
        f"th={cfg.bar_detect_dist_m:.2f} topic={lidar.active_scan_topic}"
    )
    if left_d is None or right_d is None:
        return False
    return left_d < cfg.bar_detect_dist_m and right_d < cfg.bar_detect_dist_m


def x_trim_vy_phase1(x_now: float, cfg: Config) -> float:
    # First forward leg (yaw~90): if x increases, move left.
    if x_now > cfg.target_x_m:
        return float(np.clip(abs(x_now - cfg.target_x_m) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
    return -float(np.clip(abs(x_now - cfg.target_x_m) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))


def x_trim_vy_phase2(x_now: float, cfg: Config) -> float:
    # Second forward leg (yaw~-90): if x increases, move right.
    if x_now > cfg.target_x_m:
        return -float(np.clip(abs(x_now - cfg.target_x_m) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))
    return float(np.clip(abs(x_now - cfg.target_x_m) * cfg.x_hold_kp, 0.0, cfg.x_hold_max_vy))


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
    threading.Thread(target=pos.run, daemon=True).start()

    stop_flag = {"stop": False}

    def _handle_sigint(_sig, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        publish_usergait_110()
        ctrl.to_recovery_stand()
        time.sleep(1.2)
        ctrl.to_locomotion(cfg)
        time.sleep(0.6)

        phase = "TURN_90"
        ug_end_t = 0.0
        bar1_done = False
        bar2_done = False
        last_log_t = 0.0
        last_topic_detect_t = 0.0

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

            if phase == "TURN_90":
                err = wrap_to_180(cfg.yaw_first_deg - yaw_now)
                if abs(err) <= cfg.yaw_tol_deg:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "LEG1_DEFAULT"
                    print(f"[PHASE] TURN_90 done yaw={yaw_now:.2f} -> LEG1_DEFAULT")
                    continue
                wz = (1.0 if err >= 0.0 else -1.0) * cfg.phase_wz_radps

            elif phase == "LEG1_DEFAULT":
                if (not bar1_done) and bar_detected(lidar, cfg):
                    bar1_done = True
                    ctrl.stop(cfg)
                    time.sleep(0.10)
                    phase = "LEG1_USERGAIT"
                    speak("识别到限高杆")
                    print(f"[PHASE] LEG1_DEFAULT bar detected -> LEG1_USERGAIT until y={cfg.y_after_first_usergait_m:.3f}")
                    continue
                vx = cfg.phase_vx_mps
                vy = x_trim_vy_phase1(x_now, cfg)
                yaw_err = wrap_to_180(cfg.yaw_first_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "LEG1_USERGAIT":
                if y_now >= cfg.y_after_first_usergait_m - cfg.y_tol_m:
                    ctrl.stop(cfg)
                    time.sleep(0.10)
                    ctrl.to_locomotion(cfg)
                    time.sleep(0.20)
                    phase = "TURN_M90"
                    print(f"[PHASE] LEG1_USERGAIT reached y={y_now:.3f} -> TURN_M90")
                    continue
                vx = cfg.phase_vx_mps
                vy = 0.0
                yaw_err = wrap_to_180(cfg.yaw_first_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "TURN_M90":
                err = wrap_to_180(cfg.yaw_second_deg - yaw_now)
                if abs(err) <= cfg.yaw_tol_deg:
                    ctrl.stop(cfg)
                    time.sleep(0.15)
                    phase = "LEG2_DEFAULT_PREBAR"
                    print(f"[PHASE] TURN_M90 done yaw={yaw_now:.2f} -> LEG2_DEFAULT_PREBAR")
                    continue
                wz = (1.0 if err >= 0.0 else -1.0) * cfg.phase_wz_radps

            elif phase == "LEG2_DEFAULT_PREBAR":
                if (not bar2_done) and bar_detected(lidar, cfg):
                    bar2_done = True
                    ctrl.stop(cfg)
                    time.sleep(0.10)
                    phase = "LEG2_USERGAIT"
                    ug_end_t = time.time() + cfg.usergait_forward_2_s
                    speak("识别到限高杆")
                    print(f"[PHASE] LEG2_DEFAULT_PREBAR bar detected -> LEG2_USERGAIT {cfg.usergait_forward_2_s:.1f}s")
                    continue
                # yaw~-90 forward direction toward decreasing y
                vx = cfg.phase_vx_mps
                vy = x_trim_vy_phase2(x_now, cfg)
                yaw_err = wrap_to_180(cfg.yaw_second_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "LEG2_USERGAIT":
                if time.time() >= ug_end_t:
                    ctrl.stop(cfg)
                    time.sleep(0.10)
                    ctrl.to_locomotion(cfg)
                    time.sleep(0.20)
                    phase = "LEG2_TO_FINAL_Y"
                    print(f"[PHASE] LEG2_USERGAIT timeout -> LEG2_TO_FINAL_Y")
                    continue
                vx = cfg.phase_vx_mps
                vy = 0.0
                yaw_err = wrap_to_180(cfg.yaw_second_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            elif phase == "LEG2_TO_FINAL_Y":
                if y_now <= cfg.y_final_m + cfg.y_tol_m:
                    ctrl.stop(cfg)
                    print(f"[DONE] reached y={y_now:.3f}, x={x_now:.3f}, yaw={yaw_now:.2f}")
                    break
                vx = cfg.phase_vx_mps
                vy = x_trim_vy_phase2(x_now, cfg)
                yaw_err = wrap_to_180(cfg.yaw_second_deg - yaw_now)
                wz = float(np.clip(np.deg2rad(yaw_err) * cfg.yaw_hold_kp, -cfg.yaw_hold_max_wz, cfg.yaw_hold_max_wz))

            now = time.time()
            if now - last_log_t > 0.25:
                print(
                    f"phase={phase} xyz=({x_now:.3f},{y_now:.3f},{z_now:.3f}) yaw={yaw_now:.2f} "
                    f"cmd=({vx:.3f},{vy:.3f},{wz:.3f})"
                )
                last_log_t = now

            if phase in ("LEG1_USERGAIT", "LEG2_USERGAIT"):
                ctrl.send_usergait(vx, vy, wz, cfg)
            else:
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
