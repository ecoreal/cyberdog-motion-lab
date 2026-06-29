# Fourth

路径：`fourth/`

`fourth/main.py` 实现了以下流程：

1. 左移到 `x=-0.105`
2. 转向到 `yaw=90.4deg`
3. 进入低姿态前进行进阶段（默认走用户步态 `mode=62, gait_id=110`）
4. 激光识别限高杆（只识别一次）：
   - 左扇区和右扇区同时出现 `<=0.8m` 物体
   - 触发语音播报：`识别到限高杆`
5. 切换到可乐瓶识别模式（前方扇区 `-0.3~0.3`）：
   - 前方出现 `<=0.3m` 物体
   - 触发语音播报：`识别到可乐瓶`
   - 停止自动直行
6. 向前走一段撞击可乐瓶（`hit_forward_m`）
7. 回退阶段流程：
   - 先切默认步态（`mode=11`）
   - 转向到 `yaw=-90deg`
   - 先做一次 `x` 对齐（误差小于 `0.0025`）
   - 再切回低姿态用户步态回退到 `y=7.108`

## 用户步态与周期微调

- 前进和回退阶段都可以使用低姿态用户步态（`mode=62, gait_id=110`）。
- 为防止 `x` 漂移，脚本会每隔 `12s` 触发一次 `ALIGN_X_PERIODIC`：
  - 临时切换默认步态
  - 微调 `x` 到目标附近
  - 再切回用户步态继续行进
- 若 `ALIGN_X_PERIODIC` 超过 `10s` 仍未对齐，会自动退出微调并恢复用户步态继续走。

## 运行

在工程根目录执行：

```bash
python3 fourth/main.py
```

如果你当前终端未加载 ROS2/工作区环境，先执行：

```bash
source /opt/ros/galactic/setup.bash
source /home/cyberdog_sim/install/setup.bash
python3 /home/cyberdog_sim/fourth/main.py
```

## 语音播报说明

- 脚本会尝试调用系统 `spd-say` 或 `espeak` 进行播报。
- 若系统没有这些命令，终端仍会打印 `[VOICE] ...` 提示，不影响流程运行。

## 关键参数（`fourth/main.py` 的 `Config`）

- 位姿目标
  - `goal_x_left_m = -0.105`
  - `goal_yaw_deg = 90.4`
  - `goal_yaw_back_deg = -90.0`
  - `goal_y_back_m = 7.108`
- 检测阈值
  - `bar_pair_dist_m = 0.80`（限高杆双侧阈值）
  - `coke_stop_dist_m = 0.30`（可乐瓶前向阈值）
  - `coke_front_min_rad = -0.30`
  - `coke_front_max_rad = 0.30`
- 撞击距离
  - `hit_forward_m = 0.55`
- 用户步态
  - `use_usergait_low_posture = True`
  - `low_usergait_id = 110`
  - `usergait_forward_vx_mps`（用户步态前进速度）
- 周期微调
  - `x_align_tol_m = 0.0025`
  - `align_interval_s = 12.0`
  - `align_timeout_s = 10.0`
- 运动控制
  - `phase_vx_mps`、`phase_vy_mps`、`phase_wz_radps`
  - `x_hold_kp`、`x_hold_max_vy`（直行/回退时的 x 轴微调）

## 日志观察

- `phase=...`：当前状态
- `[LIDAR] ...`：当前活跃雷达话题及扇区最小距离
- `[PHASE] ...`：阶段切换
- `[DETECTED xx/20] ...`：限高杆命中后的重复确认日志
- `[DONE] ...`：流程完成
