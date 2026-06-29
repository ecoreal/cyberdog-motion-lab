# First

路径：`first/`

这个脚本现在是一个三阶段定点动作流程（不再依赖视觉巡线）：

1. `GO_X`：沿世界 `x` 轴移动到 `x=3.05`
2. `TURN_YAW`：原地转向到 `yaw=90deg`
3. `GO_Y`：在朝向 `+y` 后继续前进，直到 `y=0.622`

并且在第 1 阶段会持续做横向纠偏：
- 当 `y` 偏大时，机器人会右移把 `y` 拉回 `0` 附近
- 纠偏通过 `vy` 实现

## 坐标与方向约定

- 初始朝向：`+x`
- 转到 `yaw=90deg` 后，机体前进方向对应世界 `+y`
- 所以：
  - 转向前要改 `x`，主控量是 `vx`
  - 转向后要改 `y`，主控量也应该是 `vx`（不是 `vy`）

## 运行

在 `cyberdog_sim` 根目录：

```bash
source /opt/ros/galactic/setup.bash
source install/setup.bash
python3 first/main.py
```

## 运行日志说明

终端会持续打印：

- `phase`：当前阶段（`GO_X` / `TURN_YAW` / `GO_Y`）
- `dog_xyz`：当前三维坐标
- `yaw`：当前偏航角（deg）
- `cmd(vx,vy,wz)`：当前发送给机器狗的速度命令
- `pianli`：当前 `y` 方向偏差（`y_err = y_target - y_now`）
  - 在 `GO_X/TURN_YAW` 阶段，`y_target` 固定为 `0`
  - 在 `GO_Y` 阶段，`y_target` 为 `0.622`

## 关键参数（main.py -> MotionConfig）

- `goal_x_m = 3.05`
- `goal_y_m = 0.622`
- `goal_yaw_deg = 90.0`
- `x_tol_m / y_tol_m / yaw_tol_deg`：三阶段停止容差
- `phase_vx_mps`：阶段 1 和阶段 3 的主前进速度
- `phase_wz_radps`：阶段 2 的转向角速度
- `y_ref_m = 0.0`：阶段 1 的横向保持目标
- `y_lock_kp / y_lock_max_vy`：阶段 1 的 `y` 纠偏强度与限幅
- `y_hold_kp / y_hold_max_wz`：阶段 3 保持 `yaw=90deg` 的角速度控制

## 防走反策略

每个阶段都内置“误差变大则反向”保护：
- 若目标误差连续变大，脚本会自动翻转该轴运动方向
- 用于抑制因坐标符号、姿态定义差异导致的反向行走
