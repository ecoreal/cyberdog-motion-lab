# fourth1

路径：`fourth1/`

`fourth1/main.py` 按以下固定流程执行：

1. 左移到 `x=0.969`
2. 调整姿态到 `yaw=90.1deg`
3. 直行，同时保持 `x≈0.969`
   - 雷达左前方扇区：`[0.0793, 0.7136] rad`
   - 雷达右前方扇区：`[-0.7136, -0.0793] rad`
   - 当两侧都出现 `<0.37m` 时停车
4. 左移到 `x=0.38`
5. 调整姿态到 `yaw=90deg`
6. 直行到 `y=8.866`
7. 右移到 `x=0.961`
8. 调整姿态到 `yaw=90deg`
9. 持续前进，并左右微调保持 `x=0.961`
   - 规则：`x` 变小则右移，`x` 变大则左移
10. 当前方扇区（`[-0.12, 0.12] rad`）出现 `<0.3m` 小球时：
   - 记录撞击前位置
   - 继续前进一段距离撞击小球
   - 后退到撞击前位置

## 运行

```bash
cd /home/cyberdog_sim
source /opt/ros/galactic/setup.bash
source install/setup.bash
python3 fourth1/main.py
```

## 可调参数（`Config`）

- `hit_forward_m`：检测到小球后继续前进的撞击距离（默认 `0.55m`）
- `phase_vx_mps / phase_vy_mps`：前进/侧移速度
- `ball_front_min_rad / ball_front_max_rad`：前方检测扇区
- `return_pos_tol_m`：回退到撞击前位置的容差
