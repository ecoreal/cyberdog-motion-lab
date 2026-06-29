# fourth2

## 流程

1. 转向到 `yaw=90deg`
2. 默认步态前进，持续微调 `x≈2.001`（`x` 增大则左移）
3. 检测到限高杆后切用户步态（`mode=62, gait_id=110`）前进 `7s`
4. 切回默认步态继续前进到 `y=10.985`
5. 转向到 `yaw=-90deg`
6. 继续前进，持续微调 `x≈2.001`（此时 `x` 增大则右移）
7. 再次检测到限高杆后切用户步态前进 `10s`
8. 切回默认步态继续前进（持续微调 `x`）直到 `y=7.164`

## 运行

```bash
cd /home/cyberdog_sim
source /opt/ros/galactic/setup.bash
source install/setup.bash
python3 fourth2/main.py
```
