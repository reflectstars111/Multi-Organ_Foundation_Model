# 80 轮训练与后续队列恢复（2026-10-05）

2026-10-05 19:09（北京时间）完成启动检查。此次恢复既有实验，不改变训练参数、数据划分或选优规则。

## 中断与断点核验

原控制器与两卡 worker 在 2026-10-03 07:32 收到 SIGTERM；日志未记录触发来源。后续队列状态文件仍显示等待，但原 PID 均已不存在。

恢复前核验冻结计划、源码、两分片记录、显存预检报告，以及全部九组 `last.pt` 和 `best.pt` 的协议和历史。多头共享 U-Net、Region 的三个种子及 Task-ID seed42/43 均为 epoch80；Task-ID seed44 为 epoch4。优化器及随机状态存在，恢复入口加载这些状态，从第 5 轮继续未完成的训练。

两卡旧失败标记已归档为 `failed_gpu*.before_restart_20261005_*.json`，避免相互读取旧失败记录而阻断恢复。旧 worker 状态和四个控制器状态也已归档；日志追加保存。未修改训练源码。

## 启动检查

以下是启动时快照，实时状态需同时检查状态文件和进程。

| 阶段 | PID | 核验状态 |
|---|---:|---|
| 80 轮三结构模型控制器 | 1149524 | `running` |
| GPU0 worker / 训练子进程 | 1149532 / 1149593 | `training`，通过原冻结入口 `--resume` 续训 |
| GPU1 worker | 1149533 | `waiting_for_peer_training`，Region 三种子训练完成标记已验证 |
| 80 轮单通道控制器 | 1149837 | `waiting_fair` |
| 80 轮并集评价控制器 | 1149838 | `waiting_fair` |
| 50 轮并集评价控制器 | 1149839 | `waiting_gpu` |

各控制器的启动命令和 PID 保存于对应日志目录的 `recovery_launch_20261005_*.json`，本次启动输出为 `restart_20261005_*.log`。四个控制器已脱离当前会话运行，继续使用项目 GPU 文件锁和显存准入。

接续顺序：Task-ID seed44 剩余训练 → 三结构模型统一测试 → 80 轮单通道三种子训练及测试 → 80 轮统一并集评价。旧 50 轮并集评价独立等待 GPU0 锁，可能在阶段间执行。新器官损失加权 [Q01](EXPERIMENT_QUEUE.md) 仍为待实现／待调度。

本次检查证明断点可读取、协议一致及后台进程启动成功；尚无恢复后完成全部训练或评价的结论。不持续在线轮询实验。

## 手动监测

查看控制器状态：

```bash
watch -n 15 'cat /ssd1/data/Multi-Organ_Foundation_Model/runs/*80_base24_batch16_512_20261001_fix-logs/controller_status.json /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_union_eval_512_20260930-logs/controller_status.json'
```

查看续训日志（每轮完成后输出新指标）：

```bash
tail -n 15 -F /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_80_base24_batch16_512_20261001_fix/logs/gpu0.log
```

核对进程是否仍在，避免把遗留状态文件当成运行证据：

```bash
pgrep -af 'python.*(scripts.defer_|unet_moe\.)'
```

各阶段最终汇总写入对应运行目录的 `summary.json`。具体输出路径沿用 [80 轮协议](UNIFIED_80_512.md)。
