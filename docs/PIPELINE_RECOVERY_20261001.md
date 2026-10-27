# 并集评价与 80 轮后续队列恢复（2026-10-01）

23:39（北京时间）完成启动核验。50 轮并集评价已恢复等待 GPU；80 轮单通道与并集评价已接到当前运行目录 `fair_80_base24_batch16_512_20261001_fix`。这次完成的是代码修复和队列接续，并集评价尚未产出新结果。

## 失败原因与修复

50 轮评价器重新聚合两卡结果后，直接与根 `summary.json` 比较。调度器在保存根汇总时额外加入了 `parallel.gpu_observations`，因此正常产物也会被判为不一致。本次用真实产物复现，确认唯一差异就是这一项；九组训练历史、测试指标、权重和记录校验均已通过。

`unet_moe/union_evaluate.py` 现在按调度器的保存规则，从两个 worker 状态文件补齐 GPU 启动记录再比较。仍完整校验指标、计划和 GPU 记录；回归测试确认修改测试指标、并行约定或 GPU 记录都会被拒绝。

原 80 轮后续控制器指向已经失败的 `20260930` 根目录，不会自动跟随另起的 `20261001_fix`。本次为当前目录启动独立的单通道和并集评价控制器，所有前置条件和输出路径均指向同一套恢复实验。

同时修复了单通道显存预检的 CUDA 初始化顺序：先初始化上下文，再重置显存峰值计数，并把这一步纳入失败报告。它复用了三结构预检已采用的处理方式；单通道实际 GPU 预检仍在前置实验完成后执行。

## 已验证状态

所有目录均位于 `/ssd1/data/Multi-Organ_Foundation_Model/runs/`。

| 控制器对应输出目录 | PID | 启动核验状态 | 前置条件 |
|---|---:|---|---|
| `fair_union_eval_512_20260930` | 3799898 | `waiting_gpu` | 50 轮三模型与单通道均已完成，等待 GPU0 文件锁和显存 |
| `single_channel_union_80_base24_batch16_512_20261001_fix` | 3799899 | `waiting_fair` | 当前 80 轮三模型完成全部训练和测试 |
| `fair_union_eval_80_base24_batch16_512_20261001_fix` | 3799900 | `waiting_fair` | 当前 80 轮三模型及新单通道三种子全部完成 |

新控制器的完整启动命令保存在各自 `<输出目录>-logs/recovery_launch.json`。旧 50 轮评价失败状态已归档为同日志目录的 `controller_status.before_recovery_*.json`，历史日志保留。原 `20260930` 的 80 轮失败状态也保留，以供追溯。

验证证据：两项回归测试先复现原故障；修复后全量 CPU 测试 **178 项通过**。真实 50 轮产物通过完整评价入口校验，包含 9 次运行与 17,698 条记录；当前 80 轮两分片的计划、源码、配置、记录及预检完成标记也已核验。启动后仅做了一次进程与状态检查，后续由用户手动监测。

## 手动查看

```bash
watch -n 15 'cat /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_union_eval_512_20260930-logs/controller_status.json /ssd1/data/Multi-Organ_Foundation_Model/runs/*80_base24_batch16_512_20261001_fix-logs/controller_status.json'
```

```bash
tail -n 20 -F /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_union_eval_512_20260930-logs/evaluation.log
tail -n 20 -F /ssd1/data/Multi-Organ_Foundation_Model/runs/single_channel_union_80_base24_batch16_512_20261001_fix-logs/single_channel.log
tail -n 20 -F /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_union_eval_80_base24_batch16_512_20261001_fix-logs/evaluation.log
```

`waiting_gpu` 表示等待文件锁或显存；即使显存看起来足够，当前训练持有锁时也会等待。`waiting_fair` 表示等待前置实验，不能解释为已经训练。各阶段完成后，最终结果写入对应输出目录的 `summary.json`。
