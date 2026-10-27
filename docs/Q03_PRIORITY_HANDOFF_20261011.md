# Q03优先级交接（2026-10-11）

用户确认的顺序已接入后台：**当前两项Q01训练自然完成 → Q03全部训练与评价 → 恢复Q01剩余任务**。Q02仍待实现与冻结。

## 启动快照

核验时间：2026-10-11 01:56，北京时间。这是启动时快照，后续进度以运行产物为准。

| 项目 | 状态 |
|---|---|
| 优先级调度器 | PID `4037674`，阶段 `draining` |
| 后台恢复进程 | PID `4037675`，监护调度器，异常退出时从持久化阶段恢复 |
| Q01已完成训练 | 19/27 |
| 当前GPU0训练 | `shared / seed44 / organ_exposure`，49/80轮，worker PID `3995767` |
| 当前GPU1训练 | `shared / seed44 / organ_exposure_fg_bg`，39/80轮，worker PID `3999776` |
| Q01后移任务 | Region/seed44与Task-ID/seed44各A/B/C，共6项训练；随后全27项测试及汇总 |
| Q03 | 4种归一化×3种子，共12项训练与12项测试，当前尚未开始GPU预检 |

旧Q01控制器PID `2123736`已暂停派发；两个训练worker继续执行并写入原有结果目录。由于旧控制器在交接期间不刷新状态，其`controller_status.json`可能保留旧完成数量；不能据此判定训练停止。Q03旧等待控制器在移交前仍显示`waiting_reference`，实际优先级阶段看下面的独立状态文件。

## 自动交接规则

1. 只暂停旧Q01控制器，不向当前worker发送停止信号；重新枚举真实子进程，防止状态写入延迟漏记任务。
2. 两个worker均自然退出后，逐个调用Q01冻结入口的`--verify-run`，核验完整80轮产物。任何核验失败都会记录错误并恢复原Q01控制器，不启动Q03。
3. 核验通过后释放已停止的旧Q01控制器，结束无训练worker的Q03等待入口，再使用独立adapter启动冻结Q03控制器。
4. adapter只改变“等Q01全部结果”的前序条件，仍核验Q01/Q03的数据与抽样指纹。原四臂512输入GPU预检、显存准入、文件锁、12项训练屏障、测试及汇总保持原逻辑。
5. Q03完整12项训练、测试、汇总通过后，按原命令恢复Q01。Q01重新核验已有产物，只继续未完成任务；确认其真正提交后续worker后，优先级调度器才标记`complete`。

调度状态、计划和调度源码独立存放于：

```text
/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_priority_q03_20261011
```

没有修改两套实验的`protocol.json`、冻结`source_snapshot`、训练超参、数据或模型权重。512输入、80轮、有效batch16、base24及原评价规则保持不变。

## 手动监测

```bash
PRIORITY_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_priority_q03_20261011
Q01_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/q01_organ_weighted_80_512_20261007
Q03_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/region_domain_norm_80_512_20261010

# 交接阶段、当前worker、后续控制器PID及错误
cat "$PRIORITY_RUN/status.json"

# 持续查看阶段；Ctrl+C仅结束监测
watch -n 15 "cat '$PRIORITY_RUN/status.json'"

# 当前两项Q01日志；后续Q01恢复后仍写这里
tail -f "$Q01_RUN/logs/gpu0.log" "$Q01_RUN/logs/gpu1.log"

# Q03尚未启动时日志可能不存在，-F会等待文件创建
tail -F "$PRIORITY_RUN/q03_controller.log"

# Q03训练／测试矩阵和GPU
cat "$Q03_RUN/controller_status.json"
nvidia-smi
```

主要阶段：`draining`等待当前Q01训练；`q03_running`执行Q03预检／训练／评价；`q01_running`恢复并核验Q01；`complete`表示交接完成且Q01已经提交剩余任务，**不表示Q01全部实验已经完成**。`failed`时查看`status.json`的`error`及本目录日志；不会把不完整Q03结果当作成功继续。

调度进程意外退出时由watchdog自动接管；启动时已持久化子进程身份，重启不会重复提交同一控制器。若二者均消失且状态不是`failed`或`complete`，手动恢复命令见[Q03协议第6—7节](Q03_REGION_DOMAIN_NORMALIZATION_512.md)。不需要在线监测助手。

## 验证与可追溯性

- 本次全量医学回归：**360 passed，2 skipped，49.36秒**。
- 覆盖真实隔离测试进程：只暂停父控制器不影响worker、自然退出和僵尸识别、PID身份验证、锁保留、完整交接顺序、失败恢复、子进程启动与状态持久化间的中断，以及watchdog重启。
- 补充故障注入检查：Q03缺任一条结果拒绝接续；Q01尚处于`recovering`时不报告恢复成功。
- Q01原29个冻结源文件、Q03原34个冻结源文件的SHA均保持一致。
- 调度计划SHA-256：`f3f5d3627b821270ab94a65558cfac2e54d8678bd9b21f32176ec9b33885a768`。

[调度计划](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_priority_q03_20261011/plan.json)、[实时交接状态](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_priority_q03_20261011/status.json)、[实施检查](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_priority_q03_20261011/implementation_checks.json)。本次仅改变执行先后，不构成任何性能提升证据。
