# UniverSeg 延期与医学队列接续

日期：2026-10-14，北京时间。用户要求“先终止跳过本项UniverSeg训练，待别的任务都做完了之后我们再来看”。

## 已停止的任务与保留产物

已向原Q04控制器发送SIGTERM，并确认其两个UniverSeg训练子进程退出。原控制器PID为869636；seed42、seed43训练PID分别为1564601、1647790。没有删除训练目录、权重或日志。

| UniverSeg种子 | 已保存的完整轮数 | best轮 | 后续安排 |
|---|---:|---:|---|
| 42 | 26/80 | 23 | 延期，保留模型、优化器和随机状态 |
| 43 | 12/80 | 11 | 延期，保留模型、优化器和随机状态 |
| 44 | 0/80 | — | 尚未开始，延期 |

检查点已在CPU上实际加载，核验了模型数值有限、last轮数与日志一致、best与选优历史一致，以及优化器和随机状态存在。尚未保存的半轮不计为完成；以后恢复时从上述完整轮次之后重新开始。

## 调度范围

新的执行顺序为：

1. Q04的Lanzhou、AImageLab、SMART、MOFO：四方法×三种子，共12项主训练及其评价。前两种方法的6项训练已经完成；MOFO仍需各自完整的先验训练。
2. Q02：原12项区域监督／路由配对训练和评价。
3. Q05：原12项视觉上下文／任务调制配对训练和评价。
4. UniverSeg保持延期，等上述任务结束后再由用户决定，不自动恢复。

原Q04五方法协议与冻结源码不改写。延期通过独立的调度覆盖程序及授权清单记录。训练、数据、512输入、80轮预算、抽样、验证选优和指标实现保持原版本；调整的是当前执行范围及前置任务完成条件。

原Q04仍是15项设计，其中3项未完成。四方法的12项训练全部通过原检查点与历史验证后，才开放这12项的best/last评价；生成独立的`remaining_summary.json`，明确`complete=false`、`active_scope_complete=true`。控制器最终状态为`complete_with_deferred`，不会将UniverSeg写成完成，也不会伪造完整五方法汇总。

Q02的调度覆盖程序需要核验这12项训练、24份best/last测试结果及3项延期记录后才放行；Q02自身的12项训练和评价门槛不变，Q05仍等待完整Q02。

以后恢复UniverSeg时，应记录其余四方法已经开放测试这一事实，不追溯声称原15项一直共同保持测试封闭。现有UniverSeg的验证分数只作为未完成训练的中间结果。

## 证据与手动查看

- [用户授权与检查点指纹](experiments/Q04_UNIVERSEG_DEFERRAL_20261014.json)
- [14:41实际接续与双卡训练核验](experiments/Q04_UNIVERSEG_HANDOFF_20261014.json)
- [CPU检查点核验](/ssd1/data/Multi-Organ_Foundation_Model/runs/q04_external_baselines_80_512_20261013/verification/universeg_user_hold_20261014/checkpoints.json)
- [Q04队列状态](/ssd1/data/Multi-Organ_Foundation_Model/runs/q04_external_baselines_80_512_20261013/controller_status.json)
- [Q02队列状态](/ssd1/data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/controller_status.json)
- [Q05队列状态](/ssd1/data/Multi-Organ_Foundation_Model/runs/q05_visual_context_task_film_80_512_20261014/controller_status.json)

```bash
watch -n 30 cat /ssd1/data/Multi-Organ_Foundation_Model/runs/q04_external_baselines_80_512_20261013/controller_status.json
```

**14:40接续核验：**Q04新控制器PID `1777399`已启动，GPU0运行SMART/42（PID1777967），GPU1运行SMART/43（PID1777973）。Q04状态为6项训练完成、2项运行、4项待执行、3项延期。Q02新控制器PID `1777775`处于`waiting_q04`；原Q05控制器PID `1730462`继续`waiting_q02`。

本次针对延期、原外部控制器／训练入口及Q02控制器的 **97项测试通过**。独立审查通过；原Q04/Q02冻结源码分别115/77份文件保持原哈希。三个调度覆盖文件分别复制到两个运行目录的`queue_overrides/user_deferred_universeg_20261014/`，同目录`deployment.json`记录源码SHA和测试证据。

## 当前恢复入口

以下命令仅在相应控制器已经停止时使用；后台已运行时无需再次执行。沿用原检查点恢复，不重建协议。原Q04控制器入口不识别此次延期，当前使用下列覆盖入口。

```bash
# Q04：继续四方法，保持UniverSeg延期
/ssd1/data/Multi-Organ_Foundation_Model/envs/q04/bin/python \
  /ssd1/data/Multi-Organ_Foundation_Model/runs/q04_external_baselines_80_512_20261013/queue_overrides/user_deferred_universeg_20261014/run_q04_deferred.py \
  --output /ssd1/data/Multi-Organ_Foundation_Model/runs/q04_external_baselines_80_512_20261013 \
  --after-root /ssd1/data/Multi-Organ_Foundation_Model/runs/q01_organ_weighted_80_512_20261007 \
  --after-evaluation-root /ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012 \
  --gpus 0,1

# Q02：等待上述四方法完整训练与评价后接续
/ssd1/code/Multi-Organ_Foundation_Model/.conda/bin/python \
  /ssd1/data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/queue_overrides/user_deferred_universeg_20261014/run_q02_after_deferred_q04.py \
  --output /ssd1/data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013 \
  --reference-root /ssd1/data/Multi-Organ_Foundation_Model/runs/q01_organ_weighted_80_512_20261007 \
  --after-root /ssd1/data/Multi-Organ_Foundation_Model/runs/q04_external_baselines_80_512_20261013 \
  --deferral /ssd1/data/Multi-Organ_Foundation_Model/runs/q04_external_baselines_80_512_20261013/queue_deferral.json \
  --gpus 0,1
```

Q05的原入口及恢复方式不变。UniverSeg的延期记录不是自动恢复命令，也没有安排在Q05后自动启动。
