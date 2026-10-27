# 医学实验状态与证据入口

更新日期：**2026-10-09，北京时间**。完整配置、数据划分、所有结果与限制统一见[医学实验总览](MEDICAL_EXPERIMENT_OVERVIEW.md)。本页为人工核验快照，不会随训练自动刷新。用户定义的单通道共享U-Net与两种MoE主对照见[专页](SHARED_UNION_VS_MOE.md)；下文Q01中的共享模型指16通道附加对照。

| 实验 | 完成状态 | 结果入口 |
|---|---|---|
| 50轮四模型、三种子 | 12次训练及原始测试完成，另9次结构模型并集评价完成 | [正式结果](experiments/FAIR_COMPARISONS_20261009.md) |
| 80轮四模型、三种子 | 12次训练及原始测试完成，另9次结构模型并集评价完成 | [正式结果](experiments/FAIR_COMPARISONS_20261009.md) |
| 腹部空标签惩罚、后处理、短程微调 | 512试验完成，候选均未采用；256 pilot与后处理v1分别替代／作废 | [历史及腹部结果](experiments/HISTORICAL_AND_ABDOMEN_20261009.md) |
| Q01器官加权27项配对 | 11:00:31快照：4项完成、2项运行、21项待执行；27项测试均未开始 | [完整矩阵及验证结果](experiments/Q01_SNAPSHOT_20261009.md) |
| 旧19结构、paper5、v6与工程检查 | 按实际产物分别标记完成、缺失、失败及工程用途 | [历史清单](experiments/HISTORICAL_AND_ABDOMEN_20261009.md) |

当前Q01已完成共享16通道U-Net/seed42的A/B/C，以及Region/seed42的A。Region B/C继续训练。初步单种子验证显示加权组末轮血管退化有所减轻，但伴随误检变化；不能作为多种子测试结论。实时状态见[控制器](/ssd1/data/Multi-Organ_Foundation_Model/runs/q01_organ_weighted_80_512_20261007/controller_status.json)，手动监测命令见[Q01协议](Q01_ORGAN_WEIGHTED_512.md)。

80轮16结构测试阳性宏Dice为16通道U-Net 72.39±0.76%、Task-ID 71.46±0.29%、Region 71.81±0.62%。80轮9域并集测试宏Dice为16通道U-Net 84.74±0.26%、Task-ID 84.48±0.19%、Region 85.13±0.31%、单通道84.96±0.30%。均为三个种子的均值±样本标准差；两个指标不能互相排名。50轮与80轮同时改变base、batch和学习率，也不能仅归因于轮数。

本次目录盘点覆盖runs根下全部**36个目录及1个资源文件**，包括日志、失败启动和被替代产物；它们不是37次独立训练。可下载[目录CSV](experiments/medical_run_inventory_20261009.csv)、[Q01矩阵CSV](experiments/q01_matrix_20261009.csv)和[数据划分CSV](experiments/data_split_20261009.csv)。

尚未形成的证据：Q01全部27项的最终测试及三种子配对汇总、同一80轮协议的九独立U-Net对照、已完成的腹部五折以及真实腹部超声外部验证。历史paper5的Region和共享原始输出在本机缺失；对应报告数字仅作历史文本保存。

原状态页保留为[截至2026-10-05的历史存档](EXPERIMENT_STATUS_ARCHIVE_20261005.md)。其中“等待”“未完成”等描述只反映当时状态。10月1日与10月5日故障恢复过程分别见[首次恢复](PIPELINE_RECOVERY_20261001.md)、[中断恢复](PIPELINE_RESTART_20261005.md)。本次仅整理文档及证据，不重跑模型，也未改动运行队列。
