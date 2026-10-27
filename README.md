# Multi-Organ Foundation Model

**完整医学结果汇报（2026-10-26）：**[主线、baseline与全部消融总结](docs/MEDICAL_MAINLINE_BASELINE_SUMMARY_20261026.md)，覆盖104次正式主训练、统一三指标、逐器官漏检、Q06/Q07及实测成本；另附[332条可导出汇总统计](docs/experiments/MEDICAL_MAINLINE_BASELINE_ALL_20261026.csv)。

**10月26日15:30完成核验：**Q07已于今天13:22完成全部三组80轮训练、正式评价和成本测量；Q06/Q07合计5项训练及5项评价全部完成，当前无本项目医学训练运行。Q07验证选出的best测试16结构宏平均：0专家DSC 73.25%、HD95 106.55 pixel、ASSD 75.23 pixel；1专家72.71%、108.69、80.56；9专家73.41%、112.76、87.30。9专家相对0专家结构DSC仅高0.16个百分点且距离更差；9域并集为85.49%／84.71%／85.89%，9专家并集距离更好。结果仅seed42，尚不能证明多专家具有全面优势。正式产物见[Q07汇总](/ssd1/data/Multi-Organ_Foundation_Model/runs/q07_task_experts_80_512_seed42_20261025/summary.json)和[三指标CSV](/ssd1/data/Multi-Organ_Foundation_Model/runs/q07_task_experts_80_512_seed42_20261025/evaluation_metrics.csv)。下方保留较早时刻快照。

多器官超声图像分割研究代码。用户定义的**共享 U-Net**是九域共同训练、**一个前景并集输出通道**的普通U-Net；Task-ID MoE和Region MoE保留原设计。该主对照的50／80轮三种子结果均已完成，见[共享U-Net与MoE对照](docs/SHARED_UNION_VS_MOE.md)。旧代码名`shared`表示16通道U-Net，现作为附加对照，不能与单通道主基线混称。仓库名称不代表已验证基础模型或临床泛化能力。

**2026-10-26 10:03进度：**Q06训练及评价全部完成。Q07的0／1专家均已完成80轮，9专家因10月25日14:33主机重启中断于42轮；本次已核验检查点并恢复GPU0从第43轮续训。新控制器PID `26364`、worker PID `26471`，三组正式评价待训练齐全后自动执行，预计还需约3—4小时。见[最新进度、重启恢复和监测](docs/MEDICAL_PROGRESS_20261026.md)。下方日期较早条目保留为历史快照。

**2026-10-25 09:26进度：**Q06两组80轮训练和评价已于06:19完成；best测试结构DSC 73.85→73.89%，HD95/ASSD变差，固定末轮有小幅改善。Q07已自动接续：GPU0无专家52/80轮，GPU1单专家42/80轮，9专家待空闲卡。全部新实验seed42、512×512、80轮。见[最新状态、Q06三指标和手动监测](docs/Q06_Q07_EXECUTION_20261025.md)。Q05完整结果仍见[10月25日报告](docs/MEDICAL_PROGRESS_20261025.md)。

**2026-10-22进度：**Q02全部12项训练和12项评价已完成。三种子best测试结构DSC：Task-ID弱均衡74.26→74.94%，Region区域监督73.86→73.61%；Region并集小升掩盖血管等器官漏检增加。Q05因GPU续步逐位比较失败而未启动，本次已定位并修复预检，恢复双卡控制器和独立验证审查；冻结训练参数及源码保持原样。见[最新进度与完整Q02三指标](docs/MEDICAL_PROGRESS_20261022.md)、[修复、启动证据与监测](docs/Q05_RECOVERY_20261022.md)。

**此前完成范围：**50/80轮主对照、Q01和Q03合计63次训练及原任务测试、439项DSC/HD95/ASSD补评均已完成。Q04四方法的12项主训练及评价完成，UniverSeg三项延期。比较口径与历史三指标见[10月16日主线与baseline分析](docs/MEDICAL_BASELINE_ANALYSIS_20261016.md)、[10月14日报告](docs/MEDICAL_PROJECT_PROGRESS_REPORT_20261014.md)，另附[逐器官三指标](docs/experiments/MEDICAL_PER_ORGAN_RESULTS_20261013.md)及[可导出CSV](docs/experiments/MEDICAL_RESULTS_BEST_TEST_20261013.csv)。

**执行顺序（2026-10-16用户确认）：Q02 → Q05 → Q06路由实验 → Q07机制对照。** Q02与Q05各12项80轮训练及评价均完成；Q06两项训练和评价完成，Q07正在双卡运行。UniverSeg保持延期，seed42/43保留26/12轮完整检查点，seed44未开始，不自动恢复。见[后续执行计划](docs/MOE_NEXT_STEPS_20261016.md)、[实际队列](docs/EXPERIMENT_QUEUE.md)、[延期记录](docs/Q04_UNIVERSEG_DEFERRAL_20261014.md)。

**2026-10-23后续种子调整：**Q06、Q07及之后新医学实验只使用 **seed42**。Q06两臂共2项，Q07的0／1／9专家共3项；当前Q05按原三种子12项继续收尾。单种子新结果报告配对差，不报告跨种子标准差或跨种子稳定性。见[机器可读策略](docs/experiments/MEDICAL_FUTURE_SEED_POLICY_20261023.json)。

**接续状态（10月26日10:03快照）：**Q06控制器已正常结束；Q07新控制器PID `26364`，9专家worker为`26471`。实时核对`controller_status.json`的更新时间及实际进程；主机重启后旧JSON不会自行变成中断状态。

| 文档 | 内容 |
|---|---|
| [最新医学进度（2026-10-26）](docs/MEDICAL_PROGRESS_20261026.md) | Q07主机重启中断、42轮断点恢复、已完成组别及当前队列 |
| [Q06／Q07实施与启动（2026-10-25）](docs/Q06_Q07_EXECUTION_20261025.md) | 单seed新机制、五组GPU预检、双卡实际队列及手动监测命令 |
| [最新医学进度（2026-10-25）](docs/MEDICAL_PROGRESS_20261025.md) | Q05完整三种子测试三指标、best/last差异及Q06启动前后快照 |
| [最新医学进度（2026-10-22）](docs/MEDICAL_PROGRESS_20261022.md) | Q02完整三种子结果、与外部baseline比较、Q05修复和双卡接续 |
| [最新主线与baseline分析（2026-10-16）](docs/MEDICAL_BASELINE_ANALYSIS_20261016.md) | Q04正式测试与MoE同口径比较、Q02单种子验证及专家路由诊断 |
| [后续执行计划（2026-10-16）](docs/MOE_NEXT_STEPS_20261016.md) | Q02、Q05接续，以及Q06条件路由实验、Q07机制对照的依赖与验收 |
| [项目进度与结果历史快照（2026-10-14）](docs/MEDICAL_PROJECT_PROGRESS_REPORT_20261014.md) | 历史完整测试三指标、逐器官结果与当日训练状态 |
| [DSC / HD95 / ASSD 统一补评](docs/MEDICAL_METRICS_REEVALUATION.md) | 全部可恢复医学结果、逐器官与并集评价、空预测对角线惩罚及后台监测命令 |
| [五种外部baseline：Q04](docs/Q04_EXTERNAL_BASELINES_512.md) | MOFO、UUSIC前三名、UniverSeg；512／80轮／三种子并集对照，已实现并接入独立队列 |
| [MoE优化：Q02](docs/Q02_EXECUTION_20261013.md) | Region区域监督与Task-ID弱负载均衡；12项配对、全验证集路由诊断、三指标与后台监测 |
| [独立模型改进：Q05](docs/Q05_MOE_REFINEMENT_512.md) | Region纯视觉区域/背景对比、Task-ID特征调制；12项配对训练及评价已完成 |
| [共享U-Net与MoE主对照](docs/SHARED_UNION_VS_MOE.md) | 单通道前景并集基线、两种原设计MoE的结果与比较边界 |
| [医学实验总览](docs/MEDICAL_EXPERIMENT_OVERVIEW.md) | 全部医学实验的统一入口、数据协议、结果口径及可导出清单 |
| [正式对照结果](docs/experiments/FAIR_COMPARISONS_20261009.md) | 50／80轮四模型、三种子、逐结构和统一并集测试 |
| [Q01快照与完整矩阵](docs/experiments/Q01_SNAPSHOT_20261009.md) | 全部27项任务、已有配对验证结果和护栏状态 |
| [实验任务队列](docs/EXPERIMENT_QUEUE.md) | 待执行实验、依赖及验收标准；器官损失加权三组配对设计 |
| [架构和数据约定](unet_moe/README.md) | 21 个 task 槽位、19 个完整结构、16 个 paper5 结构及标签处理 |
| [区域 MoE](unet_moe/README_REGION.md) | 图像条件路由、辅助损失与实际计算边界 |
| [服务器运行](REMOTE_4090.md) | 环境、固定 split seed、多种子运行及续训命令 |
| [v6 历史协议](EXPERIMENT_V6_PROTOCOL.md) | `ct_cv5` 计划与实际直接执行的 `ct_final` |
| [历史 paper5 报告](PAPER5_THREE_MODEL_ANALYSIS.md) | 2026-09-29 单种子结果；部分原始产物未迁移到当前服务器 |

本仓库仅保存代码、测试和实验文档。数据、权重与日志位于仓库外。各数据集的 `processed_png/` 提供图像、mask 与 manifest；HC18 使用 `manifest_filled.csv`，AbdomenUS 使用 `manifest_indexed.csv`，CAMUS 还需原版患者划分名单。服务器路径见运行文档。

```powershell
python -m pip install -r requirements-unet-moe.txt -r requirements-preprocessing.txt
python -m pytest tests -q
```

新实验默认只评估验证集；最终冻结配置后才显式使用 `--evaluate-test`。旧 `train`、`region_train`、`baseline_suite` 入口保留历史协议和续训支持，其 `--seed` 同时控制项目划分与训练，不能用来直接进行固定测试集的多种子比较。
