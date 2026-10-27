# 医学实验进度与 Q02 完整结果（2026-10-22）

本文依据已保存的训练记录、评价结果及验证集复核产物整理。日期采用北京时间。本次只读取既有结果，不重新读取原始测试标签，不运行新的测试集选参。

## 1. 当前进度与待恢复工作

| 阶段 | 当前状态 | 完成范围与剩余工作 |
|---|---|---|
| Q04 外部 baseline | 当前四方法范围完成 | Lanzhou、AImageLab、SMART、MOFO 各三个种子，共 12 项主训练及评价完成；UniverSeg 三项仍延期，原五方法矩阵尚未全部完成 |
| Q02 区域监督与路由优化 | 完成 | 12 项训练及 12 项评价于 10 月 17 日 14:50 完成，三种子验证复核已生成 |
| Q05 主模型改进 | 已修复并恢复双卡正式训练 | 16:39四组GPU预检全部通过；GPU0 Region/control/42、GPU1 Region/refined/42正在训练，另10项等待，12项评价待全部训练完成 |
| Q06 Region 路由优化 | 有条件待设计与实施 | 等待 Q05 完整验证结果和路由诊断，不因已查看的测试分数直接触发 |
| Q07 无专家／少专家对照 | 已登记，尚未实施训练 | 在 Q06 是否需要执行的决定之后推进 |

UniverSeg 继续遵循用户此前的延期要求，不在本次自动恢复。

状态来源：[Q02 控制器](../../../data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/controller_status.json)、[Q05 控制器](../../../data/Multi-Organ_Foundation_Model/runs/q05_visual_context_task_film_80_512_20261014/controller_status.json)、[后续步骤状态](../../../data/Multi-Organ_Foundation_Model/runs/moe_followup_review_20261016/next_steps.json)。以上状态文件会继续变化，本文是本次检查时的记录。

## 2. 评价与比较口径

- Q02 是同种子、相同初始化、抽样序列和主损失下的配对比较：Region 辅助组增加区域 proposal 监督，Task-ID 辅助组增加弱负载均衡。两组主损失均为 C，即 `organ_exposure_fg_bg`；不能把 Q01 的 B 组数字混作本轮对照。
- 三个种子为 42、43、44，正式训练为 512×512、80 轮。表格数字为三个种子的均值 ± 样本标准差。
- **结构评价**先计算每个结构，再对 16 个结构等权平均；**并集评价**先在各域计算图像平均，再对 9 个数据域等权平均。两种视图不是同一任务或同一分母。
- 三指标为 DSC (%) ↑、HD95 (pixel) ↓、ASSD (pixel) ↓。单侧掩膜为空时，距离取像素中心对角线 `hypot(511, 511) ≈ 722.66 pixel`；双方为空为 100 / 0 / 0。
- 本文三指标主表均采用 **`positive_surface_defined`**：只纳入真值有前景、且真值边界可定义的样本；仍保留预测为空的有限惩罚，不能删除漏检样本来改善距离指标。
- Q02 的 **best** 由验证集结构阳性 DSC 选点；**last** 固定为第 80 轮；**guardrail** 是符合预先定义器官保护条件的独立选点。不存在合格点时按缺失报告，不能替换成 best。
- Q04 的 best 按验证集阳性**并集** DSC 选点。因此 Q02 与 Q04 的 best 横向表是共同输出评价口径下、按各自既定选点规则的比较，**选点规则并不完全一致**。严格比较还应同时查看固定第 80 轮，不能只报告有利选点。
- 本项目测试划分在历史研究中已被查看，不能称为未见外部验证。后续 Q05/Q06 的设计和触发仍使用训练／验证证据；不据这次测试比较改写已冻结协议。

协议与原始汇总：[Q02 协议](../../../data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/protocol.json)、[Q02 评价报告](../../../data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/EVALUATION_REPORT.md)、[Q04 四方法报告](../../../data/Multi-Organ_Foundation_Model/runs/q04_external_baselines_80_512_20261013/remaining_reports/REPORT.md)。Q02 原评价报告的并集 `positive` 表存在不可定义距离 NA；下列有限距离来自 CSV 中明确的 `positive_surface_defined` 行，两种集合不可混淆。

## 3. Q02 正式测试：16 结构

以下为验证集选出的 best 检查点在测试集上的三种子结果。

| 模型 | 组别 | DSC (%) ↑ | HD95 (pixel) ↓ | ASSD (pixel) ↓ |
|---|---|---:|---:|---:|
| Region MoE | 对照 | 73.86 ± 0.88 | 82.41 ± 2.30 | 47.12 ± 1.60 |
| Region MoE | 区域辅助监督 | 73.61 ± 0.09 | 97.88 ± 8.54 | 65.41 ± 12.15 |
| Task-ID MoE | 对照 | 74.26 ± 0.75 | 102.47 ± 5.29 | 72.66 ± 6.43 |
| Task-ID MoE | 弱负载均衡 | 74.94 ± 0.53 | 100.21 ± 7.41 | 70.80 ± 8.07 |

辅助组减对照组的配对差值：

| 模型 | Δ DSC（百分点） | Δ HD95（pixel） | Δ ASSD（pixel） |
|---|---:|---:|---:|
| Region MoE | −0.24 ± 0.86 | +15.47 ± 7.33 | +18.29 ± 10.55 |
| Task-ID MoE | +0.68 ± 0.31 | −2.26 ± 2.56 | −1.86 ± 4.10 |

**Task-ID 的弱均衡获得小幅结构改善，Region 的区域辅助监督尚未实现结构改善。** 这些是三个种子的观测均值与配对差值，不等于统计显著性检验的结论。

## 4. Q02 正式测试：前景并集及外部 baseline

均采用 best/test/positive_surface_defined，按九个域等权平均；选点规则差异见第 2 节。

| 方法 | DSC (%) ↑ | HD95 (pixel) ↓ | ASSD (pixel) ↓ |
|---|---:|---:|---:|
| AImageLab（Q04） | **87.29 ± 0.07** | **38.95 ± 1.16** | **12.17 ± 0.45** |
| Task-ID 对照（Q02） | 85.98 ± 0.24 | 42.35 ± 1.84 | 14.05 ± 0.96 |
| Task-ID 弱负载均衡（Q02） | 86.06 ± 0.04 | 43.25 ± 1.50 | 14.55 ± 1.55 |
| Region 对照（Q02） | 85.52 ± 0.39 | 44.13 ± 2.24 | 13.58 ± 0.37 |
| Region 区域辅助监督（Q02） | 85.90 ± 0.44 | 43.17 ± 2.04 | 13.70 ± 0.64 |
| Lanzhou（Q04） | 84.55 ± 0.43 | 45.63 ± 3.02 | 18.36 ± 2.40 |
| SMART（Q04） | 81.58 ± 0.63 | 51.36 ± 3.28 | 20.52 ± 1.84 |
| MOFO（Q04） | 61.61 ± 28.19 | 122.12 ± 70.73 | 75.35 ± 51.14 |

Region 辅助组并集 DSC 增加 0.38 个百分点，但逐结构表现下降，说明并集会掩盖器官身份和漏检问题。Task-ID 的并集 DSC 仅增加 0.08 个百分点，HD95/ASSD 均值略变差，不能把逐结构的小幅提升描述为所有指标都提高。

当前 Q02 两条主线的并集三项均值优于 Lanzhou、SMART 和本次 MOFO 复现，仍未超过 AImageLab。Task-ID 辅助组与 AImageLab 的 DSC 差距为 1.23 个百分点，Region 辅助组差 1.39 个百分点。MOFO 的种子波动很大，且 UniverSeg 尚未完成，因此不能据此宣称全面超过外部方法，也不能将 MOFO 本次均值当作原方法性能上限。

## 5. 漏检、误检与腹部器官取舍

下列为 best 检查点测试诊断，完整漏检指真值阳性但预测完全为空；它不同于预测非空但与真值零重叠。

| 模型 | 诊断指标（%） | 对照 | 辅助组 |
|---|---|---:|---:|
| Region | 16 结构完整漏检率宏平均 | 3.66 | 6.58 |
| Region | 16 结构零重叠率宏平均 | 7.79 | 9.43 |
| Region | 空标签误检率宏平均 | 51.62 | 41.75 |
| Task-ID | 16 结构完整漏检率宏平均 | 7.86 | 7.74 |
| Task-ID | 16 结构零重叠率宏平均 | 10.22 | 9.72 |
| Task-ID | 空标签误检率宏平均 | 28.28 | 23.41 |

Region 的误检下降伴随漏检上升，腹部尤为明显：

| 模型 | 器官 | 阳性 DSC (%)：对照 → 辅助 | 完整漏检率 (%)：对照 → 辅助 |
|---|---|---:|---:|
| Region | 血管 | 34.06 → 27.23 | 34.44 → 49.89 |
| Region | 脾 | 32.99 → 32.99 | 10.22 → 28.44 |
| Region | 肾 | 46.59 → 45.02 | 1.92 → 9.39 |
| Task-ID | 血管 | 37.95 → 44.54 | 44.15 → 36.64 |
| Task-ID | 胆囊 | 68.60 → 74.14 | 17.41 → 10.00 |
| Task-ID | 肾 | 44.35 → 39.29 | 21.65 → 29.50 |

Task-ID 在血管和胆囊有收益，但肾退化；腹部并不是所有器官共同改善。后续必须继续保留逐器官诊断和预设护栏。

## 6. 验证集复核与专家路由

以下证据来自已完成的训练／验证复核，可用于后续阶段判断；它们与上述测试结果独立列出。

| 模型 | 验证指标 | 对照 → 辅助组 |
|---|---|---:|
| Region | best 16 结构阳性 DSC (%) | 75.58 → 75.92 |
| Region | best 腹部阳性 DSC (%) | 56.50 → 56.05 |
| Region | best 完整漏检率 (%) | 3.40 → 5.93 |
| Task-ID | best 16 结构阳性 DSC (%) | 75.23 → 75.85 |
| Task-ID | best 腹部阳性 DSC (%) | 54.14 → 55.31 |
| Task-ID | best 完整漏检率 (%) | 8.23 → 7.44 |
| Task-ID | last 16 结构阳性 DSC (%) | 73.98 → 73.45 |

Task-ID 的改进依赖验证选点，固定第 80 轮并未获得同样的 DSC 改善，不能只看 best。

**Region 最细层仍没有形成专家分工。** 在三个种子、第 60 和 80 轮的固定诊断中，对照组和辅助组的最细 L4 均只有 1/9 个活动专家，最大专家占比为 100%。区域辅助监督增加了部分较粗层的专家使用，却未解决最细层独占。

**Task-ID 弱均衡改变了路由分布。** 第 80 轮，辅助组三个种子的四个层级均有 9/9 个活动专家；最细 L4 最大专家占比分别为 28.46%、16.29%、24.29%，对照为 76.67%、50.22%、74.46%。这些路由数字对应固定第 80 轮，不能声称是任意 best 检查点的路由状态。

护栏需与 best 分开看。Region 两组各有 3 个独立护栏候选；Task-ID 对照仅有 2 个，seed44 无合格候选，辅助组有 3 个。缺失种子不能补为 best，也不能用 2 种子均值直接冒充完整三种子配对。

完整验证来源：[q02_validation_review.md](../../../data/Multi-Organ_Foundation_Model/runs/moe_followup_review_20261016/q02_validation_review.md)、[机器可读复核 JSON](../../../data/Multi-Organ_Foundation_Model/runs/moe_followup_review_20261016/q02_validation_review.json)。

## 7. 后续执行顺序

1. **继续 Q05，修复已完成。** 原生GPU反向计算存在微小的非逐位差异，原预检却要求续步结果逐位相等。现仅在预检的恢复证明阶段开启严格确定性，全部原比较条件保留，四组真实GPU准入通过。既定12项配对已接续，冻结训练评价协议不变；详情见[修复和监测记录](Q05_RECOVERY_20261022.md)。
2. **完成 Q05 后进行验证复核。** Region 检验纯视觉前景／背景对比特征是否改善路由；Task-ID 检验任务条件 FiLM 对解码特征的直接调制。分别检查三指标、逐器官漏检／误检、护栏和专家使用，不因一个总平均上涨就认定模型改进成功。
3. **按 Q05 验证诊断决定 Q06。** 若 Region 的最细层仍被单个专家独占，才据完整验证证据设计独立路由优化配对。Q06 目前没有正式训练，不应记作正在排队运行的已实现实验。
4. **推进 Q07 机制对照。** 在 Q06 决定之后，落实同条件无专家／少专家对照，区分收益来自任务条件、参数容量还是专家机制。该阶段尚未实施训练。

执行计划：[MOE_NEXT_STEPS_20261016.md](MOE_NEXT_STEPS_20261016.md)；动态审查状态：[controller_status.json](../../../data/Multi-Organ_Foundation_Model/runs/moe_followup_review_20261016/controller_status.json)。

本次核验：131项相关测试通过，四组真实GPU预检通过；控制器PID `212829`，两项训练worker PID `213206`／`213212`。原生训练进程未继承预检的CUBLAS环境覆盖。独立审查PID `212991`，状态`waiting_q05`，已复用Q02完整验证报告。以上为10月22日16:39启动快照，不代表已有Q05完整轮次或性能结果。

手动查看：

```bash
watch -n 60 cat /ssd1/data/Multi-Organ_Foundation_Model/runs/q05_visual_context_task_film_80_512_20261014/controller_status.json
watch -n 5 nvidia-smi
```

## 8. 原始结果索引

- [Q02 完成汇总](../../../data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/summary.json)
- [Q02 全部三指标统计](../../../data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/evaluation_metrics.csv)
- [Q02 配对差值](../../../data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/evaluation_paired.csv)
- [Q02 逐结构／逐域／腹部来源统计](../../../data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/evaluation_per_group.csv)
- [Q02 完整评价汇总及诊断](../../../data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013/evaluation_summary.json)
- [Q04 四方法评价报告](../../../data/Multi-Organ_Foundation_Model/runs/q04_external_baselines_80_512_20261013/remaining_reports/REPORT.md)

