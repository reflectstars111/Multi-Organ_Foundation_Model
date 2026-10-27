# 医学实验统一补评：DSC、HD95、ASSD

制定日期：2026-10-12，北京时间。此任务对既有医学模型重新推理和评价，不重训、不按新指标重新选择检查点。原实验及 RadarVLA 产物保持原样。

**完成状态（2026-10-13 11:32:19，北京时间）：439项补评全部完成，失败0项。**97项排除、24项原协议未选中继续保留原因，不计作完成实验。下述四项标签修订已实际完成；其所有HC18逐图真值像素数均与填充区域标签一致，CSV的672条相关统计和32个跨种子分组均含修订variant，原435项结果及原协议哈希复核一致。Q04已自动接续到两卡GPU预检，Q02等待Q04完成。[最终核验记录](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/verification/filled_label_repair_final.json)。

**2026-10-13 HC18标签修订：**用户明确要求“需要的就是使用填充后的区域标签”。已经完成的435项补评所引用记录均使用`masks_filled`，无需重跑。剩余4项旧`region_top1_v1`（best/last × val/test）修复Windows路径，并将154张验证、139张测试图的HC18旧`masks/train`路径明确映射到同名`masks_filled/train`。其余图像和标签路径按原相对位置解析；6398个文件均已定位并计算SHA-256，293张填充标签的尺寸、二值取值及前景面积已核对处理清单。

这是用户授权的**评价目标修订**：旧检查点仍按轮廓训练，本次对填充区域真值计算指标，预测本身不作填洞或后处理。四项任务所有统计统一标记`hc18_filled_eval_after_contour_training`，使逐图结果、CSV、Markdown及跨种子汇总均能识别，且保持完整模型宏平均的分母。不可把这些值解释为原轮廓评价复现；历史256输入和旧训练预算也继续单列。原训练记录、检查点、冻结指标源码和435项完成结果保持原样，目标附录及实际文件内容清单独立冻结于`repair_snapshot/20261013_hc18_filled/manifest.json`；历史失败证据归档保留。

标签修订入口及原单／双卡补评调度共**104项测试通过**，实盘只读校验通过。2026-10-13 11:30:29（北京时间）启动后台恢复控制器，PID `1060118`，允许GPU0、GPU1各执行一项。启动不是完成，实时进度仍看`status.json`。修订清单SHA-256为`3802f3b1cfb574f3f8a09c12d4d21c34dc9615582fda71fcbaeb9a378f9ed68d`；[修订说明](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/LABEL_POLICY_AMENDMENT.md)、[启动记录](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/verification/filled_label_repair_launch.json)保存在运行目录。

这四项旧模型保留原19结构名称，其中`head_contour`是历史输出槽名称；带上述修订variant的该行，其实际评价真值已经是填充区域。正式16结构模型及其数据定义不受此历史命名影响。已完成修订任务的逐图真值像素数另与填充标签按原评价分辨率逐一核验，记录在`verification/filled_label_completed_targets_verified.json`。

**2026-10-13双卡调度更新：**补评由独立调度器向GPU0、GPU1各派发最多一个任务。任一卡先空闲就接续，另一张卡的训练继续运行。调度器单独冻结于`scheduler_snapshot/20261013_dual_gpu/`；推理与指标仍执行原`source_snapshot`，原评价协议哈希和已完成结果沿用。Q01／Q04／Q02训练队列本身已支持两卡，无需更改其训练配置。

切换时核验：新控制器PID `900311`，`gpu_indices=[0,1]`，两卡均处于`waiting_gpu_lock`；原Q01两项训练继续运行，4项完成补评及原协议／推理源码哈希一致。相关调度与补评测试62项通过。[交接核验记录](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/verification/dual_gpu_handoff_verified.json)。后续状态以运行目录中的`status.json`为准。

## 结果入口

运行目录：`/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012`。

| 文件 | 内容 |
|---|---|
| [REPORT.md](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/REPORT.md) | 已完成补评的模型级结果，随队列更新 |
| [metrics.csv](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/metrics.csv) | 逐器官、逐域、腹部、CT来源与整体宏平均，包含三个指标及覆盖数量 |
| [summary.json](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/summary.json) | 完成数量和同协议跨种子均值、样本标准差 |
| [inventory.json](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/inventory.json) | 全部任务，以及暂缓、排除、未选择或不可用的原因 |
| [status.json](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/status.json) | 后台控制器与当前评价任务 |
| [protocol.json](/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012/protocol.json) | 冻结的指标定义、来源校验和任务范围 |

旧汇报中的 Dice 表仍是原实验快照，HD95/ASSD 不能从旧 Dice 数字反推。上述报表只纳入真正完成重新推理的任务；没有算完的结果不填充或推测。

## 补评范围与比较边界

盘点覆盖服务器上全部 **232 个现存检查点**。任务按检查点选择方式与划分展开，共 **560 个记录**；初始化前快照为 315 个可评价、128 个等待原训练／测试边界、97 个排除、20 个未被原协议选中。运行时以冻结初始清单和实时状态为准，任务数不是训练次数。

- 正式 50／80 轮：单通道并集 U-Net、16 通道 U-Net、Task-ID MoE、Region MoE，保留各自种子与训练预算。
- Q01 器官加权、Q03 数据域归一化：已完成训练的验证结果先补评；测试必须等待原实验的全训练完成条件及测试开放证据。
- 腹部空标签消融、18 次腹部恢复微调、有效的 80 条后处理规则，以及可恢复的早期独立 U-Net、共享和 MoE 结果。原协议仅验证的探索仍只评价验证集。
- `best`、`last`、原协议实际选中的护栏检查点分别列出。微调选回初始化权重也如实保留，不把它算成微调收益。
- 历史 256 输入保留 256 评价网格，当前正式配置保留 512。不同分辨率的 pixel 距离、不同结构集合、划分和训练预算不合并。历史 7/12 轮中止消融标记为部分训练，不混入完整训练结果。
- smoke、已作废的后处理首版、没有可恢复权重的历史文字结果和派生汇总不伪造评价值，清单保留排除／缺失原因。

结构分割与前景并集是不同任务，分开报告。单通道 U-Net 仅有并集结果；MoE 和多通道模型同时产生逐结构与可比的已标注结构并集结果。九个独立 U-Net 除逐域结果外，合并结构／域统计生成整体结果；必须九域齐全才能标为完整集合，缺域结果明确标记。

## 三个指标的统一定义

以原协议预处理后的二维二值掩膜计算，普通模型概率阈值为 0.5，已冻结的后处理筛查保留各候选规则。

| 指标 | 定义 | 方向 |
|---|---|---|
| DSC (%) | `100 × 2 × intersection / (prediction_pixels + target_pixels)`，不加平滑项 | 越大越好 |
| HD95 (pixel) | 两个方向的边界最近邻距离合并后取第 95 百分位 | 越小越好 |
| ASSD (pixel) | 两个方向的边界最近邻距离合并后取均值，按边界点数量加权 | 越小越好 |

边界由二维四邻域腐蚀提取；距离为欧氏距离，像素间距固定 `[1,1]`，单位是评价网格的 **pixel，不是 mm**。HD95/ASSD 的双向合并定义与 [MedPy 官方实现](https://github.com/loli/medpy/blob/master/medpy/metric/binary.py) 一致。

### 空掩膜和忽略像素

采用用户明确选定的**有限对角线惩罚**：仅一侧为空时，DSC=0，HD95=ASSD=`hypot(H-1,W-1)`。这里使用最远像素中心之间的距离；512×512 时为 **722.6631 pixel**。误检和漏检都计入惩罚，并报告 `n_penalized`，不删除失败样本。

两侧都空时，DSC=100，HD95=ASSD=0；全图无有效标注时排除并计数。标签 `-1` 像素不参与 DSC，接触四邻域未知像素的边界点不参与距离计算，以免把未知区域边缘当成已知器官轮廓。两侧有前景但没有可观测边界时，距离为 `null`／NA，并显式报告无定义数量，不用零替代。

### 全量结果和固定真值边界子集

当前正式测试集 AUL 的 92 张图中，`Malignant/229.png` 与 `Malignant/306.png` 仅标注了肿块，背景未知；它们没有可观测的完整边界。此问题涉及全部 1618 张测试图中的 2 张，不能用模型预测来决定是否剔除。

因此保留四种统计口径：

| cohort | 纳入范围 |
|---|---|
| `all` | 所有有有效标注的图，含空标签图 |
| `positive` | 真值前景非空的图，完整保留无定义距离及 NA |
| `all_surface_defined` | 在 `all` 中，仅剔除真值非空且没有可观测边界的图 |
| `positive_surface_defined` | 在 `positive` 中，使用同一真值边界条件 |

后两者的筛选条件**只由真值确定，所有模型一致**，不依据预测质量筛选。真值有边界但预测为空的图继续按对角线惩罚计入；预测本身没有可观测边界的情况仍保留 NA。报表同时列出原覆盖分母、实际样本数和真值边界排除数量。有限距离子集均值如出现，仅作诊断，不替代这些预定义结果。

每结构／域先对图像取均值，再等权计算结构／域宏平均；腹部及 CT 来源另外列出。跨种子仅合并模型、实验、选择方式、划分、输入尺寸、预算和 cohort 一致的结果，报告实际种子数量与样本标准差。

## 实现与可追溯性

- 入口：[run_segmentation_reevaluation.py](../scripts/run_segmentation_reevaluation.py)。指标、盘点、推理适配与汇总分别在 `unet_moe/surface_metrics.py` 和 `unet_moe/reevaluation_*.py`。
- 独立评价依赖：[requirements-evaluation.txt](../requirements-evaluation.txt)；当前项目 `.conda` 环境已有所需依赖。
- 每任务保存 `jobs/<job_id>/job.json`、`per_image.jsonl`、`result.json`、`progress.json`；日志在 `logs/<job_id>.log`，失败单独记录 `failure.json`，不冒充完成。
- 指标代码、清单与协议冻结到运行目录；校验检查点、原始划分、原实验可用的源码／数据哈希和逐图结果。旧实验没有源码清单时明确记录证据限制，不声称有完整历史冻结。
- 断点恢复复用经校验的完成结果。原实验检查点、训练协议、结果和优先级控制器不写入任何改动。

启动前验证：完整医学测试 **497 项通过、2 项跳过**；41 种历史模型／配置组合的真实检查点严格加载通过，14 类模型的真实样本标签与原始入口一致。验证记录随运行目录保存。

首个完整补评于 2026-10-12 11:12 完成：50 轮公平对照的 Region MoE／seed42／best／test，共 1618 张图、4882 条逐结构与并集记录。16 结构阳性宏平均为 **DSC 69.3124%、HD95 116.7213 pixel、ASSD 84.2120 pixel**，其中 193 条阳性结构记录为空预测并计入对角线惩罚。重算 DSC 与原测试宏平均相差 0.000095 个百分点，逐结构最大差 0.00277 个百分点；一致性记录在 `verification/first_complete_job_dsc_consistency.json`。这只是首个已完成任务，全部模型结果以动态报表为准。

## 运行与手动查看

后台补评使用 **GPU0和GPU1**，每卡最多一个进程、每进程4个边界指标CPU工作线程，与原训练共享GPU文件锁。每次派发检查训练优先级；已有训练或测试待调度时停止新派补评，已开始的补评任务完成后交还卡。Q01只剩正在运行的任务、没有待派任务时，补评可利用另一张空卡。保持至少18000 MiB空闲显存的原准入条件；原测试边界未开放的任务继续等待，不提前测试。

```bash
EVAL_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_metrics_dsc_hd95_assd_20261012
cat "$EVAL_RUN/status.json"
watch -n 30 "cat '$EVAL_RUN/status.json'"
tail -f "$EVAL_RUN/filled_label_repair_controller.log"
less "$EVAL_RUN/REPORT.md"
```

`status.json`的`gpu_indices`显示允许使用的两张卡，`active_jobs`分别给出GPU、worker PID和job ID。逐图进度查看其中的`job_id`对应目录：

```bash
cat "$EVAL_RUN/jobs/<job_id>/progress.json"
tail -f "$EVAL_RUN/logs/<job_id>.log"
```

控制器退出后的恢复命令（已有控制器时锁会拒绝重复启动）：

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
nohup .conda/bin/python "$EVAL_RUN/repair_snapshot/20261013_hc18_filled/repair_legacy_region_paths.py" \
  --run-controller --output "$EVAL_RUN" --gpus 0,1 --poll-seconds 15 \
  >> "$EVAL_RUN/filled_label_repair_controller.log" 2>&1 &
```

初始化新评价目录仍由原入口使用`--initialize --empty-policy diagonal --metric-workers 4`；当前已初始化目录通过上述标签修订入口恢复，它复用冻结双卡调度器和原指标实现，并验证目标附录、文件内容及四项白名单。禁止重新覆盖协议或同时启动旧控制器。后台完成后`state`为`complete`；`complete_with_failures`或`complete_with_unavailable`表示仍有需处理的缺口。
