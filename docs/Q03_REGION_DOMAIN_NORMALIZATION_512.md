# Q03：Region MoE 九数据域独立归一化配对实验

设计日期：2026-10-10（北京时间）。用户已明确选择按**九个数据域**设置独立BatchNorm；腹部五器官共用腹部域的归一化，Region仍一次输出16个结构。

**当前阶段（2026-10-11）：模型与实验链路已实现、测试通过、协议已冻结；执行顺序调整为当前两项Q01训练完成 → Q03全部12项训练与评价 → 恢复Q01剩余任务。** 正式Q03结果尚未产生。[优先级交接与监测](Q03_PRIORITY_HANDOFF_20261011.md)记录新的后台顺序；本页第7节保留10月10日的历史冻结记录。Q02仍是另一项待实现与冻结的独立设计。

## 1. 假设与改动边界

当前Region的卷积块实际使用`GroupNorm(4, C)`。GN按单张样本的通道组与空间位置计算统计，跨数据域共享的是可学习的缩放／偏移参数，并不存在混合多个数据域的BN运行均值。实验要检验：**为不同数据域分开归一化参数与BN运行统计，能否改善跨域分割与腹部小器官表现。**

域的顺序固定为`BUSI / MMOTU / DDTI / AUL / FALLMUD / Fetal_HC / CCA / CAMUS / AbdomenUS`。域ID只来自图像记录的`dataset`元信息；训练、验证、测试都显式传入。不得从真值掩膜、标注通道或器官是否阳性推断域ID。分域模型的输入契约是“图像＋已知数据域”，应明确报告这个额外条件。

本轮修改密集共享路径的13个卷积块、26处归一化：5个编码块、4个解码融合块、4个MoE共享分支。36个稀疏专家块的72处GN保持原实现。因为专家只接收到路由选择的图像子批，进一步按域拆分会产生更稀疏的“域×专家”统计；本轮先隔离共享特征上的归一化效应。

分域BN的每个位置有9套独立的`weight/bias/running_mean/running_var/num_batches_tracked`。空域不调用BN、不更新统计；验证／测试使用训练积累的running statistics，不在验证／测试图像上重新估计。GN四通道组；BN固定`eps=1e-5, momentum=0.1, affine=True, track_running_stats=True`。

方法依据：[GroupNorm文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)、[BatchNorm2d文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)、[Domain-Specific Batch Normalization原论文](https://arxiv.org/abs/1906.03950)。这些资料说明机制，不证明该改动在本项目上有效。

## 2. 四臂配对矩阵

所有行均为Region MoE，不包含共享U-Net或Task-ID MoE。

| 组 | 代码标识 | 密集路径归一化 | 稀疏专家 | base24参数量 | 种子 |
|---|---|---|---|---:|---|
| N0 | `shared_gn` | 当前共享GN | GN | 13,311,644 | 42/43/44 |
| N1 | `domain_gn` | 每域独立GN仿射参数；统计仍按图计算 | GN | 13,358,492 | 42/43/44 |
| N2 | `shared_bn` | 共享BN参数和运行统计 | GN | 13,311,644 | 42/43/44 |
| N3 | `domain_bn` | 每域独立BN参数和运行统计 | GN | 13,358,492 | 42/43/44 |

共**4组×3种子＝12次从头训练**。分域组增加46,848个参数（约0.35%），不增加卷积或路由网络；参数量和运行成本仍分别记录。

主要比较：

- N3−N2：在BN下分开数据域的效果，直接检验用户提出的假设。
- N1−N0：仅分开GN仿射参数的效果。
- N2−N0：在共享设置下更换GN→BN的效果。
- N3−N0：完整候选相对当前Region的效果。
- N3−N1：两者都按域独立时，BN与GN的差别。

不同归一化的统计定义不同，不能仅做N3−N0就把全部增益归因为“按域独立”。

## 3. 固定条件与公平性

| 项目 | 冻结设置 |
|---|---|
| 数据 | 复用Q01冻结的`paper5/trainmore_v6/ct7_2`记录、数据内容哈希与抽样计划 |
| 输入 | 512×512 |
| Region | base24、9专家、top-k=1、16结构输出 |
| 训练预算 | 80轮，每域每轮500次图像抽样，总4500次／轮、360000次／训练 |
| batch | 物理microbatch4，有效batch16，末尾不足批按实际图像数归一 |
| 更新次数 | 282次／轮，22560次／训练 |
| 优化器 | AdamW，lr1e-4、weight_decay1e-4、betas(0.9,0.999)、eps1e-8 |
| 学习率／精度／增强 | 恒定／float32／无增强 |
| 主损失 | 原始`image_mean`：逐图有效结构的masked BCE＋soft Dice平均，再平均图像 |
| 辅助项 | coarse、指定专家路由、负载均衡等均为0 |
| 阈值／梯度裁剪／workers | 0.5／范数5／0 |
| 配对 | 同seed的全部卷积、head、proposal、router初始参数逐张量一致；归一化gamma1、beta0 |
| 选优 | 16结构验证阳性宏Dice选best，另存last80与既有Q01统一护栏best |

Q01结果只提供已冻结的数据协议和抽样计划，不作为本实验预训练权重。四组同期从头训练；主损失统一为原平均损失，不使用test挑选损失，也不同时叠加Q01加权或Q02路由辅助项。

GN对照应与当前默认Region严格兼容；BN和GN的可学习参数初值一致，但初始函数通常不同。梯度累积只合并梯度，**不把BN统计的物理microbatch4变成16图统计**。

读取Q01三种子×80轮的全部270000个物理小批后：99.8622%的小批包含多个域；按域拆分后的非空子批中83.1191%只有1张图，平均1.18265张。空间维仍为BN提供统计元素，因此不是“batch1必然报错”，但运行统计的可靠性需要实测。首轮保持原混域小批，不单独为N3改采样分组；后续若研究同域小批，应四组一同重跑。

## 4. 实现与完整性检查

模型接口：

```python
from unet_moe.region import RegionUNet, PAPER_ALIGNED_TASK_IDS

model = RegionUNet(base=24, top_k=1, task_ids=PAPER_ALIGNED_TASK_IDS,
                   normalization='domain_bn')
# image: [B, 1, 512, 512]
# domain_ids: [B]，整型，与image同设备；范围0～8
output = model(image, domain_ids)
# output['logits']: [B, 16, 512, 512]
```

模型和实验文件：

- [Region模型](../unet_moe/region.py)、[分域归一化模块](../unet_moe/normalization.py)。
- [独立训练／评价入口](../unet_moe/region_normalization_suite.py)。
- [后台队列](../scripts/run_region_normalization.py)。

数据包装器返回显式的`image,target,domain_id`，检查同一图像的记录只属于一个域。恢复检查点同时恢复优化器、随机状态、归一化参数和全部BN缓冲区。协议记录域顺序、归一化范围、各臂参数、源文件与数据内容哈希；源文件被冻结后，从运行目录中的`source_snapshot`执行。

每轮保存逐器官和腹部CT来源指标、路由诊断、各域实际小批大小／图像曝光量，以及归一化运行统计摘要和更新次数。检查评价前后模型缓冲区不变；不对val/test执行BN校准。

必要检查包括：旧GN严格加载及输出兼容、四臂卷积／路由初始化一致、混域索引还原、未出现域不更新统计、eval单图与不同批次组合结果一致、保存恢复包含BN状态、梯度累积按实际图像数缩放、未知域拒绝、断点续训一致、数据／源文件指纹和测试屏障。

## 5. 评价与“是否提升”的回答方式

全部12项训练及验证选优完成后统一执行test。使用同一套best／last／guardrail口径，护栏未选中时保留空值，不以其他检查点代替。

报告以下结果，单位及分母与现有正式对照一致：

1. 16结构阳性宏Dice、腹部五类宏Dice、全部逐结构Dice，单列血管。
2. 阳性漏检与零重叠率、空标签误检率和误检面积，按CT来源分层。
3. 逐种子配对差值及三种子均值／样本标准差；明确正负方向是否一致。
4. 统一前景并集评价，与已有单通道共享U-Net的下游评分对接。域条件与训练目标不同应同时说明。
5. BN每域更新量、运行方差、路由使用分布，以及耗时和显存。

主要效应为N3−N2的16结构阳性宏Dice；N3−N0回答是否优于当前Region。若均值改善但种子方向不一致，只报告不稳定改善；若整体上涨而血管或腹部误检退化，报告收益／代价，不称全面提升。三个种子仅支持初步稳定性判断，不宣称统计显著或外部泛化优势。

## 6. 调度与手动监测

已冻结的运行目录：

```text
/ssd1/data/Multi-Organ_Foundation_Model/runs/region_domain_norm_80_512_20261010
```

2026-10-11按用户确认调整为：Q01当前两项训练自然完成并通过冻结入口的80轮产物核验后，优先执行Q03。独立adapter只替换原控制器“等待Q01完整结果”的门槛，仍核验Q01/Q03冻结数据和抽样指纹；之后获取现有GPU文件锁，完成四臂真实512输入的GPU预检，再按显存准入执行。12项训练完成前不得提前启动本轮test。Q03全部评价和汇总核验通过后，自动恢复原Q01控制器。用户手动查看状态即可。

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
NORM_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/region_domain_norm_80_512_20261010
PRIORITY_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_priority_q03_20261011

# 优先级交接状态；draining表示当前两项Q01仍在自然训练
cat "$PRIORITY_RUN/status.json"

# Q03预检／训练／测试／完成状态
cat "$NORM_RUN/controller_status.json"

# 持续查看主控制器日志
# Ctrl+C只退出tail，不停止训练
tail -F "$PRIORITY_RUN/q03_controller.log"

# GPU阶段开始后查看训练日志
tail -f "$NORM_RUN/logs/gpu0.log" "$NORM_RUN/logs/gpu1.log"

# 显存和GPU使用率
nvidia-smi
```

## 7. 历史落地记录（2026-10-10，调度顺序已由第6节更新）

状态核验时间：`2026-10-10T20:12:31+08:00`。控制器PID **3924781**，状态 **`waiting_reference`**；训练0/12、测试0/12完成，全部等待Q01闭环。此时控制器未申请GPU、没有训练子进程。

- 协议SHA-256：`13835cabeb0fb18e06905b4d7aa55618aaae3ea378b824c6712b602cda536442`。
- 记录SHA-256：`f621896f10234777d3067a019d96a7cb1dae3b6add305bca32815298744315e0`，与Q01一致。
- 冻结34个源文件，校验全部数据内容、三种子抽样计划、共享GN与Q01 Region初始化、四臂卷积／路由初值一致。
- 最终全量医学回归：**316 passed，2 skipped**（40.64秒）。
- 四臂CPU真实512输入均输出有限的`[1,16,512,512]`，评价前后BN缓冲不变。
- 额外合成CPU端到端检查：32输入/base4/1轮，完整执行12组训练、12组结构与并集测试、5项配对汇总；仅是工程验证，不计作正式训练或性能收益。
- 修复并覆盖回归：测试阶段不再误传训练`--resume`；快照包初始化文件和导入路径均冻结，混合工作区PYTHONPATH时仍导入快照代码。
- **GPU512预检尚未运行**：由队列在Q01完成后执行；通过后才进入正式训练。归一化是否提升性能仍待12组结果。

[冻结协议](/ssd1/data/Multi-Organ_Foundation_Model/runs/region_domain_norm_80_512_20261010/protocol.json)、[实施检查记录](/ssd1/data/Multi-Organ_Foundation_Model/runs/region_domain_norm_80_512_20261010/implementation_checks.json)、[实时状态](/ssd1/data/Multi-Organ_Foundation_Model/runs/region_domain_norm_80_512_20261010/controller_status.json)。

优先级调度器异常退出时由后台watchdog恢复；两个调度进程均消失且状态不是`failed`／`complete`时，可用以下命令恢复。存在活动调度器时独占锁会拒绝重复启动。若状态为`failed`，先查看错误日志处理原因；不要另开原Q03等待入口，否则会恢复成“等待完整Q01”的旧顺序。

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
PRIORITY_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/medical_priority_q03_20261011
nohup .conda/bin/python "$PRIORITY_RUN/source_snapshot/scripts/run_medical_priority.py" \
  --manifest "$PRIORITY_RUN/plan.json" >> "$PRIORITY_RUN/supervisor.log" 2>&1 &
```
