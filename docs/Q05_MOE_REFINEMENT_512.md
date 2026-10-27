# Q05：Region视觉上下文与Task-ID直接条件调制

**2026-10-25完成核验：**Q05于10月24日04:16完成全部12项80轮训练与12项评价，完整验证审查也已完成。正式三指标及best/last差异见[最新结果](MEDICAL_PROGRESS_20261025.md)；Q06审查条件已触发但尚未实现入队，Q07也尚未开始。下方早期运行状态按原日期保留。

**2026-10-22最新状态：**Q02完整训练和评价已完成。Q05此前因GPU续步逐位检查阻塞，本次已通过独立预检修复层恢复；16:39四组GPU预检通过，双卡运行Region/control/42与Region/refined/42，另10项待训。原模型、训练协议及源码快照保持原样。当前运行和恢复入口见[Q05修复记录](Q05_RECOVERY_20261022.md)，下文历史状态按原时间保留。

日期：2026-10-14。目标是在不改变两条主线输入约定的前提下，检验更充分的条件表达能否改善分割。用户已要求实现独立版本、准备配对实验，并于11:44（北京时间）明确要求加入训练队列。当前顺序为Q04 → Q02 → Q05；各实验的冻结协议保持不变。实现与核验状态在本文末尾记录。

**14:30调度更新：**用户要求延期UniverSeg，Q04改由其余四方法先完成训练和评价，再接Q02、Q05。Q05控制器及其完整Q02前置检查保持原样，详见[独立调度覆盖记录](Q04_UNIVERSEG_DEFERRAL_20261014.md)。

## 1. 设计边界与依据

Region保留**从图像预测区域、再根据区域视觉特征选择专家**的设计，不增加器官ID或数据域ID作为router输入。此前提出的固定器官身份嵌入已撤下：它不等于泄露真值，但会引入通道身份先验，改变本轮希望研究的纯视觉区域路由机制。

这里的“纯视觉”指推理路由的条件来自图像，并不是没有器官语义监督：原模型已有固定的16结构输出，Q02也按这些器官类别监督proposal。Q05保持这一约定，同时避免向router额外注入器官编号。

Task-ID保留已知任务ID输入与逐结构查询。现实现中任务embedding直接进入router，解码特征没有独立的任务调制路径；这不意味着原模型没有利用任务条件，top1选择及其保留的softmax幅值本就能传递条件和梯度。

Q01已经表明损失加权可以改善DSC，但Task-ID同时出现整结构漏检增加。损失权重不是梯度贡献的测量，现有结果尚未确定漏检的因果机制。本轮单独检验结构条件表达，阳性图/空标签图的曝光平衡另行设计，不与结构变化混在同一个候选中。

Q02已实现Region四层proposal辅助监督及Task-ID弱负载均衡。本轮两臂都固定这些辅助项，检验**在同样辅助监督下，新增条件表达的独立收益**；不将已有Q02功能重复描述为Q05创新。

## 2. Region：纯视觉前景与背景对比池化

现有一层输入为特征`x[B,C,H,W]`、预测区域概率`m[B,16,H,W]`，以每个区域的池化特征送入router。新增该预测区域的互补背景特征：

\[
z_r^f=\frac{\sum_p m_r(p)x(p)}{\max(\sum_p m_r(p),\epsilon)},\qquad
z_r^b=\frac{\sum_p(1-m_r(p))x(p)}{\max(\sum_p(1-m_r(p)),\epsilon)}.
\]

\[
s_r=\operatorname{Router}\left(z_r^f+A(z_r^f-z_r^b)\right).
\]

`A`为每层独立、区域间共享的`C×C`无偏置线性投影，全部零初始化。前景或背景概率总量不超过`1e-6`时，对比项置零。池化使用模型预测，不使用真值掩膜、器官身份或标注可用性决定实际路由。

四层通道数192/96/48/24，增加 **48,960个参数**。原proposal、router、专家、共享分支、空间融合及16通道输出均保留。候选初始函数与控制组一致；训练后router可同时依据区域内容和相对周围背景的区别选择专家。

可能失败的原因：proposal仍可能不准确；空间恒定的mask使前景与背景池化相同，新增投影没有有效信号；模型也可能过度利用数据域背景。Q02的proposal监督在两臂中均存在，但不能预先保证消除这些问题。需记录专家活动数、路由分布、proposal质量和耗时。

**不将top1选中概率归一为常数1。** 保留原softmax幅值和分割梯度路径；本轮不硬性规定器官对应哪个专家。

## 3. Task-ID：四层解码特征直接调制

对每层MoE输出`h[B,C,H,W]`，由原任务embedding `e_t[B,32]`产生通道级调制：

\[
h'=(1+\gamma(e_t))\odot h+\beta(e_t).
\]

每层线性投影`32→2C`的权重和偏置全部零初始化，增加 **23,760个参数**。初始预测、公共参数与原模型一致；任务条件可以直接调整特征，而不只经过router。FiLM是已发表的通道仿射条件机制，本实验检验其在本项目中的作用，不将该机制本身声称为新方法。[FiLM原论文](https://arxiv.org/abs/1709.07871)。

风险是任务特化过强或仅学到任务偏置；低DSC器官不一定因此恢复。第一步FiLM投影可获得梯度，但其零权重对应的新增embedding梯度路径为零是正常现象，原router路径仍存在。若要声称收益来自MoE而非一般条件调制，后续还需同FiLM、无专家的条件U-Net对照。

## 4. 十二项独立配对实验

| 架构 | control | refined | 种子 | 训练数 |
|---|---|---|---|---:|
| Region | C主损失＋0.1×四层proposal损失均值 | control＋视觉对比池化 | 42/43/44 | 6 |
| Task-ID | C主损失＋0.01×四层负载均衡均值 | control＋Task-FiLM | 42/43/44 | 6 |

C=`organ_exposure_fg_bg`来自Q02已冻结的验证选择，绑定原selection文件及协议指纹，不根据后来看到的测试结果改成B。控制组独立重跑，不直接把旧Q01/Q02的数字填入本轮控制结果。

固定512×512、80轮、base24、9专家、top-k1、有效batch16、物理microbatch4、AdamW lr1e-4/weight_decay1e-4、float32、GN、无增强、原9域数据及16结构标签。沿用相同训练清单与逐轮抽样，HC18使用填充区域，未知标签继续忽略。各域每轮500张，共4500张、282次更新；任务查询展开规则保持不变。

同架构同种子两臂的公共初始参数、初始模型函数和随机数状态相同；新模块不消耗额外随机初始化。完整候选state包含新增参数，不能误称两个state字节完全相同。检查点明确记录arm及初始化指纹，禁止用控制组工厂静默恢复候选权重。

## 5. 评价与解释

保留原验证16结构DSC选出的best、epoch80的last和原Q01护栏best；未通过护栏则明确记录无候选，不互相替换。12项训练全部核验完成后才开放测试，统一提供结构及并集的DSC (%)、HD95 (pixel)、ASSD (pixel)，以及逐器官、腹部来源、整器官空预测、零交集、空标签误检及面积。

评价阈值0.5；单侧空掩膜距离按512网格对角线722.6631 pixel惩罚。保留阳性全部样本和真值边界可定义口径及其分母，不删除空预测改善距离分数。

每项结论必须同时展示三种子的`refined−control`差值。DSC提高但整器官漏检或距离恶化应报告为取舍，不称全面提升；仅有一个种子的改善不称稳定收益。已有测试集曾用于历史分析，因此本轮属于同一项目内的后续探索，不能包装成完全未触碰测试集的外部验证。

诊断继续记录完整验证集路由、proposal、去掉专家残差后的预测，以及固定训练图上的专家、共享块、router和新增投影/FiLM梯度。梯度为逐图目标梯度范数的平均，不是有效batch总梯度；诊断须恢复模式、随机数和buffer，并保留既有`.grad`。

本轮不承诺超过AImageLab，也不因其验证成绩较高而扩大参数量、改变预算或为各方法分别调阈值。

## 6. 实现与执行状态

**2026-10-14 11:44:50（北京时间）已入队，控制器PID `1730462`，状态`waiting_q02`。** 12项训练、12项最终评价及4组GPU预检均待执行。Q02仍在`waiting_q04`，因此当前Q05控制器只等待、不占GPU；Q02完整训练/评价及产物核验通过后，自动开始Q05准入和双卡调度。入队前的`prepared_not_enqueued`记录保留为历史准备快照。

运行目录：`/ssd1/data/Multi-Organ_Foundation_Model/runs/q05_visual_context_task_film_80_512_20261014`。

协议SHA-256：`5f1dc7ab5b81d49ac605413862f4072de06b59c885e872618a6f6d0d0fdc4124`。已核验82份冻结源码、训练/验证文件内容、三种子共240份逐轮抽样和12组初始化指纹。测试文件此阶段不打开；Q02协议/选择原字节及哈希已绑定。

| 代码 / 证据 | 用途 |
|---|---|
| [模型](../unet_moe/refinement_models.py) | Region视觉对比、Task-FiLM；控制与候选工厂 |
| [完整训练入口](../unet_moe/refinement_suite.py) | prepare、GPU准入、训练、恢复、核验、评价、汇总 |
| [评价](../unet_moe/refinement_evaluation.py) | 结构与并集三指标、逐种子配对差 |
| [诊断](../unet_moe/refinement_diagnostics.py) | 全验证路由、专家残差对照、新增分支梯度 |
| [控制器](../scripts/run_moe_refinement.py) | 双卡调度、前驱核验、失败持久化及恢复 |
| [准备核验](/ssd1/data/Multi-Organ_Foundation_Model/runs/q05_visual_context_task_film_80_512_20261014/verification/preparation.json) | 入队前的源码/数据/初始化冻结快照 |
| [实际入队记录](/ssd1/data/Multi-Organ_Foundation_Model/runs/q05_visual_context_task_film_80_512_20261014/verification/enqueue.json) | 启动时间、PID、命令、依赖与实际等待状态 |
| [实时控制器状态](/ssd1/data/Multi-Organ_Foundation_Model/runs/q05_visual_context_task_film_80_512_20261014/controller_status.json) | 四组准入、12项训练/评价进度及失败原因 |
| [CPU 512检查](/ssd1/data/Multi-Organ_Foundation_Model/verification/q05_20261014/cpu_512.json) | 实际腹部训练图，四组各2次更新，新增参数与router梯度、严格恢复 |
| [CPU检查脚本](experiments/q05_cpu_verification_20261014.py) | 可复现的功能核验，不读取测试图 |
| [本次验收记录](experiments/Q05_PREPARATION_20261014.json) | 测试结果、正式训练未启动及来源指纹 |

**验证结果：医学测试780 passed、2 skipped；其中Q05专项90 passed。** 两项跳过来自原有reallocation测试的本地数据路径条件，不属于本次模型测试。真实512检查使用base24、单张腹部图展开5个结构查询，四组各做两次AdamW更新；初始预测与控制一致、增量参数及router梯度非零、保存恢复预测相同。这是CPU功能验证，不等于物理batch4/有效batch16已通过GPU显存准入，也不代表分割效果已有提升。

四组GPU预检需在启动正式训练前单独完成。其显存峰值反映最重查询批次在初始更新时的实际路由，不是训练后期激活更多专家时的数学上界；控制器保留显存余量和失败关闭，不会在OOM后自动缩分辨率或修改预算。

## 7. 手动检查与恢复命令

以下检查不启动任何训练：

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
Q05_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/q05_visual_context_task_film_80_512_20261014
Q02_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/q02_routing_80_512_20261013
.conda/bin/python "$Q05_RUN/source_snapshot/scripts/run_moe_refinement.py" \
  --output "$Q05_RUN" --reference-root "$Q02_RUN" --after-root "$Q02_RUN" \
  --gpus 0,1 --dry-run
```

**控制器已在后台运行，不要重复执行启动命令。** 以下仅供控制器退出后的恢复。控制器先等待并只读核验Q02的完整训练/测试，再做四组GPU准入及双卡调度，不抢占Q04/Q02。重启同一目录会保留已有检查点并恢复；失败需查明原因后显式加`--retry-failed`。

```bash
nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .conda/bin/python "$Q05_RUN/source_snapshot/scripts/run_moe_refinement.py" \
  --output "$Q05_RUN" --reference-root "$Q02_RUN" --after-root "$Q02_RUN" \
  --gpus 0,1 > "$Q05_RUN/controller.log" 2>&1 < /dev/null &

# 手动查看当前进度
watch -n 30 "cat $Q05_RUN/controller_status.json"
tail -f "$Q05_RUN/controller.log"
```
