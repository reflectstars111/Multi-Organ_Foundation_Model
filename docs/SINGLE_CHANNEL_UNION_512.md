# 共享 U-Net（单通道）：前景并集协议

**完成核查（2026-10-09）：**50轮和80轮单通道实验均完成三种子训练及测试；两套结构模型派生并集评价也已完成。9域阳性并集测试宏Dice分别为85.25±0.23%和84.96±0.30%（三种子均值±样本标准差）。完整四模型比较见[正式结果](experiments/FAIR_COMPARISONS_20261009.md)。本页下方记载50轮协议；80轮的base24、有效batch16、lr1e-4配置与产物见[80轮协议](UNIFIED_80_512.md)。

本页模型就是用户指定的共享U-Net，与两种原设计MoE的结果见[主对照表](SHARED_UNION_VS_MOE.md)。它是一套普通 U-Net、**一个二值输出通道**、只输入超声图像，不输入数据域或任务ID。现有公平对照中的`shared`实际是一套U-Net加16个结构输出通道，现称为“16通道U-Net（附加对照）”；其既有协议和结果保留原代号以便追溯。

单通道目标按**每张独立图像**合并已标注结构前景，不将同一张腹部图拆成五个互相冲突的二值样本。已知前景为任一已标注通道的前景；仅当所有需要的已标注通道在像素处都有效时，才把它当作已知背景，否则忽略。腹部沿用当前 paper5 的肝、肾、血管、胆囊、脾五类；源掩膜中未纳入 paper5 的胰腺、肾上腺、骨及 `255` 像素作为忽略区。AUL 的 3 张仅有肿块、缺少肝标注的图像保留在原划分和抽样中：肿块前景有效，其他像素按未知处理，并单独记录其贡献。

50轮配置沿用同批公平对照：同一train/val/test划分、九个数据域、512×512、base16、batch4、AdamW学习率`3e-4`与weight decay `1e-4`、每域每轮500张图、训练种子42/43/44。图像抽样哈希已与同批结构模型逐轮核对一致；每轮4500张图、1125次优化步骤。训练只用并集二值BCE+soft Dice，验证集选择检查点；完整预算结束后显式测试。

单通道模型报告**并集** Dice、空标签误检和阳性漏检，可按数据域及腹部 CT 来源分层。它不能输出肝、肾等结构身份，因此不能计算可解释的逐结构 Dice。附加的16通道 U-Net、Task-ID MoE、Region MoE 的 16 结构前景宏 Dice 与此不同，不能直接排序。可以把它们的已标注通道预测取并集，在相同图像和有效像素上另算并集指标作描述性比较；其训练目标与单通道模型仍不同，不能将差异全部归因为架构。

50轮单通道队列已完成。结果目录为`/ssd1/data/Multi-Organ_Foundation_Model/runs/single_channel_union_512_20260930`，队列状态与日志在同级目录`single_channel_union_512_20260930-logs`。运行前核对了原Region分片配置、17,698条记录和抽样协议；原公平对照源码、权重与协议保留。

手动查询状态和最近日志：

```bash
cat /ssd1/data/Multi-Organ_Foundation_Model/runs/single_channel_union_512_20260930-logs/controller_status.json
tail -n 30 /ssd1/data/Multi-Organ_Foundation_Model/runs/single_channel_union_512_20260930-logs/single_channel.log
```

训练开始后，各种子的逐轮指标位于结果目录的 `seed_42/metrics.jsonl`、`seed_43/metrics.jsonl`、`seed_44/metrics.jsonl`；完成后查看 `summary.json` 和各 `seed_*/test_metrics.json`。

派生评价队列已使用三种结构模型各三种子的最佳检查点，在相同测试图像及有效像素上计算并集Dice、空标签误检和阳性漏检，结果位于`/ssd1/data/Multi-Organ_Foundation_Model/runs/fair_union_eval_512_20260930`。这9项是推理评价，不另计训练。手动查询历史产物：

```bash
cat /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_union_eval_512_20260930-logs/controller_status.json
tail -n 30 /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_union_eval_512_20260930-logs/evaluation.log
```

并集 Dice 只与其他模型在同一并集标签上的评估相比。原对照训练目标和验证检查点选择规则不同，因此这种比较用于描述任务表现，不作为纯架构差异的因果结论。历史测试集此前已用于探索性诊断，不能称为从未接触的独立外部盲测集。
