# 器官区域特征驱动的 U-Net MoE

更新日期：2026-09-30。模型只输入灰度图，不输入 task ID、数据集 ID 或真实 mask。真实标签用于训练监督。19 类 `all8` 或 16 类 `paper5` 由 `task_ids` 决定；旧 task-ID checkpoint 与此架构不兼容。

## 结构与实际计算

共享五级编码器（默认 base=16，通道 16/32/64/128/256）→ 四级解码器 → 多通道独立 sigmoid 输出。每级解码器执行：

1. 上采样并融合 skip，得到特征 F。
2. 预测软区域 M，计算区域加权特征 `z_r = sum(M_r * F) / sum(M_r)`。
3. 共享路由 MLP 根据每个区域的视觉特征，从 9 个专家中选择 Top-k；没有显式类别 embedding。
4. 用区域概率和路由权重对专家输出做空间加权，并加上共享卷积分支。

每个解码阶段各有 9 个专家和一个共享分支，不是整个模型只有 9 个卷积模块。预测区域位于专家计算之前，推理无需真值区域。Top-k 是**每区域**的选择；多个区域的专家并集可覆盖全部专家，且被选专家计算整张特征图，不是裁剪 ROI。Top-1 不能直接解释为每图只执行一个专家或必然节省算力。

## 专家监督与标签

| 专家组 | 结构 |
|---|---|
| BUSI | breast_lesion |
| MMOTU | ovarian_lesion |
| DDTI | thyroid_nodule |
| AUL | liver、liver_mass |
| FALLMUD | aponeurosis |
| HC18 / Fetal_HC | head_contour（当前为填实 mask） |
| CCA | carotid |
| CAMUS | lv_cavity、myocardium、left_atrium |
| AbdomenUS | all8：肝、肾、胰腺、血管、肾上腺、胆囊、骨、脾；paper5：肝、肾、血管、胆囊、脾 |

9 是专家组数，19/16 是有效分割结构数，21 是 legacy task 编号槽位数。FALLMUD fascicle 和 AUL ultrasound_outline 不进入此模型。AUL 与腹部的肝脏通道分别保留，不合并不同来源标注。

旧 `region_train` 默认总损失为 `最终 BCE+Dice + 0.3 × 粗区域 BCE+Dice + 0.1 × 路由分类损失`。分割按图像平均有效通道，腹部阳性图像权重由训练标签统计。路由分类只监督有前景结构，映射到上表专家组；它引导分工，但不保证测试时正确识别器官存在性。`fair_suite` 默认所有架构辅助项为 0，先统一主目标；启用辅助项应单列为消融。

缺失通道和腹部 ignore 像素设为 -1，不作为负标签。HC18 要求全部 999 张标注图采用 `hc18_filled_v1`；只有 aponeurosis 等仍为细线/薄结构，低分辨率可能损失细节。标签与划分详见 [README.md](README.md)。

## 运行与评估

新候选和正式比较统一使用 512×512 输入、base16、batch4，见 [固定 split seed 的统一入口](../REMOTE_4090.md)。保留的单模型入口也可执行来源验证，例如下列 512 新目录模板，不计算 test；该示例不是已完成的五折结果：

```bash
python -m unet_moe.region_train --data-root /path/to/pointed_data --output /path/to/new_region_512_cv0 --abdomen-labels paper5 --abdomen-split ct_cv5 --abdomen-fold 0 --split-policy trainmore_v6 --top-k 1 --size 512 --base 16 --batch-size 4 --quota-per-domain 500 --epochs 50 --lr 0.0003 --abdomen-positive-sampling 0.5 --workers 2
```

`region_train --resume` 从最后完整 epoch 恢复模型与优化器，并校验配置与记录。该旧入口不保存完整 RNG；特别是无 `--quota-per-domain` 的随机采样续训，不能声称与不中断训练逐位相同。训练自动选可用 CUDA，可指定 `--device cpu`；尚无 AMP/DDP。

常规划分按验证阳性 Dice 宏平均选择 `best.pt`。`ct_cv5` 只写验证完成结果；`ct_final` 没有腹部验证数据，固定使用 `last.pt`，不能用其他八域验证最优轮次声称腹部也已选优。旧入口在非 CV 训练结束后自动评估 test，新统一入口需显式 `--evaluate-test`。

对于旧 `region_train` 的 `region_unet_v1` checkpoint，`region_predict` 在 CPU 上还原原图尺寸，输出 checkpoint 所含结构的 19 或 16 张二值 PNG 和四级路由概率 `routing.json`。各通道可重叠；没有未见器官置信度剔除机制，未标注类别也无负监督。现有标签不能评估所有跨域误检。

## 已完成的历史运行与边界

- `region_top1_v1`：历史 256×256、19 结构、HC18 轮廓标签；已保存测试结果，阳性宏 Dice 0.5463，选中第 35 轮。与填实 HC18 的基线不构成严格对照。
- `unet_baseline_9v1`：共享及九个独立 U-Net 已全部完成，19 结构阳性宏 Dice 为 0.5694/0.6125；旧文档“仅共享完成、BUSI 第 21 轮”已过期。
- `region_top1_512_v6_final_seed42`：80 轮完成，固定选择 last80；16 结构阳性/含空宏 Dice 为 0.7451/0.7828。它同时改变了输入分辨率、容量、训练数据和预算，不能作为旧模型仅改 Top-k 的消融。
- 旧腹部诊断三张样例显示血管、肾上腺、骨为空预测，只能解释这些样例，不能代替全测试分布。当前 paper5 不再含肾上腺和骨，但删除困难类后的宏平均不能与旧 19 类值直接比较。

新固定划分的多种子架构对照尚无完整结果。需要在相同验证协议下分别评估参数量、执行专家数、耗时、每结构 Dice、空标签误报及按 CT 来源的稳定性，才能判断区域路由的收益。[实验状态](../docs/EXPERIMENT_STATUS.md)记录证据范围。

## 九数据域独立归一化实验

Q03为Region增加可选的共享GN／分域GN／共享BN／分域BN，仅改变编码器、融合层和MoE共享分支的归一化；专家内部继续GN。分域模型显式接收记录元信息中的9域ID，腹部五器官共用腹部域，仍一次输出16结构。该变体不沿用旧的“纯图像输入”描述；默认共享GN与旧模型接口和检查点兼容。实验矩阵、独立训练入口、评价口径及后台监测见[Q03协议](../docs/Q03_REGION_DOMAIN_NORMALIZATION_512.md)。
