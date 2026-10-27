# 多器官超声分割：架构与数据约定

更新日期：2026-10-09。全部医学实验及最新证据见[医学实验总览](../docs/MEDICAL_EXPERIMENT_OVERVIEW.md)，服务器命令见[REMOTE_4090.md](../REMOTE_4090.md)。本页说明架构和数据约定；历史smoke与训练运行见文末。

用户定义的共享U-Net为所有已标注器官合并到一个前景通道；其与原设计MoE的对比见[主对照结果](../docs/SHARED_UNION_VS_MOE.md)。代码名`shared`保留历史含义，指16通道附加对照。

## 五种模型设置

| 模型 / 入口 | 输入和输出 | 主要区别 |
|---|---|---|
| 共享 U-Net（单通道主基线） / `single_channel_suite` | 灰度图 → 1个前景并集logit | 普通U-Net，不输入域或任务ID，不输出器官身份；见[并集协议](../docs/SINGLE_CHANNEL_UNION_512.md) |
| 16通道 U-Net（附加对照） / `baseline_suite --mode shared` | 灰度图 → 16 个结构 logits（paper5） | 所有数据集共享主干；这是现有 `shared` 协议，不是单通道并集基线 |
| 独立 U-Net / `baseline_suite --mode independent` | 灰度图 → 本数据集结构 logits | 九套参数；部署需知道数据集归属 |
| task-ID MoE / `train --decoder moe` | 灰度图 + task ID → 单结构 logits | 每级视觉池化特征与 task embedding 共同路由 |
| 区域 MoE / `region_train` | 灰度图 → 多结构 logits | 预测软区域特征路由，无需输入 task ID；见 [区域文档](README_REGION.md) |

`train --decoder shared` 是保留 task embedding 的共享解码器消融，与不输入 task ID 的普通共享 U-Net 不同。task-ID 模型包含五级 CNN 编码器、四级解码器、GroupNorm、共享卷积分支及稀疏专家。Top-k 权重来自完整 softmax，不对选中权重重新归一化；Top-1 仍有分割梯度进入路由器。专家数量由 `--experts` 控制，不等于输出结构数。

架构比较入口`fair_suite`及累积梯度版本`fair_suite_accum`在固定记录上统一三种结构模型的图像抽样、逐图像有效通道BCE+Dice、验证指标与训练预算。已完成的50轮对照使用512×512/base16/batch4/lr3e-4；80轮对照使用512×512/base24/有效batch16/lr1e-4，结构模型物理microbatch4。单通道各有对应配置的三种子并集实验；Q01另在80轮配置下做器官损失加权消融。历史256实验保留原记录。Task-ID为每张图的已标注结构分别前向；相同更新步数不代表相同FLOPs。正式对照辅助损失均为0，模型参数量与推理条件见[结果总览](../docs/MEDICAL_EXPERIMENT_OVERVIEW.md)。

## 标签数量与监督

`data.TASKS` 保留 **21 个兼容槽位**，不是当前所有模型都输出 21 个有效结构。

| 配置 | 有效目标 | 排除规则 |
|---|---:|---|
| 旧 task-ID `all8` 数据入口 | 20 | 保留 AUL outline；FALLMUD fascicle（ID 7）没有记录 |
| 区域/普通 U-Net `all8` | 19 | 另排除 AUL ultrasound_outline（ID 5） |
| `paper5` / 新公平对照 | 16 | 在 19 结构基础上排除腹部胰腺、肾上腺、骨（ID 15/17/19） |

paper5 含非腹部 11 类和腹部肝、肾、血管、胆囊、脾 5 类。腹部“血管”和“整肾”是本地标签语义，不等价于论文中的主动脉或肾内部细分。AbdomenUS 当前只使用 AUS 模拟超声，不能据此推断真实腹部超声泛化。

manifest 只取 `paired` 行。图像双线性缩放至正方形，mask 最近邻缩放；普通二值 mask 阈值为 128，CAMUS 按索引 1/2/3 解码，AbdomenUS 按索引 1–8 解码。腹部 ignore=255 转为 target=-1；缺失通道同样为 -1，不参与损失或指标。AUL 使用独立 liver/mass 标签，保留重叠和缺失标注，不读取合并标签作为完整多类真值。

**HC18 当前使用填实标签。** `build_records` 优先读取 `manifest_filled.csv`，记录 `label_version=hc18_filled_v1`；区域/普通 U-Net 入口校验全部 999 张标注图均为填实版本。兼容 task 名仍叫 `head_contour`，名称不代表当前 mask 是细轮廓。旧 task-ID 入口未强制填实，复现实验需检查 records 中的版本。原轮廓 mask 保留；生成脚本为 `scripts/fill_hc18_masks.py`。

## 划分

CAMUS 使用官方 400/50/50 患者名单，跳过 half_sequence；名单缺失或冲突时报错。HC18 按文件名检查编号分组，官方 335 张无标签测试图不参加有监督评分；本项目 HC18 test 来自 999 张有标注发布训练图。CCA 按序列前缀分组，AUL 同图标签保持同组；其他来源混合发布划分和项目哈希划分，不能统称患者独立或官方复现。

腹部 `ct7_2` 保留发布测试的 4 个 CT 来源，原训练来源中 ct3/ct4 留验证，独立图像数为 477/156/293。`ct_cv5` 按 5 组来源验证且不运行 test；ct15 出现于两折，因此不是九个 CT 等权且互斥的标准五折。`ct_final` 将原训练 9 个 CT 的 633 图全部用于训练，**腹部没有验证集**，固定选择最后一轮。2026-09-29 启动的两条 v6 实验实际使用 last epoch 80，没有先完成五折。

`trainmore_v6` 只将部分项目验证样本转回训练；同一 split seed 下 test 保持一致。旧三个训练入口的 `--seed` 同时改变划分和训练随机性；新 `fair_suite --split-seed 42 --seeds 42,43,44` 分开控制二者。完整 records、分组键和标签版本应随每次运行保存；仅比较 seed 数值不足以证明样本相同。

## 评估与续训

当前 Dice 在缩放后的网格上、阈值 0.5 计算，先按结构内图像平均，再对结构等权平均。`macro_positive_dice` 仅统计该结构真值有前景的图像；`macro_all_dice` 包含空标签图像，空预测+空真值计 1。旧 task-ID 的 `macro_dice` 属于后者。`macro_empty_false_positive_rate` 是空真值图像上出现任意预测前景的比例，仅覆盖已标注通道。它们不是像素全局 Dice、患者平均或 HD95；阳性与含空指标不可混比。

`train`、`region_train`、`baseline_suite` 均支持 `--resume`，需要保持配置和冻结记录一致，不能用它修改超参。旧区域/普通基线 checkpoint 未保存完整随机状态；支持续训不等于逐位重现中断前轨迹。旧 task-ID 保存 PyTorch CPU/CUDA RNG，但没有完整 Python/NumPy RNG。新统一入口保存完整 RNG 和配置校验信息，并默认不评估 test。各入口目前无 AMP/DDP 训练路径。

以下独立预测命令适用于旧 `train` / `region_train` 产出的 checkpoint。新 `fair_suite` schema 由统一 runner 评估，不能直接交给旧预测入口：

```bash
python -m unet_moe.predict --checkpoint /path/to/task_id/best.pt --image /path/to/image.png --task lv_cavity --output /path/to/prediction.png
python -m unet_moe.region_predict --checkpoint /path/to/region/best.pt --image /path/to/image.png --output /path/to/predictions
```

checkpoint 架构须与预测入口一致。两种预测都还原原图尺寸；task-ID 拒绝无训练记录的任务，多结构独立 sigmoid 输出允许重叠。

## 历史记录

2026-09-16 的 `moe_smoke*`、`moe_*check*`、`shared_check_v2`、`region_check_v1` 使用 base=4、32×32 和少量优化步骤，仅验证工程链路。早期 `moe_full_hc_v2`、`moe_full_9datasets_v3` 为 13-task/21-task 阶段实验，历史记录显示已停止，不应继续称其“当前后台训练”。其任务集合、HC18 标签与新协议不同。原 README 的验证张数为对应旧记录计数，不能用于当前图像数或精度结论。

当前服务器可核验的正式历史结果包括 `region_top1_v1`、`unet_baseline_9v1`、`unet_paper5_ct7_2_v1` 独立模型及两条 v6 80 轮实验，详见[实验状态](../docs/EXPERIMENT_STATUS.md)。
