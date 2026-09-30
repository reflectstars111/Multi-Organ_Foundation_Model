# 初版：面向多器官超声的 U-Net + Decoder MoE

> 新增图像独立输入的**器官区域特征路由版本**：见 [README_REGION.md](README_REGION.md)。使用 `region_train` / `region_predict` 入口，9 组专家、19 个结构输出，不输入 task ID。下文保留旧 task-ID 基线说明，与新版本不要混用。

这是可训练的研究基线，尚无完整训练精度结论。一个共享模型接受灰度图像与目标 task ID，返回该目标的二值分割 logits。对同一图像指定多个 task 可得到多个结构；首版不是自动识别所有器官的单次多类输出模型。

## 设计与论文的关系

参考 `relate_paper/2506.08356v2.pdf`（MedMoE）§4.2–4.6 的多尺度视觉特征、条件专家选择及辅助路由思想。原文研究视觉语言学习，不是 U-Net 分割论文；正文 §4.3 用报告 embedding 路由，图 2 描述图像 embedding 路由，二者表述不完全一致。

本实现：

1. 五级 CNN 编码器，通道默认 24/48/96/192/384，GroupNorm 适合小 batch。
2. 四级解码器：上采样 → 拼接同尺度 skip → 卷积融合 → MoE。
3. 每级路由器输入全局池化视觉特征和 32 维 task embedding，默认 4 个卷积专家，按样本选择 Top-2；仅对选中的样本执行对应专家。
4. 共享分支输出 + 被选专家输出的加权和。权重使用完整 softmax 中选中的概率（不重新归一化），保证 Top-1 时仍有分割梯度进入路由器。
5. 损失 = BCE + soft Dice + 0.01 × load balance + 0.001 × router z-loss。记录每级专家使用比例。

与论文的区别：无文本编码器、无图文对比学习、无专家内部跨尺度注意力；多尺度通过 U-Net skip 与逐级解码融合。专家没有预先固定为某个器官，器官条件来自 task embedding，专长由训练学习。Top-2、共享专家及均衡损失是本基线的设计选择。

## 数据入口与标签

从 `pointed_data/*/processed_png/manifest.csv` 自动读取 image/mask 相对路径，只使用 `paired` 行。

| 数据集 | task 名称 | 处理 |
|---|---|---|
| BUSI | breast_lesion | 二值 mask |
| MMOTU | ovarian_lesion | 二值 mask |
| DDTI | thyroid_nodule | 作者预处理版 |
| AUL | liver / liver_mass / ultrasound_outline | 各标签独立监督 |
| FALLMUD | aponeurosis / fascicle | 各标签独立监督 |
| Fetal HC | head_contour | 保留原轮廓，不填充成头部区域 |
| CCA | carotid | 原始二值标签语义 |
| CAMUS | lv_cavity / myocardium / left_atrium | 原索引 1/2/3 各自转二值目标 |
| AbdomenUS | abdomen_liver / kidney / pancreas / vessels / adrenal / gallbladder / bone / spleen（各名均带 abdomen_ 前缀） | 926 对 AUS，精确 RGB→索引 1–8；深灰为 ignore=255 |

未标注目标不参与损失。不同任务共享模型参数，不将其他数据集的缺失类别当作负标签。PNG 调色板通过 NumPy 直接读取索引；普通二值 mask 用阈值 128 去除 JPEG 灰度伪影。图像双线性、mask 最近邻缩放；首版固定缩放为正方形，可能改变长宽比，细线目标可能因降采样消失，应在正式实验中验证分辨率。

CAMUS 默认只用 ED/ES，跳过 half_sequence。划分已改为下述 source_aware_v1 协议。原始患者信息不足的数据集，分组无交集不等于患者独立；跨数据集重复患者或图像仍需核查。

## 当前划分协议（source_aware_v3_abdomen）

- CAMUS：严格读取原包 subgroup_training/validation/testing.txt，固定 400/50/50 名患者；缺名单、重复患者或患者未列入名单时报错，不回退随机划分。
- HC18：按用户要求，将原官方 train 的 999 张带标签图按检查编号哈希约 70%/15%/15% 划为项目训练/验证/测试；seed=42 实际为 706/154/139 张。335 张官方无标签测试图单独保留并计入 skipped。新的 139 张测试图可计算 Dice，但不是官方挑战赛测试集。
- FALLMUD：包含来源名称的图像分组按哈希阈值分配约 68%/17%/15%，即参考论文 85% 开发、15% 评估，再从开发部分留 20% 验证。这不是原作者固定名单，也未保证序列/患者独立。
- MMOTU：本地已有 test 固定保留，train 内留约 20% 验证；未划分的 CEUS 使用项目 70%/15%/15%。本地缺原官方名单，不能称作官方复现。
- BUSI、DDTI、AUL、CCA：项目约 70%/15%/15%；DDTI 未采用衍生代码的五折或默认 val/test 复用。CCA 按序列前缀分组，不保证患者独立；AUL 同图全部任务同组。
- AbdomenUS 已启用：优先读取 `manifest_indexed.csv`，不再使用含失效 RUS 路径的旧索引；没有转换结果时报错。8 个新增任务使总任务数变为 21；原 13 个 ID 不变。按 CT 编号划分，seed=42 的 train/val/test 为 321/312/293 张（5/4/4 个 CT），哈希在小组数下偏离目标比例明显。所有已有显式 test/val 均保留。

AbdomenUS 转换脚本为 `python scripts/convert_abdomen_masks.py`，保留原始 RGB 标签；只接受核查到的准确颜色，未知颜色报错。深灰 `(10,10,10)` 的语义未指定，编码为 255；Dataset 转为 target=-1，BCE、Dice 损失和评估屏蔽该区域。新数据集需重新训练 21-task 模型，不可将旧 13-task checkpoint 视为已经学过腹部任务。当前已启动的 `runs/moe_full_hc_v2` 仍为旧 13-task 实验，本次没有中断或重启它。

比例为确定性哈希的期望值，不是精确张数；固定 seed 和分组键即可重现，变更 seed 会改变项目划分。每条记录新增 split_protocol 和 split_version，导出时检查同图及同组不跨集合。此检查不涵盖尚未知的患者关系。训练仍仅用 val 选 best，test 不进入训练或选模；当前未提供独立测试评分命令。

```powershell
python -m unet_moe.audit_splits --data-root "数据集 (Datasets)/pointed_data" --output runs/split_audit.json --seed 42
```

审核文件含完整记录、独立图像数、分组数、协议及跳过原因；为保护旧结果，输出文件已存在时报错。已验证报告为 `runs/split_audit_v3.json`。新协议与旧实验不兼容，请使用新的输出目录重新训练；下文旧验证数量是历史运行结果。

## 运行

### 下次实验变更（不影响已启动的 v3 实验）

默认 `--experts 9`，Top-2 保持不变：每个解码阶段有 9 个可路由专家及 1 个共享分支，专家由路由器学习选择，并未固定绑定数据集。9 数据集不等于 9 个分割目标。

FALLMUD 仅使用 aponeurosis；fascicle 从新建的训练、验证、测试记录中全部排除，原始文件保留，跳过原因记录为 `FALLMUD:disabled_fascicle`。当前为 20 个有效分割目标；为兼容旧 checkpoint 的 task 编号，TASKS 的 21 个槽位保留，fascicle 的 ID=7 不再有训练数据。现有后台 v3 仍按原配置运行；下次须使用新输出目录从头训练。

当前目标计数：BUSI/MMOTU/DDTI/HC18/CCA 各 1，AUL 3，FALLMUD 1，CAMUS 3，AbdomenUS 8，共 20；先前 FALLMUD 含 2 个目标，因此共 21。若需要“每个数据集固定一个专家”，需另行改为固定数据集路由，不能仅设置专家数量实现。

当前正式实验：`runs/moe_full_9datasets_v3/`，从随机初始化训练全部 9 个数据集、21 个任务，50 epochs，256×256，batch=4，base=24，4 experts / Top-2，seed=42。日志在 `runs/moe_full_9datasets_v3-logs/stdout.log`，错误日志为同目录 `stderr.log`。旧 `moe_full_hc_v2` 已停止、结果保留。启动前已核对全部引用文件存在；训练/验证/测试任务记录数为 13,278 / 4,381 / 4,365。完成后自动对验证最优模型评估测试集，保存 `test_metrics.json`；启动不代表已完成训练。

### 本轮更新

- 增加 `--decoder shared`：关闭稀疏专家与路由，保留 task embedding 条件和共享卷积分支，作为任务条件 U-Net 对照；默认 `--decoder moe` 保持原架构。两者参数量不同，比较时应报告参数量。
- HC18 改为按文件名首段检查编号分组，例如 `010_HC` 与 `010_2HC` 不会跨训练/验证集合。这会改变旧版划分，请重新生成实验记录。
- AUL 仍读取原始独立 liver/mass/outline 标签，不读取新增 `manifest_liver_mass.csv`：合并 PNG 中 103 张存在缺失标注，不能直接作为完整多类监督；独立标签也保留了原始重叠关系。
- 新 checkpoint 记录有训练数据的 task ID，推理拒绝没有训练记录的目标。原 checkpoint 配置仍兼容。

共享 U-Net 对照运行示例（其他参数与 MoE 实验保持一致）：

```powershell
python -m unet_moe.train --data-root "数据集 (Datasets)/pointed_data" --output runs/shared_experiment01 --decoder shared --epochs 50 --batch-size 8 --size 256
python -m pytest tests/test_unet_moe.py -q
```

在项目根目录安装 `requirements-unet-moe.txt`；CUDA 版 PyTorch 按本机驱动环境安装。代码已在现有 PyTorch 环境验证。

```powershell
python -m unet_moe.smoke --data-root "数据集 (Datasets)/pointed_data"
python -m unet_moe.train --data-root "数据集 (Datasets)/pointed_data" --output runs/experiment01 --epochs 50 --batch-size 8 --size 256
```

显存不足时降低 `--batch-size` 或 `--base`（须为 4 的倍数）。`--workers 0` 适合 Windows 初次排查。训练按任务频次反比采样，避免 CAMUS 主导梯度。输出 `records.json`（固定划分及跳过原因）、`metrics.jsonl`（每任务验证 Dice/宏平均/路由负载）、`best.pt` 和 `last.pt`。best 仅由验证宏平均 Dice 选择；空真值且空预测计 Dice=1。评估为任务内样本均值，尚非患者级平均，也未加入 HD95。

```powershell
python -m unet_moe.predict --checkpoint runs/experiment01/best.pt --image "数据集 (Datasets)/pointed_data/心脏超声心动图数据集 (CAMUS)/processed_png/images/patient0001/patient0001_2CH_ED.png" --task lv_cavity --output runs/prediction.png
```

推理将 logits 还原到原图尺寸再阈值化，输出 0/255 PNG。指定目标必须是已训练任务。多目标预测可能重叠，需要进一步设计多类别融合。smoke.pt 只用于链路验证，不能作为有分割能力的预训练模型。

## 文件

本轮验证（2026-09-16）：4 项模型/标签/分组测试通过；13 个任务真实样本前后向和 checkpoint 回读通过。MoE 与 shared 两种模型均在 RTX 5070 Laptop GPU 上以 base=4、32×32、batch=8 完成 2 个优化步骤、全部 2,710 条验证记录及单图推理。当前划分为 train=11,524、val=2,710、test=382；记录数按任务计，不等于独立图像数。产物分别在 `runs/moe_smoke_v2/`、`runs/moe_check_v2/`、`runs/shared_check_v2/`，包括 checkpoint、验证指标与预测 PNG。仅为链路验证，不用于精度比较。

本地验证结果：2 个模型/标签测试通过；13 个任务真实样本完成前后向及 checkpoint 一致性检查；小模型（base=4、32×32、2 个优化步骤）已跑通完整训练与 2,723 条验证记录，并成功执行单图推理。结果保存在 `runs/moe_smoke/` 和 `runs/moe_train_check/`。这些运行仅检验工程链路，不能据此评价模型准确率或临床能力。

- `model.py`：U-Net、稀疏路由、损失。
- `data.py`：manifest 读取、任务定义、分组划分、PNG 处理。
- `train.py`：训练、验证、任务均衡采样、checkpoint。
- `predict.py`：单图单目标推理。
- `smoke.py`：真实样本逐任务前后向和 checkpoint 回读。

建议实验顺序：先训练共享 U-Net 对照，再比较 MoE 的专家数与 Top-k，最后验证患者独立划分下的跨数据集泛化。已提供任务条件共享 U-Net 对照开关；尚未实现 AMP、DDP 或断点续训。
