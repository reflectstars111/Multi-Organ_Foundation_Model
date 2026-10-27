# UUSIC 2025 第二、三名源码审查与本项目适配边界

审查日期：2026-10-12（北京时间）。本记录依据本地发布源码、Docker 元信息及检查点文件；没有导入执行第三方模型，没有启动训练，也没有修改第三方源码。

本项目主对照按 **512×512、80 轮、3 个随机种子、相同数据划分**重新训练。用户已确认 **并集作为五种方法共同的主对照**：AImageLab 主方案使用9个数据域的 FiLM 条件，按每张图的已标注前景并集训练；Lanzhou 保留原生无条件前景分割。AImageLab 的16结构条件查询只作为可选扩展另列。统一重训结果应注明“本项目统一协议下的重训练”，不表述为完整复现挑战赛训练流程或排行榜成绩。

## 1. 审查对象与来源

| 简称 | 本地真实根路径 |
|---|---|
| A | `/ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第二名源码解析 (AImageLab Rank-2 Source)` |
| L | `/ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第三名源码解析 (Lanzhou Rank-3 Source)` |

下文 `A/` 与 `L/` 均按上表展开，行号依据审查当日文件。

- AImageLab-zip：第二名；元信息说明档案保留其原提交 Docker，镜像 manifest 为 `sha256:b4a59458c65b81af2ae4f13adcd8ebb28eeee9c05763dcd2f309a1d9aec77cd9`，原镜像更新日期 2025-08-17。
- Lanzhou：第三名；原比赛 Docker 已不可用，现有镜像由组织者根据团队源码及 `fused_swin_unet_best.pth` 重建，**不是团队原比赛容器**。
- 本地发布元信息记录来源 DOI 为 `10.5281/zenodo.21807962`，标题为 *UUSIC25 Released Finalist Containers and Model Artifacts*。参赛者材料为 CC BY-NC 4.0，第三方软件保留各自许可。
- UUSIC 为联合分类与分割挑战赛。“第二／三名”不能在没有单项排名证据时改写为“分割单项第二／三名”。

来源：[AImageLab 发布说明](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 前三名元信息 (Finalist Metadata)/README_AImageLab-zip_rank2.md:3>)；[Lanzhou 发布说明](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 前三名元信息 (Finalist Metadata)/README_Lanzhou_rank3.md:3>)；[本地资源清单](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/RESOURCE_MANIFEST.md:18>)。

上述 Docker 描述属于发布元信息，并不表示本地 Docker 压缩包当前完整可用。主代理检查发现本地三个 Docker 文件仍在增长，因此本次不声称其下载完成或整体哈希验证通过。下节的核验仅针对已稳定存在的二、三名检查点文件。

## 2. 检查点只读核验

SHA-256 通过 Python 标准库 `hashlib` 分块读取完整文件计算，结果与本地 `RESOURCE_MANIFEST.md` 一致。没有使用 `torch.load`，也没有执行 pickle 中声明的重建函数。

| 检查点 | 字节数 | 本次实算 SHA-256 |
|---|---:|---|
| `A/checkpoints/fusionwrapper.pt` | 652694011 | `42931d63cf3f0ee37b3f3695ed99e06e8fe24cb5185e16392938e4e4d632bb60` |
| `L/fused_swin_unet_best.pth` | 203934462 | `431f4ff430efbc428ffc34df0ceb6929a481fc7b659d945fbf89b80a14b45a36` |

对 A 检查点，通过 `zipfile` 读取 `fusionwrapper/data.pkl`，以 `pickletools.genops` 静态定位 `model_state_dict` 的 `alpha`，对应 CPU FloatStorage `0`、一个元素。`fusionwrapper/data/0` 的四个原始字节为 `00 00 00 00`，按 float32 解码是 **0.0**。

对 L 检查点，静态检查 pickle 字段可见 `feature_enhancer.1.running_mean` 等 BN 统计；未见 `ds_weights`、`att1` 等开发版独有字段。这支持其与 `exact_model.py` 的结构对应关系，但**静态字段检查不替代后续严格加载、512 前反向与数值测试**。

## 3. 第二名 AImageLab：完整模型应保留

### 3.1 发布模型的真实结构

[发布入口 A/model.py:34](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第二名源码解析 (AImageLab Rank-2 Source)/model.py:34>) 构造 `FusionWrapper`，配置如下：

| 参数／支路 | 发布配置 |
|---|---|
| 分类特征支路 | ResNet18，CBAM 开启；分类支路 FiLM 关闭 |
| 分类类别数 | 10 |
| 分割主干 | `UNet2DFiLM`，`in_channels=3`、`num_classes=1` |
| 分割条件数 | `n_organs=6` |
| 宽度与深度 | `size=32`、`depth=5` |
| 融合 | 分类全局特征 512 → 2048，经 `tanh(alpha)` 加入分割瓶颈 |
| 输出 | 单前景通道，与分割输入同尺寸 |

`use_film=False` **不是关闭整个模型的 FiLM**。它传给分类网络；分割编码与解码各卷积块仍创建并调用 FiLM。证据为 `A/nets/net.py:448–463`、`:496–516`、`:671–687`。分割条件映射见 `A/utils/util.py:116–126`，包含 breast、cardiac、thyroid、fetal、kidney、breast_luminal。

[融合实现 A/nets/net.py:688](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第二名源码解析 (AImageLab Rank-2 Source)/nets/net.py:688>) 的实际运算为：

```text
classification view → ResNet18 + CBAM → global average pooling → [B,512]
                    → Linear(512,2048) × tanh(alpha) → [B,2048,1,1]
segmentation view + organ_id → FiLM encoder/bottleneck → additive fusion
                            → FiLM decoder → [B,1,H,W]
```

发布检查点的 `alpha=0`，因此其当前分割输出在数值上不使用分类特征贡献；这只是**该检查点的参数事实**。`alpha` 在源码中是可训练参数，从头训练时可以变化。正式重训练应保留完整融合结构，**不能删除分类特征支路后仍称为完整第二名模型**。若研究有效分割支路，必须另列“仅 FiLM U-Net”消融。

### 3.2 输入、输出与512适配

- 分割预处理先检测并裁去超声黑边，再按长边缩放并填充到512，像素缩放至 `[0,1]`：`A/model.py:65–74`。
- 分类特征视图实际调用256，归一化为 `(x−127.5)/63.75`：`A/model.py:55–63,148–159`。因此不能把发布模型描述为“两路均512”。
- ResNet 分类特征通过全局平均池化输出固定512维：`A/nets/net.py:194–232`；将其视图也设为512在结构上可行，但属于与原提交不同的输入协议，需在配置与报告中记录。
- 发布二值化采用 `sigmoid(logits)>0.5`：`A/model.py:159–162`；另有按器官形态学和连通域后处理：`:181–221,255–258`。通用对照若使用共同阈值且不加这些后处理，要明确协议差异。
- 原输出层仍是 `Conv→InstanceNorm→LeakyReLU`，不是常见的裸卷积输出：`A/nets/net.py:295–306,562–566`。忠实适配须保留；若改掉需要单列结构修改。
- 原瓶颈2048与固定 `Linear(512,2048)` 耦合。不能将本项目 U-Net 的 `base=24` 机械套入这个模型。

### 3.3 任务适配与待实现部分

**主方案：完整 FusionWrapper＋9数据域 FiLM＋每图前景并集。** 将原6个分割 FiLM 条目改成9个数据域条目；一次输入图像及其数据域ID，输出一个前景通道。损失以该图全部已标注结构的前景并集为目标，保留统一的未知像素忽略规则。分域ID只来自已有数据域元信息，不依赖测试真值。原6任务嵌入不能直接完整加载到9域适配模型，主对照必须重新训练。

**可选扩展：16结构条件查询。** 如果后续另行研究逐器官结果，可把分割 FiLM 改为16结构任务，一张图按有标注结构生成查询并计算对应 masked loss。该扩展与主方案分开命名；不得将16结构模型推理并集的结果混称为“按并集标签训练”。所有查询集需在评价协议中预先定义，不能借真值轮廓选择预测器官。

本地未提供完整训练入口。需补齐数据划分、共同采样、`-1` 忽略区、损失、优化器、混合精度、恢复随机状态和统一评价。当前分类与分割联合训练历史没有完整证据；本项目若仅提供分割监督，应明确为分割任务重训练，不能声称复现原联合训练配方。

不要直接导入 `A/model.py`：其 `:224` 起在顶层创建目录、读取输入、加载模型、执行推理，并删除和重命名输出目录。适配应使用 `nets/net.py` 的模型定义，处理其 `from utils.util ...` 的包命名隔离，避免与项目模块冲突。

## 4. 第三名 Lanzhou：发布版不等于开发训练版

### 4.1 发布入口与模型

[L/eval_competition_test.py:11](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第三名源码解析 (Lanzhou Rank-3 Source)/eval_competition_test.py:11>) 导入 `ExactFusedSwinUNet`，在 `:83–85` 构造并严格加载检查点。用于正式基线的结构依据应是 [L/exact_model.py:98](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第三名源码解析 (Lanzhou Rank-3 Source)/exact_model.py:98>)。

| 部件 | 发布 `ExactFusedSwinUNet` | 开发 `FusedSwinUNet` |
|---|---|---|
| 归一化默认 | BN，`exact_model.py:99–111` | GN，`model_fused_swin_unet.py:183–185` |
| 构造时 Swin 权重 | `weights=None`，`:116` | ImageNet1K V1，`:226–227` |
| 多尺度最末分支 dilation | 3，`:71–73` | 4，`:150–153` |
| 特征增强 | 两层3×3卷积，`:142–149` | 1×1＋ASPP＋1×1，`:265–274` |
| 分类头输入 | stage4 的768维，`:127–136` | 后三尺度拼接1344维，`:243–257` |
| 跳跃连接 attention gates | 无 | 有，`:297–315` |
| 解码 Dropout | 无对应层 | `Dropout2d(0.1)`，`:317–328` |
| 深监督融合 | 固定0.4／0.3／0.3，`:239–245` | 可训练 `ds_weights`，`:291–295` |

现有训练脚本 [L/train_fused_swin_unet.py:20](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第三名源码解析 (Lanzhou Rank-3 Source)/train_fused_swin_unet.py:20>) 导入的是右栏开发版，在 `:149` 实例化。**直接运行该脚本无法作为左栏发布结构的重训练实现。**

发布版构造器 `weights=None` 仅说明加载完整检查点前不单独下载 ImageNet 权重，不能据此认定发布检查点训练时从未用过预训练。开发版显式使用 ImageNet，也不足以证明发布版的精确训练历史。

### 4.2 输入、输出与任务语义

原发布评价为 RGB224、ImageNet mean/std 归一化，见 `L/eval_competition_test.py:19–29,51–59`。输出是两个互斥类别的 logits，按 softmax／argmax 得到背景和前景：`:61–67`。

`L/exact_model.py:197–245` 的分割路径不接收器官ID；`classification_task_name` 只选择分类头。其主要输出含义是：

```text
[B,3,H,W] → Swin-T stages + multi-scale blocks → U-Net decoder
          → final logits + fixed deep-supervision logits → [B,2,H,W]
```

输出和深监督分支在 `:235–241` 插值回输入尺寸，因此结构上允许512输入。512下的严格加载、前反向、混合精度以及显存仍需实际测试，本文没有执行。

最少改变结构的任务适配是前景并集。腹部同一图若轮流配五种器官目标但不提供条件，会给同一输入施加冲突监督。不能以这种方式伪造逐器官能力，也不能用真值把其并集预测拆成器官评分。若要求16结构输出或条件查询，应另列“Lanzhou 扩展版”，不能归入未修改的第三名结构。

二类输出可在统一二值评价中转换为 `foreground_logit = z_foreground − z_background`，其 sigmoid 与原 softmax 前景概率等价；无需强行把输出层改为一个通道。训练可使用带忽略区的 two-class CE＋Dice，具体损失协议需统一记录。

### 4.3 现有开发训练脚本的工程问题

这些问题说明应重新实现正确训练入口，而不是直接照搬脚本：

1. `random_split` 得到的 train/val 子集共享同一个 Dataset；之后修改 `val_subset.dataset.transform`／`image_transform`，也改变了训练子集。见分类路径 `:124–128`、分割路径 `:135–140`。当前效果包括训练随机增强被验证变换覆盖。
2. 若修复上条并恢复随机几何增强，还必须同步变换图像和掩膜。原图配置含随机翻转和旋转（`:92–98`），掩膜仅最近邻resize（`:133–136`），不能独立抽样后直接配对。
3. `itertools.cycle(DataLoader)` 会缓存首轮已迭代批次并重复，不能当作每轮正常重新采样与重洗牌。见 `:144–146`。
4. 脚本交替进行分类与分割训练，并假设分类 loader 非空，见 `:158–176`。直接给纯分割数据需要改训练入口。
5. 原 `CrossEntropyLoss()` 与 Dice 没有本项目 `-1` 标签忽略逻辑，见 `:24–32,152–153`。不能把未知像素当背景或直接喂给默认 CE。
6. 原验证按联合分类／分割指标选模型，见 `:217–222`；本项目应使用事先定义的共同验证选点规则，不能套用不可比的联合排名分数。

## 5. 依赖与数据来源边界

| 项目 | 发布环境证据 | 适配要求 |
|---|---|---|
| AImageLab | `A/Dockerfile:3` 为 torch2.7.1/CUDA12.8；使用 torchvision transforms v2；requirements含 OpenCV、NumPy、Pillow、tqdm | 与本项目环境做兼容性验证；避免无约束安装替换当前训练环境 |
| Lanzhou | `L/Dockerfile:3–9` 为组织者重建的 torch1.12.1/CUDA11.3 环境，特意保留匹配的 torchvision | 在目标环境严格核对 Swin 特征提取节点、checkpoint key／shape、前反向；不能认为二者原Docker依赖完全相同 |

AImageLab 的 `A/nets/net.py:135` 直接调用 `resnet_map[resnet_size]()`，没有指定 torchvision ImageNet 权重，但完整发布 checkpoint 的训练历史仍需另行证明。

AImageLab 数据映射中明确出现 BUSI、CAMUS、DDTI 等：`A/utils/util.py:38–89`；Lanzhou 分类配置也包括 BUSI 等：`L/eval_competition_test.py:73–81`。这些名称与本项目数据来源存在交集，**目前没有逐样本审计来确认或排除测试样本重合**。所以发布权重不能直接被描述为无数据泄漏的公平主对照；统一重训与外部预训练参考应分别报告。

## 6. 实施前需要关闭的缺口

- 固定使用 A 的完整融合结构和 L 的 `ExactFusedSwinUNet`，记录源码／配置／检查点哈希；保留外部材料原件。
- 按已确认的共同并集主表实现 AImageLab 的9域条件和 Lanzhou 的原生前景分割。逐器官附表仅纳入确有对应预测语义的方法，16结构扩展另外记录。
- 在共同数据协议下实现正确的训练／验证／恢复循环，严格处理未知像素和同患者划分。
- 保留方法固有宽度和模块，不能把本项目 `base=24`、专家数等模型专属参数套到外部架构。
- 测量512下可行的 microbatch。若用累积保持有效 batch，需要注明 BN 统计仍由物理 microbatch 决定，不能声称与直接 batch16 数值等价。
- 先做严格权重加载和合成／真实小样本前反向验收，再安排80轮正式训练；本记录不代表这些运行检查已通过。
- 接入本项目 DSC (%)、HD95 (pixel)、ASSD (pixel)，沿用共同空掩膜对角线惩罚和未知边界处理。不要通过各方法自己的旧 Dice 代码生成主表。
