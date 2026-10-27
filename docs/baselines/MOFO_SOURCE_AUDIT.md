# MOFO 源码与统一基线接入审查

核验日期：2026-10-12（北京时间）。范围为本地源码、Git 来源和随附论文的只读审查；未运行第三方训练入口，未加载第三方权重，也未修改 MOFO 源码。本记录不代表模型已经接入或实验已经开始。

用户已确认：MOFO、挑战赛前三名方法和 UniverSeg 作为五个外部对照，在本项目统一数据划分下，以 **512×512、80 轮、三个种子**重新训练，且以**前景并集作为五方法共同主对照**。MOFO 主方案按9个数据域建立并集任务，保留单标签分类及9个提示/掩膜头；16结构映射仅作为可能的扩展，不进入本次主对照。

## 1. 来源及本地材料

| 项目 | 已核验事实 |
|---|---|
| 本地目录 | `relate_baseline/多器官超声基础模型 (MOFO)/` |
| 上游 | [Git-HB-CHEN/MOFO](https://github.com/Git-HB-CHEN/MOFO)，README 和本地 Git remote 一致 |
| 本地提交 | `c5076f087ae68bfb7ba19d976def2dcd5224f1c8`，提交时间为 2024-09-30 |
| 工作区差异 | 9 个 tracked 文件显示修改；`git diff --ignore-space-at-eol --stat` 为空，核验时这些差异仅为行尾变化 |
| 论文 | Haobo Chen 等，*Multi-Organ Foundation Model for Universal Ultrasound Image Segmentation With Task Prompt and Anatomical Prior*，IEEE TMI 44(2)，2025，DOI `10.1109/TMI.2024.3472672` |
| 已有材料 | 主模型、训练/测试/推理脚本、pUNet 先验模型、部分损失类、两个数据域的示例数据 |
| 本地缺失 | CSWin、文本提示、解剖先验和分割模型权重；未发现 `.pth`、`.pt` 或 `.safetensors` 文件 |
| 许可证文件 | 随附 GNU GPL v3；接入时保留原文件和来源说明 |

依据：[README:22](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/README.md:22>)、[README:68](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/README.md:68>)、[本地论文](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/Multi-Organ_Foundation_Model_for_Universal_Ultrasound_Image_Segmentation_With_Task_Prompt_and_Anatomical_Prior.pdf>)、[LICENSE](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/LICENSE>)。README 中的 “Under Review” 是旧描述，不能据此认定论文仍在审稿。

## 2. 原模型的数据流与任务语义

```text
RGB 图像 [B,3,224,224]
  ├─ CSWin 编码器 → 多尺度图像特征 → 卷积解码器 → 共享像素特征
  └─ 图像特征 + 全部类别提示
       → 四级提示解码器 → [B,K,256] 任务表示
            ├─ 动态卷积参数 → K 个前景 logit 图 [B,K,224,224]
            └─ 提示均值池化 + 分类头 → [B,K] 类别 logits
```

发布训练脚本设置 `K=10`。文本提示输入为 `[K,512]` 的 CLIP 特征，经线性投影成为 256 维表示。`MOFO.forward(x)` 不接受逐样本 task-ID，而是一次生成所有任务通道；各任务的动态头共享基础像素特征。依据：[模型构造:433](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/model/MOFO.py:433>)、[文本提示:444](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/model/MOFO.py:444>)、[forward:517](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/model/MOFO.py:517>)。

原 loader 每幅图返回一个数据集编号 `setseq`、一个二值 mask，并仅把该 mask 放入相应输出通道。训练损失选择这个通道，同时用单标签交叉熵监督分类头。因此，原任务是“每图一个目标类别”，不是“每图任意多类别同时成立”。依据：[loader:37](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/dataset/mutlidomain_baseloader.py:37>)、[训练损失:97](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/train.py:97>)。

发布仓库还有两种不同的评价方式：

- `test.py` 用已知的 `setseq` 选择目标通道；这属于已知任务的分割评价。[test.py:53](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/test.py:53>)
- `infer.py` 用分类预测 `argmax` 选择唯一通道；分类出错也会影响最终分割。[infer.py:48](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/infer.py:48>)

接入时必须注明采用哪种方式，不能在结果中混淆“已知任务”和“自动选择任务”。

## 3. 512 输入需要的实际改动

仅把 loader 的 `input_size` 改为 512 无法运行原模型：

| 位置 | 原实现 | 512 下的问题及适配方向 |
|---|---|---|
| [MOFO.py:243](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/model/MOFO.py:243>)、[441](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/model/MOFO.py:441>) | 编码器 `img_size` 默认为224，MOFO 构造没有显式传入其他尺寸 | 需暴露并传递512，更新各级 attention 的空间分辨率 |
| [MOFO.py:401](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/model/MOFO.py:401>) | 提示特征展平固定 `h=7,w=7` | 512 时对应网格为16×16，需按真实空间维度展平 |
| [MOFO.py:441](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/model/MOFO.py:441>)、[窗口 reshape:191](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/model/MOFO.py:191>) | CSWin 条带宽度 `[1,2,7,7]`，窗口划分要求整除 | 第三级32×32不能整除7；需实现带 attention mask 的 padding/unpadding，或明确记录窗口参数修改 |

保留窗口宽度7、正确屏蔽补齐位置，是较接近原注意力定义的适配方向；直接改成8也是一种分辨率适配，但应记录为模型改动。这里尚未实现或验证任何方案。保持解码器原设计后，输出应恢复为512×512；灰度图复制到三个输入通道不增加额外图像信息。

## 4. 先验与 CLIP 阶段缺口

原训练入口直接加载三份资产：[train.py:59](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/train.py:59>)。

```text
model/cswin_small_224.pth           ImageNet 预训练图像编码器
task_weights/prompt_encoding.pth   CLIP 文本提示特征
prior_weights/prior_weight_sel.pth 已训练的多器官先验网络
```

README 提供权重下载入口，但不表示这些权重已经位于本地或适合本项目的新任务定义。特别是原10类提示和先验的类别语义不等同于本项目9个域、16个结构。

论文第三节、印刷页1008–1009及第四节C描述了完整方法：

1. 用 CLIP ViT-B/32 文本编码器和器官描述构造提示，原模板为超声图像描述。需要补齐提示生成脚本、类别顺序及权重来源；涉及非超声域时，应记录模板适配。
2. 先训练多器官 mask autoencoder，用 mask 重建损失和器官标签参与的多尺度一致性损失构造解剖先验空间。
3. 冻结先验编码器，把预测 mask 和 GT mask 投影到同一空间，再约束二者的特征差异。

本地 [pUNet.py:73](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/prior_model/pUNet.py:73>) 提供模型结构及多尺度特征，但没有该网络的完整预训练入口。[prior_loss.py:79](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/utils/prior_loss.py:79>) 的 `SelectedCLoss` 无器官标签参数，只把相似度矩阵的对角配对作为正对，不能直接等同于论文 Eq8 的同器官多正对约束。

因此，完整 baseline 需要补写先验训练和提示生成；随机初始化先验或随机提示不能冒充论文中的这两个组件。先验只可使用本项目训练集，且先验训练的计算预算必须额外披露。80轮是已确认的统一主训练预算，不应把先验训练开销隐藏其中。

## 5. 发布训练入口的正确性问题

### 5.1 先验损失对分割模型的梯度被截断

[train.py:92](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/train.py:92>) 先对预测做 `sigmoid`，再执行 `torch.where(prob > 0.5, 1., 0)`，最后将硬二值结果送入先验网络。这个离散选择切断了先验特征损失到分割 logits 的可微路径，因而该项无法按论文意图约束分割网络。

此外，脚本只调用 `prior_model.eval()`，没有冻结先验参数；优化器又只包含分割模型参数。先验参数仍会得到无用梯度，且不会被该优化器清零或更新。依据：[优化器:73](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/train.py:73>)、[eval:80](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/train.py:80>)。

正确适配应向先验编码器输入连续预测概率，显式冻结先验参数，同时保留对预测输入的求导；GT分支可不建梯度图。需以“先验损失产生有限且非零的分割参数梯度、先验参数保持不变”的检查验证，不能仅以 loss 数值变化判断有效。

### 5.2 发布代码与论文损失并不完全一致

| 项目 | 论文 | 发布训练代码 |
|---|---|---|
| 分割损失 | Eq5：Dice + BCE | Dice + 0.5 × Focal |
| 解剖特征距离 | Eq6：欧氏 L2 | `nn.MSELoss()` |
| 辅助分类权重 | λ1 = 0.5 | 0.3 |
| 分类类别不平衡 | Eq4 描述类别加权 CE | 默认未加权 `nn.CrossEntropyLoss()`，另由 loader 做样本加权 |
| 批内聚合 | 需按选定实现统一说明 | 分割损失按 batch 求和，CE/MSE 默认求均值 |

依据：[train.py:69](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/train.py:69>)、[train.py:102](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/train.py:102>)、[loss.py:28](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/utils/loss.py:28>)、[loss.py:74](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/utils/loss.py:74>)、本地论文 Eq4–9 与第四节C。

实现时应冻结选用“论文损失”还是“发布代码损失”的明确配置，并记录梯度修复。不能将不同版本随意混合后声称严格复现。尤其改 batch 或使用梯度累积时，求和与均值混用会改变各项实际权重。

### 5.3 数据、保存和评价入口需要统一工程适配

- 原 loader 使用 `PIL.convert('1')` 读取二值 mask，原损失没有 `-1` 忽略像素逻辑；不能直接读取本项目部分标注的多结构标签。先验重建及约束也必须显式处理未知区域，不能把未知像素解释成真实背景。
- 原 mask 损失通过 `setseq` 选通道；未选通道虽然在载入时填零，但没有作为已知背景监督。接入时应保留这一标注语义，避免跨域伪负标签。
- [train.py:174](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/train.py:174>) 仅保存模型参数，缺少优化器、调度器和随机状态；需接入本项目完整断点恢复协议。
- [metric.py:19](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/utils/metric.py:19>) 的 Dice 对双空样本会产生0/0，且没有 HD95/ASSD。正式比较应直接使用本项目三指标入口和已确定的空掩膜对角线惩罚。

## 6. 已确认的并集主方案及逐结构扩展的差异

| 目标 | 对 MOFO 的映射 | 需要说明的变化 | 能比较的结果 |
|---|---|---|---|
| **主对照：每图所有已标注结构的前景并集** | 9个数据域各对应一个并集任务；9个提示、9个 mask 通道、9类辅助分类，每图监督其所在域通道 | 网络仍有9个内部条件通道，但最终每图取一个二值前景结果；这是 MOFO 的任务条件设计，不等同于普通 U-Net 的单输出通道。提示需准确描述该域的并集目标 | 与单通道 U-Net 等比较同一并集目标；不能由并集结果可靠拆出单器官 Dice |
| 扩展：16个结构分别分割（本次不实施） | 16个结构提示和 mask 通道，每图只监督已标注结构 | 同图五腹部器官与原单标签分类冲突；可将辅助分类头解耦为9域分类，或另明确定义多标签分类，但这都是需记录的架构/损失适配 | 与 Task-ID MoE、Region MoE 的逐器官结果对应，也可派生并集结果 |

不能仅设置 `class_num=16`，再让同一腹部图在五个互斥分类标签之间反复竞争；也不能将9域并集训练结果写成16结构分割结果。已确认的并集协议中，腹部沿用本项目已确定的五类目标，其他源标签及未知像素按统一有效区域规则处理。下一步按该主方案冻结类别映射和提示资产。主评价中如何选择域通道，应在统一推理协议中明确记录；这不改变已经确定的并集训练目标。

## 7. 接入所需交付与验证

已确认的512×512、80轮、三种子应沿用；原模型通道宽度和注意力深度属于模型结构，不能直接套用本项目 U-Net 的 `base=24`。原始代码默认 batch28、200轮、lr1e-3、224输入；论文报告的是 batch80、四张 RTX3090。统一协议重训应以实际配置命名，而不是声称重现原论文全部实验条件。

需要完成的具体交付：

1. 保留上游源码，建立可追溯的512适配补丁及原始/修改后行为说明。
2. 按已确认的9域并集目标冻结任务名称、标签映射、CLIP提示文本和资产哈希；保持统一训练/验证/测试隔离。
3. 实现先验预训练、连续预测的先验约束、正确的冻结行为、部分标注损失和完整恢复。
4. 接入统一采样、训练预算、模型选择和三指标评价，披露额外预训练资产与先验阶段预算。
5. 验证512前向/反向、部分标注不产生伪负监督、先验项梯度有效、断点恢复及输出掩膜映射；随后才进入正式实验队列。

原 [requirements.txt](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/多器官超声基础模型 (MOFO)/requirements.txt>) 使用单等号版本写法，不是有效的 pip requirements 格式，还漏列实际导入的 `torch`、`torchvision`、`PyYAML`、`tensorboard`。应另写可复现环境配置，保留第三方原文件。

推荐在结果表中明确标注 **MOFO（512适配，统一协议重训）**，并附任务映射与先验预算。模型架构可以借助官方代码忠实接入，但当前状态仍是“已完成源码审查，必要训练阶段与适配尚待实现”。
