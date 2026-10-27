# SMART / QMOS 源码核查与接入说明

核查日期：2026-10-12（北京时间）。本次只读审查本地源码与官方发布记录，未执行外部模型、加载权重或启动训练。

## 结论与已确认范围

SMART 的正式提交模型为 **QMOS：Swin-U-Net 编解码器、共享可学习 query、动态二值分割头，以及诊断分类支路**。它可以作为本项目的对比方法，但必须明确其原实现和统一协议适配之间的区别。

用户已确认在本项目数据上采用 **512×512、80 轮、三个随机种子重新训练，以前景并集作为五种外部方法的共同主对照目标**。SMART 保留原单通道、无任务条件的分割接口，使用统一的前景并集标签及忽略区域协议。已有 16 个 query 是模型内部隐变量，不是 16 个器官标签，本实验不把它们改成逐器官输出。

当前源码足够实现分割训练，但没有完整原挑战赛联合训练入口。正式结果建议命名为 **SMART-seg-512（本项目统一协议重训）**；不能直接称为完整复现原冠军的分类与分割联合训练。

## 来源与材料

官方发布为 [UUSIC25 Released Finalist Containers and Model Artifacts（Zenodo，版本 1.0）](https://zenodo.org/records/21807962)，DOI 为 `10.5281/zenodo.21807962`，发布日期为 2026-08-10。发布页列明第一名 SMART、第二名 AImageLab-zip、第三名 Lanzhou，材料是挑战结束后冻结的提交容器。此排名来自挑战总体结果，本文不据此推断 SMART 在本项目分割任务上的表现。

本地目录为：

```text
relate_baseline/挑战赛与资源索引 (Challenge Resources)/
  UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/
```

其中包含模型实现、Dockerfile、依赖清单及 `utils/best_model.pth`（本地约 295 MiB）。权重的存在不代表已验证可加载，亦不代表可作为无数据重叠风险的受控比较初始化。本次没有读取权重对象或做数值推理，也没有将本地所有文件与容器内文件逐一做哈希一致性核验。

官方材料声明团队自有内容采用 CC BY-NC 4.0，第三方组件保留各自许可证，见[本地官方发布 README](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 前三名元信息 (Finalist Metadata)/README_SMART_rank1.md:15>)。

## 实际结构与接口

| 项目 | 实际源码行为 | 证据 |
|---|---|---|
| 推理预处理 | 灰度图，双三次缩放至 224×224，`ToTensor()` 转为 `[0,1]` | [model.py](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/model.py:24>) |
| 输入 | 外部 `[B,1,224,224]`；内部复制为 3 通道；没有器官 ID、数据域 ID、支持集输入 | [QMOS.forward](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QMOS.py:803>) |
| 编解码器 | Swin-U-Net，patch=4、embed_dim=96、编码深度 `[2,2,6,2]`、heads=`[3,6,12,24]`、window=7 | [QMOS 构造函数](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QMOS.py:787>) |
| 共享 query | 可学习参数 `[1,16,384]`，为全部图像共享的隐变量，供分割和分类支路使用 | [query_tokens](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QMOS.py:799>) |
| 分割条件提取 | 4 层 Q-former；query 与深层图像特征做 cross-attention，经注意力池化产生图像条件向量 | [QSegmenterHead](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QSegmenter.py:78>) |
| 动态分割头 | 为全尺寸和 1/4 尺寸特征分别生成 449 个卷积参数；两层 1×1 动态卷积产生各一个前景通道 | [参数定义](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QSegmenter.py:89>)、[前向计算](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QSegmenter.py:132>) |
| 分类支路 | 多尺度特征与共享 query 交互，经 Transformer 融合，输出 2 类及 4 类 logits | [QClassifierHead](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QClassifier.py:30>) |
| 返回值 | 主分割 `[B,1,224,224]`、辅助分割 `[B,1,56,56]`、分类 `[B,2]` 和 `[B,4]` | [返回接口](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QMOS.py:808>) |
| 官方分割后处理 | 主分割 sigmoid、阈值 0.5，再将二值掩膜最近邻恢复原图尺寸 | [分割推理](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/model.py:73>) |

449 个参数来自 `32×12 + 32 + 1×32 + 1`：首层对 96 通道特征采用 8 组卷积生成 32 通道，第二层生成 1 通道。**16 个 query 经注意力池化共同生成分割头参数，不会分别输出 16 张器官掩膜。**

## 512×512 适配必须完成的工作

仅把官方推理的 resize 改为 512，或仅把模型 `img_size=224` 改为 512，均不成立。

1. `QMOS` 构造函数硬编码 224，且 [PatchEmbed.forward](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QMOS.py:560>) 断言输入尺寸等于初始化尺寸。
2. 512 输入的四级特征空间尺寸为 `128、64、32、16`，均不能被窗口 7 整除；原 [window_partition](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QMOS.py:40>) 直接 `view`，没有 padding。
3. [shifted-window mask 构造](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QMOS.py:216>) 与 [Swin block 前向](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/QMOS.py:241>) 同样依赖原始网格尺寸。

保留窗口 7 的适配应使每层按窗口补齐空间维度，为补齐 token 和 shifted-window 边界构造正确 attention mask，并在反分窗、逆 shift 后裁剪回原尺寸，再进入后续残差、patch merging 或解码路径。编解码器所有 Swin block 都需要覆盖，不能只修编码器。应验证原 224 路径的数值一致性，以及 512 前向、反向、输出空间对齐和 padding mask 行为。

改窗口为 8 可以解决整除问题，但会改变注意力窗口及相对位置偏置规格；应作为明确记录的结构变体，不应默认为与官方模型完全相同。本文不把该变体定为最终实现。

模型通道配置也不能直接套用本项目 U-Net 的 `base=24`：官方动态头依赖 96 通道和 8 组卷积。统一训练预算不意味着任意改写各方法的固有结构。

## 训练补齐与监督边界

本地可执行文件提供模型定义和挑战推理入口，未提供数据加载、训练循环、分割与分类损失、联合采样或原损失权重。源码中有两个分割输出，但不能仅据其存在推断原训练采用的辅助损失系数。

可以补齐的内容包括：本项目固定划分的数据适配、`-1` 忽略标签损失、统一训练采样、主/辅助分割监督、优化器与学习率策略、混合精度和梯度累积、完整断点恢复、固定验证选模，以及 DSC (%)、HD95 (pixel)、ASSD (pixel) 评价。这些配置必须作为本项目重训协议记录，不冒充原作者训练配方。

原推理根据 `dataset_name == 'Breast_luminal'` 选择四分类，其余分类任务选择二分类，见[分类输出选择](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/model.py:42>)。本项目现有分割训练协议并未提供原挑战赛完整且对齐的诊断分类监督。**器官标签、数据域标签不能代替这些诊断类别。** 若仅以分割标签重训，分类支路是否保留但不计算、或从训练适配器中旁路，应记录清楚；不能宣称已复现原联合任务带来的收益。

## 依赖与接入状态

官方 [Dockerfile](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/Dockerfile:3>) 基于 `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`；[requirements.txt](</ssd1/code/Multi-Organ_Foundation_Model/relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/requirements.txt:1>) 指定 torchvision 0.16.0、einops 0.8.0。源码还使用 NumPy 和 Pillow。当前环境的兼容性、512 显存需求、有效查询 batch 的实现方式都尚未实测。

当前状态是 **源码核查完成，训练适配尚未实施**。前景并集主对照目标已确认；工程上需要补齐的是正确的 512 窗口处理和完整训练入口，并按分割单任务重训记录方法名称和实际训练配置。以上缺口不需要伪造原训练细节，也不需要先下载更多大权重才能解决。
