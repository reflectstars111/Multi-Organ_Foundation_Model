# SMART-seg-512 实现记录

更新：2026-10-13。网络适配及 CPU 单元验证已完成；实际 512 GPU 训练预检、统一训练器、队列冻结由 Q04 总入口负责。本文不包含性能结果，也不将代码测试等同于完整训练完成。

## 源码与保留的结构

源材料为 `relate_baseline/挑战赛与资源索引 (Challenge Resources)/UUSIC 2025 第一名源码解析 (SMART Rank-1 Source)/utils/`，发布记录为 https://zenodo.org/records/21807962。原目录没有修改。使用网络文件的独立副本位于 `unet_moe/external_baselines/vendor/smart/`；`PROVENANCE.json` 保存每个原文件 SHA-256，`RELEASE_LICENSE.md` 保留组织者的许可说明。团队自有部分是 CC BY-NC 4.0，第三方部分保留其许可及版权头。

对外入口为 `unet_moe.external_baselines.smart.build_model()`，默认构造 512 模型，不下载或加载任何比赛／预训练分割权重。保留发布网络的完整宽度和分割路径：

| 部分 | 参数／张量 |
|---|---|
| 输入 | `[B,1,512,512]`，沿用统一数据入口的 `[0,1]` 灰度，网络内部复制为 3 通道 |
| Swin patch | `4×4`、96 通道 |
| 编码器 | 深度 `[2,2,6,2]`；heads `[3,6,12,24]`；窗口 7 |
| 四级 token 网格 | `128²×96 → 64²×192 → 32²×384 → 16²×768` |
| 解码器 | 原 skip、concat projection、PatchExpand、FinalPatchExpand，不缩小宽度 |
| 共享 query | `[1,16,384]` 可训练；不代表器官类别，也不使用域 ID 条件 |
| 分割 Q-former | 4 层、8 heads，使用原最深层 encoder feature 接口 |
| 动态头 | 两个独立参数生成器，各产生 449 个动态卷积参数；保留第一层 8 组卷积及原 ReLU |
| 原始输出 | 主 logit `[B,1,512,512]`；辅助 logit `[B,1,128,128]` |
| 适配器输出 | `{"logits": 主logit, "auxiliary_logits": [双线性上采样至512的辅助logit]}` |
| 全部参数 | 76,471,149 |
| 实际可训练参数 | 52,527,239 |
| 保留但冻结的分类参数 | 23,943,910 |

源网络的最深 `features[-1]` 是进入第四级 block 前的张量；本实现保留这一发布代码行为，没有擅自换成归一化后的瓶颈输出。

## 512 窗口处理

每个编码器／解码器 Swin block 根据真实网格独立补齐到窗口 7 的倍数，然后才进行 cyclic shift 和分窗。attention mask 同时处理 shifted-window 区域边界和补齐 token：

1. 真实 query 对所有补齐 key 的权重严格为零（logit mask 为负无穷）。
2. 补齐 query 只允许访问其自身，使 softmax 至少有一个有限元素，避免全负无穷行产生 NaN；这类输出随后被裁掉。
3. 反分窗、逆 shift 后裁剪至真实网格，再进入原残差、MLP、patch merging／expand。
4. 原来可整除窗口的 224 路径保持发布实现的 shifted mask。严格加载同一参数后，全网主／辅助分割及两个分类输出逐元素完全一致。

相对位置偏置、窗口宽度、编码深度均没有改变。activation checkpoint 使用 PyTorch 的 `use_reentrant=False`，保持 RNG 语义并允许正常梯度传播；默认启用以降低训练激活占用，不是减少网络。

## 分类支路和训练监督

分类支路从多尺度特征、辅助特征和同一份 query 参数读取输入，结果只返回二分类和四分类 logits；它没有将分类 token、分类输出或更新后的 query 写回分割前向路径。分割 Q-former 与共享 query 保留并接受分割梯度。

本项目没有原比赛完整诊断标签，因此 `q_cls_head` 完整保留在 state dict 中但冻结，并在默认前向中旁路计算。可用 `compute_classification=True` 执行原分类路径核查，但其随机初始化输出不作为诊断预测结果或监督。域 ID 只用于统一接口校验，不输入网络，也不生成伪诊断标签。

训练使用统一的并集标签和 `-1` 忽略规则，冻结适配损失为：

`L = L_seg(main, target) + 0.4 × L_seg(upsample(aux), target)`

其中 `L_seg` 为公共训练器的 masked BCE + soft Dice。只上采样辅助 logits，不缩小标签，避免丢失小结构或将未知像素变成背景。系数 0.4 是本项目在训练前冻结的适配设置，原发布没有可核验的联合训练损失配方，不能写成原作者设置。best/last、80 轮、三种子、有效 batch16 和三指标评价由 Q04 共同协议控制。

## 已完成验证与依赖

`tests/test_external_smart.py`：**18 passed**（2026-10-13，CPU，3.70 秒）。验证包括独立空间均值 oracle 的 shift／边界行为、四级 512 网格无效 key 屏蔽与有限 softmax、原深度／宽度／query／动态头保留、两个分割头及 query／主干梯度、域 ID 不改变输出，以及原 224 全网严格数值一致性。原 224 对照直接读取独立的本地原源码；在不含原源码的环境，该项会明确 skip，其他项仍执行。

依赖只有 PyTorch、NumPy、einops；不要求 timm 或 torchvision。验证环境版本为 `torch=2.5.1+cu124`、`numpy=1.26.4`、`einops=0.8.2`。正式依赖由 Q04 独立环境统一冻结。

运行命令：

```bash
.conda/bin/python -m pytest -q tests/test_external_smart.py
```

CPU 梯度测试使用原宽度但较小空间输入；它不能代替正式 512、真实有效 batch、两次优化更新的 GPU 预检。后者的物理 batch、累积方式、峰值显存和运行记录应由 Q04 总入口保存。
