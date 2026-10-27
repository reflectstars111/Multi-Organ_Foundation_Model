# Q04 AImageLab / Lanzhou 适配实现

更新：2026-10-13。纯模型适配和 CPU 结构／梯度检查已完成；正式 512、真实数据、有效 batch16 的两次优化更新由 Q04 独立 GPU preflight 验收，不能将本页 CPU 检查描述为已通过正式预检。

两个接口均为 `build_model(num_domains=9)`，输入 `images: float[B,1,H,W]`、取值 `[0,1]`，以及 `domain_ids: long[B]`，返回 `{"logits": [B,1,H,W], "auxiliary_logits": []}`。输入空间尺寸由共同数据入口设为 512。模型不下载权重，不加载比赛预训练模型。依赖为现有 PyTorch 2.5.1、torchvision 0.20.1、NumPy；没有修改医学环境。

## AImageLab-zip-seg-512

代码：[aimagelab.py](../../unet_moe/external_baselines/aimagelab.py)。保留完整 `FusionWrapper`，包括 ResNet18、四级 CBAM、512→2048 投影、可训练 `tanh(alpha)` 融合，以及 size32/depth5 的 FiLM U-Net。分割 FiLM 的 embedding 从 6 改为 9 个数据域。分类支路 FiLM 保持原先关闭状态；分割 FiLM 仍开启。

- 分割视图：灰度复制为 RGB，强度 `[0,1]`。
- 分类特征视图：同一 512 网格，`4*x-2`，等价于原 uint8 `(x-127.5)/63.75`。原提交分类视图为256，本次按共同512协议改动。
- 最终输出仍为 Conv→InstanceNorm→LeakyReLU。
- 原诊断分类／框回归模块保留，当前分割任务不为其编造标签。分类特征提取支路通过融合参与分割训练。alpha 初始化为0意味着第一步特征支路梯度为0、alpha可学习，门打开后支路能够获得分割梯度。
- 不使用比赛的黑边裁剪、分器官形态学与连通域后处理，统一采用公共数据几何和公共评价阈值。

`vendor/aimagelab/net.py` 来源是发布的 `nets/net.py`，没有导入具有顶层推理／文件删除副作用的 `model.py`。只提取张量 padding 工具和默认类别数映射；修复了 padding 判定误检查通道维以及空间已整除时 `pads` 未初始化的问题。512 输入输出数值路径不因此改变。来源哈希、具体改动与许可保存在同目录 `SOURCE.json`、`LICENSE.txt`。

## Lanzhou-exact-seg-512

代码：[lanzhou.py](../../unet_moe/external_baselines/lanzhou.py)。`vendor/lanzhou/exact_model.py` 与发布源码逐字一致，包含原 Swin-T、BN、dilation2/3、多尺度块、普通两层3×3特征增强及固定深监督融合：

`z = final + 0.4*ds1 + 0.3*ds2 + 0.3*ds3`。

输入灰度复制为 RGB 后使用 ImageNet mean/std 强度归一化。网络仍输出原背景／前景两通道，适配层返回 `z_fg-z_bg`；其 sigmoid 与原 softmax 前景概率相同，二值 BCE 与两类 CE 的值和梯度均等价。深监督已在输出内融合，所以不重复生成外加辅助损失。七个发布分类头保留，不使用域ID作为诊断标注；域ID不参与分割路径。

torchvision Swin 的动态窗口补齐支持512，原结构宽度不变。没有引用 `model_fused_swin_unet.py` 或其 GN/ASPP/门控开发模型，也没有复用开发训练脚本。

## 检查证据

运行 `.conda/bin/python -m pytest -q tests/test_external_finalists.py`：**9 passed，3.05秒**。

检查覆盖：完整网络宽度前反向、AImageLab 融合门及分类／分割两支梯度、9域FiLM可学习性和恒等初始化、原输出激活保留、Lanzhou 所有深监督头梯度、BN与膨胀率、无条件域不变性、双类／单logit概率及CE梯度等价、奇数及整除网格padding、Lanzhou原文件SHA一致。CPU图像64×64只是降低测试成本，没有缩小任何模型宽度；正式512预检结果以运行目录 `preflight/{method}.json` 为准。
