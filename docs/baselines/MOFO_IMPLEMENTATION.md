# Q04 MOFO-512-adapted 实现

实现日期：2026-10-13。此页描述可执行实现与验证口径，不代表先验或主分割模型已经完成训练。

## 结构、来源与输入

原始来源为 `Git-HB-CHEN/MOFO`，提交 `c5076f087ae68bfb7ba19d976def2dcd5224f1c8`。`unet_moe/external_baselines/vendor/mofo/` 保留原模型及 GPL-3.0 许可证，第三方原目录未修改。

- 保留 CSWin-S 编码器的 `embed_dim=64`、深度 `[2,4,32,2]`、注意力头 `[2,4,8,16]`、条带宽度 `[1,2,7,7]`，原卷积解码器、四级提示解码器和三层动态卷积头。
- 输入 `[B,1,512,512]` 灰度值沿用公共 loader 的 `[0,1]`，复制为三通道。视觉主干从头训练，不加载官方分割或 ImageNet 权重。
- 512 时第三级 32×32 网格按宽度 7 补齐。补齐发生在 Q/K/V 投影后，attention 明确屏蔽补齐 key，LePE 卷积保留零边界，随后裁剪回原网格。整除分辨率上的输出与原注意力实现保持一致。
- 四级提示压缩后的真实网格均为 16×16，去除原 `7×7` reshape 常量；没有把输入缩回224，也没有改变窗口7为8。
- 9个域的顺序严格使用 `unet_moe.region.DOMAINS`。内部仍生成 `[B,9,512,512]`，按已知域选择为 `[B,1,512,512]`，另保留 `[B,9]` 辅助域分类。该额外域条件在结果表中单列。

## 文本资产

执行 `python -m unet_moe.external_baselines.mofo_prompts --output ASSETS_DIR`，使用官方 `openai/clip-vit-base-patch32`，固定 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。下载权重 SHA256 必须等于 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`。

只保留官方文本编码器与投影的真实输出，不对 CLIP 向量再做 L2 归一化，与 `encode_text` 输出口径一致。9个具体并集描述、模型 revision、权重及最终张量哈希保存在 `text_prompts.json`，真实 `[9,512]` 张量保存在 `text_prompts.pt`。腹部使用 medical image 描述，避免把含CT来源的数据统称超声。

正式 factory 若缺少文本资产、哈希不符或先验未完成会立即报错，不能回退随机提示或随机先验。

## 部分标注与解剖先验

仅筛选完整全图会使腹部训练域完全消失，因为该域所有图像都包含本次排除器官的未知区域。因此先验使用明确的可观察 ROI 策略：

1. 完整标注图保留全图。部分标注图使用最大完全已知、同时含前景与背景的矩形；最短边至少32像素，并至少保留该图已标注前景的25%。该阈值在训练前冻结，避免把极少量正像素的背景大框当作器官形状先验。未知像素不填为背景。
2. ROI最近邻调整为512×512，作为原始单通道 `pUNet` 的掩膜重建输入和目标。原网络通道32/64/128/256/512及全部8组多尺度特征保留。
3. 先验预训练复用冻结支持池，所有像素均来自 train；同一 ROI 的面积与保留前景比例记录在公共支持池清单。全空、全前景或无合适ROI的样本不进入先验阶段。
4. 主分割查询仍是完整512图，损失仍遵守原ignore规则；仅先验辅助项采用对应ROI。预测概率先裁剪后双线性调整，GT最近邻调整，两者送入同一冻结先验编码器。跳过先验项不跳过分割及域分类损失。

这是统一部分标注协议下的显式适配，不声称完全复现原论文全图先验实验。尤其腹部先验学习的是可观察局部结构，必须与全图分割效果区分。

## 损失与额外预算

采用论文损失版本，避免发布训练代码中硬阈值导致先验梯度截断的问题：

```text
Lseg = masked_BCE + soft_Dice + 0.5*CE(domain) + 0.2*Lprior
Lprior = ||encoder(sigmoid(logits_ROI)) - encoder(GT_ROI)||_2
Lae = BCE(reconstruction,mask) + soft_Dice + 0.5*Lconsistency
```

公共分割项按有效像素和样本平均，Dice沿用公共代码的平滑常量1。域均衡查询的有效类频率一致，CE各类等权。先验项逐图求512维瓶颈表示的欧氏范数，再除整个查询batch大小；无有效ROI的图贡献0，因此梯度累积不会因可用先验样本数量改变权重。GT编码无梯度，先验参数 `requires_grad=False`，连续预测输入保留梯度。

一致性损失按论文 Eq8 使用同域非自身样本作正对、异域样本作负对，以 cosine similarity 的 exp 和 log 实现，分母只包含负对（并非 InfoNCE）；8组多尺度特征平均。某个anchor在batch中缺少正对或负对时该项记0并维持全batch归约。实现用稳定 logsumexp，不使用原发布代码的“矩阵对角配对”替代同域多正对。

三个种子分别训练独立先验，额外各80轮、每域每轮500次掩膜抽样，共360000次；有效batch16、AdamW 1e-4、weight_decay 1e-4、恒定学习率、无增强。该阶段不计入分割主训练80轮预算。使用物理batch1和精确梯度缓存：先不建图提取整个batch的池化特征，计算批级一致性梯度，再逐microbatch重放，结合重建损失反传。pUNet使用逐样本InstanceNorm且没有随机层，避免把有效batch16的一致性损失错误拆成16个batch1。代价是每个mask两次完整前向，必须额外记录。

## API 与队列调用

```python
from unet_moe.external_baselines.mofo import build_model
model = build_model(assets_dir, prior_checkpoint, image_size=512,
                    activation_checkpointing=True)
output = model(images, domain_ids)
# output: logits [B,1,512,512], classification_logits [B,9], auxiliary_logits=[]
loss = common_loss(output['logits'], targets)
loss += model.auxiliary_loss(output, targets, domain_ids)
```

先验运行入口（由统一控制器管理GPU锁，不在本模块抢占GPU）：

```bash
python -m unet_moe.external_baselines.mofo_prior \
  --records /path/to/frozen/records.json \
  --pool-manifest /path/to/frozen/support_manifest.json \
  --output /path/to/mofo_prior/seed_42 \
  --seed 42 --epochs 80 --size 512 --batch-size 16 \
  --micro-batch-size 1 --quota-per-domain 500 --device cuda
```

恢复加 `--resume /path/to/mofo_prior/seed_42/last.pt`。每个已完成epoch原子保存模型、优化器、Python/NumPy/Torch/CUDA RNG、完整协议、历史与epoch；确定性抽样由seed/epoch重建，JSONL历史由检查点恢复。所有检查点支持 `weights_only=True`。未完成epoch从上一次完整epoch重做，不把部分更新当完成。已有协议不匹配会拒绝覆盖；有旧checkpoint但未显式resume也会拒绝覆盖。完成标志仅在真正达到配置轮数时写入。保留 `protocol.json`、`prior_pool.json`、`status.json`、`history.jsonl` 和 `last.pt`。

主模型 factory 核验先验checkpoint完成状态、尺寸、域顺序和train-only ROI协议；不允许加载随机或尚未训练完的先验。外层统一入口负责记录特定seed prior依赖、统一三指标评价及15项主训练全部完成后的test屏障。

## 验证

`tests/test_external_mofo.py` 的8项测试通过，覆盖补齐token不能稀释有效值、原整除尺寸的数值等价、16×16动态提示网格、冻结先验的非零输入梯度、未知像素梯度为0、精确梯度缓存与整批训练参数更新等价、优化器/RNG恢复及协议不匹配拒绝。

2026-10-13另完成真实数据CPU集成验证：使用当前CLIP资产、冻结训练集9域各2次抽样，512尺寸、有效batch16、物理batch1，把先验CLI实际运行1轮（仅验证用18次mask抽样），随后从检查点恢复。`weights_only=True`读取成功、优化器累计2次更新、恢复后的JSONL与检查点历史一致，无重复epoch。再用真实BUSI和AbdomenUS完整512查询及该短预算先验，通过 `MOFOAdapter` 完成2次主损失＋先验损失更新，所有梯度有限、先验参数始终冻结、动态头参数实际改变。正式factory默认要求80轮先验，已验证它拒绝这个1轮fixture。

证据：[end_to_end_cpu.json](/ssd1/data/Multi-Organ_Foundation_Model/runs/q04_preparation_20261013/verification/mofo/end_to_end_cpu.json)。该验证不是正式实验结果，也不是可以投入80轮主训练的先验资产。512完整网络前后向和完整有效batch16先验两次更新另有CPU检查；真实GPU两次更新和资源配置由统一队列预检阶段负责，未通过不得进入正式主训练。
