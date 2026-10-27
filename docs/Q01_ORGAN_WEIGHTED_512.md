# Q01 器官损失加权配对实验

**进度快照（2026-10-09 11:00:31，北京时间）：**27项中4项完成、2项运行、21项待执行，测试尚未开始。共享U-Net/seed42 A/B/C和Region/seed42 A完成80轮；Region B/C分别为62/43轮。已有验证结果、主best／末轮／护栏选优及全部任务见[Q01完整矩阵](experiments/Q01_SNAPSHOT_20261009.md)，其他医学实验见[总览](MEDICAL_EXPERIMENT_OVERVIEW.md)。下方启动情况和回归测试数量记录的是10月7日当时的验证。

2026-10-07，双卡后台队列已启动（控制器启动PID 2123736）。启动时GPU0执行shared/seed42/A，GPU1执行shared/seed42/B，25次待执行；实时状态以运行产物为准。执行依据：[原始任务设计](EXPERIMENT_QUEUE.md)、[实施与验收计划](superpowers/plans/2026-10-07-q01-organ-weighting.md)。

运行目录：`/ssd1/data/Multi-Organ_Foundation_Model/runs/q01_organ_weighted_80_512_20261007`。真实状态以目录内 `controller_status.json`、各实验 `metrics.jsonl` 和 `run_summary.json` 为准；本页不持续自动更新。

## 配对矩阵和固定参数

共享 16 通道 U-Net、Region MoE、Task-ID MoE，各自运行 A/B/C 三组，种子 42/43/44，共 27 次从头训练。每个架构同种子的三组使用相同初始化与逐轮抽样。

512×512，80 轮，base24，有效 batch16、物理 microbatch4，AdamW `lr=1e-4, weight_decay=1e-4`，恒定学习率，float32，梯度范数裁剪 5，workers0，无增强、无辅助损失。MoE 为 9 专家、top-k1。每域每轮抽 500 张，合计 4500 张、282 次更新。

沿用 `paper5/trainmore_v6/ct7_2` 清单、忽略规则及 CT3/CT4 验证划分。记录 SHA-256 为 `f621896f10234777d3067a019d96a7cb1dae3b6add305bca32815298744315e0`。数据文件内容另外逐文件计算 SHA-256，不能用元数据哈希代替内容校验。

## 已冻结的损失

每图有效结构数记为 K，每结构损失为有效像素 BCE + soft Dice（平滑常数1）。

- A `image_mean`：先对每图有效结构平均，再对图像平均，与历史入口相同。
- B `organ_exposure`：按真实域均衡抽样分布统计 `s_c=E[I_ic/K_i]`，取 `w_c=1/(16*s_c)`。在原通道损失上乘 w，分母仍为 K，不除以每图权重和。
- C `organ_exposure_fg_bg`：在 B 上，仅对阳性图通道的 BCE 加温和前景权重。`q=(训练阳性图前景像素数+1)/(训练阳性图有效像素数+2)`，`r=clip(sqrt((1-q)/q),1,4)`，`z=1+(r-1)*q`；前景像素乘 `r/z`，背景乘 `1/z`。空标签图 BCE 保持不变，Dice 不变。

实际训练统计覆盖6737张训练图，固定探针去重后15张；数据完整性清单另覆盖全部train/val/test的19356个唯一图像／掩膜文件（约2.08 GiB），不代表权重统计使用了验证或测试标签。腹部五类的B系数均为2.8125；C的前景／背景比r依次为肝1.7986、肾4、血管4、胆囊4、脾2.8176。血管训练阳性153张、空标签324张。

统计只来自训练集，未用验证／测试筛选权重。具体数值、阳性／空标签图数、有效像素、分组与来源分布保存在 `train_statistics.json`。分组不自动等同于患者；腹部 CT 来源不宣称为已确认患者编号。

## 诊断、选优和测试

固定探针为每结构在训练清单中首个阳性与首个空标签图，去重后所有种子和实验组共用；轮次为 0/1/10/20/40/60/80。记录原始／加权通道损失、共享编码器梯度 L2 范数、阳性与空标签预测面积。探针按单图执行，梯度贡献含 `1/K/N_probe`；报告的是这些单图梯度范数的均值和和，不是联合梯度的范数。诊断恢复 RNG、缓冲区与训练状态，不改优化器或已有梯度。

每轮验证保存16结构前景宏 Dice、腹部五结构宏 Dice、逐器官阳性 Dice、零重叠率、空标签误检率及面积，并按腹部 CT 来源分层。阈值固定0.5。

同时保留以下检查点：

1. `best.pt`：验证集16结构前景宏 Dice 最佳，作为损失主比较。
2. `last.pt`：固定 epoch80，用于检查选优是否掩盖末期退化。
3. `guardrail.pt`：在统一护栏内选验证宏 Dice 最佳；无候选通过时记为“未选中”，不回退成另一种检查点。

护栏已冻结：腹部五类各自阳性 Dice≥0.10、血管≥0.20、腹部宏 Dice≥0.40；每类阳性零重叠率≤0.50、空标签误检面积比例≤0.02。缺少必要分母则不通过。这些是研究选优条件，不是临床有效性标准。

全部27次训练和验证选优完成后，控制器才统一评价测试集的 best/last/guardrail。测试集已有历史分析，不称为未接触过的外部验证。最终 `summary.json` 汇总三种子均值／样本标准差，以及同种子的 B−A、C−B 差值；保留逐器官及来源分层，不能只凭总体均值判定腹部改善。

## 运行和手动监测

新入口为 `unet_moe/organ_weighted_suite.py`，控制器为 `scripts/run_organ_weighted.py`。实际后台进程使用运行目录内 `source_snapshot`，之后编辑工作区不会改变正在运行的实验。控制器自动领取任务，复用 `/tmp/multi_organ_fair_gpu_0.lock` 和 `_1.lock`，根据各架构真实预检峰值加 3072 MiB 余量进行显存准入。

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
Q01_RUN=/ssd1/data/Multi-Organ_Foundation_Model/runs/q01_organ_weighted_80_512_20261007

# 调度状态：27个任务的训练/测试阶段、GPU和进程号
cat "$Q01_RUN/controller_status.json"

# 两张卡的训练日志；Ctrl-C仅退出日志查看
tail -f "$Q01_RUN/logs/gpu0.log" "$Q01_RUN/logs/gpu1.log"

# GPU占用
nvidia-smi
```

中断后恢复（使用同一冻结入口，已完成且通过产物校验的任务会跳过）：

```bash
nohup .conda/bin/python "$Q01_RUN/source_snapshot/scripts/run_organ_weighted.py" \
  --output "$Q01_RUN" --gpus 0,1 \
  >> "$Q01_RUN/logs/controller.log" 2>&1 < /dev/null &
```

根目录锁拒绝重复启动；子进程继承 GPU 锁。控制器失败或被 SIGTERM 停止时会终止其训练子进程组，恢复从最近完整轮次继续。停止命令只针对本实验控制器：

```bash
kill -TERM "$(cat "$Q01_RUN/controller.pid")"
```

每项完成后会校验全部80轮采样／更新预算、选优检查点和固定诊断记录；仅有 `summary.json` 或进程退出码不能使任务被误认为完成。

## 启动验证

医学回归224通过、2跳过。三架构各完成最重计划批次的两次真实优化器更新以及固定梯度诊断。预检最大物理任务查询数20，对应本次预检有效批16图共50次查询。峰值预留显存／准入要求（MiB）：shared 4380／7452，Region 8248／11320，Task-ID 31212／34284。所有预检使用C组权重并已通过，报告位于运行目录`preflight/`。

协议SHA-256：`80bb6228ed72d1400be050cc60adda2a2dbea58db1f988913a8a2d6d72f553a5`。以上为启动及运行链路验证；10月9日已有部分单种子验证结果，但尚无完整三种子测试结论。
