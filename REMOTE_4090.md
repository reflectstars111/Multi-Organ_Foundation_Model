# 4090 服务器运行约定

## 2026-09-29：v6 / 512 / Top-1 双路线训练（进行中）

新任务分别使用物理 GPU 0（旧 task-ID MoE）与物理 GPU 1（无 task-ID 区域 MoT），**均为 Top-1**、80 epoch、512×512、base 24、batch 16、9 专家、AdamW 学习率 1e-4、seed 42。两条路线使用完全相同的 17,698 条图像—结构记录（独立图像数 6893/851/1618），`paper5` 共 16 类、`trainmore_v6 + ct_final`；腹部最终训练 633 张，原发布测试 293 张不动。最终固定使用第 80 轮而非腹部缺席的验证集挑最佳轮，测试只在训练结束后运行。旧 task-ID 路线每轮 282 次更新（4512 条单任务抽样）；区域 MoT 每域抽 500 张图，合计约 282 次更新，但一张图可有多个目标，因此两者仍不是严格等信息量对照。

服务器目录：代码 `/ssd1/code/Multi-Organ_Foundation_Model`；数据、日志与 checkpoint 位于 `/ssd1/data/Multi-Organ_Foundation_Model/runs`。启动脚本为 `scripts/run_top1_512_v6_remote.sh`，单步真实数据显存检查脚本为 `scripts/smoke_top1_512.py`。真实数据 batch 16 的训练单步峰值显存：旧 task-ID 约 21.2 GiB，区域 MoT 约 29.8 GiB。其他进程仍会占用部分显存，需关注是否出现 OOM。

```bash
LOG=/ssd1/data/Multi-Organ_Foundation_Model/runs/top1_512_v6_logs
tail -f "$LOG/task_id.stdout.log"
tail -f "$LOG/region.stdout.log"
ps -p "$(cat "$LOG/task_id.pid")","$(cat "$LOG/region.pid")" -o pid,etime,stat,%cpu,args
nvidia-smi
```

输出目录：`runs/task_id_top1_512_v6_final_seed42` 与 `runs/region_top1_512_v6_final_seed42`。每轮写 `last.pt`、验证指标，完成后写 `test_metrics.json`。如果任务被外部原因打断，先核对日志和 `last.pt`，随后在**相同参数与输出目录**下为相应 Python 命令加 `--resume`；不可新开第二个进程同时写入同一目录。服务器原代码已在 `/ssd1/code/Multi-Organ_Foundation_Model/.backup_top1_512_20260929/` 备份，新旧实验输出均未覆盖。

## 新版三方案实验（2026-09-29）

服务器代码已更新并通过 23 项相关测试。新实验统一使用 `paper5/ct7_2`、16 个结构、9 个数据集、256×256、base 16、batch 2、50 epoch、每个数据集每轮抽样 500 张、相同随机种子 42；三方案的训练图像抽样和主分割损失一致。主线为图像区域特征 Top-1 MoE（9 专家），另外两组为共享 U-Net 和 9 个独立 U-Net。MoE 另有粗分割与路由辅助损失，架构本身不同。

本地和服务器真实数据审计均通过：腹部训练/验证/测试为 477/156/293 张、按 CT 来源隔离；第 1 轮三方案各数据集抽样图像一致。旧实验不覆盖。

两卡曾被其他 GPU 任务占满，旧的服务器三方案等待队列尚未创建训练输出就已停止。随后在本地启动三方案训练；2026-09-29 17:30 本地 Top-1 MoE 和共享 U-Net 均已完成。用户要求把 9 个独立 U-Net 改在服务器跑；本地串行队列停止，保留已完成的 BUSI、MMOTU 和 DDTI 中断前 checkpoint。**目前是 MoE/共享 U-Net 在本地、9 个独立 U-Net 在服务器的新混合执行方案。**

本地 RTX 5070 Laptop GPU（8GB）的 Top-1 MoE 测试集 16 类阳性 Dice 宏平均为 0.7199，共享 U-Net 为 0.7108。结果在本地 `runs/region_paper5_ct7_2_v1` 和 `runs/unet_paper5_ct7_2_v1/shared`。服务器独立模型输出在 `/ssd1/data/Multi-Organ_Foundation_Model/runs/unet_paper5_ct7_2_v1`，日志在 `/ssd1/data/Multi-Organ_Foundation_Model/runs/paper5_ct7_2_independent-logs`。服务器 GPU 0 分配 BUSI/MMOTU/DDTI/AUL/FALLMUD，GPU 1 分配 Fetal_HC/CCA/CAMUS/AbdomenUS；启动脚本会检查 GPU 利用率和温度，全部完成后生成 `independent_comparison.json`。

本地与服务器 `records.json` 中的绝对路径不同，因此原始 SHA-256 不同。最终三方案比较前须按 `scripts/rebase_baseline_run.py` 的 `semantic_key` 逻辑验证记录在去除机器路径前缀后完全一致，不能直接把不同哈希的结果混在一起。

本地查看进度（PowerShell）：

```powershell
Get-Content runs/paper5_ct7_2_local_v1-logs/controller.stdout.log -Tail 20
Get-Content runs/paper5_ct7_2_local_v1-logs/moe.stdout.log -Tail 20
Get-Content runs/paper5_ct7_2_local_v1-logs/controller.stderr.log -Tail 20
```

服务器独立模型队列进度：

```bash
LOG=/ssd1/data/Multi-Organ_Foundation_Model/runs/paper5_ct7_2_independent-logs
tail -f "$LOG/controller_gpu0.log" "$LOG/controller_gpu1.log"
tail -f "$LOG/BUSI.log" "$LOG/Fetal_HC.log"
tail -f "$LOG/finalizer.log"
```

`scripts/run_independent_paper5_remote.sh` 与 `scripts/finalize_independent_paper5_remote.sh` 是当前服务器任务。旧的 `run_three_way_paper5_remote.sh` 和 `finalize_three_way_paper5_remote.sh` 未运行，不要重启，以免重复训练。

代码与项目环境：`/ssd1/code/Multi-Organ_Foundation_Model/`。

训练用的 `pointed_data`、checkpoint、日志与迁移包：`/ssd1/data/Multi-Organ_Foundation_Model/`。不把训练内容放到 `/hdd`；`/ssd2` 保留为 `/ssd1` 空间不足时的备用盘。迁移仅包含九个数据集本项目实际使用的 `processed_png` 图像、标签、索引及 CAMUS 原版患者划分文件；原始发布包、论文等不参与本次训练。

项目 Python：`/ssd1/code/Multi-Organ_Foundation_Model/.conda/bin/python`。环境由服务器现有 `pytorch_bin` 环境克隆到项目目录，避免修改其他项目环境。

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
DATA=/ssd1/data/Multi-Organ_Foundation_Model
PY=$PWD/.conda/bin/python

# 若训练中断，检查已有最后一轮 checkpoint 后运行 --resume。
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 "$PY" -u -m unet_moe.baseline_suite \
  --data-root "$DATA/pointed_data" --output "$DATA/runs/unet_baseline_9v1" \
  --mode all --epochs 50 --size 256 --base 16 --batch-size 2 \
  --quota 500 --workers 2 --seed 42 --device cuda --resume
```

训练和验证按数据集来源使用相同图像抽样；独立模型有九份参数，推理时需知道图像对应哪个数据集。`comparison.json` 在全部测试完成后汇总 19 个结构的 Dice 和参数量。日志在 `/ssd1/data/Multi-Organ_Foundation_Model/runs/unet_baseline_9v1-logs/`。

本批基线已全部完成，`runs/unet_baseline_9v1/comparison.json` 已生成。共享 U-Net 的 19 类阳性 Dice 宏平均为 0.5694，九个独立 U-Net 合并后为 0.6125；AbdomenUS 八类分别为 0.1952 和 0.2836。上述 `--resume` 命令仅用于原实验复现，不需要再次启动。脚本 `scripts/run_remote_worker.sh 0` / `1` 与 `mofm_summary` 用于原实验的两卡续训和结果汇总，日志分别在 `gpu0.log`、`gpu1.log`、`finalizer.log`。

`mofm_summary` tmux 会话已启动，执行 `scripts/finalize_remote_baselines.sh`，待全部 10 个测试结果齐备后自动生成 `comparison.json`，过程写入 `finalizer.log`。下述命令可在需要时手动重算汇总：

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
DATA=/ssd1/data/Multi-Organ_Foundation_Model
.conda/bin/python -m unet_moe.baseline_suite \
  --data-root "$DATA/pointed_data" --output "$DATA/runs/unet_baseline_9v1" \
  --mode summarize --epochs 50 --size 256 --base 16 --batch-size 2 \
  --quota 500 --workers 2 --seed 42 --device cpu --resume
```

迁移流程校验：本地数据 tar 和服务器数据 tar 做 SHA-256 比较；服务器解压后重新扫描图像、mask；Windows 与 Linux 记录逐条比较数据集、任务、划分、标签版本和相对路径。仅路径前缀不同才重写 checkpoint 中的记录校验值。`migration.json` 记录新旧校验值。

## 腹部稀有类修复（已部署，尚未正式训练）

`unet_moe/region.py` 和 `unet_moe/region_train.py` 已针对主线 Top-1 MoE 修改损失：每张图先平均有效标注通道，腹部各类的阳性图再按训练集阳性/空图数量比加权（上限 8），同一权重同时用于最终分割和四级粗区域监督；空标签仍参与负例约束，`-1` 缺失/忽略区域仍不参与损失。训练启动前强制检查 HC18 的 999 条记录使用实心标签。权重从训练划分自动计算，不读取验证或测试标签。

2026-09-29 已同步上述两个代码文件和对应测试到服务器，逐文件 SHA-256 与本地一致。服务器项目环境补装了 `pytest`，相关 19 项测试通过；真实肾上腺阳性训练样本的单步 GPU 前向/反向有限且粗区域提议分支收到梯度。旧版三个文件备份在服务器代码目录 `.backup_abdomen_fix_20260929/`，旧 `region_top1_v1` 权重与结果保持不变。

服务器数据审计：AbdomenUS 的 926 对图像/索引 mask 均尺寸匹配且类别值仅为 0–8/255；13 个 CT 来源按训练/验证/测试 5/4/4 分开，无交叉。HC18 共 999 条记录均使用 `hc18_filled_v1`，mask 文件存在。训练划分中，肾上腺仅 44/321 张图有前景，自动阳性权重为 6.295；血管 93/321、权重 2.452；骨骼 115/321、权重 1.791。

**边界：**数据格式、路径及已知类别映射通过检查，稀有类损失权重的代码问题已修复；但训练数据仅来自 5 个模拟 CT 来源，类别稀少与跨来源分布差异仍在。没有新模型测试成绩，因此不能断言分割效果已改善，也不能声称真实超声泛化问题已解决。后续必须使用新输出目录进行受控验证，不能覆盖旧实验。

## 论文范围对齐的新协议（已通过服务器数据审计，等待训练）

原 8 类 AbdomenUS 实验保留。新增可选 `--abdomen-labels paper5 --abdomen-split ct7_2`：腹部只监督肝、肾、血管、胆囊、脾，连同其他数据集共 16 个输出，仍为 9 个专家。当地数据的红色标签是“血管”，不保证只含主动脉；黄色标签是整肾，不能等同于论文细分的肾盂/皮质。因此这是按本地可用标签**近似对齐**论文器官范围，不是逐像素复现论文标注。

原发布包测试 CT `ct1/ct10/ct11/ct12` 及其 293 张图保持不动；原训练包中 `ct3/ct4` 的 156 张作验证，其余 7 个 CT 的 477 张作训练。分割协议与旧 5/4/4 划分独立记录；`--abdomen-positive-sampling 0.5` 仅在训练抽样时按 CT 均衡并温和提高稀有前景样本概率，验证/测试不使用真值选图。旧命令默认 `legacy/all8`，旧实验不应覆写。后续共享 U-Net、9 个独立 U-Net 和主线 MoE 都应使用同一新协议才能正式比较。
