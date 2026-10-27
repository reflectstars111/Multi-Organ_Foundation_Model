# 4090 服务器运行约定

更新日期：2026-09-30。代码与项目环境在 `/ssd1/code/Multi-Organ_Foundation_Model/`；训练数据、checkpoint、日志与迁移包在 `/ssd1/data/Multi-Organ_Foundation_Model/`。训练不写入 `/hdd`；`/ssd2` 保留为 `/ssd1` 空间不足时的备用盘。环境使用项目 `.conda/bin/python`，避免修改其他项目环境。运行前查看 `nvidia-smi`，只选择空闲 GPU，不终止其他项目任务。

## 当前状态

2026-09-30 本轮已启动 GPU1 的 512×512 自动实验队列。先运行两组各 12 轮的腹部空标签惩罚消融，只做验证；汇总后自动运行三模型×三种子、各 50 轮的公平对照，全部训练完成才评估冻结测试。实时阶段在 `/ssd1/data/Multi-Organ_Foundation_Model/runs/research_512_20260930-logs/status.json`，详细日志为同目录 `ablation.log` 和 `fair.log`。启动不表示准确率问题已经解决。

队列入口为 `bash scripts/run_512_research_remote.sh /ssd1/data/Multi-Organ_Foundation_Model/runs 1`；只有中断后才用相同命令加 `--resume`。正在运行时不要重复启动。源码快照保存在数据目录的 `transfer/research_512_source_20260930.tar.gz`，续训必须保持源码和协议一致。

本轮改为两卡分片：GPU0 运行现有协议的**16 通道多头共享 U-Net**（代码代号 `shared`）与 task-ID MoE 的三个种子；GPU1 完成腹部消融后运行 Region MoE 的三个种子。仍保持 512×512、base16、batch4、50 轮、每域每轮 500 张以及全部辅助损失为零。用户指定的纯单通道 U-Net 另作前景并集实验，其指标口径不同。GPU0 同时保留另一个项目的既有任务，因此计时仅为实际运行耗时，不能用于证明架构速度优势。

公平对照根目录 `runs/fair_512_seed42_43_44_20260930/` 的 `parallel_plan.json` 保存任务分配；`shards/gpu0/`、`shards/gpu1/` 各自保存协议、checkpoint 与指标。两卡分别写 `worker_gpu0.json` / `worker_gpu1.json` 状态和 `logs/gpu0.log` / `logs/gpu1.log`。全部九次训练通过完整性及抽样一致性检查后才允许测试，最终根目录 `summary.json` 统一汇总。上层研究队列的 `status.json` 主要表示 GPU1 的阶段，GPU0 的并行进度应另看 worker 状态。

GPU0 分片已经启动。手动查看其日志可运行 `tail -n 30 -F /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_512_seed42_43_44_20260930/logs/gpu0.log`。并行版本源码快照为 `transfer/research_512_parallel_source_20260930.tar.gz`，本次完整测试为 108 项通过。单卡 worker 中断后可在项目目录运行 `.conda/bin/python -m scripts.fair_parallel --output /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_512_seed42_43_44_20260930 --gpu 0` 恢复对应分片；GPU1 对应参数为 `--gpu 1`，已有 worker 正在运行时不要重复启动。

`unet_baseline_9v1` 的共享及九个独立模型均已完成。`unet_paper5_ct7_2_v1` 当前服务器有九个独立模型完成产物，但缺该批共享/区域模型原始结果。两条 `*_top1_512_v6_final_seed42` 均已完成 80 轮并用 last80 评估；腹部没有 validation，五折计划尚未完成。完整结果和指标差异见 [实验状态](docs/EXPERIMENT_STATUS.md)。旧迁移文档的“共享完成、BUSI 21轮”和继续启动 worker 指令属于历史，不再是当前任务。

本轮新训练统一为 **512×512、base16、batch4**。旧 256 候选目录 `empty_ablation_20260930` 已停止并标记 `superseded`，保留其产物供追溯，不在其 checkpoint 上改变尺寸续训。512 候选使用新目录 `empty_ablation_512_20260930` 重跑，完整比较尚未完成；正式三架构三种子使用 `fair_512_seed42_43_44_20260930`，当前待启动。小尺寸 `fair_smoke_20260930` 是已完成的旧源码链路检查，不能当作 512 精度结果或用于当前源码续训。

## 固定划分的三架构比较

新入口 `unet_moe.fair_suite` 固定 `paper5 + ct7_2 + trainmore_v6`，划分 seed 固定为 42，训练种子独立控制。本轮正式脚本采用 512×512、base16、batch4、Top-1、9 专家、每域每轮 500 张图、50 轮、AdamW lr=3e-4、weight decay=1e-4。三架构每轮均看相同 4500 张抽样图像，以相同逐图像 BCE+Dice 训练，使用验证阳性 Dice 选模，runner 默认不读 test PNG 或计算 test 指标。

以下是另开单进程实验的命令模板，不适用于已经配置 `parallel_plan.json` 的本轮根目录，也不表示完整三种子实验已完成。先固定全程预算、只运行第一轮估算开销；重新运行同一命令增加 `--resume` 并移除 `--stop-after-epoch 1` 即继续。

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
DATA_ROOT=/ssd1/data/Multi-Organ_Foundation_Model
PROJECT_PY="$PWD/.conda/bin/python"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 "$PROJECT_PY" -u -m unet_moe.fair_suite \
  --data-root "$DATA_ROOT/pointed_data" \
  --output "$DATA_ROOT/runs/fair_512_single_example" \
  --models shared,task_id,region --seeds 42,43,44 --split-seed 42 \
  --size 512 --base 16 --batch-size 4 --top-k 1 --experts 9 \
  --epochs 50 --quota-per-domain 500 --lr 0.0003 --workers 2 \
  --device cuda:0 --stop-after-epoch 1
```

`CUDA_VISIBLE_DEVICES=1` 将物理 GPU1 映射为进程内 `cuda:0`。示例应按实际空闲 GPU 调整。默认辅助损失为 0；`--coarse-weight`、`--route-weight`、`--balance-weight`、`--router-z-weight` 的非零设置须另设输出目录。`--empty-weight 0.1 --empty-fraction 0.01` 是已标注腹部空通道的候选困难负像素损失，只能当作待验证消融；不把缺失标签当作背景，也没有证实能提升性能。

封装脚本可顺序执行；未配置并行计划时检查 GPU 空闲，已配置时进入对应 GPU 分片并按剩余显存准入。两种方式均保留进程锁：

```bash
bash scripts/run_empty_ablation_remote.sh "$DATA_ROOT/runs/empty_ablation_512_20260930" 1
bash scripts/run_fair_suite_remote.sh "$DATA_ROOT/runs/fair_512_seed42_43_44_20260930" 1
```

候选脚本对比区域 MoE 的 `empty_weight=0/0.1`，均为 512×512、base16、batch4、12 epoch、每域每轮 128 图、seed42、coarse/route=0.3/0.1，只计算验证集。正式脚本使用上方 50 轮、三种子、三架构配置、辅助项全为 0，并在全部训练结束后显式追加 `--evaluate-test`。两个脚本的协议与预算不同，不能混合汇总。目录已存在时须以原配置加第三个参数 `--resume`；运行脚本与上方逐轮暂停示例应择一使用，不并行操作同一目录。

只检查协议与划分时，可在上述命令加 `--dry-run`：写入 protocol/records，不读取 PNG、不训练。以后必须使用完全相同配置加 `--resume` 开始实际运行。`--stop-after-epoch` 只是暂停点，不能改变预设 `--epochs`、模型集合、种子列表、输入尺寸或损失。源码、记录、配置校验不一致时拒绝续训；需保留原代码或使用新目录。依赖环境版本会记录，但代码哈希不代表 PNG 内容哈希。

冻结配置并完成全部预算后，复用相同命令、移除暂停参数并加 `--resume --evaluate-test`，才对验证选中的 checkpoint 计算 test。训练 seed、模型清单和设备参数也须保持一致。测试已在历史项目中被查看过，不能称为全新外部测试。

输出顶层为 `protocol.json`、`records.json`、`summary.json`；模型子目录为 `<model>/seed_<seed>/`，包含 `last.pt`、`best.pt`、`metrics.jsonl` 和显式评估时的 `test_metrics.json`。协议记录代码摘要、Git 提交、依赖版本、图像/分组数和预算；每轮保存样本顺序摘要、参数量及训练/验证时间。checkpoint 保存模型、优化器与 Python/NumPy/PyTorch CPU/CUDA RNG。CUDA 跨版本不保证逐位一致。

`summary.json` 汇总种子均值和样本标准差；只有一个种子时标准差为 null。阳性 Dice、含空 Dice、空图误报比例/面积与阳性漏检分别报告。新 checkpoint 当前通过 `fair_suite` 统一评估，不能直接传给旧 `predict`/`region_predict`。

## 检查与历史续训

```bash
"$PROJECT_PY" -m pip install -r requirements-unet-moe.txt -r requirements-preprocessing.txt
UNET_MOE_DATA_ROOT="$DATA_ROOT/pointed_data" "$PROJECT_PY" -m pytest tests -q
```

设置 `UNET_MOE_DATA_ROOT` 才能在本服务器执行依赖真实数据的划分检查；只看到 skip 不能认为真实数据协议已验证。训练不会因文档示例自动启动。

旧 `train`、`region_train`、`baseline_suite` 均支持 `--resume`，但其 `--seed` 同时改变项目划分；不得用改变旧 seed 的方式声称固定 test 的多种子对照。旧区域/基线 checkpoint 也没有完整 RNG 状态。只有确实未完成且配置一致的旧任务才应续训；现有完成目录应保持不动。

迁移核对仍包括数据 tar 的 SHA-256、图像和 mask 路径扫描、Windows/Linux 记录的相对路径语义键比较。仅路径前缀变化才可按迁移脚本重建记录哈希，`migration.json` 保留新旧对应；迁移不改变样本划分或标签语义。旧脚本 `scripts/run_remote_worker.sh` 面向已完成的历史基线，不是新公平对照的启动入口。
