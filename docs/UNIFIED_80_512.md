# 512×512、80 轮统一参数对照（2026-09-30）

**完成核查（2026-10-09）：**三结构模型×三种子的9次训练和测试、单通道3次训练和测试、9次派生并集评价均已完成，四模型均达到80轮。结果见[正式对照明细](experiments/FAIR_COMPARISONS_20261009.md)，全项目见[医学实验总览](MEDICAL_EXPERIMENT_OVERVIEW.md)。本页保留实际协议及历史监测入口；当前在运行的是独立的[Q01器官加权实验](Q01_ORGAN_WEIGHTED_512.md)。

**历史恢复：**原`20260930`队列在预检阶段失败，实际训练与评价产物位于`20261001_fix`系列。10月1日修复链路，10月5日恢复SIGTERM中断后的Task-ID seed44及后续控制器；这些阶段现已结束。[10月1日记录](PIPELINE_RECOVERY_20261001.md)、[10月5日记录](PIPELINE_RESTART_20261005.md)仅反映各自当时状态。

首次启动前全量 CPU 测试 **172 项通过**；本次流程修复后 **178 项通过**。真实数据元信息演练核验两分片均为 6737/1007/1618 张 train/val/test 图像，17,698 条记录 SHA 为 `f621896f10234777d3067a019d96a7cb1dae3b6add305bca32815298744315e0`，与旧公平协议完全相同；原 50 轮训练源码 SHA `2833331b3b2fdf30ee37841cc19a49b427681fe8cbd8c856b71e671760aef901` 未变。

本协议回应用户指定的 `epochs=80, batch_size=16, size=512, base=24, experts=9, top_k=1, lr=1e-4, seed=42` 参数组。它**另开结果目录**；已完成的 50 轮、base16、batch4、lr3e-4 对照及其单通道实验保留为另一套协议，不把不同超参的结果直接排名。

`unet_moe.train` 中这些是旧 Task-ID MoE 入口的默认超参，不包含相同数据协议：该入口另默认 `legacy/all8`，而旧 v6 实际运行显式设置 `paper5/trainmore_v6/ct_final` 和 `steps-per-epoch=282`。新的统一对照固定 `paper5/trainmore_v6/ct7_2`、划分种子 42、训练种子 42/43/44，保留腹部验证 CT；三种结构模型和单通道前景并集 U-Net 使用相同的 17,698 条 paper5 记录和每轮图像抽样。每域每轮抽 500 张，共 4,500 张；有效 batch 16 为每轮 282 次优化步骤。旧入口的 `steps-per-epoch=0` 意味着按“图像×任务”记录数抽样，不能表示这一套按图像配对的预算，因此不沿用该哨兵值。

四种模型统一使用 80 轮、512×512、base24、有效 batch16、AdamW `lr=1e-4, weight_decay=1e-4`、无数据增强，结构模型辅助损失权重为 0。`experts=9` 与 `top_k=1` 只适用于 MoE；`decoder=moe` 只适用于 Task-ID 入口，不适用于普通 U-Net。单通道模型只输入图像、输出一个前景并集通道；腹部按 paper5 五类合并，排除的 3/5/7/255 像素忽略，AUL 仅肿块标注的三张图像保留并把其余像素视作未知。

为避免 Task-ID 在一批 16 张图中展开多个结构查询时超出显存，三种结构模型采用**有效 batch16、物理 microbatch4**：连续四个 microbatch 的平均损失按图像数加权累积梯度，然后只更新一次参数；最后不足 16 张的批次按实际图像数处理。验证和测试可使用物理 batch4，逐图像指标不变。正式训练前对预先确定的最重训练 microbatch 执行两次前反向和 AdamW 更新的显存预检，计入优化器状态常驻后的峰值；若失败，停止并记录失败，不悄悄改变有效 batch 或模型。单通道模型若使用物理 batch16，需独立通过显存预检。

训练沿用完整预算，基于验证阳性 Dice 选择最佳检查点。三个结构模型的九次训练全部结束后显式测试；随后单通道模型完成三种子训练后显式测试，再统一计算其他模型的并集测试指标。四种模型的训练配置在启动前固定，不根据先出的测试结果调整。单通道只能与其他模型在相同并集标签上的测试指标比较；原结构模型按逐结构目标训练，二者差异不能全部归因为架构。腹部恢复实验是依赖旧 50 轮检查点的单独 8 轮微调，不并入新的 80 轮主对照。

后台链路：旧 50 轮公平对照完成 → 新三结构模型显存预检 → 新三结构模型 80 轮训练和测试 → 单通道预检、80 轮训练和测试 → 九个结构模型检查点的并集评价。GPU0 负责多头共享 U-Net / Task-ID 与后续单通道；GPU1 负责 Region。所有 GPU 阶段沿用项目文件锁，等待显存准入，不终止其他项目进程。旧 50 轮派生实验也使用这些锁，故可能先执行并推迟新阶段。

结果均位于 `/ssd1/data/Multi-Organ_Foundation_Model/runs/`：

| 目录 | 内容 |
|---|---|
| `fair_80_base24_batch16_512_20261001_fix` | 三结构模型×三种子；根 `summary.json` 为完整测试汇总 |
| `single_channel_union_80_base24_batch16_512_20261001_fix` | 单通道 U-Net×三种子；根 `summary.json` 为并集测试汇总 |
| `fair_union_eval_80_base24_batch16_512_20261001_fix` | 前述九个结构模型在相同并集目标上的测试评价 |

每个目录对应一个兄弟目录 `<目录名>-logs/`，其中 `controller_status.json` 记录各阶段。三结构模型与单通道预检均已通过；三个控制器现均为完成状态，完整逐轮及测试产物已核验。下面命令用于追溯该已完成队列。

手动查看三个控制器：

```bash
watch -n 15 'cat /ssd1/data/Multi-Organ_Foundation_Model/runs/*80_base24_batch16_512_20261001_fix-logs/controller_status.json'
```

查看正式训练逐轮日志（开始训练后文件才出现）：

```bash
tail -n 20 -F /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_80_base24_batch16_512_20261001_fix/logs/gpu0.log /ssd1/data/Multi-Organ_Foundation_Model/runs/fair_80_base24_batch16_512_20261001_fix/logs/gpu1.log
tail -n 20 -F /ssd1/data/Multi-Organ_Foundation_Model/runs/single_channel_union_80_base24_batch16_512_20261001_fix-logs/single_channel.log
```

失败时先看对应控制器状态和 `controller.log`；结构模型显存预检报告位于新公平根目录的 `preflight/`，单通道报告位于其日志目录 `gpu_preflight.json`。预检失败会停止该链路并保留失败原因，不自动减小有效 batch 或重写配置。
