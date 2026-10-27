# 医学历史实验、腹部探索及失效记录

核查日期：2026-10-09（北京时间）。本页依据服务器 `runs/` 中实际存在的配置、逐轮日志、测试/比较 JSON 和失效标记整理，未加载大检查点或重新运行推理。机器可读明细见[历史实验清单](historical_experiments_20261009.json)。这里的“完成”表示对应历史协议的训练或筛查完成；“未采用”表示实验完成但效果未过预设条件，两者不等同于程序失败。

正式 50/80 轮对照和当前 Q01 另见[医学实验总览](../MEDICAL_EXPERIMENT_OVERVIEW.md)。以下不同协议的结果不能合并成一张架构排行榜；旧 256 输入实验保留其真实尺寸，不因后续统一 512 的要求改写历史。

## 1. 已有原始产物的历史训练

表中 Dice 均为百分数；没有多种子标准差的行只有 seed42。历史 `shared` 代表有结构通道的共享 U-Net，不是后续单通道前景并集模型。

| 运行目录/模型 | 配置与监督范围 | 完成证据 | 已保存测试指标 |
|---|---|---|---:|
| `region_top1_v1` | 256、base16、batch2、lr3e-4、50轮；19结构、腹部8类；旧 HC18 轮廓版本 | 50轮日志；测试采用epoch35 | 19结构前景宏 Dice **54.63%** |
| `unet_baseline_9v1/shared` | 256、base16、batch2、lr3e-4、50轮；19结构；HC18 filled | 50轮日志与测试 JSON | 19结构前景宏 Dice **56.94%** |
| `unet_baseline_9v1` 的9独立模型 | 同批历史19结构协议；每域独立训练一套U-Net | 9份50轮日志、9份测试 JSON、比较汇总 | 合并19结构前景宏 Dice **61.25%** |
| `unet_paper5_ct7_2_v1` 的9独立模型 | 256、base16、batch2、lr3e-4、50轮；16结构；腹部五类、ct7_2 | 9份50轮日志、9份测试 JSON；各域选优epoch24–39 | 16结构前景宏 Dice **73.01%** |
| `region_top1_512_v6_final_seed42` | 512、base24、batch16、lr1e-4、80轮；16结构；trainmore_v6 + ct_final | 80轮日志；测试明确采用固定last80 | 前景宏 Dice **74.51%**；含空宏 Dice **78.28%** |
| `task_id_top1_512_v6_final_seed42` | 512、base24、batch16、lr1e-4、80轮；相同v6记录；每轮282步 | 80轮日志；测试明确采用固定last80 | 旧任务含空宏 Dice **78.05%**；未保存对应阳性宏 Dice |

原始来源：[旧Region测试](/ssd1/data/Multi-Organ_Foundation_Model/runs/region_top1_v1/test_metrics.json)、[19结构U-Net对照](/ssd1/data/Multi-Organ_Foundation_Model/runs/unet_baseline_9v1/comparison.json)、[paper5独立模型汇总](/ssd1/data/Multi-Organ_Foundation_Model/runs/unet_paper5_ct7_2_v1/independent_comparison.json)、[Region v6测试](/ssd1/data/Multi-Organ_Foundation_Model/runs/region_top1_512_v6_final_seed42/test_metrics.json)、[Task-ID v6测试](/ssd1/data/Multi-Organ_Foundation_Model/runs/task_id_top1_512_v6_final_seed42/test_metrics.json)。

两项 v6 实验采用固定最后一轮；腹部 train/val/test 为 **633/0/293**，并没有腹部验证选优或已完成的腹部五折。它们共享记录 SHA `ce45e1b973a4d9cb7f73ecc18412561bd77db26312b68236989bd5fce212e88a`，但 Region 每域抽图、旧 Task-ID 抽“图像×任务”记录，曝光预算与目标函数不相同；Region 还启用粗区域/路由辅助权重0.3/0.1及腹部阳性平衡。因此这两项也不能直接代替现在的公平配对对照。

历史 paper5 独立模型的腹部五器官测试前景 Dice 为 **69.83/37.28/29.67/66.88/30.07%**（肝/肾/血管/胆囊/脾）。这批9模型合计约17.69M参数，只能证明当时独立模型的保存结果，不能作为现有512、base24、80轮三种子的独立模型结果。根 `protocol.json` 中的 `domains=BUSI` 是一次调用留下的参数；完成数量以9个实际子目录的50轮日志和测试文件为准。

`unet_baseline_9v1/migration.json` 记录了20476条语义记录保持一致的迁移过程以及路径哈希变化。迁移不是一次新训练，不能另计实验次数。

## 2. 仅有历史文本、当前缺少原始输出的比较

[旧paper5报告](../../PAPER5_THREE_MODEL_ANALYSIS.md)列出的 Region 71.99%、共享U-Net 71.08% 没有在本次服务器核查中得到原始产物复核：`region_paper5_ct7_2_v1/` 和 `unet_paper5_ct7_2_v1/shared/` 不存在。不得将这两个数字并入“本机已核验”的结果表，也不能据旧报告再生成缺失的原始测试数据。

`paper5_ct7_2_v1-logs` 中控制器主要保留GPU等待记录，finalizer末条仍在等待三方产物；它没有证明旧计划完整完成。后来 `paper5_ct7_2_independent-logs/finalizer.log` 明确记录九个独立U-Net完成，并写出实际存在的 `independent_comparison.json`。这两份日志不是两批独立模型结果，也不能把旧三方计划当成三方已完成。

## 3. 固定旧Region的腹部诊断

`abdomen_audit_20260930` 已完成，使用 v6 Region 的固定last80、512输入、阈值0.5，对现有926张腹部图做训练/测试诊断。它没有训练模型或调阈值，是解释旧结果的探索性分析。

- 测试腹部前景 Dice：肝 **73.22%**、肾 **43.56%**、血管 **37.20%**、胆囊 **74.02%**、脾 **30.62%**。
- 肝空标签测试图 **65/66** 出现预测，图像误检率 **98.48%**；误报图面积中位数4246像素，均超过16像素。全部空图有效像素上的误检比例为5.15%，与图像误检率是两个不同分母。
- 肾、血管、脾阳性测试图的零交集分别为 **46/174、78/151、34/75**；不能仅通过压低预测面积来宣称改善。
- 449×464→512最近邻标签缩放没有造成这五类阳性图消失；ignore像素参与指标时被排除。五腹部通道在四层都选择专家8，未支持“主要因为选错专家”的简单解释。
- 原始RGB掩膜目录当时未在本机保存；审计核实的是索引标签与缩放后的标签存在性，不能声称本次重新验证了RGB转换全过程。未标注非腹部通道的激活也不能按已确认假阳性计数。

证据：[诊断协议](/ssd1/data/Multi-Organ_Foundation_Model/runs/abdomen_audit_20260930/protocol.json)、[发现及限制](/ssd1/data/Multi-Organ_Foundation_Model/runs/abdomen_audit_20260930/findings.json)。该诊断已经查看历史测试集；后续默认“不看test”不会使这组历史测试重新成为从未查看的外部验证。

## 4. 空标签惩罚的12轮试验

`empty_ablation_20260930` 为原256输入试跑：weight0完成12轮，weight0.1只完成7轮；随后按用户统一512要求中断并由[明确的superseded标记](/ssd1/data/Multi-Organ_Foundation_Model/runs/empty_ablation_20260930/status.json)指向512版本。这批不能描述为两组完整512实验。

`empty_ablation_512_20260930` 两组均完成12轮，仅Region、seed42，base16、batch4、lr3e-4，每域每轮128图。两组每轮抽样哈希一致，采用验证16结构前景宏Dice选优，不使用测试集；共同保留粗区域/路由辅助权重0.3/0.1，与后来的辅助项全零公平主对照不同。

| 所选验证检查点 | 原权重0（epoch11） | 候选权重0.1（epoch12） |
|---|---:|---:|
| 16结构前景宏Dice | 51.22% | 46.24% |
| 腹部五类前景宏Dice | 15.62% | 0.36% |
| 腹部空标签图像误检率 | 44.95% | 0.00% |
| 腹部空标签误检面积比例 | 4.86% | 0.00% |
| 腹部阳性零交集率 | 69.57% | 99.50% |

候选通过“降低误检”条件，却未通过前景Dice和漏检护栏，最终 **not_selected**，默认empty权重保持0。它证明这次单种子、12轮、权重0.1的候选有明显问题，不能外推为所有空标签加权均无效。[原始比较及判定](/ssd1/data/Multi-Organ_Foundation_Model/runs/empty_ablation_512_20260930/comparison.json)

## 5. 验证集80条后处理规则

有效版本为 `abdomen_postprocess_512_20260930_v2`：固定旧50轮公平Region seed42的best epoch39，只在腹部156张验证图（CT3=66、CT4=90）上筛查阈值与连通域组合。网格中有 **80种规则，每种应用于5个器官**，不是80次新训练。默认阈值0.5，最低候选阈值0.05。

肝、血管、胆囊、脾均无规则通过第一阶段条件；肾的precision规则（阈值0.6、最小连通域128像素、不只留最大域）进入另一CT检查但 **not_confirmed**：CT4阳性零交集从3/43增加到4/43。因此最终没有采用任何新规则。血管在最低阈值0.05的筛选CT上仍是全空预测，不能依靠这批阈值/连通域操作恢复。

CT3/CT4都用于原始模型选优，并且已在v1检查过；另一CT检查不构成独立外部确认，156张图也不是156个独立CT来源。[有效筛查及协议](/ssd1/data/Multi-Organ_Foundation_Model/runs/abdomen_postprocess_512_20260930_v2/screen.json)

旧 `abdomen_postprocess_512_20260930` **明确作废**：v1在筛选连通域之前与真值有效区域相交，使用了实际推理不能获得的ignore掩膜。其 `screen.json` 不应用于选规则或报告后处理收益。[作废原因和替代路径](/ssd1/data/Multi-Organ_Foundation_Model/runs/abdomen_postprocess_512_20260930/superseded.json)

## 6. 六组×三种子的8轮腹部恢复微调

`abdomen_recovery_512_20260930` 的 **18/18次训练完成8轮**，完成时间为2026-10-01 06:24（北京时间），没有测试评价。每个种子从旧50轮公平Region的best.pt出发（42/43/44分别为epoch39/50/31），重置AdamW；统一512、base16、batch4、lr1e-4、每域128图/轮，共1152图和288步/轮。

先在epoch0至8中按固定护栏选腹部验证前景Dice最高的检查点，再将各候选与同seed的微调control配对。下表差值单位为百分点；epoch0代表选择微调前模型。

| 实验组 | 相对control改变 | 所选epoch（42/43/44） | 腹部Dice配对均值差 | 最终判定 |
|---|---|---|---:|---|
| control | 原损失和均匀腹部抽样 | 0/6/1 | — | 配对参照 |
| empty_0p002 | 空标签top1%惩罚0.002 | 0/8/0 | -0.37 | 未采用 |
| empty_0p01 | 空标签top1%惩罚0.01 | 0/0/0 | -0.61 | 未采用 |
| rare_positive_x2 | 血管/脾阳性通道损失×2 | 0/6/0 | -0.18 | 未采用 |
| abdomen_positive_sampling_x3 | 血管或脾阳性训练图抽样权重×3 | 0/2/0 | -0.45 | 未采用 |
| rare_positive_x2_empty_0p002 | 阳性×2和空标签0.002组合 | 0/4/0 | -0.28 | 未采用 |

**12/18次最终选epoch0，5个候选均未通过预设比较条件。** 所选检查点的血管验证前景Dice在所有组的三种子均值均为0；本批短程微调没有可靠恢复血管。采样组保持每域抽样数量不变，但改变域内图像分布，其逐轮抽样序列不与control相同。

原条件要求腹部Dice平均至少增加1个百分点，同时约束总体Dice、空标签误检率/面积、阳性零交集及单种子退化。不能仅以部分误检下降改判候选成功，也不能将这批微调无收益解释为后来从头训练Q01的加权策略无效。完整条件、每seed差值、未过条件和检查点来源见[冻结计划](/ssd1/data/Multi-Organ_Foundation_Model/runs/abdomen_recovery_512_20260930/plan.json)及[最终配对比较](/ssd1/data/Multi-Organ_Foundation_Model/runs/abdomen_recovery_512_20260930/comparison.json)。

## 7. 失败、工程预检与日志的归档关系

| 目录或产物 | 实际状态 | 归档解释/替代关系 |
|---|---|---|
| `fair_80_base24_batch16_512_20260930` | 预检失败，未开始训练 | 分片summary为dry_run=true、runs为空；CUDA reset_peak_memory_stats报Invalid device argument；由`20261001_fix`重建 |
| `fair_80_base24_batch16_512_20260930-logs` | failed | 保存原始失败证据，不应视为当前80轮仍失败 |
| `single_channel_union_80_base24_batch16_512_20260930-logs` | 上游失败导致failed | 同名输出目录不存在；新的`20261001_fix`单通道结果已完成 |
| `fair_union_eval_80_base24_batch16_512_20260930-logs` | 上游失败导致failed | 同名输出目录不存在；新的`20261001_fix`并集结果已完成 |
| `fair80_preflight_fix_20261001` | 三模型passed | 显存与两步优化器预检；不计为80轮训练 |
| `fair_smoke_20260930` | 三模型各1轮完成 | 64输入、base4、每域2图，无正式测试；仅链路检查 |
| `migration_smoke_v1` | 共享模型1轮完成 | 32输入、base4、每域2图；虽存测试JSON，不能用于精度主结论 |
| `research_512_20260930-logs` | complete | 旧50轮对照及512空标签消融的外层控制器 |
| `top1_512_v6_logs` | 对应两项v6历史训练完成 | 日志和pid文件不单计实验；pid文件存在不说明当前进程存活 |
| `unet_baseline_9v1-logs` | 对应旧10模型完成 | 不另计一批训练 |
| `abdomen_recovery_512_20260930-logs` | complete | 对应18次微调，不另计18次 |
| `resources_20260930.json` | 历史资源快照 | 非实验，不代表当前GPU负载 |

旧50轮并集评价曾因聚合元数据校验失败，之后在原结果/日志目录恢复并已完成；不能只读归档的 `controller_status.before_recovery_*.json` 判定当前失败。有效80轮`20261001_fix`队列也曾中断后恢复；其最终完成状态应读取现有主 `controller_status.json` 和完整 `summary.json`。恢复过程见[10月1日修复记录](../PIPELINE_RECOVERY_20261001.md)、[10月5日恢复记录](../PIPELINE_RESTART_20261005.md)。

所有目录分类、状态证据和替代路径均列于[JSON清单的directory_classification](historical_experiments_20261009.json)。本页未移动、删除或改写任何运行产物。
