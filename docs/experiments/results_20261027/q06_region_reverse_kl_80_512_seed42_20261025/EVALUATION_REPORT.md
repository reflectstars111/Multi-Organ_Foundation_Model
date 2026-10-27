# Q06 机制实验：配对评价

冻结协议：`fb5713e8bdaf3afdd245f779af4345d094ac70c4784e5d38fc5c3c2c2e39aade`。固定 seed42；单种子标准差不可估计，报告 null。

Q06 比较最后层阳性查询路由正则；Q07 比较 0/1/9 个专家残差。公共参数初始化、图像抽样和 C 主损失保持配对；Q07 仅 9 专家组保留原负载均衡项。

结构表按16个结构等权平均；并集表先对每数据域图像平均，再对9个域等权平均。二者不是同一个任务或同一个分母。

DSC (%) 越大越好，HD95 (pixel)、ASSD (pixel) 越小越好。单侧空掩膜使用像素中心对角线 hypot(511,511) ≈ 722.66 pixel；双方为空为100/0/0。

`positive` 只按真值是否有前景划分。`*_surface_defined` 仅排除真值边界完全不可观测的样本，保留预测为空或失效的惩罚。完整队列遇不可定义距离时报告NA。

`best` 为验证集结构阳性DSC最佳点，`last` 为第80轮，`guardrail` 为同时满足器官保护条件的最佳点。无合格点时不拿其他检查点替代，配对表明确报告缺失种子。

| 模型 | 组别 | 选点 | 视图 | 集合 | 入选种子 | DSC (%) | HD95 (pixel) | ASSD (pixel) |
|---|---|---|---|---|---:|---:|---:|---:|
| region | control | best | structure | positive_surface_defined | 1 | 73.85 | 100.52 | 71.36 |
| region | control | best | structure | all_surface_defined | 1 | 79.13 | 82.36 | 60.06 |
| region | control | best | union | positive_surface_defined | 1 | 86.20 | 40.73 | 13.28 |
| region | control | best | union | all_surface_defined | 1 | 86.15 | 42.70 | 15.89 |
| region | control | last | structure | positive_surface_defined | 1 | 73.31 | 108.09 | 80.07 |
| region | control | last | structure | all_surface_defined | 1 | 79.27 | 81.64 | 59.45 |
| region | control | last | union | positive_surface_defined | 1 | 86.10 | 42.26 | 14.53 |
| region | control | last | union | all_surface_defined | 1 | 85.85 | 45.63 | 18.55 |
| region | control | guardrail | structure | positive_surface_defined | 1 | 73.85 | 100.52 | 71.36 |
| region | control | guardrail | structure | all_surface_defined | 1 | 79.13 | 82.36 | 60.06 |
| region | control | guardrail | union | positive_surface_defined | 1 | 86.20 | 40.73 | 13.28 |
| region | control | guardrail | union | all_surface_defined | 1 | 86.15 | 42.70 | 15.89 |
| region | routing | best | structure | positive_surface_defined | 1 | 73.89 | 104.12 | 76.30 |
| region | routing | best | structure | all_surface_defined | 1 | 78.89 | 86.12 | 64.41 |
| region | routing | best | union | positive_surface_defined | 1 | 85.89 | 42.68 | 15.32 |
| region | routing | best | union | all_surface_defined | 1 | 85.56 | 46.68 | 19.95 |
| region | routing | last | structure | positive_surface_defined | 1 | 74.11 | 103.44 | 75.90 |
| region | routing | last | structure | all_surface_defined | 1 | 80.00 | 78.38 | 57.43 |
| region | routing | last | union | positive_surface_defined | 1 | 86.36 | 39.84 | 13.05 |
| region | routing | last | union | all_surface_defined | 1 | 86.21 | 42.55 | 16.43 |
| region | routing | guardrail | structure | positive_surface_defined | 1 | 73.27 | 81.29 | 46.48 |
| region | routing | guardrail | structure | all_surface_defined | 1 | 75.20 | 99.56 | 73.62 |
| region | routing | guardrail | union | positive_surface_defined | 1 | 84.94 | 45.40 | 13.83 |
| region | routing | guardrail | union | all_surface_defined | 1 | 83.95 | 54.13 | 23.43 |

各组配对差值方向由 comparison 字段明确标记，见 [evaluation_paired.csv](evaluation_paired.csv)。腹部血管、阳性漏检、空标签误检、CT来源及缺失选点在 [evaluation_summary.json](evaluation_summary.json) 中分别保留；逐结构/域/来源的三指标及覆盖数量见 [evaluation_per_group.csv](evaluation_per_group.csv)。测试划分已在历史研究中查看，不应称为未见外部验证。
