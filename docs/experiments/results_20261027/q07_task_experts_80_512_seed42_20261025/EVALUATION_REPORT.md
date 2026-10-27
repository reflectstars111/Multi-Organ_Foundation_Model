# Q07 机制实验：配对评价

冻结协议：`8422facedcbb0c7761822d9e6c69b7efda1f71cf88e9236fc5eb1f95012f2d34`。固定 seed42；单种子标准差不可估计，报告 null。

Q06 比较最后层阳性查询路由正则；Q07 比较 0/1/9 个专家残差。公共参数初始化、图像抽样和 C 主损失保持配对；Q07 仅 9 专家组保留原负载均衡项。

结构表按16个结构等权平均；并集表先对每数据域图像平均，再对9个域等权平均。二者不是同一个任务或同一个分母。

DSC (%) 越大越好，HD95 (pixel)、ASSD (pixel) 越小越好。单侧空掩膜使用像素中心对角线 hypot(511,511) ≈ 722.66 pixel；双方为空为100/0/0。

`positive` 只按真值是否有前景划分。`*_surface_defined` 仅排除真值边界完全不可观测的样本，保留预测为空或失效的惩罚。完整队列遇不可定义距离时报告NA。

`best` 为验证集结构阳性DSC最佳点，`last` 为第80轮，`guardrail` 为同时满足器官保护条件的最佳点。无合格点时不拿其他检查点替代，配对表明确报告缺失种子。

| 模型 | 组别 | 选点 | 视图 | 集合 | 入选种子 | DSC (%) | HD95 (pixel) | ASSD (pixel) |
|---|---|---|---|---|---:|---:|---:|---:|
| task_id | experts0 | best | structure | positive_surface_defined | 1 | 73.25 | 106.55 | 75.23 |
| task_id | experts0 | best | structure | all_surface_defined | 1 | 78.07 | 91.69 | 68.59 |
| task_id | experts0 | best | union | positive_surface_defined | 1 | 85.49 | 47.09 | 19.62 |
| task_id | experts0 | best | union | all_surface_defined | 1 | 85.70 | 46.80 | 19.87 |
| task_id | experts0 | last | structure | positive_surface_defined | 1 | 72.63 | 113.26 | 86.57 |
| task_id | experts0 | last | structure | all_surface_defined | 1 | 78.67 | 88.13 | 67.27 |
| task_id | experts0 | last | union | positive_surface_defined | 1 | 85.55 | 44.85 | 16.53 |
| task_id | experts0 | last | union | all_surface_defined | 1 | 84.89 | 51.09 | 23.49 |
| task_id | experts0 | guardrail | structure | positive_surface_defined | 1 | 72.00 | 107.11 | 77.15 |
| task_id | experts0 | guardrail | structure | all_surface_defined | 1 | 77.14 | 91.47 | 67.21 |
| task_id | experts0 | guardrail | union | positive_surface_defined | 1 | 84.72 | 47.15 | 15.53 |
| task_id | experts0 | guardrail | union | all_surface_defined | 1 | 83.51 | 57.57 | 26.69 |
| task_id | experts1 | best | structure | positive_surface_defined | 1 | 72.71 | 108.69 | 80.56 |
| task_id | experts1 | best | structure | all_surface_defined | 1 | 78.14 | 90.99 | 69.88 |
| task_id | experts1 | best | union | positive_surface_defined | 1 | 84.71 | 51.21 | 24.32 |
| task_id | experts1 | best | union | all_surface_defined | 1 | 85.15 | 49.38 | 23.08 |
| task_id | experts1 | last | structure | positive_surface_defined | 1 | 72.18 | 124.69 | 96.19 |
| task_id | experts1 | last | structure | all_surface_defined | 1 | 78.90 | 91.12 | 69.70 |
| task_id | experts1 | last | union | positive_surface_defined | 1 | 84.93 | 48.72 | 20.69 |
| task_id | experts1 | last | union | all_surface_defined | 1 | 84.94 | 50.02 | 22.67 |
| task_id | experts1 | guardrail | structure | positive_surface_defined | 1 | 73.97 | 97.76 | 67.98 |
| task_id | experts1 | guardrail | structure | all_surface_defined | 1 | 78.33 | 86.94 | 64.67 |
| task_id | experts1 | guardrail | union | positive_surface_defined | 1 | 85.32 | 42.82 | 14.20 |
| task_id | experts1 | guardrail | union | all_surface_defined | 1 | 84.47 | 50.74 | 22.89 |
| task_id | experts9 | best | structure | positive_surface_defined | 1 | 73.41 | 112.76 | 87.30 |
| task_id | experts9 | best | structure | all_surface_defined | 1 | 79.17 | 87.36 | 66.66 |
| task_id | experts9 | best | union | positive_surface_defined | 1 | 85.89 | 43.21 | 13.88 |
| task_id | experts9 | best | union | all_surface_defined | 1 | 84.65 | 53.68 | 25.09 |
| task_id | experts9 | last | structure | positive_surface_defined | 1 | 71.95 | 125.21 | 95.97 |
| task_id | experts9 | last | structure | all_surface_defined | 1 | 78.78 | 89.57 | 66.04 |
| task_id | experts9 | last | union | positive_surface_defined | 1 | 84.70 | 49.89 | 19.19 |
| task_id | experts9 | last | union | all_surface_defined | 1 | 84.78 | 50.62 | 20.62 |
| task_id | experts9 | guardrail | structure | positive_surface_defined | 1 | 74.11 | 99.26 | 70.15 |
| task_id | experts9 | guardrail | structure | all_surface_defined | 1 | 78.24 | 88.21 | 65.14 |
| task_id | experts9 | guardrail | union | positive_surface_defined | 1 | 84.91 | 44.48 | 13.87 |
| task_id | experts9 | guardrail | union | all_surface_defined | 1 | 83.71 | 54.94 | 25.06 |

各组配对差值方向由 comparison 字段明确标记，见 [evaluation_paired.csv](evaluation_paired.csv)。腹部血管、阳性漏检、空标签误检、CT来源及缺失选点在 [evaluation_summary.json](evaluation_summary.json) 中分别保留；逐结构/域/来源的三指标及覆盖数量见 [evaluation_per_group.csv](evaluation_per_group.csv)。测试划分已在历史研究中查看，不应称为未见外部验证。
