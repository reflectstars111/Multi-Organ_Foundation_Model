# 医学外部 baseline 资源

2026-10-12 已确认：**MOFO、UUSIC 2025 总体前三名 SMART／AImageLab-zip／Lanzhou、UniverSeg**，共5种方法，作为本项目外部对照。

主协议：现有数据划分，512×512，80轮，seed 42/43/44，各方法保留原结构与条件机制，重新训练；统一评价已标注器官前景并集。新增15项主训练，方法特有的准备阶段另外记录。

- [Q04完整接入与实验登记](../docs/Q04_EXTERNAL_BASELINES_512.md)
- [机器可读方法、源码哈希及15项清单](baseline_registry.json)
- [MOFO源码审查](../docs/baselines/MOFO_SOURCE_AUDIT.md)
- [SMART源码审查](../docs/baselines/SMART_SOURCE_AUDIT.md)
- [AImageLab与Lanzhou源码审查](../docs/baselines/AIMAGELAB_LANZHOU_SOURCE_AUDIT.md)
- [UniverSeg本地官方说明](UniverSeg/README.md)

**2026-10-13：五方法适配、完整训练／恢复／评价及独立队列已实现；15项主训练等待前置任务及GPU准入，MOFO另有3项80轮先验。** 见[执行与监测说明](../docs/Q04_EXECUTION_20261013.md)。原始第三方文件保留；适配实现在 `unet_moe/external_baselines/`。大权重、Docker、样例数据和嵌套Git仓库不随父仓库登记文件提交。
