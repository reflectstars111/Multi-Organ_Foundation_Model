# Multi-Organ Foundation Model

多器官超声图像分割研究代码。包含共享 U-Net、九个独立 U-Net、task-ID MoE，以及无 task-ID 的区域特征路由 MoT 实验。模型和训练入口见 [unet_moe/README.md](unet_moe/README.md) 与 [unet_moe/README_REGION.md](unet_moe/README_REGION.md)。

本仓库仅保存代码、测试和实验文档。数据集、处理后的图像、训练权重与日志不在版本控制中；运行时需自行准备 `数据集 (Datasets)/pointed_data/*/processed_png/`，其中每个数据集提供 `images/`、`masks/`、`manifest.csv` 和 `report.json`。实验划分见 [EXPERIMENT_V6_PROTOCOL.md](EXPERIMENT_V6_PROTOCOL.md)。

```powershell
pip install -r requirements-unet-moe.txt
python -m pytest tests -q
```

数据预处理依赖另见 [requirements-preprocessing.txt](requirements-preprocessing.txt)。
