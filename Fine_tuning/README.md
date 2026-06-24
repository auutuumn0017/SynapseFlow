# Fine-Tuning Module

这个目录包含针对 DeepSeek 1.5B 模型在 RTX 3050 上的后训练代码（基于 Unsloth 的 QLoRA 和 ORPO）。

## 目录结构规划
- `data/`: 存放用于 SFT 和 ORPO 训练的 JSON 数据集。
- `scripts/`: 存放数据预处理、格式转换脚本。
- `train_sft.py`: Supervised Fine-Tuning (SFT) 训练脚本。
- `train_orpo.py`: Odds Ratio Preference Optimization (ORPO) 训练脚本。
- `merge_adapter.py`: 将训练好的 LoRA 权重与基础模型合并或用于 RAG 挂载的脚本。
