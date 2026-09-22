# EC-LoRA: Energy-Driven Continual LoRA Implicit Generation

[English](README.md) | 中文

## 目录

```text
EC_LoRA/
├── checkpoints/  # 各任务的 LoRA checkpoint
├── requirements.txt
└── src/
    ├── common/
    ├── vit_b32/
    ├── deepseek_7b/
    ├── qwen3_8b/
    ├── qwen3_vl_8b/
    └── llava_v1_5_7b/
        ├── model/   # 能量模型与参数处理
        ├── config/  # 训练和评测配置
        ├── eval/    # 评测代码
        └── tools/   # 训练工具
```

五个模型目录采用相同结构。

## 安装

创建环境并安装依赖；GPU 环境请选择匹配的 PyTorch/CUDA 版本。

```bash
conda create -n ec_lora python=3.10
conda activate ec_lora
pip install -r requirements.txt
pip install -r src/vit_b32/requirements.txt
```

## 主干模型

| 模型 | 链接 |
| --- | --- |
| ViT-B/32 | [OpenAI CLIP](https://github.com/openai/CLIP) |
| DeepSeek-7B | [DeepSeek LLM 7B Base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base) |
| Qwen3-8B | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Qwen3-VL-8B | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| LLaVA-v1.5-7B | [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) |

## 配置

```text
src/vit_b32/config/8tasks/config_meta_ebm_8task.yaml
src/deepseek_7b/config/meta_ebm_deepseek.yaml
src/qwen3_8b/config/meta_ebm_qwen.yaml
src/qwen3_vl_8b/config/meta_ebm_qwenvl.yaml
src/llava_v1_5_7b/config/meta_ebm_llava.yaml
```

## LoRA Checkpoint

将各任务的 LoRA checkpoint 放在 `checkpoints/` 下：

```text
checkpoints/
├── vit_b32/normalized_data_<task>/
├── deepseek_7b/output_<task>/
├── qwen3_8b/output_<task>_qwen3_8b/
├── qwen3_vl_8b/output_<task>/
└── llava_v1_5_7b/output_<task>/
```

语言及多模态任务目录需包含 `top_checkpoints.json`。每个 adapter 的
`adapter_config.json` 和 `adapter_model.safetensors`（或 `.bin`）可直接放在
任务目录，也可放在 `checkpoint-*/` 子目录。
ViT 目录需包含 `normalized_<task>_*.pth` 和验证集排序清单
`top_checkpoints_<task>.json`；其 source adapter 从
`<project_root>/ICM-LoRA-ViT/checkpoints/output_<task>/best_<task>_lora_vit.pt`
读取。

## 训练与评测（ViT-B/32）

设置 `src/vit_b32/config/8tasks/config_meta_ebm_8task.yaml` 中的路径后，
在仓库根目录运行：

```bash
python src/vit_b32/train_meta_ebm.py --config src/vit_b32/config/8tasks/config_meta_ebm_8task.yaml
```

ViT 每个训练阶段结束后自动评测。其他主干模型使用
`src/<backbone>/config/` 下各自的 YAML 配置。

## 参考仓库

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [LoRA](https://github.com/microsoft/LoRA)
- [DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM)
- [Qwen3](https://github.com/QwenLM/Qwen3)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [RobustMerge / MM-MergeBench](https://github.com/AuroraZengfh/RobustMerge)
- [RobustMerge 固定版本](https://github.com/AuroraZengfh/RobustMerge/tree/12128022190516cc93a549cc4931112ce4bdcda3)
- [LLaVA-v1.5 论文](https://arxiv.org/abs/2310.03744)
- [RobustMerge 论文](https://arxiv.org/abs/2502.17159)
