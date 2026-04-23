# CaptionPS - 电力安全场景多模态大模型微调与评估

## 📋 项目简介

本项目基于 **LLaMA-Factory** 框架，针对电力安全场景进行多模态大语言模型（MLLM）的微调与评估。项目支持多种主流视觉语言模型的 LoRA 微调、推理和全方位性能评估，涵盖图像描述生成、安全状态判断、危险区域定位等任务。

### 核心功能
- ✅ 支持 6+ 种主流多模态大模型的 LoRA 微调
- ✅ 批量推理与结果生成
- ✅ 多维度评估指标（BLEU、ROUGE、BERTScore、IoU/GIoU/DIoU/CIoU、分类准确率等）
- ✅ 电力安全场景专用数据集处理

---

## 🏗️ 项目结构

```
CaptionPS/
├── train_lora/              # LoRA 微调配置文件
│   ├── InternVL3_5.yaml     # InternVL3.5 系列配置
│   ├── Qwen2.5.yaml         # Qwen2.5-VL 配置
│   ├── Qwen3-VL.yaml        # Qwen3-VL 配置
│   ├── Llama-3.2.yaml       # Llama-3.2 Vision 配置
│   ├── llava-1.5.yaml       # LLaVA-1.5 配置
│   └── gemma3-pt.yaml       # Gemma-3 配置
│
├── Interference/            # 模型推理脚本
│   ├── eval_qwen3-vl-2B.py          # Qwen3-VL-2B 基础模型推理
│   ├── eval_qwen3-vl-2B-lora.py     # Qwen3-VL-2B LoRA 微调后推理
│   ├── eval_qwen2.5-vl-3B.py        # Qwen2.5-VL-3B 基础模型推理
│   ├── eval_qwen2.5-vl-3B-lora.py   # Qwen2.5-VL-3B LoRA 微调后推理
│   ├── eval_InternVL3_5-2B*.py      # InternVL3.5-2B 推理（基础/LoRA）
│   ├── eval_InternVL3_5-8B*.py      # InternVL3.5-8B 推理（基础/LoRA）
│   ├── eval_InternVL3_5-14B*.py     # InternVL3.5-14B 推理（基础/LoRA）
│   ├── eval_llama3.2_11b*.py        # Llama-3.2-11B 推理（基础/LoRA）
│   ├── eval_llava-1.5-7b*.py        # LLaVA-1.5-7B 推理（基础/LoRA）
│   ├── eval_llava-1.5-13b*.py       # LLaVA-1.5-13B 推理（基础/LoRA）
│   └── eval_gemma-3-12b-pt*.py      # Gemma-3-12B 推理（基础/LoRA）
│
├── eval_metric/             # 评估指标计算脚本
│   ├── BLEU-4_ROUGE-L_token-acc.py  # BLEU-4、ROUGE-L、Token 准确率
│   ├── BERTScore.py                 # BERTScore 语义相似度评估
│   ├── IOU.py                       # IoU/GIoU/DIoU/CIoU 边界框重叠度 + 分类准确率
│   └── predrict_classes.py          # 安全状态与场景类别预测准确率
│
├── Dataset_files/           # 数据集文件
│   ├── train-11913.json     # 训练集（11,913 条样本）
│   └── test-2350.jsonl      # 测试集（2,350 条样本）
│
└── README.md                # 项目说明文档
```

---

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- LLaMA-Factory（用于微调训练）
- Transformers >= 4.37
- PEFT（LoRA 微调支持）
- CUDA >= 11.8（推荐）

### 安装依赖

```bash
# 克隆 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 安装额外依赖
pip install bert_score rouge_score sacrebleu Pillow tqdm
```

---

## 📊 数据集说明

### 数据格式

#### 训练集 (`train-11913.json`)
```json
{
  "image_path": "/path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n请描述这张图片中的安全状况。"
    },
    {
      "from": "gpt",
      "value": "[unsafe] <crossing the fence> 工人正在穿越围栏..."
    }
  ],
  "coordinates": "[[[x1,y1],[x2,y2]]]"  // 危险区域边界框（可选）
}
```

#### 测试集 (`test-2350.jsonl`)
```json
{
  "image_path": "/path/to/image.jpg",
  "question": "Which of the following...",
  "answer": "[safe] <welding or cutting operations> ...",
  "coordinates": "NULL"  // 或边界框坐标
}
```

### 安全场景类别

项目涵盖 6 类电力安全场景：
1. **welding or cutting operations** - 焊接或切割作业
2. **work at heights** - 高空作业
3. **staying under the crane** - 停留在起重机下方
4. **crossing the fence** - 穿越围栏
5. **electrical inspection pole operation** - 电杆检修作业
6. **others** - 其他场景

---

## 🔧 模型微调（LoRA）

### 支持的模型

| 模型 | 参数量 | 配置文件 | 模板名称 |
|------|--------|----------|----------|
| Qwen3-VL | 2B | `Qwen3-VL.yaml` | `qwen3_vl` |
| Qwen2.5-VL | 3B | `Qwen2.5.yaml` | `qwen2_vl` |
| InternVL3.5 | 2B/8B/14B | `InternVL3_5.yaml` | `intern_vl` |
| Llama-3.2-Vision | 11B | `Llama-3.2.yaml` | `mllama` |
| LLaVA-1.5 | 7B/13B | `llava-1.5.yaml` | `llava` |
| Gemma-3-PT | 12B | `gemma3-pt.yaml` | `gemma3` |

### 微调步骤

1. **准备数据集**
   ```bash
   # 将数据集转换为 LLaMA-Factory 格式
   # 在 LLaMA-Factory/data/dataset_info.json 中添加数据集配置
   ```

2. **修改配置文件**
   
   以 `train_lora/Qwen3-VL.yaml` 为例：
   ```yaml
   ### model
   model_name_or_path: /root/autodl-tmp/Qwen3-VL-2B-Instruct
   image_max_pixels: 4000000
   
   ### method
   stage: sft
   do_train: true
   finetuning_type: lora
   lora_rank: 8
   lora_target: all
   
   ### dataset
   dataset: 12346-6000  # 数据集名称（需在 dataset_info.json 中定义）
   template: qwen3_vl
   cutoff_len: 4096
   max_samples: 10000
   
   ### train
   per_device_train_batch_size: 2
   gradient_accumulation_steps: 8
   learning_rate: 1.0e-4
   num_train_epochs: 5.0
   
   ### output
   output_dir: saves/Qwen3-VL-2B-Instruct-12346-6000/lora/sft
   ```

3. **启动训练**
   ```bash
   cd LLaMA-Factory
   
   # 单卡训练
   llamafactory-cli train ../CaptionPS/train_lora/Qwen3-VL.yaml
   
   # 多卡分布式训练（使用 DeepSpeed）
   deepspeed --num_gpus=4 src/train_bash.py \
     --deepspeed examples/deepspeed/ds_z3_config.json \
     ../CaptionPS/train_lora/Qwen3-VL.yaml
   ```

### 关键参数说明

- **lora_rank**: LoRA 秩，推荐 4-16（Gemma-3 使用 4，其他模型使用 8）
- **learning_rate**: 学习率，推荐 1e-4 到 5e-5
- **cutoff_len**: 最大序列长度，根据模型调整（Qwen 系列可用 4096，其他建议 1024-2048）
- **bf16**: 是否启用 BF16 精度（Gemma-3 启用，其他可选）
- **gradient_checkpointing**: 显存优化，大模型建议开启

---

## 🧪 模型推理

### 推理脚本说明

每个模型对应两个推理脚本：
- `eval_<model>.py`: 基础模型推理
- `eval_<model>-lora.py`: LoRA 微调后模型推理

### 运行推理

```bash
cd Interference

# 示例：Qwen3-VL-2B LoRA 微调后推理
python eval_qwen3-vl-2B-lora.py

# 修改脚本中的关键路径
# - model_path: 基础模型路径
# - PeftModel.from_pretrained(): LoRA 权重路径
# - load_data(): 测试集路径
```

### 输出格式

推理结果保存为 JSONL 文件，每行包含：
```json
{
  "image_path": "/path/to/image.jpg",
  "output": "[unsafe] <crossing the fence> 工人正在穿越围栏...",
  "answer": "[unsafe] <crossing the fence> ...",
  "coordinates": "[[[x1,y1],[x2,y2]]]"
}
```

---

## 📈 模型评估

### 1. 文本生成质量评估

#### BLEU-4 / ROUGE-L / Token Accuracy
```bash
cd eval_metric
python BLEU-4_ROUGE-L_token-acc.py
```

**输出指标：**
- **BLEU-4**: n-gram 精确度（侧重流畅性）
- **ROUGE-L**: 最长公共子序列召回率（侧重内容覆盖）
- **Token-Accuracy**: Token 级别匹配率

#### BERTScore
```bash
python BERTScore.py
```

**特点：**
- 基于 DeBERTa-XLarge 的上下文嵌入相似度
- 支持分批处理（默认 batch_size=20）
- 输出 Precision、Recall、F1 三个维度

**配置修改：**
```python
jsonl_file_path = "/path/to/predictions.jsonl"
model_type = "microsoft/deberta-xlarge-mnli"
batch_size = 20  # 根据显存调整
```

### 2. 空间定位能力评估（IoU 系列）

```bash
python IOU.py
```

**评估指标：**
- **IoU (Intersection over Union)**: 标准交并比
- **GIoU (Generalized IoU)**: 考虑包围框的广义 IoU
- **DIoU (Distance IoU)**: 考虑中心点距离
- **CIoU (Complete IoU)**: 综合考虑重叠、距离、长宽比

**同时输出：**
- 分类准确率（安全状态判断）
- Precision / Recall / F1（类别预测）

**适用场景：**
- 需要边界框坐标的任务（如危险区域定位）
- 当 `coordinates` 字段为 `"NULL"` 时自动跳过 IoU 计算

### 3. 分类任务评估

```bash
python predrict_classes.py
```

**评估内容：**
- **安全状态分类**: `[safe]` vs `[unsafe]`
- **场景类别分类**: 6 类电力安全场景

**匹配策略：**
- 精确匹配：完全一致的类别标签
- 模糊匹配：关键词重叠检测
- 同义词映射：支持近义词归一化

**输出示例：**
```
total: 2350
安全状态预测正确数: 2100
安全状态预测正确率: 0.8936
场景类别预测正确数: 1950
场景类别预测正确率: 0.8298
```

---

## 📊 评估流程示例

### 完整评估流水线

```bash
# 1. 推理生成预测结果
cd Interference
python eval_qwen3-vl-2B-lora.py  # 生成 predictions.jsonl

# 2. 文本质量评估
cd ../eval_metric
python BLEU-4_ROUGE-L_token-acc.py  # 修改文件路径
python BERTScore.py                  # 修改文件路径

# 3. 空间定位评估（如有坐标标注）
python IOU.py                        # 修改文件路径

# 4. 分类任务评估
python predrict_classes.py           # 修改文件路径
```

---

## ⚙️ 配置文件详解

### 训练配置对比

| 参数 | Qwen 系列 | Llama/LLaVA | Gemma-3 |
|------|-----------|-------------|---------|
| cutoff_len | 4096 | 1024-2048 | 1024 |
| lora_rank | 8 | 8 | 4 |
| batch_size | 2 | 2 | 1 |
| bf16 | false | false | true |
| gradient_checkpointing | - | - | true |

### 为什么不同模型配置不同？

- **Qwen 系列**: 支持更长上下文，可使用更大 `cutoff_len`
- **Gemma-3**: 显存占用高，需降低 batch_size 并启用梯度检查点
- **Llama-3.2**: 需要 `trust_remote_code=true`

---

## 💡 最佳实践

### 1. 显存优化

```yaml
# 对于显存受限的场景（< 24GB）
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
bf16: true
```

### 2. 提升训练稳定性

```yaml
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0  # 梯度裁剪
```

### 3. 多卡训练加速

```bash
# 使用 DeepSpeed ZeRO-3
deepspeed --num_gpus=8 src/train_bash.py \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  config.yaml
```

### 4. 评估批处理优化

```python
# BERTScore.py 中调整 batch_size
batch_size = 20  # 32GB 显存推荐值
# 显存不足时降至 10 或 5
```

---

## 🐛 常见问题

### Q1: 训练时显存溢出（OOM）
**解决方案：**
- 降低 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing: true`
- 使用 `bf16: true`（如果硬件支持）
- 减小 `cutoff_len`

### Q2: LoRA 权重加载失败
**解决方案：**
```python
# 确保路径正确
model = PeftModel.from_pretrained(
    model, 
    "/path/to/saves/model-name/lora/sft"  # 检查此目录是否存在 adapter_model.bin
)
```

### Q3: 推理速度慢
**解决方案：**
- 使用 `torch.bfloat16` 或 `torch.float16`
- 启用 Flash Attention 2（如果支持）
- 减少 `image_max_pixels`

### Q4: IoU 评估结果为 0
**可能原因：**
- 预测结果中未提取到有效坐标
- 坐标格式不匹配
- `coordinates` 字段为 `"NULL"`

**调试方法：**
查看 `IOU.py` 输出的 `extraction_failures` 列表

---

## 📝 引用

如果使用本项目的代码或数据，请引用：

```bibtex
@misc{captionps2024,
  title={CaptionPS: Multi-modal Large Language Model Fine-tuning for Power Safety Scenarios},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/CaptionPS}}
}
```

---

## 📄 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系。
