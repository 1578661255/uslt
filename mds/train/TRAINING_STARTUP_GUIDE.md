# Stage 3 多模态融合 - 训练启动指南

## 📋 系统现状概览

### ✅ 已完成
1. **新模块开发** (100%)
   - ✓ `temporal_alignment.py` - 描述加载与时间对齐
   - ✓ `text_fusion_modules.py` - 文本编码与融合

2. **现有代码集成** (100%)
   - ✓ `models.py` - 多模态融合集成
   - ✓ `datasets.py` - 描述加载支持
   - ✓ `config.py` - 路径配置化
   - ✓ `utils.py` - CLI 参数扩展
   - ✓ `fine_tuning.py` - 训练循环支持

3. **验证** (100%)
   - ✓ 所有合法性检查通过
   - ✓ Python 语法检查通过
   - ✓ 配置集成验证通过
   - ✓ 数据流兼容性通过

---

## 🚀 快速启动

### 方式 1：基础训练（不使用描述文本）

```bash
cd d:\home\pc\code\slt\Uni-Sign

python fine_tuning.py \
    --dataset CSL_Daily \
    --epochs 20 \
    --batch-size 16 \
    --output_dir ./output_baseline
```

**说明**：与 Stage 2 完全相同，不使用多模态融合

---

### 方式 2：启用多模态融合训练

```bash
python fine_tuning.py \
    --dataset CSL_Daily \
    --epochs 20 \
    --batch-size 16 \
    --use_descriptions \
    --text_dropout_p 0.1 \
    --output_dir ./output_multimodal
```

**关键参数**：
- `--use_descriptions`：启用文本描述多模态融合
- `--text_dropout_p 0.1`：文本 dropout 概率（防止过拟合）

---

### 方式 3：冻结文本编码器（仅训练融合模块）

```bash
python fine_tuning.py \
    --dataset CSL_Daily \
    --epochs 20 \
    --batch-size 16 \
    --use_descriptions \
    --text_encoder_freeze \
    --output_dir ./output_fusion_only
```

**说明**：
- mT5 编码器冻结，仅调整 GatingFusion 和 LearnableMaskEmbedding
- 推荐用于显存受限的情况
- 训练速度快，参数量最小

---

### 方式 4：完整微调（带融合检查点）

```bash
python fine_tuning.py \
    --dataset CSL_Daily \
    --epochs 20 \
    --batch-size 16 \
    --use_descriptions \
    --text_dropout_p 0.1 \
    --fusion_checkpoint ./pretrained_fusion.pth \
    --output_dir ./output_finetune
```

**说明**：从预训练的融合模块检查点开始

---

## 📊 数据流验证

系统完整的数据处理流程：

```
原始数据
  ↓
─── 姿态分支 ─────────────────────────── 多部位 GCN
  ↓
文本描述 → DescriptionLoader → 原始描述字典
  ↓
     TemporalAligner → 对齐描述列表 + 缺失指示符
  ↓
S2T_Dataset.__getitem__() → 7元组
  ├─ name_sample, pose_sample, text, gloss, rgb_dict
  ├─ descriptions (新)
  └─ has_description (新)
  ↓
collate_fn() → src_input 字典
  ├─ 原有字段保持不变
  ├─ descriptions (List[List[str or None]])
  └─ has_description (List[List[int]])
  ↓
Uni_Sign.forward()
  ├─ 姿态特征提取 → inputs_embeds (B, T, 768)
  ├─ IF descriptions exist:
  │  ├─ TextEncoder → text_features (B, T, 768)
  │  ├─ _apply_text_dropout → text_features (训练时)
  │  └─ GatingFusion → fused_embeds = pose + gate × text
  │
  └─ MT5 编码器-解码器 → 翻译输出
```

---

## ⚙️ 配置检查表

运行以下命令验证所有配置是否正确：

```bash
python pre_training_checklist.py
```

预期输出：
```
✓ utils.py 参数
✓ fine_tuning.py 集成
✓ models.py 集成
✓ datasets.py 集成
✓ config.py 配置

✓ 所有检查通过！系统准备就绪
```

---

## 🔧 高级选项

### 调整文本 Dropout 强度

```bash
# 较强的正则化（推荐用于防止过拟合）
--text_dropout_p 0.2

# 较弱的正则化（推荐用于数据充足）
--text_dropout_p 0.05

# 无 Dropout（仅调试用）
--text_dropout_p 0.0
```

### 显存优化

如果遇到 OOM（内存不足）错误：

```bash
# 方案 1：减小批大小
--batch-size 8

# 方案 2：冻结文本编码器（显存减少 ~2GB）
--text_encoder_freeze

# 方案 3：启用梯度累积
--gradient-accumulation-steps 16  # 等效于 batch_size=512

# 方案 4：启用 ZeRO 优化
--zero_stage 2 --offload
```

### 推理模式

评估已训练模型的性能：

```bash
python fine_tuning.py \
    --dataset CSL_Daily \
    --use_descriptions \
    --finetune ./output_multimodal/best_checkpoint.pth \
    --eval
```

---

## 🐛 常见问题排查

### Q1：运行报错 "No module named 'temporal_alignment'"
**原因**：未在正确目录  
**解决**：确保在 `Uni-Sign` 目录下运行
```bash
cd d:\home\pc\code\slt\Uni-Sign
python fine_tuning.py ...
```

### Q2：报错 "descriptions not found in src_input"
**原因**：--use_descriptions 但描述文件缺失  
**解决**：检查 `./description/CSL-Daily/split_data/` 是否存在
```bash
# 检查文件是否存在
ls -la ./description/CSL-Daily/split_data/train/ | head -10
```

### Q3：OOM (GPU 显存溢出)
**原因**：TextEncoder 占用较多显存  
**解决方案**（按优先级）：
1. 使用 `--text_encoder_freeze` 降低优化器状态大小
2. 减小 `--batch-size`
3. 增加 `--gradient-accumulation-steps`
4. 启用 `--zero_stage 2 --offload`

### Q4：训练速度明显变慢
**原因**：文本编码开销  
**优化方案**：
- 使用 `--text_encoder_freeze` 跳过梯度计算
- 增加 `--num_workers` 加速数据加载
- 使用 `--pin-mem` 锁定内存

---

## 📈 训练效果预期

### 基础模型（Stage 2）
```
BLEU-4 on test: ~35.5
```

### 多模态融合（Stage 3，使用描述）
```
BLEU-4 on test: ~37-39
```

**预期提升**：+2-4 BLEU 视描述质量而定

---

## 📝 核心参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_descriptions` | False | 启用多模态文本融合 |
| `--text_dropout_p` | 0.1 | 文本 dropout 概率 |
| `--text_encoder_freeze` | False | 冻结 TextEncoder 参数 |
| `--fusion_checkpoint` | "" | 融合模块检查点路径 |
| `--dataset` | CSL_Daily | 数据集选择 |
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 16 | 批大小 |
| `--lr` | 1e-3 | 学习率 |

---

## 📦 关键文件清单

### 新增文件
- `temporal_alignment.py` (292 行) - 描述加载和对齐
- `text_fusion_modules.py` (442 行) - 文本编码和融合
- `test_description_loading.py` - 描述加载验证
- `test_models_integration.py` - 模型集成验证
- `pre_training_checklist.py` - 训练前检查

### 修改文件
- `models.py` - 多模态融合集成 (+180 行)
- `datasets.py` - 描述加载支持 (+150 行)
- `config.py` - 路径配置化 (+3 行)
- `utils.py` - CLI 参数扩展 (+13 行)
- `fine_tuning.py` - 训练循环支持 (+15 行)

### 数据文件
- `description/CSL-Daily/split_data/train/` - 162 个样本
- `description/CSL-Daily/split_data/dev/` - 样本
- `description/CSL-Daily/split_data/test/` - 样本

---

## 🎯 建议的训练计划

### 第一阶段：基础验证（1 天）
```bash
# 1. 确保基础模型能正常运行
python fine_tuning.py --dataset CSL_Daily --epochs 1 --batch-size 16 \
    --output_dir ./test_baseline

# 2. 启用描述文本进行验证
python fine_tuning.py --dataset CSL_Daily --epochs 1 --batch-size 16 \
    --use_descriptions --output_dir ./test_multimodal
```

### 第二阶段：融合模块训练（3-5 天）
```bash
# 冻结文本编码器，仅训练融合模块（速度快）
python fine_tuning.py --dataset CSL_Daily --epochs 10 --batch-size 16 \
    --use_descriptions --text_encoder_freeze \
    --output_dir ./fusion_training
```

### 第三阶段：完整微调（5-10 天）
```bash
# 完整训练，包括文本编码器微调
python fine_tuning.py --dataset CSL_Daily --epochs 20 --batch-size 16 \
    --use_descriptions --text_dropout_p 0.1 \
    --finetune ./fusion_training/best_checkpoint.pth \
    --output_dir ./full_training
```

---

## ✅ 最后检查清单

在启动训练前，确保：

- [ ] 运行 `pre_training_checklist.py` 全部通过
- [ ] 数据文件夹存在：`./description/CSL-Daily/split_data/`
- [ ] 配置文件包含 `description_dirs` 设置
- [ ] GPU 显存充足（建议 ≥ 24GB for 完整训练）
- [ ] 选择合适的启动方式（基础/多模态/冻结等）
- [ ] 指定 `--output_dir` 保存结果

---

**更新时间**：训练前完整集成完成  
**系统状态**：✅ 准备就绪  
**下一步**：选择上述启动方式之一，开始训练
