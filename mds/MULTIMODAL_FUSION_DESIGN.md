# Uni-Sign Stage 3 多模态特征融合架构优化方案

## 核心建议总结

### 📊 文本编码器：推荐 mT5-base
- **理由**: 与主模型特征空间一致，多语言支持，已集成
- **替代**: 轻量级(DistilBERT) + LoRA微调

### 🔄 融合机制：推荐 Cross-Attention (最优) / Gating (轻量)

**两阶段方案**:
1. **Phase 1**: Gating Fusion (+3-5% BLEU，快速验证)
2. **Phase 2**: Cross-Attention (+5-8% BLEU，性能优化)

### ⏱️ 时间对齐：级别2 智能插值（推荐）
```
if 帧i有描述:
    使用
elif 最近帧有描述:
    使用最近
else:
    线性插值合并
```

---

## 📂 改动清单

### 新建文件
- `text_fusion_modules.py` - 融合模块实现
- `temporal_alignment.py` - 时间对齐模块

### 修改现有文件
- `datasets.py`: 加载描述文本 + 保存帧索引 (+80行)
- `models.py`: 集成融合模块 (+40行)
- `fine_tuning.py`: 训练循环支持 (+30行)
- `utils.py`: CLI参数 (+15行)

---

## 🚀 快速命令

```bash
# Gating 融合版本（推荐首选）
deepspeed --include localhost:0,1,2,3 fine_tuning.py \
    --batch-size 16 --epochs 20 --dataset CSL_Daily \
    --use_descriptions --text_fusion_type gating \
    --rgb_support --finetune out/stage2/best.pth

# Cross-Attention 版本（高性能）
deepspeed --include localhost:0,1,2,3 fine_tuning.py \
    --batch-size 16 --epochs 20 --dataset CSL_Daily \
    --use_descriptions --text_fusion_type cross_attention \
    --rgb_support --finetune out/stage2/best.pth
```

---

## 📈 性能预期

| 方案 | BLEU | 速度 | 参数 | 推荐 |
|-----|------|------|------|------|
| Gating | +3-5% | -3% | +1% | ⭐⭐⭐⭐ |
| Cross-Attn | +5-8% | -14% | +2% | ⭐⭐⭐⭐⭐ |

---

详见完整文档
