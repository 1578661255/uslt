# 快速参考卡片

## 🎯 一句话
融合逐帧动作描述文本，用 Gating 或 Cross-Attention，预期 +3-8% BLEU

## 🚀 快速命令
```bash
# Gating 版本
deepspeed fine_tuning.py --use_descriptions --text_fusion_type gating \
    --batch-size 16 --epochs 20 --dataset CSL_Daily --rgb_support

# Cross-Attention 版本  
deepspeed fine_tuning.py --use_descriptions --text_fusion_type cross_attention \
    --batch-size 16 --epochs 20 --dataset CSL_Daily --rgb_support
```

## 📊 决策矩阵
```
需要快速验证 → Gating (+3-5%, 快)
需要最优性能 → Cross-Attention (+5-8%, 慢)
参数有限制 → Concat (最轻)
```

## 📁 关键改动
| 文件 | 改动 |
|-----|------|
| datasets.py | 加载描述+保存索引 |
| models.py | 集成融合模块 |
| fine_tuning.py | 训练循环 |

## ✅ 验证清单
- [ ] 单batch 前向成功
- [ ] Loss 在 0.1-1.0
- [ ] Dev BLEU ≥ baseline
- [ ] 推理速度可接受

## 💡 关键技巧
1. **必须保存 `__frame_indices__`** - 时间对齐关键
2. **先用 Gating 验证** - 快速确认可行性
3. **渐进融合** - 从低权重逐步增加
4. **向后兼容** - `--use_descriptions=False` 要能用

## 🔴 常见错误
- 忘记保存帧索引 → 时间错位
- 直接用 Cross-Attn → 显存溢出
- 融合权重过大 → BLEU 下降
- 描述文本路径错 → 加载失败
