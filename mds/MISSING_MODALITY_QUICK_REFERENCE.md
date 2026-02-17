# 缺失模态处理 - 快速参考卡

## 🎯 一句话总结

```
占位符用【可学习掩码】→ 训练加【Text Dropout】→ 推理用【缺失指示符】
```

---

## 📊 决策矩阵速查

### Q1: 应该用什么占位符？

| 占位符方案 | 简单性 | 泛化 | 一致性 | 建议场景 |
|-----------|-------|------|--------|---------|
| ❌ 零向量 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 不推荐 |
| ✅ **可学习掩码** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **第一选择** |
| ❌ 随机噪声 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 数据增强 |
| ✅ 条件零向量 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 升级方案 |

**采用**: 可学习掩码 + 缺失指示符

---

### Q2: 需要特殊的训练策略吗？

| 策略 | BLEU 提升 | 代价 | 推荐度 |
|-----|---------|------|--------|
| ❌ 无特殊处理 | +2-3% | 0 | ⭐⭐ |
| ✅ **固定 Text Dropout (0.3)** | +3-5% | 小 | ⭐⭐⭐⭐⭐ |
| ✅ **自适应 Text Dropout** | +4-6% | 中等 | ⭐⭐⭐⭐ |
| ❌ 掩码交替 | +3-5% | 大 | ⭐ |

**采用**: 固定 Text Dropout (rate=0.3)

---

### Q3: 推理时如何保证一致性？

| 方案 | BLEU 差异 | 一致性 | 复杂度 | 推荐度 |
|-----|---------|--------|--------|--------|
| ⚠️ 保守融合 | <2% | ⭐⭐⭐ | 低 | ⭐⭐ |
| ✅ **缺失感知融合** | <2% | ⭐⭐⭐⭐ | 中 | ⭐⭐⭐⭐⭐ |
| ❌ 集合投票 | <1% | ⭐⭐⭐⭐⭐ | 低 | ⭐ (太慢) |
| ❌ 自适应推理 | <1% | ⭐⭐ | 高 | ⭐ |

**采用**: 缺失感知融合（在 Gating 中加入缺失指示符）

---

## 💡 三个核心问题的答案

### 1️⃣ 占位符策略

```
【结论】: 使用可学习的掩码嵌入

为什么?
  ✅ 零向量会导致梯度黑洞，融合网络无法学习
  ✅ 可学习掩码有充分的梯度流，收敛稳定
  ✅ 推理一致性好（确定性占位符）

初始化:
  mask_embedding = mean(encode_descriptions)
  
优势:
  • 初始值接近真实文本特征
  • 训练快速收敛
  • 梯度流量充足
```

### 2️⃣ 训练策略

```
【结论】: 加入 Text Dropout (dropout_rate=0.3)

为什么?
  ✅ 强制模型学会纯视频推理
  ✅ 有无文本样本的梯度更平衡
  ✅ 提升 BLEU 同时改善一致性

效果:
  • BLEU: +3-5%
  • 一致性: BLEU差异 < 2%
  • 稳定性: Loss 曲线平滑

超参建议:
  dropout_rate = 0.3  (数据平衡)
            或 0.4  (数据不平衡)
```

### 3️⃣ 推理一致性

```
【结论】: 在 Gating 层加入缺失指示符

为什么?
  ✅ 显式告诉网络哪些是缺失的
  ✅ 网络自动学会"有文本→融合, 无文本→保留"
  ✅ 梯度清晰，学习稳健

实现:
  gate = σ(W @ [pose, text, indicator])
  
  indicator = [1.0] if has_text else [0.0]

效果:
  • 有文本 BLEU: +3-5%
  • 无文本 BLEU: 不下降或微升
  • BLEU 差异: < 2%
```

---

## ⚡ 快速命令参考

### 占位符初始化代码

```python
# 方案 1: 随机初始化（不推荐）
self.mask_embedding = nn.Parameter(torch.randn(1, 768) * 0.01)

# 方案 2: 基于已有文本特征（推荐）
with torch.no_grad():
    text_features = [encode(desc) for desc in descriptions]
    init_mask = torch.stack(text_features).mean(dim=0)
    self.mask_embedding = nn.Parameter(init_mask.unsqueeze(0))
```

### Text Dropout 应用

```python
def forward(self, text_feature, training=True, dropout_rate=0.3):
    if not training:
        return text_feature
    
    # 训练时以 dropout_rate 概率丢弃文本
    if torch.rand(1).item() < dropout_rate:
        return torch.zeros_like(text_feature)
    return text_feature
```

### 缺失感知 Gating

```python
def forward(self, pose_feat, text_feat, has_text_indicator):
    # has_text_indicator: (B, 1) 张量，1.0 表示有文本，0.0 表示无文本
    
    # Gating 融合
    combined = torch.cat([pose_feat, text_feat, has_text_indicator], dim=-1)
    gate = torch.sigmoid(self.gate_layer(combined))  # (B, T, 1)
    
    fused = pose_feat + gate * text_feat
    return fused
```

### 一致性验证伪代码

```python
# 计算 BLEU 差异
for video in test_set:
    if has_description(video):
        # 场景 A: 有文本
        text_feat = encode(description)
        pred_A = model.forward(video_feat, text_feat, has_text=[1.0])
        bleu_A = compute_bleu(pred_A, reference)
    
    # 场景 B: 无文本
    placeholder = model.mask_embedding
    pred_B = model.forward(video_feat, placeholder, has_text=[0.0])
    bleu_B = compute_bleu(pred_B, reference)
    
    delta = abs(bleu_A - bleu_B)
    assert delta < 0.02, f"差异过大: {delta:.4f}"
```

---

## 📋 分阶段实施清单

### 阶段 1: 基础验证 (1 周)

- [ ] 实现可学习掩码 (方案 B 初始化)
- [ ] 单 batch 前向测试
- [ ] 监控掩码嵌入的学习
- [ ] 验证 BLEU 相对基线 +2-3%
- [ ] 检查无 NaN/Inf 异常

### 阶段 2: 鲁棒性增强 (1-2 周)

- [ ] 添加缺失指示符 (B+D 方案)
- [ ] 实现 Text Dropout (rate=0.3)
- [ ] 修改 Gating 层接收指示符
- [ ] 验证一致性指标:
  - [ ] BLEU 差异 < 2%
  - [ ] KL divergence < 0.1
- [ ] 验证 BLEU +3-5%

### 阶段 3: 高级优化 (可选)

- [ ] 实现自适应 Text Dropout
- [ ] 梯度裁剪和学习率调整
- [ ] 消融实验验证各组件
- [ ] 推理速度和显存评估

---

## ⚠️ 常见错误及解决

| 错误现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| Loss 不下降，掩码无法学习 | 初始化过随机，梯度太大 | 用方案 B 初始化 + 降低学习率 |
| BLEU 差异仍 > 3% | 缺失指示符权重学得差 | 增加指示符的学习比例 |
| 有文本 BLEU 下降 | Text Dropout 比例太高 | 降低 dropout_rate 到 0.2 |
| 推理显存突增 | 集合投票或 batch 处理问题 | 改用缺失感知融合 |
| 掩码偏离特征空间 | 学习率过高，没有约束 | 添加 L2 正则项 |

---

## 🎯 性能目标

### BLEU 提升目标

```
基线: 25.0 BLEU

阶段 1: +2-3% → 25.5-25.8
阶段 2: +3-5% → 25.8-26.3
阶段 3: +4-6% → 26.0-26.6
```

### 一致性指标目标

```
BLEU 差异 (有文本 vs 无文本):
  阶段 1: 3-4%  ❌
  阶段 2: <2%   ✅
  阶段 3: <1%   ✅✅

KL Divergence:
  阶段 1: 0.3-0.4  ❌
  阶段 2: <0.1     ✅
  阶段 3: <0.05    ✅✅

置信度稳定性:
  阶段 1: ±8%   ❌
  阶段 2: ±3%   ✅
  阶段 3: ±1%   ✅✅
```

---

## 📚 文档导航

```
你在这里 ← MISSING_MODALITY_STRATEGY.md (快速卡)

需要详细信息 → MISSING_MODALITY_ANALYSIS.md (完整分析)
需要实施指南 → 对应实施文档
需要代码示例 → IMPLEMENTATION_REFERENCE.md
```

---

## 💬 快速 FAQ

**Q: 可学习掩码会占用显存吗?**
A: 几乎不会。仅增加 768 个参数 (0.003M)，相对 ~350M 的模型可忽略。

**Q: Text Dropout 会显著减慢训练吗?**
A: 否。仅增加简单的条件分支，计算开销 <1%。可能需要更多 epoch (10-20% 更多)。

**Q: 无文本视频必然性能更差吗?**
A: 不必然。通过缺失指示符，模型学会"显式忽视缺失"，性能差异可以 <1%。

**Q: 需要单独的验证集来测试一致性吗?**
A: 不需要。可以在现有验证集上，人为构造"无文本"版本来评估。

**Q: 生产环境部署麻烦吗?**
A: 不麻烦。仅需追踪 `has_text_indicator` 和一个额外的掩码嵌入参数，无需改动推理框架。

---

**立即开始**: 从阶段 1 的可学习掩码开始，1 周内看到 +2-3% BLEU 提升！

