# 实现参考代码

## 核心融合模块 (text_fusion_modules.py)

### Gating Fusion (推荐)
```python
class GatingFusion(nn.Module):
    def __init__(self, hidden_dim=768):
        self.text_proj = nn.Linear(768, hidden_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, video_feat, text_feat):
        # 文本投影 + 对齐 + Gate加权
        text_proj = self.text_proj(text_feat)
        # ... (时间对齐逻辑)
        gate_score = self.gate_net(concat_feat)
        fused = gate_score * text_aligned + (1-gate_score) * video_feat
        return fused
```

### Cross-Attention (最优)
```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        self.text_proj = nn.Linear(768, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
    
    def forward(self, video_feat, text_feat):
        text_proj = self.text_proj(text_feat)
        attn_out, _ = self.cross_attention(video_feat, text_proj, text_proj)
        return video_feat + attn_out
```

---

## 数据加载改动 (datasets.py)

```python
# __init__ 中
self.description_bank = self.load_descriptions(phase) if args.use_descriptions else {}

# __getitem__ 中
return name_sample, pose_sample, text, gloss, support_rgb_dict, description_features

# load_pose 中
kps_with_scores['__frame_indices__'] = torch.tensor(tmp)
```

---

## 模型集成 (models.py)

```python
# __init__ 中
if args.use_descriptions:
    self.text_fusion = GatingFusion(hidden_dim)

# forward 中
if self.use_descriptions and description_features is not None:
    inputs_embeds = self.text_fusion(inputs_embeds, desc_embeddings)
```

---

## 训练脚本 (fine_tuning.py)

```python
# 训练循环中
description_features = src_input.get('description_features', None)
output = model(src_input, tgt_input, description_features=description_features)
```

详见完整代码库
