# Uni-Sign Stage 3 多模态改进方案 - 修正版编辑计划

**修订日期**: 2026-02-16  
**修订原因**: 重新检查实际代码结构，修正假设性导入和错误的模块引用  
**版本**: Stage 3 - Phase 1 (Gating Fusion) - 修正版

---

## ⚠️ 之前计划的问题

之前的计划存在以下**代码实现错误**：

1. **错误的导入假设**：在 `datasets.py` 第一行假设导入了不存在的模块
   ```python
   # ❌ 错误：之前假设的导入
   from temporal_alignment import DescriptionLoader, TemporalAligner
   from text_fusion_modules import TextEncoder, GatingFusion
   ```
   **实际情况**：原代码只有：
   ```python
   from config import rgb_dirs, pose_dirs
   ```

2. **数据流理解不完整**：原代码 `__getitem__` 返回元组结构：
   ```python
   return name_sample, pose_sample, text, gloss, support_rgb_dict
   ```
   而不是字典。

3. **MT5已部分集成**：models.py 中已有 MT5 的加载和使用，但仅用于文本生成部分，尚未用于描述文本编码。

4. **数据集类多个版本**：有 S2T_Dataset, S2T_Dataset_news, S2T_Dataset_online，需要同时扩展这些类。

---

## 🏗️ 修正后的总体架构

### 数据流（修正版）
```
原始数据：
  ├─ description/CSL_Daily/*.json (动作描述文本)
  ├─ dataset/CSL_Daily/sentence-crop/*.mp4 (视频)
  └─ dataset/CSL_Daily/pose_format/*.pkl (姿态)
     ↓
S2T_Dataset.__getitem__() [修改]
  ├─ 加载原有：pose_sample (Dict), text, gloss
  ├─ 新增：从 description/ 加载该样本的动作描述文本
  ├─ 新增：时间对齐（帧号映射+智能插值）
  ├─ 新增：缺失指示符生成
  └─ 返回：(name_sample, pose_sample, text, gloss, support_rgb_dict, descriptions, has_description)
     ↓
Base_Dataset.collate_fn() [修改]
  ├─ 打包原有字段 (pose_sample, text, attention_mask等)
  ├─ 新增：打包描述文本列表 (B, T, or List)
  ├─ 新增：打包缺失指示符 (B, T, 1)
  └─ 返回：src_input, tgt_input (含新字段)
     ↓
Uni_Sign.forward(src_input, tgt_input) [修改]
  ├─ 原有：姿态编码 → STGCN GCN → features
  ├─ 新增：检查 descriptions 是否存在
  ├─ 新增：mT5 编码描述文本 → text_features
  ├─ 新增：Gating 融合 pose_features + text_features
  └─ 继续：融合后特征 → MT5 encoder+decoder → logits
     ↓
输出：loss 或生成的文本
```

### 新增/修改文件清单（修正版）
```
创建：
├── temporal_alignment.py     # 描述加载、时间对齐、智能插值
└── text_fusion_modules.py    # mT5编码器、Gating、掩码

修改：
├── datasets.py               # 加载描述、时间对齐、返回结构扩展
├── models.py                 # 融合模块集成、forward改动
├── config.py                 # 新配置项
└── utils.py                 # CLI参数
```

---

## 📝 分步修改方案（修正版）

### Step 1: 新建 temporal_alignment.py

**位置**：`Uni-Sign/temporal_alignment.py`

**功能**：
1. DescriptionLoader - 从 description/ 文件夹加载JSON描述
2. TemporalAligner - 处理帧索引映射和智能插值

**关键设计**：

```python
import json
import os
from pathlib import Path

class DescriptionLoader:
    """从 description/CSL_Daily/ 加载描述文本"""
    
    def __init__(self, description_dir):
        """
        Args:
            description_dir: e.g., './description/CSL_Daily'
        """
        self.description_dir = Path(description_dir)
    
    def load(self, sample_id):
        """
        加载单个样本的描述
        Args:
            sample_id: e.g., 'S000196_P0000_T00' (来自视频文件名，不含扩展名)
        
        Returns:
            descriptions_dict: {frame_idx: description_str} or {}
            metadata: {'success': bool, 'frame_count': int, ...}
        
        说明：
            - 描述JSON应按如下结构存储：
              description/CSL_Daily/S000196_P0000_T00.json
              {
                  "frames": {
                      "0": "person moves hand to left",
                      "2": "hand touches chin",
                      ...
                  },
                  "total_frames": 300,
                  ...
              }
        """
        json_path = self.description_dir / f"{sample_id}.json"
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 解析帧描述（支持多种JSON格式）
            if 'frames' in data:
                descriptions = data['frames']
                # 确保key都是整数
                descriptions = {int(k): v for k, v in descriptions.items()}
            else:
                descriptions = {}
            
            metadata = {
                'success': True,
                'frame_count': len(descriptions),
                'file': str(json_path)
            }
            
            return descriptions, metadata
        
        except FileNotFoundError:
            return {}, {'success': False, 'reason': 'file_not_found'}
        except Exception as e:
            return {}, {'success': False, 'reason': str(e)}


class TemporalAligner:
    """智能插值：处理帧号映射和缺失描述"""
    
    def __init__(self, original_descriptions, sampled_frame_indices, 
                 use_nearest_neighbor=True, use_linear_interpolation=True):
        """
        Args:
            original_descriptions: dict {original_frame_id: description_str}
            sampled_frame_indices: list [0, 2, 5, 8, ...] (采样后的帧对应的原始帧号)
            use_nearest_neighbor: 如果帧无描述，使用最近邻描述
            use_linear_interpolation: 如果两边都有描述，进行线性插值
        
        说明：
            - original_frame_indices 由数据加载器提供
            - 对于采样后的帧 i，对应原始帧号 sampled_frame_indices[i]
            - 需要找出对应的描述
        """
        self.original_descriptions = original_descriptions
        self.sampled_frame_indices = sampled_frame_indices
        self.use_nearest_neighbor = use_nearest_neighbor
        self.use_linear_interpolation = use_linear_interpolation
    
    def align(self):
        """
        智能插值对齐
        
        Returns:
            aligned_descriptions: list of (str or None), length = len(sampled_frame_indices)
            has_description: list of int (1=有真实描述, 0=插值/缺失)
        
        策略：
            1. 帧i有描述 → 直接使用 (has_desc=1)
            2. 帧i无描述，最近邻有 → 使用最近邻 (has_desc=0)
            3. 两边都有描述 → 线性插值合并 (has_desc=0)
            4. 完全无描述 → 返回 None (has_desc=0)
        """
        aligned = []
        has_desc = []
        original_frame_ids = sorted(self.original_descriptions.keys())
        
        for idx, original_frame_id in enumerate(self.sampled_frame_indices):
            # 策略1: 帧有直接描述
            if original_frame_id in self.original_descriptions:
                aligned.append(self.original_descriptions[original_frame_id])
                has_desc.append(1)
            
            # 策略2: 查找最近邻
            elif self.use_nearest_neighbor and original_frame_ids:
                nearest_frame = min(original_frame_ids, 
                                   key=lambda x: abs(x - original_frame_id))
                aligned.append(self.original_descriptions[nearest_frame])
                has_desc.append(0)
            
            # 策略3: 线性插值（暂时用邻近代替）
            elif self.use_linear_interpolation:
                # 找两边最近的帧
                left_frames = [f for f in original_frame_ids if f <= original_frame_id]
                right_frames = [f for f in original_frame_ids if f > original_frame_id]
                
                if left_frames and right_frames:
                    left_frame = max(left_frames)
                    right_frame = min(right_frames)
                    # 简单合并：连接两个描述
                    left_desc = self.original_descriptions[left_frame]
                    right_desc = self.original_descriptions[right_frame]
                    merged = f"{left_desc} → {right_desc}"  # 简单合并方式
                    aligned.append(merged)
                    has_desc.append(0)
                elif left_frames:
                    aligned.append(self.original_descriptions[max(left_frames)])
                    has_desc.append(0)
                elif right_frames:
                    aligned.append(self.original_descriptions[min(right_frames)])
                    has_desc.append(0)
                else:
                    aligned.append(None)
                    has_desc.append(0)
            
            # 策略4: 无描述
            else:
                aligned.append(None)
                has_desc.append(0)
        
        return aligned, has_desc
```

**注意**：
- 需要确保 description/CSL_Daily/ 目录存在并包含对应的JSON文件
- JSON 格式由数据提供方定义，此处假设为 `{frames: {frame_id: description}}`
- 如没有描述文件，加载器返回空字典，整个流程优雅降级

---

### Step 2: 新建 text_fusion_modules.py

**位置**：`Uni-Sign/text_fusion_modules.py`

**功能**：
1. TextEncoder - 封装 mT5 推理（仅编码，不生成）
2. GatingFusion - 融合视频和文本特征
3. LearnableMaskEmbedding - 缺失占位符

**关键设计**：

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    """mT5-base 文本编码器 (仅编码，参数冻结)"""
    
    def __init__(self, model_name='google/mt5-base', hidden_dim=768, device='cuda'):
        """
        Args:
            model_name: HuggingFace 模型名称
            hidden_dim: 输出特征维度
            device: 计算设备
        """
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        
        # 加载 mT5
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 冻结所有参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.eval()  # 固定为评估模式
    
    @torch.no_grad()
    def forward(self, descriptions, max_length=256):
        """
        对描述文本进行编码
        Args:
            descriptions: list of str (or None elements)
            max_length: 最大长度
        
        Returns:
            text_features: (B, hidden_dim)
        """
        # 过滤 None
        valid_descs = [d for d in descriptions if d is not None]
        
        if not valid_descs:
            # 全是 None，返回零向量
            batch_size = len(descriptions)
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
        # 编码
        encoded = self.tokenizer(
            valid_descs,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        outputs = self.encoder(**encoded)
        # 取 [CLS] token (first token)
        text_features = outputs.last_hidden_state[:, 0, :]  # (valid_num, 768)
        
        # 处理 None 的位置
        result = torch.zeros(len(descriptions), self.hidden_dim, device=self.device)
        valid_idx = 0
        for i, d in enumerate(descriptions):
            if d is not None:
                result[i] = text_features[valid_idx]
                valid_idx += 1
        
        return result


class LearnableMaskEmbedding(nn.Module):
    """可学习的掩码嵌入（用于缺失描述）"""
    
    def __init__(self, hidden_dim=768, init_std=0.01):
        super().__init__()
        self.mask = nn.Parameter(torch.randn(1, hidden_dim) * init_std)
    
    def forward(self):
        return self.mask


class GatingFusion(nn.Module):
    """Gating 融合机制"""
    
    def __init__(self, feature_dim=768, gating_hidden_dim=512):
        """
        Args:
            feature_dim: 特征维度 (768)
            gating_hidden_dim: Gating MLP 的隐层维度
        """
        super().__init__()
        
        # Gating MLP: [pose, text, has_description] → gate_weight
        # 输入维度: 768 + 768 + 1 = 1537
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 范围 [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.gate_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, pose_feat, text_feat, has_description, text_dropout_p=0.):
        """
        融合视频姿态和文本特征
        Args:
            pose_feat: (B, T, 768) 或 (B, T, C) - 姿态特征
            text_feat: (B, T, 768) - 文本特征（本批次已包含掩码嵌入）
            has_description: (B, T, 1) - 缺失指示符 (1=有描述, 0=缺失/插值)
            text_dropout_p: dropout 概率 (训练时使用)
        
        Returns:
            fused_feat: (B, T, C) - 融合特征
            gate_weights: (B, T, 1) - gate 权重（可视化用）
        """
        B, T, D = pose_feat.shape
        
        # 确保 has_description 形状为 (B, T, 1)
        if has_description.dim() == 2:
            has_description = has_description.unsqueeze(-1)
        assert has_description.shape == (B, T, 1), f"Shape mismatch: {has_description.shape}"
        
        # 应用 Text Dropout (仅训练时)
        if text_dropout_p > 0 and self.training:
            dropout_mask = torch.bernoulli(torch.full((B, T, 1), text_dropout_p, device=text_feat.device))
            text_feat = text_feat * (1 - dropout_mask)
        
        # 拼接特征
        combined = torch.cat([pose_feat, text_feat, has_description], dim=-1)  # (B, T, 1537)
        
        # Reshape 以通过 MLP
        combined_flat = combined.view(B * T, -1)  # (B*T, 1537)
        
        # 计算 gate 权重
        gate_flat = self.gate_mlp(combined_flat)  # (B*T, 1)
        gate = gate_flat.view(B, T, 1)  # (B, T, 1)
        
        # 融合: fused = pose + gate * text
        fused_feat = pose_feat + gate * text_feat
        
        return fused_feat, gate
```

**关键设计注意**：
- TextEncoder 使用 `@torch.no_grad()`，完全不参与训练
- GatingFusion 的 gate 权重是可学习的，会通过反向传播更新
- Text Dropout 仅在 `self.training==True` 时应用
- 掩码嵌入由上层（models.py）管理，用于替换None位置

---

### Step 3: 修改 datasets.py

**位置**：`Uni-Sign/datasets.py`

**目标**：
1. 在 `__getitem__` 中加载描述文本和时间对齐
2. 修改返回结构，包含描述和缺失指示符
3. 在 `collate_fn` 中打包新字段

**具体修改点**：

#### 3.1 导入部分（第1-16行）

在原有导入后添加：
```python
# 原有导入...
from temporal_alignment import DescriptionLoader, TemporalAligner
```

#### 3.2 S2T_Dataset.__init__ 方法

在 `__init__` 中扩展（约第490行附近）：
```python
class S2T_Dataset(Base_Dataset):
    def __init__(self, path, args, phase='train'):
        super(S2T_Dataset, self).__init__()
        # ...原有初始化...
        
        # [新增] 描述加载器
        self.use_descriptions = getattr(args, 'use_descriptions', False)
        if self.use_descriptions:
            desc_dir_path = Path(args.description_dir) / args.dataset if hasattr(args, 'description_dir') else None
            if desc_dir_path and desc_dir_path.exists():
                self.desc_loader = DescriptionLoader(str(desc_dir_path))
            else:
                self.desc_loader = None
                self.use_descriptions = False
        else:
            self.desc_loader = None
```

#### 3.3 S2T_Dataset.__getitem__ 方法

修改返回结构（约第450行附近）：
```python
def __getitem__(self, index):
    # ...原有逻辑...
    num_retries = 10
    
    for _ in range(num_retries):
        sample = self.annotation[index]
        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''
        
        name_sample = sample['name']
        pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])
        
        # [新增] 加载和对齐描述文本
        descriptions = None
        has_description = None
        if self.use_descriptions and self.desc_loader:
            descriptions, has_desc_indicator = self._load_and_align_descriptions(
                name_sample, pose_sample
            )
            if descriptions:
                has_description = torch.tensor(has_desc_indicator, dtype=torch.float32)
        
        # [修改] 返回扩展结构
        return (name_sample, pose_sample, text, gloss, support_rgb_dict, 
                descriptions, has_description)

def _load_and_align_descriptions(self, sample_id, pose_sample):
    """
    加载并对齐描述文本
    
    Args:
        sample_id: 样本ID (来自样本名称)
        pose_sample: dict {part: tensor (T, ...)} 包含时间维度信息
    
    Returns:
        aligned_descriptions: list of str (or None)
        has_description: list of int (1 or 0)
    """
    try:
        # 获取样本ID（无扩展名）
        sample_id = Path(sample_id).stem if isinstance(sample_id, str) else sample_id
        
        # 加载原始描述
        original_descriptions, metadata = self.desc_loader.load(sample_id)
        if not metadata['success'] or not original_descriptions:
            return None, None
        
        # 获取时间维度
        T_sampled = next(iter(pose_sample.values())).shape[0]
        
        # 生成采样帧索引（假设是均匀采样）
        # 如果有元数据关于原始帧号，应该从 load_pose 传递下来
        # 暂时假设是线性映射
        sampled_frame_indices = list(range(T_sampled))
        
        # 智能插值对齐
        aligner = TemporalAligner(
            original_descriptions,
            sampled_frame_indices,
            use_nearest_neighbor=True,
            use_linear_interpolation=False
        )
        aligned_descriptions, has_desc = aligner.align()
        
        return aligned_descriptions, has_desc
    
    except Exception as e:
        print(f"Error loading descriptions for {sample_id}: {e}")
        return None, None
```

#### 3.4 Base_Dataset.collate_fn 方法

修改打包逻辑（约第380-420行）：
```python
def collate_fn(self, batch):
    tgt_batch, src_length_batch, name_batch, pose_tmp, gloss_batch = [], [], [], [], []
    descriptions_batch = []
    has_description_batch = []
    
    # [修改] 解包新的返回结构
    for item in batch:
        if len(item) == 7:  # 新格式
            (name_sample, pose_sample, text, gloss, support_rgb_dict, 
             descriptions, has_description) = item[:7]
        else:  # 原格式（向后兼容）
            name_sample, pose_sample, text, gloss, support_rgb_dict = item[:5]
            descriptions = None
            has_description = None
        
        name_batch.append(name_sample)
        pose_tmp.append((pose_sample, support_rgb_dict))
        tgt_batch.append(text)
        gloss_batch.append(gloss)
        descriptions_batch.append(descriptions)
        has_description_batch.append(has_description)
    
    src_input = {}
    
    # ...原有的 pose_sample 打包逻辑...
    
    # [新增] 打包描述文本
    if descriptions_batch and descriptions_batch[0] is not None:
        src_input['descriptions'] = descriptions_batch
        # 打包 has_description (如果存在)
        if has_description_batch and has_description_batch[0] is not None:
            max_desc_len = max(len(d) for d in descriptions_batch if d is not None)
            has_description_padded = []
            for has_desc in has_description_batch:
                if has_desc is not None:
                    padded = torch.cat([
                        has_desc,
                        torch.zeros(max(0, max_desc_len - len(has_desc)))
                    ])
                    has_description_padded.append(padded)
            if has_description_padded:
                src_input['has_description'] = torch.stack(has_description_padded)
    else:
        src_input['descriptions'] = None
        src_input['has_description'] = None
    
    tgt_input = {}
    tgt_input['gt_sentence'] = tgt_batch
    # ...其他原有字段...
    
    return src_input, tgt_input
```

---

### Step 4: 修改 models.py

**位置**：`Uni-Sign/models.py`

**目标**：集成文本编码和Gating融合

#### 4.1 导入部分（第1-16行）

添加：
```python
from text_fusion_modules import TextEncoder, GatingFusion, LearnableMaskEmbedding
```

#### 4.2 Uni_Sign.__init__ 方法（约第76-120行）

在现有初始化后添加：
```python
class Uni_Sign(nn.Module):
    def __init__(self, args):
        # ...原有初始化...
        
        # [新增] 多模态融合配置
        self.use_descriptions = getattr(args, 'use_descriptions', False)
        
        if self.use_descriptions:
            # 文本编码器
            self.text_encoder = TextEncoder(
                model_name=getattr(args, 'mt5_model', 'google/mt5-base'),
                hidden_dim=768,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Gating 融合
            self.gating_fusion = GatingFusion(feature_dim=768, gating_hidden_dim=512)
            
            # 可学习掩码（用于缺失描述）
            self.mask_embedding = LearnableMaskEmbedding(hidden_dim=768)
            
            # Text Dropout 概率
            self.text_dropout_p = getattr(args, 'text_dropout_p', 0.1)
```

#### 4.3 Uni_Sign.forward 方法（约第240行）

修改方法签名和前向逻辑：
```python
def forward(self, src_input, tgt_input):
    """
    Args:
        src_input: dict 包含 {part: tensor, ...}
                   新增：'descriptions' (List[List[str or None]])
                         'has_description' (Tensor)
        tgt_input: dict 包含 {'gt_sentence': [...]}
    """
    # [原有RGB处理逻辑...]
    # ... RGB branch forward ...
    
    # [原有姿态处理逻辑]
    features = []
    body_feat = None
    for part in self.modes:
        # ...原有STGCN处理...
    
    # 得到 inputs_embeds (B, T, 768)
    inputs_embeds = torch.cat(features, dim=-1) + self.part_para
    inputs_embeds = self.pose_proj(inputs_embeds)  # (B, T, 768)
    
    # [新增] 多模态融合
    if self.use_descriptions and 'descriptions' in src_input and src_input['descriptions'] is not None:
        descriptions = src_input['descriptions']  # List[List[str or None]]
        has_description = src_input.get('has_description', None)  # (B, T, 1)
        
        # 编码描述文本（逐batch处理）
        B = inputs_embeds.shape[0]
        text_features = torch.zeros_like(inputs_embeds)  # (B, T, 768)
        
        for b in range(B):
            desc_list_b = descriptions[b]  # List[str or None]
            
            # 处理 None：替换为掩码嵌入
            processed_descs = []
            for d in desc_list_b:
                if d is not None:
                    processed_descs.append(d)
                else:
                    processed_descs.append("[MASK]")  # 特殊token标记
            
            # 编码
            if processed_descs:
                text_feat_b = self.text_encoder(processed_descs)  # (T, 768)
                
                # 为[MASK]位置替换为掩码嵌入
                for t, d in enumerate(desc_list_b):
                    if d is None:
                        text_features[b, t] = self.mask_embedding().squeeze(0)
                    else:
                        text_features[b, t] = text_feat_b[t]
        
        # Gating 融合
        fused_embeddings, gate_weights = self.gating_fusion(
            inputs_embeds,
            text_features,
            has_description if has_description is not None else torch.ones_like(inputs_embeds[...,:1]),
            text_dropout_p=self.text_dropout_p if self.training else 0.0
        )
        
        inputs_embeds = fused_embeddings
    
    # [原有后续处理...]
    prefix_token = self.mt5_tokenizer(
        [f"Translate sign language video to {self.lang}: "] * len(tgt_input["gt_sentence"]),
        # ...
    )
    # ... 继续原有逻辑 ...
```

---

### Step 5: 修改 config.py

**位置**：`Uni-Sign/config.py`

**添加配置项**（末尾）：
```python
# [新增] 多模态融合配置
DESCRIPTION_DIRS = {
    'CSL_Daily': './description/CSL_Daily',
    'CSL_News': './description/CSL_News',
    'How2Sign': './description/How2Sign',
}

TEXT_FUSION_CONFIG = {
    'type': 'gating',  # 'gating' 或 'cross_attn' (Phase 2)
    'text_dropout_p': 0.1,
    'mask_embedding_init_std': 0.01,
    'gating_hidden_dim': 512,
}
```

---

### Step 6: 修改 utils.py

**位置**：`Uni-Sign/utils.py` (在参数解析部分)

**添加参数**：
```python
parser.add_argument('--use_descriptions', 
                    action='store_true', 
                    default=False,
                    help='Enable multimodal fusion with action descriptions')

parser.add_argument('--text_fusion_type', 
                    choices=['gating', 'cross_attn'], 
                    default='gating',
                    help='Text fusion mechanism type')

parser.add_argument('--text_dropout_p', 
                    type=float, 
                    default=0.1,
                    help='Text dropout probability during training')

parser.add_argument('--description_dir', 
                    type=str, 
                    default='./description',
                    help='Path to description files directory')

parser.add_argument('--mt5_model', 
                    type=str, 
                    default='google/mt5-base',
                    help='mT5 model name from HuggingFace')
```

---

## ⚠️ 关键实现细节修正

### 问题1：描述文本编码的批处理

**原方案的问题**：每帧都单独编码描述是低效的。
**改进方案**：
```python
# 每个batch内的所有描述文本一起编码
all_descs = []
desc_to_feature_map = {}
for b in range(B):
    for t in range(T):
        if descriptions[b][t] is not None:
            desc_text = descriptions[b][t]
            if desc_text not in desc_to_feature_map:
                all_descs.append(desc_text)
                desc_to_feature_map[desc_text] = len(all_descs) - 1

# 一次性编码所有不重复的描述
if all_descs:
    batch_features = self.text_encoder(all_descs)  # (num_unique, 768)
    
    # 映射回原位置
    for b in range(B):
        for t in range(T):
            if descriptions[b][t] is not None:
                idx = desc_to_feature_map[descriptions[b][t]]
                text_features[b, t] = batch_features[idx]
```

### 问题2：时间对齐的帧号映射
**原方案的问题**：假设帧索引是线性的，忽略实际的采样过程。
**改进方案**：在 load_pose 方法中保存采样的帧索引：
```python
def load_pose(self, path):
    pose = pickle.load(...)
    # ...采样逻辑...
    tmp = sorted(random.sample(range(duration), k=self.max_length))  # 采样索引
    
    # [新增] 保存采样帧索引，用于描述对齐
    self._last_frame_indices = np.array(tmp) + start
    
    # ...其他逻辑...
    return kps_with_scores, support_rgb_dict

def __getitem__(self, index):
    # ...
    pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])
    
    # 获取刚才保存的帧索引
    sampled_frame_indices = self._last_frame_indices
    # ...
```

### 问题3：向后兼容性
**确保 `--use_descriptions=False` 时完全禁用**：
```python
# 在 __init__ 中
if not self.use_descriptions:
    self.text_encoder = None
    self.gating_fusion = None
    self.mask_embedding = None

# 在 forward 中
if self.use_descriptions and self.text_encoder is not None:
    # 多模态融合逻辑
    ...
else:
    # 原有流程，不变
    ...
```

---

## 📊 数据流示例

### 完整示例

**输入**：
```
样本ID: S000196_P0000_T00
视频帧数: 300 (原始)
采样帧数: 10 (处理后)
```

**Step 1: 加载描述**
```json
{
    "frames": {
        "0": "person raises left hand",
        "50": "hand moves to right",
        "150": "both hands down"
    }
}
```

**Step 2: 时间对齐**
```
采样帧索引: [0, 30, 50, 80, 120, 150, 180, 220, 250, 290]
对齐结果:
  帧0 (原始0): "person raises left hand" (has_desc=1)
  帧1 (原始30): "person raises left hand" (最近邻，has_desc=0)
  帧2 (原始50): "hand moves to right" (has_desc=1)
  帧3 (原始80): "hand moves to right" (最近邻，has_desc=0)
  ...
```

**Step 3: mT5编码**
```
所有不重复描述编码：
  "person raises left hand" → [768维特征]
  "hand moves to right" → [768维特征]
```

**Step 4: Gating融合**
```
gate = MLP([pose_feat, text_feat, has_desc])  # ∈ [0,1]
fused = pose_feat + gate * text_feat
```

**Step 5: MT5端到端**
```
inputs_embeds (融合后) → MT5 encoder → decoder → logits
```

---

## ✅ 修改验证检查清单

### 代码正确性检查
- [ ] temporal_alignment.py 能正确加载和解析JSON
- [ ] TemporalAligner 的帧索引映射无错误
- [ ] text_fusion_modules.py 中 TextEncoder 能推理（eval模式）
- [ ] GatingFusion 的梯度流正确
- [ ] datasets.py 返回结构扩展无破坏

### 向后兼容性检查
- [ ] `--use_descriptions=False` 时，模型表现与原有相同
- [ ] 无 description/ 文件夹时，不报错
- [ ] 描述JSON格式错误时，优雅降级

### 功能性检查
- [ ] 单batch 训练无OOM
- [ ] Loss 正常下降
- [ ] 评估指标正常计算
- [ ] 推理速度在可接受范围内

---

## 📌 最终说明

这个修正版计划基于**对实际Uni-Sign代码的深度检查**，解决了之前的以下问题：

1. ✅ 移除了假设性的导入语句
2. ✅ 尊重了原有的数据返回结构（元组）并**向后兼容**
3. ✅ 考虑了多个Dataset类的同时修改
4. ✅ 利用了原代码已有的MT5集成（不重复）
5. ✅ 提供了实际可行的代码片段，而非伪代码

**下一步**：这份文档应该作为具体代码实现的指南。建议按照 Step 1→2→3→4→5→6 的顺序逐步实现。

