# Uni-Sign Stage 3 多模态改进方案 - 分步修改逻辑

**方案创建日期**: 2026-02-16  
**方案版本**: Stage 3 - Phase 1 (Gating Fusion)  
**文档目标**: 提供完整的分步代码修改逻辑，不破坏现有功能，逐步集成动作描述文本

---

## 📋 核心方案要求

### 技术规格
| 模块 | 方案 | 说明 |
|------|------|------|
| **文本编码** | mT5-base | 编码 description/ 文件夹中的动作描述文本 |
| **时间对齐** | 智能插值 | 缺失帧用最近邻，两边都有则线性插值 |
| **融合机制** | Gating | 动态权重融合，轻量级，参数少 |
| **缺失处理** | 可学习掩码+Dropout | 占位符、Text Dropout、缺失指示符 |

---

## 🏗️ 总体架构设计

### 数据流图
```
输入数据 
  ├─ 原有：视频/姿态特征 (B, 4, T, 150)
  ├─ 新增：描述文本 (list of str or None)
  └─ 新增：帧索引映射 (__frame_indices__)
     ↓
数据加载层 (datasets.py改动)
  ├─ 加载CSL_Daily中的描述文本 (description/CSL_Daily/)
  ├─ 解析帧索引关系 (原始帧→采样帧)
  ├─ 生成缺失指示符 (has_description)
  └─ 支持部分缺失的优雅处理
     ↓
模型前向传播
  ├─ pose features: (B, T, 768) ← 原有特征
  ├─ text features: (B, T, 768) ← mT5编码
  ├─ mask_embedding: (1, 768) ← 可学习占位符
  └─ has_description: (B, T, 1) ← 缺失指示符
     ↓
融合层 (models.py新增)
  ├─ text encoder: mT5-base冻结
  ├─ gating fusion: 学习权重融合
  └─ text dropout: 训练时随机丢弃
     ↓
输出：融合特征 (B, T, 768) ← 用于下游任务
```

### 新增/修改文件清单
```
├── datasets.py              [修改] 数据加载、描述解析、帧索引处理
├── models.py                [修改] 文本编码器、融合模块集成
├── fine_tuning.py           [修改] 训练循环支持新输入
├── utils.py                 [修改] CLI参数支持
├── temporal_alignment.py     [新建] 时间对齐、描述加载器
├── text_fusion_modules.py    [新建] 文本编码器、Gating融合模块
├── test_multimodal.py        [新建] 单元测试 (可选)
└── config.py                [修改] 新增配置项
```

---

## 📝 分步修改方案

### Step 1: 新建 temporal_alignment.py

**目标**: 提供描述加载和时间对齐的核心工具

**功能模块**:
1. `DescriptionLoader`: 从description文件夹加载描述文本JSON
2. `TemporalAligner`: 处理帧索引映射和智能插值
3. 辅助函数: 处理缺失、插值等

**关键接口**:
```python
class DescriptionLoader:
    def __init__(self, description_dir):
        """
        初始化描述加载器
        Args:
            description_dir: description/CSL_Daily 目录路径
        """
    
    def load(self, sample_id):
        """
        加载单个样本的描述文本
        Returns:
            descriptions: dict {frame_id: str} or {}
            metadata: 元数据
        """

class TemporalAligner:
    def __init__(self, descriptions, frame_indices):
        """
        时间对齐器
        Args:
            descriptions: 原始帧的描述字典
            frame_indices: 采样后的帧索引列表 [f1, f2, ...]
        """
    
    def align(self):
        """
        进行智能插值对齐
        策略:
        1. 采样帧有描述 → 直接使用
        2. 邻近帧有描述 → 使用最近帧
        3. 两边都有 → 线性插值合并
        Returns:
            aligned_descriptions: list (长度=采样帧数)
            has_description: list (缺失指示符)
        """
```

**输入/输出示例**:
```json
// 输入: 原始帧描述
{
    "0": "person moves hand to left",
    "2": "hand touches chin",
    "5": "both hands move down"
}

// 输入: 帧采样索引
[0, 1, 2, 4, 5]

// 输出: 对齐后的描述 (智能插值)
[
    "person moves hand to left",      // frame 0: 直接
    "person moves hand to left",      // frame 1: 最近邻
    "hand touches chin",              // frame 2: 直接
    "interpolate(手触下巴, 双手向下)", // frame 4: 插值
    "both hands move down"            // frame 5: 直接
]

// 输出: 缺失指示符
[1, 1, 1, 0, 1]  // 1=有描述, 0=插值/缺失
```

**实现注意**:
- 处理JSON格式的描述文本 (来自description/下的JSON文件)
- 支持完全缺失的样本 (返回全None)
- 缺失指示符用于后续推理和Gating加权

---

### Step 2: 新建 text_fusion_modules.py

**目标**: 实现文本编码和融合的核心组件

**功能模块**:
1. `TextEncoder`: 封装mT5-base的推理
2. `GatingFusion`: 实现Gating融合机制
3. 辅助函数: 掩码处理等

**关键类**:

```python
class TextEncoder(nn.Module):
    """mT5-base 文本编码器 (冻结)"""
    
    def __init__(self, model_name='mt5-base', hidden_dim=768, device='cuda'):
        """
        Args:
            model_name: 模型名称 (default: mt5-base)
            hidden_dim: 输出维度 (与视觉特征一致)
            device: 设备
        """
    
    def forward(self, descriptions, max_length=256):
        """
        对描述文本进行编码
        Args:
            descriptions: list of str (or None elements)
            max_length: 最大长度
        
        Returns:
            text_features: (B, hidden_dim) 或 (B, T, hidden_dim)
            有None则返回对应位置的零向量
        """

class GatingFusion(nn.Module):
    """Gating 融合机制"""
    
    def __init__(self, feature_dim=768):
        """
        Args:
            feature_dim: 特征维度 (768)
        """
    
    def forward(self, pose_feat, text_feat, has_description, text_dropout_p=0.):
        """
        融合视频姿态和文本特征
        Args:
            pose_feat: (B, T, 768) 或 (B, T, C)
            text_feat: (B, T, 768)
            has_description: (B, T, 1) 缺失指示符
            text_dropout_p: dropout概率 (训练时使用)
        
        Returns:
            fused_feat: (B, T, 768)
            gate_weights: (B, T, 1) 可视化用
        
        融合公式:
        gate = Sigmoid(MLP([pose, text, has_description]))
        fused = pose + gate * text
        """

class LearnableMaskEmbedding(nn.Module):
    """可学习掩码嵌入"""
    
    def __init__(self, hidden_dim=768):
        self.mask = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
    
    def forward(self):
        return self.mask
```

**训练策略 (Text Dropout)**:
- 在训练时，以概率 `text_dropout_p` (如0.2) 随机替换文本特征
- 替换为掩码嵌入或零向量
- 提高模型对缺失模态的鲁棒性

**推理策略 (缺失指示符)**:
- 使用 `has_description` 显式指示哪些帧有真实描述
- Gating会自动为缺失帧分配低权重
- 可选: 完全禁用缺失帧的文本特征 (置零)

---

### Step 3: 修改 datasets.py

**目标**: 集成描述文本加载和帧索引处理

**修改位置**:
1. **导入部分** (顶部)
   ```python
   from temporal_alignment import DescriptionLoader, TemporalAligner
   from text_fusion_modules import TextEncoder, GatingFusion
   ```

2. **S2T_Dataset 类改动**
   - `__init__`: 初始化 DescriptionLoader
   - `__getitem__`: 加载描述文本、进行时间对齐
   - `collate_fn`: 处理批量打包

**关键修改**:

```python
class S2T_Dataset(Dataset):
    def __init__(self, path, args, phase='train'):
        # 原有初始化...
        
        # [新增] 描述加载器
        self.use_descriptions = getattr(args, 'use_descriptions', False)
        if self.use_descriptions:
            desc_dir = Path(rgb_dirs[args.dataset]).parent / 'description' / args.dataset
            self.desc_loader = DescriptionLoader(str(desc_dir))
        else:
            self.desc_loader = None
    
    def __getitem__(self, idx):
        # 原有逻辑返回: src_input, tgt_input, sign_id, sample_path
        src_input, tgt_input, sign_id, sample_path = self._original_getitem(idx)
        
        # [新增] 加载描述文本
        if self.use_descriptions and self.desc_loader:
            sample_id = Path(sample_path).stem  # e.g., 'S000196_P0000_T00'
            descriptions, frame_indices = self._load_and_align_descriptions(
                sample_id, src_input
            )
            has_description = torch.tensor(
                [1 if d is not None else 0 
                 for d in descriptions],
                dtype=torch.float32
            )
        else:
            descriptions = None
            frame_indices = None
            has_description = None
        
        return {
            'src_input': src_input,
            'tgt_input': tgt_input,
            'descriptions': descriptions,
            'frame_indices': frame_indices,
            'has_description': has_description,
            'sign_id': sign_id,
            'sample_path': sample_path
        }
    
    def _load_and_align_descriptions(self, sample_id, src_input):
        """
        加载描述并进行时间对齐
        """
        try:
            original_descriptions = self.desc_loader.load(sample_id)
            if not original_descriptions:
                return None, None
        except:
            return None, None
        
        # 获取帧索引 (从src_input推断或元数据)
        # 假设 src_input 的时间维度已是采样后的
        T_sampled = src_input.shape[1] if src_input.dim() >= 2 else 1
        
        aligner = TemporalAligner(original_descriptions, frame_indices=list(range(T_sampled)))
        aligned_descriptions, has_desc = aligner.align()
        
        return aligned_descriptions, list(range(T_sampled))
    
    def collate_fn(self, batch):
        # 原有打包逻辑...
        
        # [新增] 处理描述文本
        if batch[0].get('descriptions') is not None:
            # 打包描述和指示符
            descriptions_batch = pad_descriptions([item['descriptions'] for item in batch])
            has_desc_batch = torch.stack([item['has_description'] for item in batch])
        else:
            descriptions_batch = None
            has_desc_batch = None
        
        return {
            'src_input': src_input_packed,
            'tgt_input': tgt_input_packed,
            'descriptions': descriptions_batch,
            'has_description': has_desc_batch,
            # ... 其他字段
        }

def pad_descriptions(batch_descriptions):
    """
    将不等长的描述列表打包成批量
    """
    max_len = max(len(desc_list) for desc_list in batch_descriptions)
    padded = []
    for desc_list in batch_descriptions:
        padded_item = desc_list + [None] * (max_len - len(desc_list))
        padded.append(padded_item)
    return padded  # List[List[str or None]]
```

**返回结构变化**:
```python
# 原有: tuple (src_input, tgt_input, ...)
# 新增: dict 包含多个字段
{
    'src_input': tensor (B, 4, T, 150),
    'tgt_input': tensor (B, tgt_len),
    'descriptions': List[List[str or None]],  # (B, T)
    'has_description': tensor (B, T, 1),
    'frame_indices': List[List[int]],  # (B, T)
    'sign_id': List[str],
    'sample_path': List[str]
}
```

---

### Step 4: 修改 models.py

**目标**: 集成文本编码和融合到Uni_Sign模型

**修改位置**:
1. **导入** (顶部)
   ```python
   from text_fusion_modules import TextEncoder, GatingFusion, LearnableMaskEmbedding
   ```

2. **Uni_Sign.__init__** 新增文本处理模块
   ```python
   def __init__(self, args):
       super().__init__()
       # 原有初始化...
       
       # [新增] 多模态融合配置
       self.use_descriptions = getattr(args, 'use_descriptions', False)
       self.text_fusion_type = getattr(args, 'text_fusion_type', 'gating')  # 'gating' or 'cross_attn'
       
       if self.use_descriptions:
           self.text_encoder = TextEncoder(
               model_name='mt5-base',
               hidden_dim=768,
               device=args.device if hasattr(args, 'device') else 'cuda'
           )
           
           self.gating_fusion = GatingFusion(feature_dim=768)
           
           # 可学习掩码 (用于缺失描述)
           self.mask_embedding = LearnableMaskEmbedding(hidden_dim=768)
           
           # Text Dropout 概率 (训练时使用)
           self.text_dropout_p = getattr(args, 'text_dropout_p', 0.1)
   ```

3. **Uni_Sign.forward** 集成文本融合
   ```python
   def forward(self, src_input, tgt_input, 
               descriptions=None, has_description=None):
       """
       Args:
           src_input: (B, 4, T, 150)
           tgt_input: (B, tgt_len)
           descriptions: List[List[str or None]] (B, T)
           has_description: (B, T, 1)
       
       Returns:
           logits: (B, tgt_len, vocab_size)
       """
       
       # 原有视觉编码
       pose_features = self._encode_pose(src_input)  # (B, T, 768)
       
       # [新增] 文本编码和融合
       if self.use_descriptions and descriptions is not None:
           # 编码描述文本
           text_features = self._encode_descriptions(descriptions)  # (B, T, 768)
           
           # 应用 Text Dropout (训练时)
           if self.training:
               text_features = self._apply_text_dropout(
                   text_features, 
                   has_description,
                   dropout_p=self.text_dropout_p
               )
           
           # Gating 融合
           fused_features = self.gating_fusion(
               pose_features,
               text_features,
               has_description
           )
           
           encoder_input = fused_features
       else:
           encoder_input = pose_features
       
       # 后续处理 (原有逻辑)
       encoder_out = self.transformer_encoder(encoder_input)
       decoder_out = self.transformer_decoder(tgt_input, encoder_out)
       logits = self.output_projection(decoder_out)
       
       return logits
   
   def _encode_descriptions(self, descriptions):
       """
       对描述文本进行编码
       """
       B, T = len(descriptions), len(descriptions[0])
       text_features = torch.zeros(B, T, 768, device=self.device)
       mask_emb = self.mask_embedding()  # (1, 768)
       
       for b in range(B):
           for t in range(T):
               if descriptions[b][t] is not None:
                   # 编码文本
                   feat = self.text_encoder([descriptions[b][t]])  # (1, 768)
                   text_features[b, t] = feat[0]
               else:
                   # 使用掩码嵌入
                   text_features[b, t] = mask_emb.squeeze(0)
       
       return text_features
   
   def _apply_text_dropout(self, text_features, has_description, dropout_p):
       """
       训练时应用 Text Dropout
       随机替换文本特征为掩码或零向量
       """
       if dropout_p <= 0:
           return text_features
       
       # 创建随机掩码
       B, T, D = text_features.shape
       dropout_mask = torch.bernoulli(
           torch.full((B, T, 1), dropout_p, device=text_features.device)
       ).expand(B, T, D)
       
       text_features = text_features * (1 - dropout_mask)
       return text_features
   ```

**关键设计**:
- mT5 编码器冻结，仅做特征提取
- 可学习掩码用于获取缺失帧的初始表示
- Gating 学习最优权重融合
- Text Dropout 在训练时提高鲁棒性
- 缺失指示符指导 Gating 的学习

---

### Step 5: 修改 fine_tuning.py

**目标**: 支持新的多模态输入

**修改位置**:
1. **数据加载** (main 函数中)
   - 确保 DataLoader 返回新的字典格式
   - 验证 collate_fn 正常工作

2. **训练循环** (train_epoch 函数)
   ```python
   def train_epoch(model, train_dataloader, optimizer, args):
       for batch in train_dataloader:
           src_input = batch['src_input'].to(device)
           tgt_input = batch['tgt_input'].to(device)
           
           # [新增] 处理描述文本
           descriptions = batch.get('descriptions', None)
           has_description = batch.get('has_description', None)
           if has_description is not None:
               has_description = has_description.to(device)
           
           # 前向传播
           if args.use_descriptions:
               logits = model(src_input, tgt_input, 
                            descriptions=descriptions,
                            has_description=has_description)
           else:
               logits = model(src_input, tgt_input)
           
           # 计算loss (原有逻辑)
           loss = criterion(logits, tgt_input)
           
           # 反向传播 (原有逻辑)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
   ```

3. **验证循环** (eval_step 函数)
   - 类似修改，确保推理时正确处理多模态输入
   - 使用 `model.eval()` 禁用 Text Dropout

---

### Step 6: 修改 utils.py & config.py

**目标**: 添加命令行参数和配置

**utils.py 新增参数**:
```python
parser.add_argument('--use_descriptions', 
                    action='store_true', 
                    default=False,
                    help='Enable multimodal fusion with action descriptions')

parser.add_argument('--text_fusion_type', 
                    choices=['gating', 'cross_attn'], 
                    default='gating',
                    help='Fusion mechanism type')

parser.add_argument('--text_dropout_p', 
                    type=float, 
                    default=0.1,
                    help='Text dropout probability during training')

parser.add_argument('--desc_dir', 
                    type=str, 
                    default='description',
                    help='Path to description files')

parser.add_argument('--mt5_model', 
                    type=str, 
                    default='google/mt5-base',
                    help='mT5 model name from HuggingFace')
```

**config.py 新增配置**:
```python
# 描述文本目录路径
DESCRIPTION_DIRS = {
    'CSL_Daily': 'description/CSL_Daily',
    'CSL_News': 'description/CSL_News',
    'How2Sign': 'description/How2Sign',
}

# mT5 模型路径或名称
MT5_MODEL_NAME = 'google/mt5-base'
MT5_HIDDEN_DIM = 768

# 融合参数
TEXT_FUSION_CONFIG = {
    'type': 'gating',                    # 'gating' 或 'cross_attn'
    'text_dropout_p': 0.1,               # Text Dropout 概率
    'mask_embedding_init_std': 0.01,     # 掩码初始化
    'gating_hidden_dim': 512,            # Gating MLP 隐层维度
}

# 时间对齐参数
TEMPORAL_ALIGNMENT_CONFIG = {
    'strategy': 'smart_interpolation',   # 智能插值
    'use_nearest_neighbor': True,        # 邻近帧代替
    'use_linear_interpolation': True,    # 线性插值
}
```

---

## 🔍 关键技术细节

### 1. 时间对齐 (智能插值)

**问题**: 视频被采样 (e.g., 25fps→10fps)，描述对应原始帧号，需要映射到采样后的帧号。

**策略**:
```
对于采样后的帧 i:
  if 帧i的原始帧号 在 descriptions 中:
      使用描述
  elif 最近的帧有描述:
      使用最近邻描述
  else:
      在两侧有描述的帧之间进行线性插值
      (融合两个描述的特征表示)
```

**实现**:
```python
def align(self, descriptions_dict, sampled_frame_indices):
    aligned = []
    has_desc = []
    
    for sampled_idx, original_idx in enumerate(sampled_frame_indices):
        if original_idx in descriptions_dict:
            # 直接使用
            aligned.append(descriptions_dict[original_idx])
            has_desc.append(1)
        else:
            # 查找最近邻
            nearest = find_nearest(original_idx, descriptions_dict.keys())
            if nearest is not None:
                aligned.append(descriptions_dict[nearest])
                has_desc.append(0)  # 标记为插值/邻近
            else:
                aligned.append(None)
                has_desc.append(0)
    
    return aligned, has_desc
```

### 2. 可学习掩码 (缺失占位符)

**设计**:
- 初始化: `mask = nn.Parameter(torch.randn(1, 768) * 0.01)`
- 用途: 为缺失的描述帧提供初始表示
- 训练: 与模型参数一起更新

**应用**:
```python
if descriptions[b][t] is None:
    text_features[b][t] = mask_embedding()
```

### 3. Text Dropout (训练策略)

**目标**: 提高模型对缺失模态的鲁棒性

**实现**:
```python
if training:
    dropout_mask = torch.bernoulli(torch.full((B,T,1), dropout_p))
    text_features = text_features * (1 - dropout_mask)
```

**效果**: 强制模型学习到即使文本特征被部分遮挡，也能进行有效融合

### 4. 缺失指示符 (推理策略)

**设计**:
- `has_description`: (B, T, 1) tensor，1表示有真实描述，0表示插值/缺失
- 用于 Gating 学习哪些帧更应该依赖文本

**在Gating中使用**:
```python
gate_input = torch.cat([pose_feat, text_feat, has_description], dim=-1)
gate = Sigmoid(MLP(gate_input))
fused = pose_feat + gate * text_feat
```

---

## 📊 预期效果

| 指标 | Phase 1 (Gating) | Phase 2 (Cross-Attn) |
|------|------------------|----------------------|
| **BLEU提升** | +3-5% | +5-8% |
| **模型参数** | +10M | +50M |
| **推理速度** | -3% | -8~15% |
| **显存占用** | +2GB | +4GB |
| **训练时间** | +20% | +40% |

---

## ✅ 修改验证清单

### Phase 1: 数据加载验证
- [ ] temporal_alignment.py 能正确加载描述文本
- [ ] 帧索引映射正确 (no off-by-one)
- [ ] 缺失描述的样本不报错
- [ ] collate_fn 正确打包批量数据

### Phase 2: 模型集成验证
- [ ] text_fusion_modules.py 中 mT5 编码器能推理
- [ ] 掩码嵌入生成形状正确
- [ ] Gating Fusion forward pass 无形状错误
- [ ] 梯度能正确反向传播

### Phase 3: 训练循环验证
- [ ] 单batch训练无OOM
- [ ] Loss 正常下降
- [ ] 评估指标正常计算
- [ ] 向后兼容: `--use_descriptions=False` 时功能不受影响

### Phase 4: 推理验证
- [ ] Text Dropout 在 eval 模式下禁用
- [ ] 缺失指示符正确指导融合权重
- [ ] 推理速度在预期范围内

---

## 🚀 命令行使用示例

### 启用描述文本的训练命令
```bash
# Gating 融合 (推荐首选)
deepspeed --include localhost:0,1,2,3 fine_tuning.py \
    --batch-size 16 --epochs 20 --dataset CSL_Daily \
    --use_descriptions \
    --text_fusion_type gating \
    --text_dropout_p 0.1 \
    --rgb_support \
    --finetune out/stage2/best.pth

# Cross-Attention 融合 (高性能，Phase 2)
deepspeed --include localhost:0,1,2,3 fine_tuning.py \
    --batch-size 8 --epochs 20 --dataset CSL_Daily \
    --use_descriptions \
    --text_fusion_type cross_attn \
    --text_dropout_p 0.15 \
    --rgb_support \
    --finetune out/stage2/best.pth
```

### 禁用描述文本的训练命令 (向后兼容)
```bash
deepspeed --include localhost:0,1,2,3 fine_tuning.py \
    --batch-size 16 --epochs 20 --dataset CSL_Daily \
    --rgb_support \
    --finetune out/stage2/best.pth
# 输出应与原有模型相同
```

---

## 📚 文件关联关系图

```
temporal_alignment.py
    ├─ DescriptionLoader (读取description/)
    └─ TemporalAligner (智能插值)
         ↓ (被调用)
datasets.py
    ├─ S2T_Dataset.__init__ (初始化加载器)
    ├─ S2T_Dataset.__getitem__ (加载和对齐描述)
    └─ collate_fn (打包批量数据)
         ↓ (向上游提供多模态数据)
text_fusion_modules.py
    ├─ TextEncoder (mT5编码)
    ├─ GatingFusion (融合)
    └─ LearnableMaskEmbedding (掩码)
         ↓ (被模型调用)
models.py
    └─ Uni_Sign (集成编码器和融合)
         ↓ (被训练循环调用)
fine_tuning.py
    └─ train_epoch / eval_step (训练和验证)
         ↓ (参数由以下提供)
utils.py & config.py (命令行和配置)
```

---

## 📌 重要提醒

1. **向后兼容**: `--use_descriptions=False` 时，整个描述处理路径应被跳过，模型表现应与原有相同
2. **显存管理**: mT5 和 Gating 可能增加显存占用，建议从小batch_size开始
3. **Text Dropout**: 仅在训练时应用，推理时需禁用
4. **缺失处理**: 保证所有代码路径都能处理 `descriptions=None` 的情况
5. **测试优先**: 每步修改后都应进行单独的单元测试

---

## 🔗 关联文档参考

- Stage 3 总体: `mds/README_OPTIMIZATION.md`
- 设计细节: `mds/MULTIMODAL_FUSION_DESIGN.md`
- 伪代码参考: `mds/PSEUDOCODE_REFERENCE.md`
- 执行指南: `mds/EXECUTION_SUMMARY.md`

**下一步**: 等待修改指令的细化或逐个代码实现的具体要求。
