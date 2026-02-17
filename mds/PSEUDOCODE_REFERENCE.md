# Stage 3 实施 - 代码伪代码参考

**用途**: 参考伪代码，指导实际代码编写  
**说明**: 包含所有 4 个文件的关键代码段逻辑

---

## 📄 文件 1: datasets.py - 数据加载改动

### 代码段 1: 描述加载器

```python
# ============================================================================
# 代码位置: datasets.py 中新增类
# 功能: 从 JSON 文件加载和管理文本描述
# ============================================================================

class DescriptionLoader:
    """加载和检索文本描述"""
    
    def __init__(self, description_root_dir):
        self.descriptions = {}  # {video_key: {frame_id: description_text}}
        self._load_all_descriptions(description_root_dir)
    
    def _load_all_descriptions(self, root_dir):
        """遍历所有 JSON 文件并加载"""
        for phase in ['train', 'dev', 'test']:  # 三个阶段
            phase_dir = f"{root_dir}/{phase}"
            # 伪代码: 遍历 phase_dir 下的所有 JSON 文件
            for json_file in glob(f"{phase_dir}/*.json"):
                video_id = extract_video_id(json_file)
                video_key = f"{phase}/{video_id}"
                
                # 读取 JSON
                data = json.load(json_file)  # [{filename, description}, ...]
                
                # 构建帧-描述映射
                self.descriptions[video_key] = {}
                for entry in data:
                    frame_id = entry['filename'].split('.')[0]  # "000000"
                    self.descriptions[video_key][frame_id] = entry['description']
    
    def get_description(self, video_key, frame_id):
        """获取特定帧的描述"""
        # 伪代码:
        if video_key not in self.descriptions:
            return None
        
        frame_key = format_frame_id(frame_id)  # "000000" 格式
        return self.descriptions[video_key].get(frame_key, None)
```

### 代码段 2: 时间对齐器

```python
# ============================================================================
# 代码位置: datasets.py 中新增类
# 功能: 处理帧采样导致的时间对齐问题
# ============================================================================

class TemporalAligner:
    """智能时间对齐"""
    
    def __init__(self, strategy='intelligent_interpolation'):
        self.strategy = strategy
    
    def align_descriptions(self, frame_indices, description_func):
        """
        将描述对齐到采样后的帧位置
        
        输入:
          frame_indices: [10, 25, 50, ...] 采样帧的原始索引
          description_func: callable(frame_id) -> str or None
        
        输出:
          aligned_descs: [desc_at_10, desc_at_25, ...]
        """
        # 伪代码:
        aligned = []
        for frame_idx in frame_indices:
            desc = description_func(frame_idx)
            
            if desc is not None:
                # 情况 1: 直接有描述
                aligned.append(desc)
            else:
                # 情况 2: 无描述，寻找最近的
                nearest_desc = self._find_nearest_description(
                    frame_idx, 
                    frame_indices, 
                    description_func
                )
                aligned.append(nearest_desc)
        
        return aligned
    
    def _find_nearest_description(self, frame_idx, frame_indices, desc_func):
        """寻找最近的有描述的帧"""
        # 伪代码:
        best_desc = None
        min_distance = float('inf')
        
        for search_idx in range(0, 1000):  # 向两边搜索
            # 尝试左边
            left_idx = frame_idx - search_idx
            if left_idx >= 0:
                desc = desc_func(left_idx)
                if desc is not None and search_idx < min_distance:
                    best_desc = desc
                    min_distance = search_idx
            
            # 尝试右边
            right_idx = frame_idx + search_idx
            desc = desc_func(right_idx)
            if desc is not None and search_idx < min_distance:
                best_desc = desc
                min_distance = search_idx
            
            if best_desc is not None:
                break
        
        return best_desc
```

### 代码段 3: S2T_Dataset 修改

```python
# ============================================================================
# 代码位置: 修改 S2T_Dataset 类
# 改动: __init__() 和 __getitem__()
# ============================================================================

class S2T_Dataset:
    
    def __init__(self, ..., use_descriptions=True, text_dropout_rate=0.3):
        # 原有初始化代码 ...
        
        # [新增]
        self.use_descriptions = use_descriptions
        self.text_dropout_rate = text_dropout_rate
        
        # [新增] 初始化描述加载器
        if use_descriptions:
            self.desc_loader = DescriptionLoader(
                'description/CSL-Daily/split_data'
            )
            self.temporal_aligner = TemporalAligner(
                strategy='intelligent_interpolation'
            )
    
    def __getitem__(self, idx):
        name = self.data[idx]
        
        # 伪代码: 加载视频特征
        pose_dict = self.load_pose(name)
        pose_sample = pose_dict['pose']
        frame_indices = pose_dict['__frame_indices__']
        
        # 伪代码: 加载文本目标
        text = self.load_text(name)
        gloss = self.load_gloss(name) if self.load_gloss else None
        rgb_dict = self.load_rgb(name)
        
        # [新增] 加载描述
        description = None
        has_description = False
        
        if self.use_descriptions:
            video_key = f"{self.phase}/{name}"
            
            # 定义获取描述的函数
            def get_desc(frame_idx):
                return self.desc_loader.get_description(video_key, frame_idx)
            
            # 时间对齐
            aligned_descs = self.temporal_aligner.align_descriptions(
                frame_indices=frame_indices,
                description_func=get_desc
            )
            
            # 合并描述
            valid_descs = [d for d in aligned_descs if d is not None]
            if valid_descs:
                description = " ".join(valid_descs)
                has_description = True
        
        # 返回增强的批数据
        return {
            'name': name,
            'pose_sample': pose_sample,
            'text': text,
            'gloss': gloss,
            'rgb_dict': rgb_dict,
            'description': description,  # [新增]
            'has_description': has_description,  # [新增]
            'frame_indices': frame_indices  # [新增]
        }
    
    def load_pose(self, name):
        """伪代码: 加载并采样姿态，记录帧索引"""
        # 从 pickle 加载
        pose_feat = load_pickle(f"pose_path/{name}.pkl")
        duration = len(pose_feat)
        
        if self.max_length >= duration:
            pose_sample = pose_feat
            frame_indices = list(range(duration))
        else:
            # 随机采样
            indices = np.random.choice(duration, self.max_length, replace=False)
            indices = np.sort(indices)  # 排序保证顺序
            pose_sample = pose_feat[indices]
            frame_indices = indices.tolist()  # [新增]
        
        return {
            'pose': pose_sample,
            '__frame_indices__': frame_indices  # [新增]
        }
```

---

## 📄 文件 2: models.py - 模型改动

### 代码段 1: 文本编码器

```python
# ============================================================================
# 代码位置: models.py 中新增类
# 功能: 使用 mT5-base 编码文本描述
# ============================================================================

class TextEncoder(nn.Module):
    """mT5-base 文本编码器"""
    
    def __init__(self, model_name='mt5-base', hidden_dim=768):
        super().__init__()
        
        # 伪代码: 加载预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 冻结参数 (不训练 mT5)
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, descriptions, max_length=256):
        """
        输入: descriptions (list of str)
        输出: text_features (B, 768)
        """
        # 伪代码:
        if descriptions is None or all(d is None for d in descriptions):
            # 全是 None，返回零向量或使用 mask
            return None
        
        # Tokenize and encode
        encoded = self.tokenizer(
            descriptions,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.encoder(**encoded)
            # 取 [CLS] token
            text_features = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        
        return text_features
```

### 代码段 2: Gating 融合模块

```python
# ============================================================================
# 代码位置: models.py 中新增类
# 功能: 融合视频和文本特征
# ============================================================================

class GatingFusion(nn.Module):
    """Gating 融合机制"""
    
    def __init__(self, feature_dim=768):
        super().__init__()
        
        # Gating MLP: [pose(768), text(768), indicator(1)] → gate(1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 范围 [0, 1]
        )
    
    def forward(self, pose_feat, text_feat, has_text_indicator):
        """
        输入:
          pose_feat: (B, T, 768)
          text_feat: (B, T, 768)
          has_text_indicator: (B, 1) 或 (B, T, 1)
        
        输出:
          fused_feat: (B, T, 768)
        """
        # 伪代码:
        B, T, D = pose_feat.shape
        
        # 确保 indicator 形状为 (B, T, 1)
        if has_text_indicator.dim() == 2:
            has_text_indicator = has_text_indicator.unsqueeze(1).expand(B, T, 1)
        
        # 拼接特征 (批处理)
        combined = torch.cat([pose_feat, text_feat, has_text_indicator], dim=2)
        # shape: (B, T, 1537)
        
        # Reshape 为 2D 以通过 MLP
        combined_flat = combined.view(B*T, -1)  # (B*T, 1537)
        
        # 计算 gate
        gate_flat = self.gate_mlp(combined_flat)  # (B*T, 1)
        gate = gate_flat.view(B, T, 1)  # (B, T, 1)
        
        # 融合: fused = pose + gate * text
        fused_feat = pose_feat + gate * text_feat
        
        return fused_feat
```

### 代码段 3: Uni_Sign 改动

```python
# ============================================================================
# 代码位置: 修改 Uni_Sign 类的 __init__() 和 forward()
# ============================================================================

class Uni_Sign(nn.Module):
    
    def __init__(self, ..., use_description=True):
        super().__init__()
        
        # 原有初始化 ...
        
        # [新增] 文本处理模块
        self.use_description = use_description
        
        if use_description:
            self.text_encoder = TextEncoder(
                model_name='mt5-base',
                hidden_dim=768
            )
            
            # 可学习掩码嵌入
            self.mask_embedding = nn.Parameter(
                torch.randn(1, 768) * 0.01
            )
            
            self.gating_fusion = GatingFusion(feature_dim=768)
    
    def forward(self, src_input, tgt_input, 
                description=None, has_description=None):
        """
        输入:
          src_input: (B, 4, T, 150) 姿态特征
          tgt_input: (B, text_len) 目标文本
          description: list of str (或 list of None)
          has_description: (B,) torch.bool
        
        输出:
          output: 模型输出
        """
        # 伪代码:
        
        # 第 1 步: 提取视频特征
        pose_features = self.encode_pose(src_input)  # (B, T, 768)
        
        # [新增] 第 2 步: 融合文本特征
        if self.use_description and description is not None:
            # 2.1 编码文本
            text_features = self.text_encoder(description)  # (B, 768) or None
            
            # 2.2 处理缺失，生成 (B, T, 768) 的文本特征
            B, T, _ = pose_features.shape
            text_features_t = torch.zeros(B, T, 768)
            
            for b in range(B):
                if text_features is not None and has_description[b]:
                    # 有真实文本: 复制到所有时间步
                    text_features_t[b] = text_features[b].unsqueeze(0).expand(T, -1)
                else:
                    # 无文本: 使用掩码
                    text_features_t[b] = self.mask_embedding.expand(T, -1)
            
            # 2.3 创建缺失指示符
            has_text_indicator = has_description.float().unsqueeze(1)  # (B, 1)
            
            # 2.4 融合
            fused_features = self.gating_fusion(
                pose_features,
                text_features_t,
                has_text_indicator
            )  # (B, T, 768)
        else:
            fused_features = pose_features
        
        # 第 3 步: 后续处理 (原有逻辑)
        output = self.decode(fused_features, tgt_input)
        
        return output
```

---

## 📄 文件 3: fine_tuning.py - 训练改动

### 代码段 1: Text Dropout 实现

```python
# ============================================================================
# 代码位置: fine_tuning.py 中的 train_one_epoch() 函数
# 功能: 应用 Text Dropout 正则化
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    text_dropout_rate=0.3):
    """伪代码: 训练一个 epoch，含 Text Dropout"""
    
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 解包批数据
        pose = batch['pose_sample'].to(device)
        text = batch['text'].to(device)
        gloss = batch['gloss'].to(device)
        description = batch['description']  # list of str
        has_description = batch['has_description'].to(device)  # (B,)
        
        # [新增] ===== Text Dropout 应用 =====
        description_after_dropout = []
        has_description_after_dropout = []
        
        for b in range(len(description)):
            if has_description[b].item():
                # 该样本有描述
                if torch.rand(1).item() < text_dropout_rate:
                    # 以 text_dropout_rate 的概率丢弃
                    description_after_dropout.append(None)
                    has_description_after_dropout.append(False)
                else:
                    # 保留描述
                    description_after_dropout.append(description[b])
                    has_description_after_dropout.append(True)
            else:
                # 本身无描述
                description_after_dropout.append(None)
                has_description_after_dropout.append(False)
        
        has_description_after_dropout = torch.tensor(
            has_description_after_dropout,
            dtype=torch.bool
        ).to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        outputs = model(
            src_input=pose,
            tgt_input=text,
            description=description_after_dropout,  # [新增]
            has_description=has_description_after_dropout  # [新增]
        )
        
        # 损失和反向传播 (原有逻辑)
        loss = criterion(outputs, gloss)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 代码段 2: DataLoader 和 Collate 函数

```python
# ============================================================================
# 代码位置: fine_tuning.py 中的数据加载部分
# 功能: 创建带自定义 collate 的 DataLoader
# ============================================================================

def create_dataloaders(batch_size, num_workers=4):
    """伪代码: 创建数据加载器"""
    
    # 创建数据集
    train_dataset = S2T_Dataset(
        phase='train',
        use_descriptions=True,  # [新增]
        text_dropout_rate=0.3  # [新增]
    )
    
    dev_dataset = S2T_Dataset(
        phase='dev',
        use_descriptions=True  # [新增]
    )
    
    # [新增] 自定义 collate 函数
    def custom_collate_fn(batch):
        """处理不同类型的数据字段"""
        collated = {}
        
        # Tensor 字段
        tensor_fields = ['pose_sample', 'text', 'gloss']
        for field in tensor_fields:
            collated[field] = torch.stack([b[field] for b in batch])
        
        # 列表字段
        collated['name'] = [b['name'] for b in batch]
        collated['description'] = [b['description'] for b in batch]
        collated['has_description'] = torch.tensor(
            [b['has_description'] for b in batch],
            dtype=torch.bool
        )
        
        # 字典字段 (RGB)
        if 'rgb_dict' in batch[0]:
            collated['rgb_dict'] = {
                k: torch.stack([b['rgb_dict'][k] for b in batch])
                for k in batch[0]['rgb_dict'].keys()
            }
        
        return collated
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,  # [新增]
        num_workers=num_workers
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,  # [新增]
        num_workers=num_workers
    )
    
    return train_loader, dev_loader
```

---

## 📄 文件 4: inference.py - 推理改动

### 代码段 1: 推理函数

```python
# ============================================================================
# 代码位置: inference.py 或 test.py 中的推理函数
# 功能: 包含文本描述的推理
# ============================================================================

def inference(model, video_path, description_db=None, use_description=True, device='cuda'):
    """伪代码: 单个视频推理"""
    
    model.eval()
    
    # 1. 提取视频特征
    pose_features = extract_pose_features(video_path)  # (1, T, 150)
    pose_features = pose_features.to(device)
    
    # [新增] 2. 加载或生成文本描述
    description = None
    has_description = False
    
    if use_description:
        video_id = extract_video_id(video_path)
        
        # 从数据库或文件加载描述
        if description_db is not None:
            description = description_db.get(video_id, None)
        else:
            description_path = f"description/{video_id}.json"
            if os.path.exists(description_path):
                data = json.load(description_path)
                descriptions = [d['description'] for d in data]
                description = " ".join(descriptions)
        
        if description is not None:
            has_description = True
    
    # 3. 前向传播
    with torch.no_grad():
        outputs = model(
            src_input=pose_features,
            tgt_input=None,
            description=[description] if description else [None],  # (1,)
            has_description=torch.tensor([has_description], dtype=torch.bool)
        )
    
    # 4. 解码输出
    translation_result = decode_output(outputs)
    
    return translation_result


def batch_inference(model, video_list, description_db=None, 
                   use_description=True, device='cuda'):
    """伪代码: 批量推理"""
    
    results = {}
    
    for video_path in video_list:
        result = inference(
            model, video_path, description_db, use_description, device
        )
        results[video_path] = result
    
    return results
```

### 代码段 2: 一致性验证函数

```python
# ============================================================================
# 代码位置: inference.py 中的验证函数
# 功能: 验证推理的一致性 (有/无文本差异)
# ============================================================================

def verify_inference_consistency(model, test_video, reference_text, 
                                description_db, device='cuda'):
    """伪代码: 验证一致性"""
    
    model.eval()
    
    # 1. 提取特征和描述
    pose_features = extract_pose_features(test_video)
    video_id = extract_video_id(test_video)
    description = description_db.get(video_id, None)
    
    if description is None:
        print(f"Skip: no description for {video_id}")
        return None
    
    # 2. 推理：有文本版本
    with torch.no_grad():
        outputs_with = model(
            src_input=pose_features.to(device),
            description=[description],
            has_description=torch.tensor([True])
        )
        pred_with = decode_output(outputs_with)
        bleu_with = compute_bleu(pred_with, reference_text)
    
    # 3. 推理：无文本版本 (丢弃描述)
    with torch.no_grad():
        outputs_without = model(
            src_input=pose_features.to(device),
            description=[None],
            has_description=torch.tensor([False])
        )
        pred_without = decode_output(outputs_without)
        bleu_without = compute_bleu(pred_without, reference_text)
    
    # 4. 计算一致性指标
    delta_bleu = abs(bleu_with - bleu_without)
    
    # KL 散度 (输出分布)
    probs_with = torch.softmax(outputs_with, dim=-1)
    probs_without = torch.softmax(outputs_without, dim=-1)
    kl = torch.nn.functional.kl_div(
        probs_without.log(),
        probs_with,
        reduction='mean'
    ).item()
    
    # 5. 判断是否通过
    consistency_pass = (delta_bleu < 0.02) and (kl < 0.1)
    
    return {
        'video_id': video_id,
        'bleu_with_text': bleu_with,
        'bleu_without_text': bleu_without,
        'delta_bleu': delta_bleu,
        'kl_divergence': kl,
        'consistency_pass': consistency_pass
    }
```

---

## 🔑 关键代码模式总结

### 模式 1: 描述的三态处理

```python
if has_description:
    # 情况 A: 有文本
    text_feature = text_encoder(description)  # (B, 768)
else:
    # 情况 B: 无文本，使用掩码
    text_feature = mask_embedding.expand(B, -1)  # (B, 768)

# 两种情况统一返回 (B, 768) 的特征
```

### 模式 2: 时间维度处理

```python
# 文本特征的时间扩展
B, T = pose_features.shape[0:2]

text_feature_t = torch.zeros(B, T, 768)
for b in range(B):
    # 对每个样本
    if has_description[b]:
        # 扩展文本特征到所有时间步
        text_feature_t[b] = text_features[b].unsqueeze(0).expand(T, -1)
    else:
        text_feature_t[b] = mask_embedding.expand(T, -1)
```

### 模式 3: Gating 融合

```python
# 三个信号输入
signal = torch.cat([pose_feat, text_feat, has_text_indicator], dim=-1)

# 计算权重
gate = self.gate_mlp(signal)  # 范围 [0, 1]

# 加权融合
fused = pose_feat + gate * text_feat
# 或: fused = (1 - gate) * pose_feat + gate * text_feat
```

### 模式 4: Text Dropout

```python
# 训练时
if self.training:
    for b in range(batch_size):
        if has_description[b] and torch.rand(1) < dropout_rate:
            description[b] = None  # 丢弃
            has_description[b] = False

# 推理时
# 不应用 dropout，正常使用
```

---

## 💡 实施建议

### 优先级顺序

**第 1 优先级** (必须实施):
1. DescriptionLoader - 数据加载
2. TemporalAligner - 时间对齐
3. 修改 S2T_Dataset - 集成描述

**第 2 优先级** (核心功能):
4. TextEncoder - 文本编码
5. GatingFusion - 融合机制
6. Learnable Mask - 缺失处理

**第 3 优先级** (训练优化):
7. Text Dropout - 正则化
8. 自定义 Collate 函数
9. 修改 train_one_epoch()

**第 4 优先级** (推理和验证):
10. 推理函数改动
11. 一致性验证函数

---

**提示**: 逐个实施上述代码段，完成每个模块后进行单元测试，确保数据流畅通。
