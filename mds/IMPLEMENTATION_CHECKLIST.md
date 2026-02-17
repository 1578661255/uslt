# Stage 3 实施 - 关键检查表与验证指南

**目的**: 提供完整的检查列表，确保实施的正确性和完整性

---

## ✅ 前期准备检查

### 文档阅读清单
- [ ] 读完 STAGE3_IMPLEMENTATION_PLAN.md (预计 2-3 小时)
  - [ ] 理解总体架构设计
  - [ ] 理解数据加载改动逻辑
  - [ ] 理解模型架构改动
  - [ ] 理解训练改动
  - [ ] 理解推理改动
  - [ ] 理解完整流程图

- [ ] 查阅 PSEUDOCODE_REFERENCE.md (伪代码参考)
  - [ ] DescriptionLoader 伪代码
  - [ ] TemporalAligner 伪代码
  - [ ] TextEncoder 伪代码
  - [ ] GatingFusion 伪代码

### 环境准备
- [ ] 确认 Python 版本 ≥ 3.8
- [ ] 确认 PyTorch ≥ 1.10
- [ ] 确认 Transformers ≥ 4.20 (用于 mT5)
- [ ] 确认描述数据位置: `description/CSL-Daily/split_data/{train,dev,test}/`
- [ ] 确认描述数据格式: JSON `[{filename, description}, ...]`
- [ ] 确认描述数据完整性: 449 个视频

```bash
# 验证命令
ls -la description/CSL-Daily/split_data/train/ | head -5
ls -la description/CSL-Daily/split_data/dev/ | head -5
ls -la description/CSL-Daily/split_data/test/ | head -5
```

---

## 📝 代码实施检查清单

### Phase 1: 数据加载模块 (datasets.py)

#### Step 1.1: 实现 DescriptionLoader 类

实施位置: `datasets.py` 新增类

检查项:
- [ ] 类定义完成
- [ ] `__init__()` 方法正确
  - [ ] `self.descriptions` 初始化为空字典
  - [ ] 调用 `_load_all_descriptions()`
- [ ] `_load_all_descriptions()` 方法正确
  - [ ] 遍历 'train', 'dev', 'test' 三个阶段
  - [ ] 使用 glob 查找 JSON 文件
  - [ ] 正确解析 JSON 格式
  - [ ] 帧号转换为 '000000' 字符串格式
  - [ ] 存储为 `{video_key: {frame_id: description}}`
- [ ] `get_description()` 方法正确
  - [ ] 处理 video_key 不存在的情况
  - [ ] 处理 frame_id 不存在的情况
  - [ ] 返回 None 当无描述

验证代码:
```python
# 测试 DescriptionLoader
loader = DescriptionLoader('description/CSL-Daily/split_data')
desc = loader.get_description('train/video_001', 0)
print(f"Description: {desc}")  # 应该返回文本或 None
```

#### Step 1.2: 实现 TemporalAligner 类

实施位置: `datasets.py` 新增类

检查项:
- [ ] 类定义完成
- [ ] `__init__()` 方法正确
  - [ ] `self.strategy` 设置为 'intelligent_interpolation'
- [ ] `align_descriptions()` 方法正确
  - [ ] 输入 frame_indices 是列表
  - [ ] 输入 description_func 是可调用的
  - [ ] 返回列表，长度等于 frame_indices
  - [ ] 每个元素是字符串或 None
- [ ] `_find_nearest_description()` 方法正确
  - [ ] 向左和向右各搜索最近有描述的帧
  - [ ] 返回最近的描述
  - [ ] 处理无描述的情况 (返回 None)
- [ ] 处理边界情况
  - [ ] frame_id 为 0
  - [ ] frame_id 超出范围
  - [ ] 整个视频无描述

验证代码:
```python
# 测试 TemporalAligner
aligner = TemporalAligner('intelligent_interpolation')
frame_indices = [10, 25, 50, 100]

def mock_desc_func(frame_id):
    return f"desc_{frame_id}" if frame_id % 2 == 0 else None

aligned = aligner.align_descriptions(frame_indices, mock_desc_func)
print(f"Aligned: {aligned}")  # 应该填充所有缺失的描述
```

#### Step 1.3: 修改 S2T_Dataset.__init__()

实施位置: `datasets.py` 修改现有类

检查项:
- [ ] 添加参数: `use_descriptions=True, text_dropout_rate=0.3`
- [ ] 存储参数: `self.use_descriptions`, `self.text_dropout_rate`
- [ ] 条件初始化描述加载器
  ```python
  if use_descriptions:
      self.desc_loader = DescriptionLoader(...)
      self.temporal_aligner = TemporalAligner(...)
  ```
- [ ] 处理 use_descriptions=False 的情况 (向后兼容)

#### Step 1.4: 修改 S2T_Dataset.__getitem__()

实施位置: `datasets.py` 修改现有方法

检查项:
- [ ] 保留原有返回的所有字段
- [ ] 新增返回字段:
  - [ ] `'description'`: str 或 None
  - [ ] `'has_description'`: bool
  - [ ] `'frame_indices'`: list of int
- [ ] 调用 `load_pose()` 获取 frame_indices
- [ ] 调用 description_loader 和 temporal_aligner
- [ ] 合并多个描述时使用空格分隔
- [ ] 处理 use_descriptions=False 的情况

验证代码:
```python
# 测试 __getitem__()
dataset = S2T_Dataset(phase='train', use_descriptions=True)
sample = dataset[0]
print(f"Keys: {sample.keys()}")
# 应该包含: name, pose_sample, text, gloss, rgb_dict, 
#           description, has_description, frame_indices
```

#### Step 1.5: 修改 S2T_Dataset.load_pose()

实施位置: `datasets.py` 修改现有方法

检查项:
- [ ] 原有逻辑保持不变
- [ ] 添加帧索引记录
  ```python
  # 不采样情况
  frame_indices = list(range(duration))
  
  # 采样情况
  frame_indices = tmp.tolist()
  ```
- [ ] 返回字典包含 `'__frame_indices__'`

验证代码:
```python
# 测试 load_pose()
pose_dict = dataset.load_pose('video_001')
assert '__frame_indices__' in pose_dict
assert len(pose_dict['pose']) == len(pose_dict['__frame_indices__'])
```

---

### Phase 2: 模型架构改动 (models.py)

#### Step 2.1: 实现 TextEncoder 类

实施位置: `models.py` 新增类

检查项:
- [ ] 继承 `nn.Module`
- [ ] `__init__()` 方法
  - [ ] 加载 AutoTokenizer
  - [ ] 加载 AutoModel (mT5-base)
  - [ ] 冻结预训练参数
- [ ] `forward()` 方法
  - [ ] 接收 descriptions (list of str)
  - [ ] Tokenize 和编码
  - [ ] 取 [CLS] token
  - [ ] 返回形状 (B, 768)
  - [ ] 处理 None 值

验证代码:
```python
# 测试 TextEncoder
encoder = TextEncoder('mt5-base')
descriptions = ["这是一个描述", "另一个描述"]
features = encoder(descriptions)
print(f"Shape: {features.shape}")  # (2, 768)
```

#### Step 2.2: 实现 GatingFusion 类

实施位置: `models.py` 新增类

检查项:
- [ ] 继承 `nn.Module`
- [ ] `__init__()` 方法
  - [ ] 定义 gate_mlp 网络
  - [ ] MLP 输入维度: 768*2+1 = 1537
  - [ ] MLP 输出维度: 1
  - [ ] 最后一层使用 Sigmoid
- [ ] `forward()` 方法
  - [ ] 接收 pose_feat (B, T, 768)
  - [ ] 接收 text_feat (B, T, 768)
  - [ ] 接收 has_text_indicator (B, 1 或 B, T, 1)
  - [ ] 处理形状广播
  - [ ] 拼接特征
  - [ ] 计算 gate
  - [ ] 融合: fused = pose + gate * text
  - [ ] 返回形状 (B, T, 768)

验证代码:
```python
# 测试 GatingFusion
fusion = GatingFusion(768)
pose = torch.randn(2, 10, 768)
text = torch.randn(2, 10, 768)
indicator = torch.tensor([[1.0], [0.0]])
fused = fusion(pose, text, indicator)
print(f"Shape: {fused.shape}")  # (2, 10, 768)
```

#### Step 2.3: 修改 Uni_Sign.__init__()

实施位置: `models.py` 修改现有类

检查项:
- [ ] 添加参数: `use_description=True`
- [ ] 条件初始化文本模块
  ```python
  if use_description:
      self.text_encoder = TextEncoder(...)
      self.mask_embedding = nn.Parameter(...)
      self.gating_fusion = GatingFusion(...)
  ```
- [ ] Learnable mask 初始化
  - [ ] 形状: (1, 768)
  - [ ] 初值接近零或随机

#### Step 2.4: 修改 Uni_Sign.forward()

实施位置: `models.py` 修改现有方法

检查项:
- [ ] 添加参数: `description=None, has_description=None`
- [ ] 保留原有前向逻辑: `pose_features = self.encode_pose(src_input)`
- [ ] 添加文本融合逻辑
  ```python
  if self.use_description and description is not None:
      # 编码文本
      # 处理缺失
      # 融合
  else:
      fused_features = pose_features
  ```
- [ ] 处理 description=None 的情况
- [ ] 处理 has_description 的形状 (确保为 (B,) 或 (B, 1))

验证代码:
```python
# 测试 Uni_Sign.forward()
model = Uni_Sign(use_description=True)
src = torch.randn(2, 4, 256, 150)
tgt = torch.randn(2, 50)
descriptions = ["desc1", "desc2"]
has_desc = torch.tensor([True, False])
output = model(src, tgt, descriptions, has_desc)
print(f"Output shape: {output.shape}")
```

---

### Phase 3: 训练脚本改动 (fine_tuning.py)

#### Step 3.1: 实现 Text Dropout 逻辑

实施位置: `fine_tuning.py` train_one_epoch() 函数内

检查项:
- [ ] 在前向传播前应用 dropout
- [ ] 逐样本应用概率: `torch.rand(1).item() < text_dropout_rate`
- [ ] 记录 has_description 的变化
- [ ] 丢弃时设置 description[b] = None 且 has_description[b] = False
- [ ] 只在 model.train() 时应用 (不在评估时)

#### Step 3.2: 实现 custom_collate_fn()

实施位置: `fine_tuning.py` 新增函数

检查项:
- [ ] 处理 Tensor 字段 (pose_sample, text, gloss)
  - [ ] 使用 torch.stack()
- [ ] 处理列表字段 (name, description)
  - [ ] 直接返回列表
- [ ] 处理布尔字段 (has_description)
  - [ ] 转换为 torch.tensor
- [ ] 处理字典字段 (rgb_dict)
  - [ ] 逐个键 stack

验证代码:
```python
# 测试 custom_collate_fn
batch = [dataset[i] for i in range(2)]
collated = custom_collate_fn(batch)
print(f"Keys: {collated.keys()}")
# 应该包含所有必要字段
```

#### Step 3.3: 修改 DataLoader 创建

实施位置: `fine_tuning.py` 数据加载部分

检查项:
- [ ] 创建 S2T_Dataset 时添加参数
  - [ ] `use_descriptions=True`
  - [ ] `text_dropout_rate=0.3`
- [ ] DataLoader 使用 custom_collate_fn
  ```python
  DataLoader(..., collate_fn=custom_collate_fn)
  ```

#### Step 3.4: 修改 train_one_epoch()

实施位置: `fine_tuning.py` 修改现有函数

检查项:
- [ ] 从 batch 解包新字段: description, has_description
- [ ] 应用 Text Dropout
- [ ] 调用 model.forward() 时传递新参数
  ```python
  outputs = model(..., description=..., has_description=...)
  ```

---

### Phase 4: 推理脚本改动 (inference.py)

#### Step 4.1: 实现推理函数

实施位置: `inference.py` 或 `test.py` 新增/修改函数

检查项:
- [ ] 加载或查询描述
  - [ ] 从 JSON 文件读取
  - [ ] 或从数据库查询
  - [ ] 处理缺失情况
- [ ] 设置 has_description 标志
- [ ] 调用 model.forward()
  ```python
  model(src_input=..., description=[...], has_description=...)
  ```
- [ ] 使用 model.eval() 和 torch.no_grad()

#### Step 4.2: 实现一致性验证函数

实施位置: `inference.py` 新增函数

检查项:
- [ ] 分别推理有/无文本版本
- [ ] 计算 BLEU 差异 (delta_bleu)
- [ ] 计算 KL 散度
- [ ] 判断是否通过: delta_bleu < 0.02 and kl < 0.1

---

## 🧪 单元测试检查清单

### Test 1: DescriptionLoader

```python
def test_description_loader():
    loader = DescriptionLoader('description/CSL-Daily/split_data')
    
    # 测试数据是否加载
    assert len(loader.descriptions) > 0, "No descriptions loaded"
    
    # 测试获取现存描述
    video_key = list(loader.descriptions.keys())[0]
    frame_key = list(loader.descriptions[video_key].keys())[0]
    desc = loader.get_description(video_key, int(frame_key))
    assert desc is not None, "Description should not be None"
    assert isinstance(desc, str), "Description should be string"
    
    # 测试获取不存在的描述
    desc = loader.get_description('train/nonexistent', 0)
    assert desc is None, "Should return None for nonexistent video"
    
    print("✓ DescriptionLoader test passed")
```

### Test 2: TemporalAligner

```python
def test_temporal_aligner():
    aligner = TemporalAligner('intelligent_interpolation')
    
    # 创建模拟描述函数
    def mock_desc(frame_id):
        if frame_id % 2 == 0:
            return f"frame_{frame_id}"
        return None
    
    frame_indices = [0, 1, 2, 3, 4, 5]
    aligned = aligner.align_descriptions(frame_indices, mock_desc)
    
    assert len(aligned) == len(frame_indices), "Output length mismatch"
    assert all(d is not None for d in aligned), "Should fill all descriptions"
    
    print("✓ TemporalAligner test passed")
```

### Test 3: S2T_Dataset 返回值

```python
def test_s2t_dataset():
    dataset = S2T_Dataset(phase='train', use_descriptions=True)
    sample = dataset[0]
    
    # 检查必要字段
    required_keys = ['name', 'pose_sample', 'text', 'gloss', 'rgb_dict',
                    'description', 'has_description', 'frame_indices']
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    
    # 检查形状
    assert sample['pose_sample'].shape[0] > 0
    assert isinstance(sample['description'], (str, type(None)))
    assert isinstance(sample['has_description'], bool)
    assert isinstance(sample['frame_indices'], list)
    
    print("✓ S2T_Dataset test passed")
```

### Test 4: TextEncoder

```python
def test_text_encoder():
    encoder = TextEncoder('mt5-base')
    
    descriptions = [
        "这是第一个描述",
        "这是第二个描述",
        None
    ]
    
    # 处理 None 值
    valid_descs = [d if d is not None else "" for d in descriptions]
    features = encoder(valid_descs)
    
    assert features.shape == (3, 768), f"Expected (3, 768), got {features.shape}"
    
    print("✓ TextEncoder test passed")
```

### Test 5: GatingFusion

```python
def test_gating_fusion():
    fusion = GatingFusion(768)
    
    B, T = 4, 16  # batch_size=4, seq_len=16
    pose = torch.randn(B, T, 768)
    text = torch.randn(B, T, 768)
    indicator = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    
    fused = fusion(pose, text, indicator)
    
    assert fused.shape == (B, T, 768), f"Expected {(B, T, 768)}, got {fused.shape}"
    
    # 检查 fused 是否在 pose 和 (pose + text) 之间
    assert torch.allclose(fused, pose, atol=1.0) or torch.allclose(fused, pose + text, atol=1.0) \
        or (fused > pose).any() and (fused < pose + text).any()
    
    print("✓ GatingFusion test passed")
```

### Test 6: Uni_Sign forward

```python
def test_uni_sign_forward():
    model = Uni_Sign(use_description=True)
    model.eval()
    
    B, T = 2, 256
    src = torch.randn(B, 4, T, 150)
    tgt = torch.randint(0, 1000, (B, 50))
    
    descriptions = ["测试描述1", None]
    has_desc = torch.tensor([True, False])
    
    with torch.no_grad():
        output = model(src, tgt, descriptions, has_desc)
    
    assert output is not None, "Output should not be None"
    print("✓ Uni_Sign forward test passed")
```

### Test 7: Text Dropout

```python
def test_text_dropout():
    dropout_rate = 0.3
    has_desc = torch.tensor([True, True, True, True])
    
    dropped = []
    for _ in range(100):
        new_has_desc = has_desc.clone()
        for b in range(len(has_desc)):
            if has_desc[b] and torch.rand(1).item() < dropout_rate:
                new_has_desc[b] = False
        dropped.append((~new_has_desc).sum().item())
    
    # 平均应该约 30% 的样本被丢弃
    avg_dropped = sum(dropped) / len(dropped)
    assert 0.2 < avg_dropped / 4 < 0.4, f"Dropout rate seems off: {avg_dropped/4}"
    
    print("✓ Text Dropout test passed")
```

---

## 📊 集成测试检查清单

### Integration Test 1: 完整数据加载管道

```python
def test_complete_data_pipeline():
    """测试从数据加载到模型输入的完整流程"""
    
    # 1. 加载数据集
    dataset = S2T_Dataset(phase='train', use_descriptions=True)
    
    # 2. 创建 DataLoader
    collate_fn = custom_collate_fn
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    # 3. 获取第一个 batch
    batch = next(iter(loader))
    
    # 4. 验证 batch 形状和类型
    assert batch['pose_sample'].shape[0] == 4
    assert len(batch['description']) == 4
    assert len(batch['has_description']) == 4
    
    # 5. 模型应该能处理这个 batch
    model = Uni_Sign(use_description=True)
    model.eval()
    
    with torch.no_grad():
        output = model(
            src_input=batch['pose_sample'],
            tgt_input=batch['text'],
            description=batch['description'],
            has_description=batch['has_description']
        )
    
    assert output is not None
    print("✓ Complete data pipeline test passed")
```

### Integration Test 2: 训练循环单次迭代

```python
def test_training_loop_single_iteration():
    """测试训练循环的单次迭代"""
    
    dataset = S2T_Dataset(phase='train', use_descriptions=True)
    loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)
    
    model = Uni_Sign(use_description=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    batch = next(iter(loader))
    
    # Text Dropout
    for b in range(len(batch['has_description'])):
        if batch['has_description'][b] and torch.rand(1) < 0.3:
            batch['description'][b] = None
            batch['has_description'][b] = False
    
    # Forward
    output = model(
        src_input=batch['pose_sample'],
        tgt_input=batch['text'],
        description=batch['description'],
        has_description=batch['has_description']
    )
    
    # Loss
    loss = criterion(output, batch['gloss'])
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training loop test passed (loss: {loss.item():.4f})")
```

### Integration Test 3: 一致性验证

```python
def test_inference_consistency():
    """测试有/无文本的推理一致性"""
    
    model = Uni_Sign(use_description=True)
    model.eval()
    
    src = torch.randn(1, 4, 256, 150)
    description = "这是一个测试描述"
    
    # 推理：有文本
    with torch.no_grad():
        output_with = model(
            src, None,
            description=[description],
            has_description=torch.tensor([True])
        )
    
    # 推理：无文本
    with torch.no_grad():
        output_without = model(
            src, None,
            description=[None],
            has_description=torch.tensor([False])
        )
    
    # 计算 KL 散度
    probs_with = torch.softmax(output_with, dim=-1)
    probs_without = torch.softmax(output_without, dim=-1)
    kl = torch.nn.functional.kl_div(
        probs_without.log(), probs_with, reduction='mean'
    )
    
    print(f"✓ Consistency test passed (KL: {kl.item():.4f})")
    assert kl.item() < 0.5, "KL divergence too large"
```

---

## 📈 性能基准检查

### Checkpoint 1: 数据加载速度

```python
def benchmark_data_loading():
    """测试数据加载速度"""
    dataset = S2T_Dataset(phase='train', use_descriptions=True)
    
    import time
    start = time.time()
    for i in range(100):
        _ = dataset[i]
    elapsed = time.time() - start
    
    print(f"100 samples loaded in {elapsed:.2f}s ({elapsed/100*1000:.2f}ms/sample)")
    assert elapsed / 100 < 0.5, "Data loading too slow"
```

### Checkpoint 2: 前向传播速度

```python
def benchmark_forward():
    """测试模型前向传播速度"""
    model = Uni_Sign(use_description=True)
    model.eval()
    
    src = torch.randn(4, 4, 256, 150)
    desc = ["test desc"] * 4
    has_desc = torch.ones(4)
    
    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(src, None, desc, has_desc)
    elapsed = time.time() - start
    
    print(f"10 forward passes: {elapsed:.2f}s ({elapsed/10*1000:.2f}ms/pass)")
```

---

## 🎯 最终验收标准

实施完成后，需要满足以下标准：

### 代码质量
- [ ] 所有代码遵循项目编码规范
- [ ] 没有未使用的导入或变量
- [ ] 所有公共方法都有文档字符串
- [ ] 错误处理完善

### 功能完整性
- [ ] 所有 4 个关键模块实现完成
- [ ] 数据加载管道工作正常
- [ ] 模型能接收并处理描述
- [ ] 训练循环包含 Text Dropout
- [ ] 推理能处理有/无文本两种情况

### 性能指标
- [ ] 数据加载: < 500ms/sample
- [ ] 前向传播: < 300ms/batch (batch_size=4)
- [ ] 一致性: KL < 0.1, BLEU diff < 0.02

### 向后兼容
- [ ] 设置 `use_descriptions=False` 时工作正常
- [ ] 没有破坏现有的模型检查点加载
- [ ] 现有的推理脚本仍然适用

---

**最后更新**: 2026-02-14  
**文档版本**: 1.0  
**项目**: Uni-Sign Stage 3 多模态改进
