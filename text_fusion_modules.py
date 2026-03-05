"""
文件功能：
    文本融合模块
    
    负责：
    1. 多编码器文本编码（mT5-base, BERT-base, CLIP-Text 等）
    2. Gating 融合机制（学习视频和文本的融合权重）
    3. 可学习掩码嵌入（为缺失的描述提供初始表示）
    
    特点：
    - 完全独立，仅依赖 PyTorch 和 Transformers
    - 所有编码器参数冻结，节省显存和计算
    - 支持多种编码器的动态切换（工厂模式）
    - Gating 融合是轻量级设计，适合快速原型验证
    - 支持 Text Dropout 用于训练时的正则化
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPModel
from typing import List, Optional, Tuple, Union, Dict


# ======================== 编码器基类 ========================

class BaseTextEncoder(nn.Module):
    """
    文本编码器基类
    
    所有文本编码器（mT5, BERT, CLIP 等）都应继承此类，
    以确保统一的接口和行为约定。
    """
    
    def __init__(self, 
                 model_name: str = None,
                 hidden_dim: int = 768,
                 device: str = 'cuda'):
        """
        初始化基类
        
        参数：
            model_name (str): 模型标识符
            hidden_dim (int): 输出特征维度
            device (str): 计算设备
        """
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = None
        self.tokenizer = None
    
    def forward(self, descriptions: List[Optional[str]], 
                max_length: int = 256) -> torch.Tensor:
        """
        编码文本描述
        
        参数：
            descriptions: 文本列表，支持 None（表示缺失）
            max_length: 最大序列长度
        
        返回：
            编码后的特征张量，形状 (B, hidden_dim)
        """
        raise NotImplementedError("子类必须实现 forward 方法")
    
    def _freeze_encoder(self):
        """冻结编码器参数，仅做推理"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()


# ======================== mT5 编码器实现 ========================

class MT5TextEncoder(BaseTextEncoder):
    """
    mT5-base 文本编码器（多语言预训练）
    
    特点：
    - 支持 100+ 种语言
    - 250M 参数，多语言表征能力强
    - 编码器维度 768
    - 适合低资源手语识别（缺乏中英文对齐数据时）
    """
    
    def __init__(self, 
                 model_name: str = 'google/mt5-base',
                 hidden_dim: int = 768,
                 device: str = 'cuda'):
        super().__init__(model_name, hidden_dim, device)
        
        # 加载分词器和编码器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 仅加载编码器部分（MT5Model 包含编编码器+解码器，取 .encoder）
        full_model = AutoModel.from_pretrained(model_name)
        self.encoder = full_model.encoder
        
        # 设备转移
        self.encoder = self.encoder.to(device)
        
        # 冻结参数
        self._freeze_encoder()
    
    @torch.no_grad()
    def forward(self, 
                descriptions: List[Optional[str]], 
                max_length: int = 256) -> torch.Tensor:
        """
        MT5 编码器的前向传播
        
        说明：
            - 使用 [CLS] token（第 0 位）作为句子表示
            - 输出维度为 768（mT5-base 编码器维度）
        """
        batch_size = len(descriptions)
        
        # 过滤出非 None 的描述
        valid_descs = [d for d in descriptions if d is not None]
        
        # 如果全是 None，直接返回零向量
        if not valid_descs:
            return torch.zeros(batch_size, self.hidden_dim, 
                             dtype=torch.float32, 
                             device=self.device)
        
        # 分词处理
        encoded = self.tokenizer(
            valid_descs,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # 编码器推理
        outputs = self.encoder(**encoded)
        
        # 取 [CLS] token（第 0 位）作为句子表示
        text_features = outputs.last_hidden_state[:, 0, :]
        
        # 构造结果（对齐 None 位置）
        result = torch.zeros(batch_size, self.hidden_dim, 
                           dtype=torch.float32,
                           device=self.device)
        
        valid_idx = 0
        for i, desc in enumerate(descriptions):
            if desc is not None:
                result[i] = text_features[valid_idx]
                valid_idx += 1
        
        return result


# ======================== BERT 编码器实现 ========================

class BERTTextEncoder(BaseTextEncoder):
    """
    BERT-base 文本编码器（英文预训练）
    
    特点：
    - 110M 参数，参数量少
    - 仅支持英文及部分欧洲语言
    - 编码器维度 768
    - 推理速度快，适合低延迟应用
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 hidden_dim: int = 768,
                 device: str = 'cuda'):
        super().__init__(model_name, hidden_dim, device)
        
        # 加载分词器和编码器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 设备转移
        self.encoder = self.encoder.to(device)
        
        # 冻结参数
        self._freeze_encoder()
    
    @torch.no_grad()
    def forward(self, 
                descriptions: List[Optional[str]], 
                max_length: int = 256) -> torch.Tensor:
        """
        BERT 编码器的前向传播
        
        特性：
            - 输出为 pooled_output（已应用 [CLS] 和全连接层）
            - 相对 mT5 更精炼的表示
        """
        batch_size = len(descriptions)
        
        # 过滤非 None 描述
        valid_descs = [d for d in descriptions if d is not None]
        
        if not valid_descs:
            return torch.zeros(batch_size, self.hidden_dim, 
                             dtype=torch.float32, 
                             device=self.device)
        
        # 分词处理
        encoded = self.tokenizer(
            valid_descs,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # BERT 编码器推理
        outputs = self.encoder(**encoded)
        
        # BERT 的 pooled output（已应用池化 + 全连接）
        text_features = outputs.pooler_output
        
        # 构造结果
        result = torch.zeros(batch_size, self.hidden_dim, 
                           dtype=torch.float32,
                           device=self.device)
        
        valid_idx = 0
        for i, desc in enumerate(descriptions):
            if desc is not None:
                result[i] = text_features[valid_idx]
                valid_idx += 1
        
        return result


# ======================== CLIP 文本编码器实现 ========================

class CLIPTextEncoder(BaseTextEncoder):
    """
    CLIP 文本编码器（视觉-语言对齐预训练）
    
    特点：
    - 63M 参数（相对轻量）
    - 视觉-文本对齐特性：可能增强视频-文本语义一致性
    - 输出维度 512（需要投影到 768）
    - 推理快速，适合多模态应用
    """
    
    def __init__(self, 
                 model_name: str = 'openai/clip-vit-base-patch32',
                 hidden_dim: int = 768,
                 device: str = 'cuda'):
        super().__init__(model_name, hidden_dim, device)
        
        # CLIP 的实际输出维度是 512
        self.clip_output_dim = 512
        
        # 加载 CLIP 文本编码器
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name)
        self.encoder = clip_model.text_model
        
        # 设备转移
        self.encoder = self.encoder.to(device)
        
        # 冻结参数
        self._freeze_encoder()
        
        # 投影层：512 -> 768（与其他编码器对齐）
        self.projection = nn.Linear(self.clip_output_dim, hidden_dim)
        self.projection = self.projection.to(device)
    
    @torch.no_grad()
    def forward(self, 
                descriptions: List[Optional[str]], 
                max_length: int = 77) -> torch.Tensor:
        """
        CLIP 文本编码器的前向传播
        
        注意：
            - CLIP tokenizer 的默认最大长度是 77，比 mT5 短
            - 输出需要投影到 768 维以保持一致性
        """
        batch_size = len(descriptions)
        
        # 过滤非 None 描述
        valid_descs = [d for d in descriptions if d is not None]
        
        if not valid_descs:
            return torch.zeros(batch_size, self.hidden_dim, 
                             dtype=torch.float32, 
                             device=self.device)
        
        # CLIP 分词处理
        encoded = self.tokenizer(
            valid_descs,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # CLIP 文本编码器推理
        outputs = self.encoder(**encoded)
        
        # 取 pooled output（已应用 pool 和激活）
        text_features = outputs.pooled_output  # shape: (N, 512)
        
        # 投影到目标维度 768
        text_features = self.projection(text_features)  # shape: (N, 768)
        
        # 构造结果
        result = torch.zeros(batch_size, self.hidden_dim, 
                           dtype=torch.float32,
                           device=self.device)
        
        valid_idx = 0
        for i, desc in enumerate(descriptions):
            if desc is not None:
                result[i] = text_features[valid_idx]
                valid_idx += 1
        
        return result


# ======================== 编码器工厂函数 ========================

def create_text_encoder(encoder_type: str = 'mt5', 
                       hidden_dim: int = 768,
                       device: str = 'cuda') -> BaseTextEncoder:
    """
    编码器工厂函数
    
    根据指定的编码器类型，返回对应的编码器实例。
    支持的类型：'mt5', 'bert', 'clip'
    
    参数：
        encoder_type (str): 编码器类型
            - 'mt5': mT5-base（多语言，推荐）
            - 'bert': BERT-base（英文，轻量）
            - 'clip': CLIP-text（视觉对齐，轻量）
        
        hidden_dim (int): 输出特征维度（默认 768）
        device (str): 计算设备（'cuda' 或 'cpu'）
    
    返回：
        编码器实例（继承自 BaseTextEncoder）
    
    示例：
        >>> encoder = create_text_encoder('mt5')
        >>> features = encoder(['描述1', '描述2'])
        >>> features.shape
        torch.Size([2, 768])
    """
    # 编码器配置字典
    encoder_configs = {
        'mt5': {
            'class': MT5TextEncoder,
            'model_name': 'google/mt5-base',
            'output_dim': 768
        },
        'bert': {
            'class': BERTTextEncoder,
            'model_name': 'bert-base-uncased',
            'output_dim': 768
        },
        'clip': {
            'class': CLIPTextEncoder,
            'model_name': 'openai/clip-vit-base-patch32',
            'output_dim': 768  # 投影后维度
        }
    }
    
    # 验证编码器类型
    encoder_type = encoder_type.lower()
    if encoder_type not in encoder_configs:
        raise ValueError(
            f"不支持的编码器类型: {encoder_type}。"
            f"支持的类型: {list(encoder_configs.keys())}"
        )
    
    # 获取编码器配置
    config = encoder_configs[encoder_type]
    encoder_class = config['class']
    model_name = config['model_name']
    
    # 创建编码器实例
    print(f"[文本编码器] 初始化 {encoder_type.upper()} 编码器...")
    print(f"  模型: {model_name}")
    print(f"  输出维度: {config['output_dim']}")
    print(f"  设备: {device}")
    
    encoder = encoder_class(
        model_name=model_name,
        hidden_dim=hidden_dim,
        device=device
    )
    
    return encoder# ======================== 原有的 TextEncoder（兼容性） ========================

class TextEncoder(MT5TextEncoder):
    """
    调用兼容层：保持向后兼容性
    
    直接使用本类时，默认使用 mT5-base 编码器。
    如需使用其他编码器，推荐使用 create_text_encoder() 工厂函数。
    """
    pass


# ======================== 掩码嵌入 ========================

class LearnableMaskEmbedding(nn.Module):
    """
    可学习的掩码嵌入
    
    功能：
        - 为缺失的描述文本提供可学习的初始表示
        - 通过反向传播与模型一起训练
        - 用于替换 None 位置的文本特征
    
    设计思路：
        - 缺失的描述无法用 mT5 编码
        - 使用一个可学习的向量作为占位符
        - 这个向量会通过训练逐渐学到有用的信息
        - 初始化为小的随机值，避免过大的初始化
    
    示例：
        >>> mask = LearnableMaskEmbedding(768)
        >>> mask_feat = mask()  # shape: (1, 768)
        >>> # 用于替换 text_features 中的缺失位置
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 init_std: float = 0.01):
        """
        初始化可学习掩码嵌入
        
        参数：
            hidden_dim (int): 嵌入维度，应与文本编码器输出维度一致
                            默认 768
            
            init_std (float): 初始化标准差
                            使用 torch.randn(1, hidden_dim) * init_std
                            默认 0.01（较小的初始化）
        
        说明：
            - init_std 越小，初始化越接近零
            - 较小的初始化有助于训练稳定性
        """
        super().__init__()
        
        # 创建可学习参数
        # shape: (1, hidden_dim)
        self.mask = nn.Parameter(
            torch.randn(1, hidden_dim) * init_std
        )
    
    def forward(self) -> torch.Tensor:
        """
        返回掩码嵌入向量
        
        返回：
            torch.Tensor: 掩码嵌入，shape (1, hidden_dim) = (1, 768)
        
        用途：
            >>> mask_embedding = LearnableMaskEmbedding(768)
            >>> for b in range(batch_size):
            ...     for t in range(seq_len):
            ...         if descriptions[b][t] is None:
            ...             text_features[b, t] = mask_embedding()
        
        注意：
            - 每次调用返回同一个向量（参数）
            - 梯度会累积到 self.mask 中
        """
        return self.mask


class GatingFusion(nn.Module):
    """
    Gating 融合机制
    
    功能：
        - 学习如何融合视频特征和文本特征
        - 动态计算融合权重（gate），范围 [0, 1]
        - 支持 Text Dropout 用于训练时的正则化
    
    融合公式：
        fused = pose_feat + gate * text_feat
        其中 gate = Sigmoid(MLP([pose_feat, text_feat, has_description]))
    
    特点：
        - 参数少（仅一个小 MLP），推理快
        - 对时间对齐错误鲁棒（通过 gate 学习权重）
        - 可微，可与模型端到端训练
    
    示例：
        >>> fusion = GatingFusion(feature_dim=768)
        >>> pose = torch.randn(2, 10, 768)  # (B=2, T=10, D=768)
        >>> text = torch.randn(2, 10, 768)
        >>> has_desc = torch.ones(2, 10, 1)
        >>> fused, gates = fusion(pose, text, has_desc)
        >>> fused.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(self,
                 feature_dim: int = 768,
                 gating_hidden_dim: int = 512):
        """
        初始化 Gating 融合模块
        
        参数：
            feature_dim (int): 输入特征维度
                            默认 768（与 mT5-base 输出维度一致）
            
            gating_hidden_dim (int): Gating MLP 的隐层维度
                                   默认 512
        
        MLP 结构：
            输入: [pose_feat, text_feat, has_description]
            维度: (feature_dim * 2 + 1) = 1537
                  ↓
            隐层 1: gating_hidden_dim = 512
            ReLU 激活
                  ↓
            隐层 2: 256
            ReLU 激活
                  ↓
            输出: 1 (gate 权重)
            Sigmoid 激活 → [0, 1]
        
        说明：
            - 输入维度是 768*2 + 1 = 1537
              * 768：pose_feat
              * 768：text_feat
              * 1：has_description 指示符
            - 参数量约 1537*512 + 512*256 + 256*1 ≈ 900K（较小）
            - 所有权重使用 Xavier 初始化
        """
        super().__init__()
        
        # Gating MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()                    # 输出范围 [0, 1]
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化 MLP 权重
        
        策略：
            - Linear 层：Xavier uniform 初始化
            - 偏置：全 0
        
        目的：
            - 确保梯度流动不会过大或过小
            - 使训练更稳定
        """
        for layer in self.gate_mlp:
            if isinstance(layer, nn.Linear):
                # Xavier uniform 初始化
                nn.init.xavier_uniform_(layer.weight)
                # 偏置初始化为 0
                nn.init.zeros_(layer.bias)
    
    def forward(self,
                pose_feat: torch.Tensor,
                text_feat: torch.Tensor,
                has_description: torch.Tensor,
                text_dropout_p: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行 Gating 融合
        
        参数：
            pose_feat (torch.Tensor): 视频姿态特征
                                     shape: (B, T, D) 或 (B, T, C)
                                     例如 (2, 10, 768)
            
            text_feat (torch.Tensor): 文本特征
                                     shape: (B, T, 768)
                                     例如 (2, 10, 768)
                                     注意：缺失位置应为掩码嵌入或零向量
            
            has_description (torch.Tensor): 缺失指示符
                                           shape: (B, T, 1)
                                           1 = 有真实描述
                                           0 = 插值/缺失/最近邻
                                           例如 (2, 10, 1)
            
            text_dropout_p (float): Text Dropout 概率
                                   仅在 self.training=True 时应用
                                   默认 0.0（不使用 dropout）
                                   范围：[0, 1)
        
        返回：
            fused_feat (torch.Tensor): 融合后的特征
                                      shape: (B, T, D) 与 pose_feat 相同
            
            gate_weights (torch.Tensor): Gating 权重（用于可视化）
                                        shape: (B, T, 1)
                                        范围 [0, 1]
        
        工作流程：
            1. 检查输入形状，规范化 has_description
            2. 应用 Text Dropout（如果启用）
            3. 拼接 [pose_feat, text_feat, has_description]
            4. Reshape 以通过 MLP
            5. 计算 gate 权重（0-1）
            6. 融合：fused = pose_feat + gate * text_feat
        
        说明：
            - gate = 0 时，完全保留 pose_feat（忽视 text_feat）
            - gate = 1 时，完全使用 pose_feat + text_feat
            - gate = 0.5 时，文本特征贡献 50%
        
        示例：
            >>> fusion = GatingFusion()
            >>> pose = torch.randn(2, 10, 768)
            >>> text = torch.randn(2, 10, 768)
            >>> has_desc = torch.ones(2, 10, 1)
            >>> fused, gates = fusion(pose, text, has_desc, text_dropout_p=0.1)
            >>> gates.min(), gates.max()
            (tensor(0.2), tensor(0.9))
        """
        B, T, D = pose_feat.shape
        
        # 规范化 has_description 的形状
        if has_description.dim() == 2:
            has_description = has_description.unsqueeze(-1)
        
        # 处理形状不匹配的情况（可能因为 text_feat 长度不同）
        if has_description.shape[1] != T:
            if has_description.shape[1] > T:
                # 截断到 T
                has_description = has_description[:, :T, :]
            else:
                # 补零到 T
                pad_size = T - has_description.shape[1]
                has_description = torch.cat([
                    has_description,
                    torch.zeros(B, pad_size, 1, device=has_description.device, dtype=has_description.dtype)
                ], dim=1)
        
        assert has_description.shape == (B, T, 1), \
            f"形状错误：expected (B, T, 1), got {has_description.shape}"
        
        # 处理 text_feat 的形状不匹配
        if text_feat.shape[1] != T:
            if text_feat.shape[1] > T:
                # 截断到 T
                text_feat = text_feat[:, :T, :]
            else:
                # 补零到 T
                pad_size = T - text_feat.shape[1]
                text_feat = torch.cat([
                    text_feat,
                    torch.zeros(B, pad_size, text_feat.shape[2], device=text_feat.device, dtype=text_feat.dtype)
                ], dim=1)
        
        # 应用 Text Dropout (仅训练时)
        # 概率为 text_dropout_p 的文本特征被随机清为 0
        if text_dropout_p > 0 and self.training:
            dropout_mask = torch.bernoulli(
                torch.full((B, T, 1), text_dropout_p, device=text_feat.device)
            )
            text_feat = text_feat * (1 - dropout_mask)
        
        # 拼接特征向量
        # [pose_feat, text_feat, has_description]
        # shape: (B, T, 768 + 768 + 1) = (B, T, 1537)
        combined = torch.cat([pose_feat, text_feat, has_description], dim=-1)
        
        # Reshape 为 2D 以通过 MLP
        # (B, T, 1537) → (B*T, 1537)
        combined_flat = combined.view(B * T, -1)
        
        # 通过 MLP 计算 gate 权重
        # (B*T, 1537) → (B*T, 1)
        gate_flat = self.gate_mlp(combined_flat)
        
        # Reshape 回 3D
        # (B*T, 1) → (B, T, 1)
        gate = gate_flat.view(B, T, 1)
        
        # 融合：fused = pose + gate * text
        # 当 gate ≈ 0 时，融合特征 ≈ pose_feat
        # 当 gate ≈ 1 时，融合特征 ≈ pose_feat + text_feat
        fused_feat = pose_feat + gate * text_feat
        
        return fused_feat, gate
