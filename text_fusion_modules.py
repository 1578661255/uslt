"""
文件功能：
    文本融合模块
    
    负责：
    1. mT5-base 文本编码器（冻结参数，仅推理）
    2. Gating 融合机制（学习视频和文本的融合权重）
    3. 可学习掩码嵌入（为缺失的描述提供初始表示）
    
    特点：
    - 完全独立，仅依赖 PyTorch 和 Transformers
    - 所有编码器参数冻结，节省显存和计算
    - Gating 融合是轻量级设计，适合快速原型验证
    - 支持 Text Dropout 用于训练时的正则化
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Tuple


class TextEncoder(nn.Module):
    """
    mT5-base 文本编码器
    
    功能：
        - 将文本描述编码为固定维度的特征向量
        - 参数完全冻结，仅做推理，不参与训练
        - 支持 None 输入的优雅处理
        - 支持批处理
    
    技术细节：
        - 使用 mT5 的 [CLS] token 作为句子表示（取第一个 token）
        - 输出维度固定为 768（与 mT5-base 一致）
        - 使用 @torch.no_grad() 装饰符确保不参与计算图
    
    示例：
        >>> encoder = TextEncoder('google/mt5-base')
        >>> features = encoder(['描述文本1', '描述文本2', None])
        >>> features.shape
        torch.Size([3, 768])
    """
    
    def __init__(self, 
                 model_name: str = 'google/mt5-base',
                 hidden_dim: int = 768,
                 device: str = 'cuda'):
        """
        初始化 mT5 文本编码器
        
        参数：
            model_name (str): HuggingFace 模型名称或本地路径
                            默认 'google/mt5-base'
            
            hidden_dim (int): 输出特征维度
                            默认 768（mT5-base 的编码器维度）
            
            device (str): 计算设备，'cuda' 或 'cpu'
                         默认 'cuda'
        
        说明：
            - 模型会自动从 HuggingFace Hub 下载（首次）或加载本地权重
            - 如果是本地路径，需要包含 config.json 和 pytorch_model.bin
            - Tokenizer 也会从同一位置加载
        """
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        
        # 加载分词器和编码器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 将模型移到目标设备
        self.encoder = self.encoder.to(device)
        
        # 参数冻结：设置 requires_grad=False
        # 这样反向传播时不会计算梯度，节省显存
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 固定为评估模式
        # 这确保 dropout 和 batch norm 等在推理时表现一致
        self.encoder.eval()
    
    @torch.no_grad()
    def forward(self, 
                descriptions: List[Optional[str]], 
                max_length: int = 256) -> torch.Tensor:
        """
        对描述文本进行编码
        
        参数：
            descriptions (List[Optional[str]]): 文本列表或 None 元素
                                              例如 ['描述1', '描述2', None, '描述3']
                                              None 表示缺失的描述，返回零向量
            
            max_length (int): tokenizer 的最大长度
                            默认 256（足够应对大多数动作描述）
        
        返回：
            torch.Tensor: 编码后的特征向量
                         形状 (B, hidden_dim) = (B, 768)
                         B = len(descriptions)
                         如果 descriptions[i] 为 None，返回零向量
        
        工作流程：
            1. 过滤出非 None 的描述
            2. 用 tokenizer 转换为 token IDs
            3. 送入 mT5 编码器获得隐层表示
            4. 取第 0 个 token（[CLS] 位置）作为句子表示
            5. 将零向量填充回 None 的位置
        
        示例：
            >>> encoder = TextEncoder()
            >>> descs = ['起手', None, '手放下']
            >>> feat = encoder(descs)
            >>> feat.shape
            torch.Size([3, 768])
            >>> (feat[1] == 0).all()  # None 位置全为 0
            True
        """
        # 处理输入
        batch_size = len(descriptions)
        
        # 过滤出非 None 的描述
        valid_descs = [d for d in descriptions if d is not None]
        
        # 如果全是 None，直接返回零向量
        if not valid_descs:
            return torch.zeros(batch_size, self.hidden_dim, 
                             dtype=torch.float32, 
                             device=self.device)
        
        # 用 tokenizer 处理文本
        encoded = self.tokenizer(
            valid_descs,
            max_length=max_length,
            padding='max_length',           # 补齐至 max_length
            truncation=True,                # 超过 max_length 则截断
            return_tensors='pt'             # 返回 PyTorch tensor
        ).to(self.device)                   # 移到目标设备
        
        # 推理：获得编码器的输出
        # outputs.last_hidden_state shape: (valid_num, max_length, 768)
        with torch.no_grad():
            outputs = self.encoder(**encoded)
        
        # 取第 0 个 token（[CLS] 位置）作为句子表示
        # shape: (valid_num, 768)
        text_features = outputs.last_hidden_state[:, 0, :]
        
        # 构造结果：将零向量填入 None 的位置
        result = torch.zeros(batch_size, self.hidden_dim, 
                           dtype=torch.float32,
                           device=self.device)
        
        valid_idx = 0
        for i, desc in enumerate(descriptions):
            if desc is not None:
                result[i] = text_features[valid_idx]
                valid_idx += 1
            # 否则保持为零向量（已初始化）
        
        return result


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
        assert has_description.shape == (B, T, 1), \
            f"形状错误：expected (B, T, 1), got {has_description.shape}"
        
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
