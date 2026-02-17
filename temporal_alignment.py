"""
文件功能：
    时间对齐和描述加载模块
    
    负责：
    1. 从 description/ 文件夹加载动作描述文本 JSON
    2. 处理视频采样过程中的帧索引映射
    3. 使用智能插值策略处理缺失的描述文本
    
    特点：
    - 完全独立，不依赖其他自定义模块
    - 支持优雅降级（无描述文件时不报错）
    - 支持多种 JSON 格式
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class DescriptionLoader:
    """
    描述文本加载器
    
    功能：
        - 从本地 JSON 文件加载动作描述文本
        - 按样本 ID 索引
        - 支持多种 JSON 格式自动识别
        - 返回元数据便于调试
    
    路径结构（以 CSL-Daily 为例）：
        description/CSL_Daily/split_data/
        ├── train/
        │   ├── S000196_P0000_T00.json
        │   ├── S000196_P0004_T00.json
        │   └── ...
        ├── dev/
        │   ├── S000196_P0000_T00.json
        │   └── ...
        └── test/
            └── ...
    
    预期 JSON 格式示例：
        {
            "frames": {
                "0": "person raises left hand",
                "2": "hand touches chin",
                "5": "both hands move down"
            },
            "total_frames": 300,
            "metadata": {...}
        }
    
    使用示例：
        >>> loader = DescriptionLoader('./description/CSL_Daily/split_data/train')
        >>> descs, meta = loader.load('S000196_P0000_T00')
        >>> descs
        {0: 'person raises hand', 2: 'hand touches chin', ...}
    """
    
    def __init__(self, description_dir: str):
        """
        初始化描述加载器
        
        参数：
            description_dir (str): 描述文件夹路径
                                  例如 './description/CSL_Daily/split_data/train'
                                  该文件夹应包含若干 {sample_id}.json 文件
        """
        self.description_dir = Path(description_dir)
        
        # 验证目录存在
        if not self.description_dir.exists():
            print(f"[警告] 描述文件夹不存在: {self.description_dir}")
    
    def load(self, sample_id: str) -> Tuple[Dict[int, str], Dict[str, Any]]:
        """
        加载单个样本的描述文本
        
        参数：
            sample_id (str): 样本 ID，例如 'S000196_P0000_T00'
                            无需包含 .json 扩展名
        
        返回：
            descriptions_dict (Dict[int, str]): 帧号到描述文本的映射
                                               {frame_id: description_text}
                                               如果无描述文件，返回空字典 {}
            
            metadata (Dict[str, Any]): 元数据
                {
                    'success': bool,              # 加载是否成功
                    'frame_count': int,           # 描述文本条数
                    'file': str,                  # 文件路径
                    'reason': str (optional)      # 失败原因
                }
        
        示例：
            >>> loader = DescriptionLoader('./description/CSL_Daily')
            >>> descs, meta = loader.load('S000196_P0000_T00')
            >>> descs
            {0: 'person raises hand', 2: 'hand touches chin', ...}
            >>> meta
            {'success': True, 'frame_count': 12, 'file': '...'}
        """
        json_path = self.description_dir / f"{sample_id}.json"
        
        try:
            # 打开并解析 JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 自动识别描述的位置（支持多种格式）
            descriptions = self._extract_descriptions(data)
            
            # 确保所有 key 都是整数
            descriptions = {int(k): v for k, v in descriptions.items()}
            
            metadata = {
                'success': True,
                'frame_count': len(descriptions),
                'file': str(json_path)
            }
            
            return descriptions, metadata
        
        except FileNotFoundError:
            return {}, {
                'success': False,
                'reason': 'file_not_found',
                'file': str(json_path)
            }
        
        except json.JSONDecodeError as e:
            return {}, {
                'success': False,
                'reason': f'invalid_json: {str(e)}',
                'file': str(json_path)
            }
        
        except Exception as e:
            return {}, {
                'success': False,
                'reason': f'unexpected_error: {str(e)}',
                'file': str(json_path)
            }
    
    def _extract_descriptions(self, data) -> Dict[str, str]:
        """
        从 JSON 数据中提取描述文本（自动识别格式）
        
        参数：
            data: 解析后的 JSON 数据（可能是 Dict 或 List）
        
        返回：
            Dict[str, str]: {frame_id: description_text}
        
        支持的 JSON 格式：
            1. [{"filename": "000000.jpg", "description": "..."}, ...]  ← 数组格式（当前实际格式）
            2. {"frames": {frame_id: text}} ← mT5 多模态论文常用format
            3. {"descriptions": {frame_id: text}}
            4. {frame_id: text} ← 直接format
            5. {"data": {frame_id: text}}
        """
        # 格式0: 数组格式（当前实际格式）
        if isinstance(data, list):
            descriptions = {}
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    # 优先使用 description 字段
                    if 'description' in item:
                        descriptions[str(idx)] = item['description']
                    # 备选：使用 text 字段
                    elif 'text' in item:
                        descriptions[str(idx)] = item['text']
            if descriptions:
                return descriptions
        
        # 格式1: 标准格式（推荐）
        if isinstance(data, dict) and isinstance(data.get('frames'), dict):
            return data['frames']
        
        # 格式2: 备选名称
        if isinstance(data, dict) and isinstance(data.get('descriptions'), dict):
            return data['descriptions']
        
        # 格式3: 直接映射
        if isinstance(data, dict):
            metadata_keys = {'metadata', 'info', 'total_frames', 'version', 'source'}
            filtered = {k: v for k, v in data.items() 
                       if k not in metadata_keys and isinstance(v, str)}
            if filtered:
                return filtered
        
        # 格式4: 嵌套在 data 字段中
        if isinstance(data, dict) and isinstance(data.get('data'), dict):
            return data['data']
        
        # 未识别格式，返回空
        return {}


class TemporalAligner:
    """
    时间对齐器
    
    功能：
        - 处理视频采样引起的帧号变化
        - 将原始帧的描述映射到采样后的帧
        - 使用智能插值策略处理缺失的描述
        - 生成缺失指示符，指导模型学习
    
    背景说明：
        原始视频可能有 300 帧，但模型处理时可能采样到 10 帧
        原始帧号和采样后帧号的对应关系由采样过程决定
        此模块负责这个映射和缺失处理
    
    示例：
        原始帧: 0, 1, 2, 3, 4, 5, ..., 299
        原始描述: {0: "hand up", 5: "hand down", 10: "hand left"}
        采样帧: [0, 5, 10, 15, 20, ...]  (采样的原始帧号)
        
        对齐后：
        采样帧 0 → 原始帧 0 → "hand up" (直接)
        采样帧 1 → 原始帧 5 → "hand down" (直接)
        采样帧 2 → 原始帧 10 → "hand left" (直接)
    """
    
    def __init__(self,
                 original_descriptions: Dict[int, str],
                 sampled_frame_indices: List[int],
                 use_nearest_neighbor: bool = True,
                 use_linear_interpolation: bool = False):
        """
        初始化时间对齐器
        
        参数：
            original_descriptions (Dict[int, str]): 原始帧的描述映射
                                                    {original_frame_id: description_text}
                                                    例如 {0: "hand up", 5: "hand down"}
            
            sampled_frame_indices (List[int]): 采样后每一帧对应的原始帧号
                                               例如 [0, 5, 10, 15, 20, ...]
                                               长度就是采样后的视频长度
            
            use_nearest_neighbor (bool): 帧无直接描述时，是否使用最近邻描述
                                        默认 True（推荐）
            
            use_linear_interpolation (bool): 两边都有描述时，是否进行线性插值
                                            目前简单实现为合并描述，不做特征插值
                                            默认 False
        
        示例：
            >>> descs = {0: "hand up", 10: "hand down"}
            >>> indices = [0, 2, 5, 8, 10, 15, 20]
            >>> aligner = TemporalAligner(descs, indices)
            >>> aligned, has_desc = aligner.align()
            >>> aligned
            ['hand up', 'hand up', 'hand up', 'hand up', 'hand down', 
             'hand down', 'hand down']
            >>> has_desc
            [1, 0, 0, 0, 1, 0, 0]
        """
        self.original_descriptions = original_descriptions
        self.sampled_frame_indices = sampled_frame_indices
        self.use_nearest_neighbor = use_nearest_neighbor
        self.use_linear_interpolation = use_linear_interpolation
        
        # 预计算：排序的原始帧号列表
        self.original_frame_ids = sorted(original_descriptions.keys()) \
            if original_descriptions else []
    
    def align(self) -> Tuple[List[Optional[str]], List[int]]:
        """
        执行智能插值对齐
        
        返回：
            aligned_descriptions (List[Optional[str]]): 对齐后的描述文本列表
                                                        长度 = len(sampled_frame_indices)
                                                        如果无描述，对应位置为 None
            
            has_description (List[int]): 缺失指示符列表
                                        长度 = len(sampled_frame_indices)
                                        1 = 帧有真实描述，0 = 插值/缺失/最近邻
        
        对齐策略（优先级从高到低）：
            1. 采样帧对应的原始帧有直接描述 → 使用
               has_description = 1
            
            2. 最近邻帧有描述 → 使用最近邻
               use_nearest_neighbor = True 时应用
               has_description = 0
            
            3. 两边都有描述 → 线性插值合并
               use_linear_interpolation = True 时应用
               has_description = 0
            
            4. 完全无描述 → 返回 None
               has_description = 0
        
        示例：
            >>> descs = {0: 'A', 3: 'B', 7: 'C'}
            >>> indices = [0, 1, 2, 3, 4, 5, 6, 7]
            >>> aligner = TemporalAligner(descs, indices)
            >>> aligned, has_desc = aligner.align()
            >>> listed(zip(indices, aligned, has_desc))
            [(0, 'A', 1), (1, 'A', 0), (2, 'A', 0), (3, 'B', 1),
             (4, 'B', 0), (5, 'B', 0), (6, 'C', 0), (7, 'C', 1)]
        """
        aligned = []
        has_desc = []
        
        # 如果没有描述，直接返回全空
        if not self.original_descriptions:
            return [None] * len(self.sampled_frame_indices), \
                   [0] * len(self.sampled_frame_indices)
        
        for frame_idx, original_frame_id in enumerate(self.sampled_frame_indices):
            # 策略1: 帧有直接描述
            if original_frame_id in self.original_descriptions:
                aligned.append(self.original_descriptions[original_frame_id])
                has_desc.append(1)
            
            # 策略2: 使用最近邻描述
            elif self.use_nearest_neighbor and self.original_frame_ids:
                nearest_frame = self._find_nearest(original_frame_id)
                aligned.append(self.original_descriptions[nearest_frame])
                has_desc.append(0)
            
            # 策略3: 线性插值（简单实现：合并左右两边的描述）
            elif self.use_linear_interpolation and self.original_frame_ids:
                merged_desc = self._interpolate(original_frame_id)
                if merged_desc:
                    aligned.append(merged_desc)
                    has_desc.append(0)
                else:
                    aligned.append(None)
                    has_desc.append(0)
            
            # 策略4: 完全无描述
            else:
                aligned.append(None)
                has_desc.append(0)
        
        return aligned, has_desc
    
    def _find_nearest(self, frame_id: int) -> Optional[int]:
        """
        查找最近的有描述的帧号
        
        参数：
            frame_id (int): 目标帧号
        
        返回：
            int: 最近的有描述的原始帧号
                 根据 abs(frame_id - nearest_id) 最小
        
        示例：
            >>> aligner = TemporalAligner({0: 'A', 10: 'B'}, [])
            >>> aligner._find_nearest(3)
            0
            >>> aligner._find_nearest(7)
            10
        """
        if not self.original_frame_ids:
            return None
        
        nearest = min(self.original_frame_ids,
                     key=lambda x: abs(x - frame_id))
        return nearest
    
    def _interpolate(self, frame_id: int) -> Optional[str]:
        """
        线性插值：如果两边都有描述，合并它们
        
        参数：
            frame_id (int): 目标帧号
        
        返回：
            str: 合并的描述文本
                 格式："{left_desc} → {right_desc}"
            None: 如果无法进行插值（只有单边描述）
        
        实现说明：
            当前实现是简单的文本级别合并
            未来可扩展为特征级别的线性插值
        
        示例：
            >>> aligner = TemporalAligner({0: 'hand up', 10: 'hand down'}, [])
            >>> aligner._interpolate(5)
            'hand up → hand down'
            >>> aligner._interpolate(12)  # 只有左边
            None
        """
        # 找左边最近的描述
        left_frames = [f for f in self.original_frame_ids if f <= frame_id]
        # 找右边最近的描述
        right_frames = [f for f in self.original_frame_ids if f > frame_id]
        
        # 两边都有描述时才进行插值
        if left_frames and right_frames:
            left_frame = max(left_frames)
            right_frame = min(right_frames)
            
            left_desc = self.original_descriptions[left_frame]
            right_desc = self.original_descriptions[right_frame]
            
            # 简单合并：在两个描述之间加箭头
            # 未来可扩展为加权平均或特征级插值
            merged = f"{left_desc} → {right_desc}"
            return merged
        
        return None
    
    def align_with_metadata(self) -> Tuple[List[Optional[str]], List[int], Dict[str, Any]]:
        """
        执行对齐并返回详细的元数据信息
        
        返回：
            aligned_descriptions (List[Optional[str]]): 对齐后的描述
            has_description (List[int]): 缺失指示符
            metadata (Dict[str, Any]): 诊断元数据
                {
                    'total_sampled_frames': int,        # 采样帧总数
                    'frames_with_description': int,     # 有真实描述的帧数
                    'frames_with_nearest': int,         # 使用最近邻的帧数
                    'frames_with_interpolation': int,   # 使用插值的帧数
                    'frames_without_description': int,  # 完全无描述的帧数
                    'coverage_ratio': float,            # 覆盖率 (0-1)
                }
        
        用途：
            - 诊断对齐质量
            - 评估模型可能的性能
            - 调整参数
        
        示例：
            >>> descs = {0: 'A', 10: 'B'}
            >>> indices = list(range(20))
            >>> aligner = TemporalAligner(descs, indices)
            >>> aligned, has_desc, meta = aligner.align_with_metadata()
            >>> meta
            {
                'total_sampled_frames': 20,
                'frames_with_description': 2,
                'frames_with_nearest': 18,
                'coverage_ratio': 1.0
            }
        """
        aligned, has_desc = self.align()
        
        total_frames = len(aligned)
        frames_with_description = sum(has_desc)
        frames_without_description = total_frames - frames_with_description
        
        # 计算覆盖率
        if total_frames > 0:
            coverage_ratio = frames_with_description / total_frames
        else:
            coverage_ratio = 0.0
        
        metadata = {
            'total_sampled_frames': total_frames,
            'frames_with_description': frames_with_description,
            'frames_without_description': frames_without_description,
            'coverage_ratio': coverage_ratio,
            'strategy': {
                'use_nearest_neighbor': self.use_nearest_neighbor,
                'use_linear_interpolation': self.use_linear_interpolation,
            }
        }
        
        return aligned, has_desc, metadata


# ============================================================================
# 辅助函数：用于数据集集成
# ============================================================================

def load_descriptions_for_sample(sample_id: str, 
                                 description_dir: str) -> Tuple[List[Optional[str]], 
                                                                 List[int],
                                                                 Dict[str, Any]]:
    """
    便捷函数：一步加载和对齐描述文本
    
    参数：
        sample_id (str): 样本 ID，例如 'S000196_P0000_T00'
        description_dir (str): 描述文件夹路径
    
    返回：
        aligned_descriptions (List[Optional[str]]): 对齐后的描述
        has_description (List[int]): 缺失指示符
        metadata (Dict[str, Any]): 元数据
    
    使用场景：
        在 datasets.py 中快速加载
    
    示例：
        >>> descs, has_desc, meta = load_descriptions_for_sample(
        ...     'S000196_P0000_T00',
        ...     './description/CSL_Daily'
        ... )
    """
    loader = DescriptionLoader(description_dir)
    original_descs, load_meta = loader.load(sample_id)
    
    if not load_meta['success'] or not original_descs:
        # 加载失败或无描述
        return None, None, load_meta
    
    # 假设采样帧索引为 0 到 len(descriptions)-1
    # 实际的采样帧索引应该从 load_pose 方法传递下来
    # 这里仅作示例
    sampled_indices = list(range(len(original_descs)))
    
    aligner = TemporalAligner(original_descs, sampled_indices)
    aligned, has_desc, align_meta = aligner.align_with_metadata()
    
    # 合并元数据
    metadata = {**load_meta, **align_meta}
    
    return aligned, has_desc, metadata
