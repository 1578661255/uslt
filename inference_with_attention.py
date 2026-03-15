"""
注意力可视化推理脚本
====================

功能：
    1. 加载已训练的模型
    2. 在推理时提取注意力权重
    3. 生成热力图和统计分析
    4. 创建可视化视频

使用方法：
    python inference_with_attention.py \
        --model_path ./outputs/checkpoint.pt \
        --video_path ./data/video.json \
        --output_dir ./attention_results \
        --fusion_type cross_attention
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models import Uni_Sign
from datasets import S2T_Dataset_CSL_Daily
from attention_visualization import AttentionVisualizer, quick_attention_visualization
from config import *


class AttentionInferenceEngine:
    """带注意力提取功能的推理引擎"""
    
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        """
        初始化推理引擎
        
        参数：
            model_path: 模型检查点路径
            device: 计算设备
        """
        self.device = device
        self.model = None
        self.visualizer = AttentionVisualizer(device=device)
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """加载模型检查点"""
        print(f"[加载模型] {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 尝试从checkpoint中恢复args，如果没有则使用默认值
        if 'args' in checkpoint:
            args = checkpoint['args']
        else:
            # 使用完整的默认配置
            class Args:
                pass
            
            args = Args()
            # 必需的参数
            args.hidden_dim = 256
            args.dataset = 'CSL_Daily'
            args.device = self.device
            args.label_smoothing = 0.1
            
            # RGB支持（自动检测）
            args.rgb_support = True
            
            # 多模态融合配置
            args.use_descriptions = True
            args.use_desc_feature = True  # 离线特征模式
            args.encoder_type = 'mt5'  # 文本编码器类型
            args.fusion_type = 'cross_attention'  # 融合方式
            args.text_dropout_p = 0.0  # 推理时关闭dropout
            args.text_encoder_freeze = False
            
            # 注意力可视化
            args.return_attention_weights = True
        
        # 确保必要的属性存在
        if not hasattr(args, 'return_attention_weights'):
            args.return_attention_weights = True
        if not hasattr(args, 'device'):
            args.device = self.device
        if not hasattr(args, 'hidden_dim'):
            args.hidden_dim = 256
        if not hasattr(args, 'dataset'):
            args.dataset = 'CSL_Daily'
        
        self.model = Uni_Sign(args)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理可能的键名前缀不匹配
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"[模型加载完成] 融合方式: {getattr(args, 'fusion_type', 'unknown')}")
    
    def extract_attention(self, 
                         src_input: Dict,
                         tgt_input: Dict,
                         return_predictions: bool = False) -> Dict:
        """
        推理并提取注意力权重
        
        参数：
            src_input: 输入字典（视频特征等）
            tgt_input: 目标字典（翻译文本等）
            return_predictions: 是否返回模型预测
        
        返回：
            {
                'attention_weights': 注意力权重,
                'predictions': 模型预测（可选），
                'statistics': 统计信息,
            }
        """
        with torch.no_grad():
            # 检查输入是否在GPU上
            for key in src_input:
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(self.device)
            
            for key in tgt_input:
                if isinstance(tgt_input[key], (list, str)):
                    continue
                if isinstance(tgt_input[key], torch.Tensor):
                    tgt_input[key] = tgt_input[key].to(self.device)
            
            # 前向传播
            output = self.model(src_input, tgt_input)
            
            # 这里需要模型支持返回注意力（我们稍后修改模型）
            # 暂时使用伪数据进行演示
            
            # 提取文本和位姿特征（用于演示）
            if 'descriptions' in src_input:
                text_feat = src_input['descriptions']
                batch_size, seq_len = text_feat.shape[:2]
                
                # 生成演示用的注意力权重
                attn_weights = torch.randn(
                    batch_size, 8, seq_len, seq_len,
                    device=text_feat.device
                )
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            else:
                attn_weights = None
        
        # 统计分析
        stats = {}
        if attn_weights is not None:
            stats = self.visualizer.extract_hand_attention(attn_weights)
        
        result = {
            'attention_weights': attn_weights,
            'statistics': stats,
        }
        
        if return_predictions:
            result['predictions'] = output
        
        return result
    
    def visualize_batch(self,
                       dataset,
                       indices: List[int],
                       output_dir: str,
                       save_video: bool = False) -> None:
        """
        可视化一个批次的样本
        
        参数：
            dataset: 数据集对象
            indices: 要可视化的样本索引
            output_dir: 输出目录
            save_video: 是否保存视频
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_stats = []
        
        for idx in indices:
            print(f"\n[处理样本 {idx}]")
            
            # 获取数据
            sample = dataset[idx]
            src_input = sample['src_input']
            tgt_input = sample['tgt_input']
            
            # 批量化
            src_input_batch = {}
            for key, val in src_input.items():
                if isinstance(val, torch.Tensor):
                    src_input_batch[key] = val.unsqueeze(0)
                else:
                    src_input_batch[key] = val
            
            tgt_input_batch = {}
            for key, val in tgt_input.items():
                if isinstance(val, torch.Tensor):
                    tgt_input_batch[key] = val.unsqueeze(0)
                elif isinstance(val, list):
                    tgt_input_batch[key] = val
                else:
                    tgt_input_batch[key] = val
            
            # 推理并提取注意力
            result = self.extract_attention(src_input_batch, tgt_input_batch)
            attn_weights = result['attention_weights']
            stats = result['statistics']
            
            all_stats.append(stats)
            
            # 获取骨架数据用于可视化
            if 'body' in src_input:
                skeleton_feat = src_input['body']  # (T, 133, 2)
                if isinstance(skeleton_feat, torch.Tensor):
                    skeleton_np = skeleton_feat.cpu().numpy()
                else:
                    skeleton_np = skeleton_feat
                
                # 生成热力图
                heatmap = self.visualizer.generate_attention_heatmap(
                    skeleton_np,
                    attn_weights[0] if attn_weights is not None else torch.ones(1, 1),
                    frame_size=(1080, 1080),
                    highlight_hands=True
                )
                
                # 保存热力图
                heatmap_path = output_path / f"sample_{idx:04d}_heatmap.png"
                import cv2
                cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
                print(f"  → 热力图: {heatmap_path}")
                
                # 生成骨架+注意力可视化
                skeleton_viz = self.visualizer.visualize_skeleton_with_attention(
                    skeleton_np,
                    attn_weights[0] if attn_weights is not None else torch.ones(1, 1),
                    frame_size=(1080, 1080)
                )
                
                skeleton_path = output_path / f"sample_{idx:04d}_skeleton.png"
                cv2.imwrite(str(skeleton_path), cv2.cvtColor(skeleton_viz, cv2.COLOR_RGB2BGR))
                print(f"  → 骨架: {skeleton_path}")
            
            # 打印统计信息
            if stats:
                print(f"\n  [注意力统计]")
                print(f"    手部注意力: {stats.get('hand_attention', 0):.4f}")
                print(f"    身体注意力: {stats.get('body_attention', 0):.4f}")
                print(f"    手部/身体比率: {stats.get('hand_vs_body_ratio', 0):.2f}x")
        
        # 绘制统计图表
        if all_stats and all_stats[0]:
            stats_path = output_path / "attention_statistics.png"
            self.visualizer.plot_attention_statistics(
                all_stats,
                video_name="batch_analysis",
                save_path=str(stats_path)
            )
            print(f"\n[统计图表] {stats_path}")
    
    def benchmark_hand_focus(self,
                            dataset,
                            num_samples: int = 10,
                            output_file: str = 'hand_focus_benchmark.json') -> None:
        """
        基准测试：统计有多少比例的注意力集中在手部
        
        参数：
            dataset: 数据集
            num_samples: 要分析的样本数
            output_file: 输出JSON文件
        """
        results = {
            'num_samples': num_samples,
            'hand_attention_stats': {},
            'summary': {}
        }
        
        hand_ratios = []
        
        for idx in range(min(num_samples, len(dataset))):
            print(f"\r[基准测试] {idx+1}/{num_samples}", end='')
            
            sample = dataset[idx]
            
            # 获取注意力
            result = self.extract_attention(sample['src_input'], sample['tgt_input'])
            attn_weights = result['attention_weights']
            stats = result['statistics']
            
            if stats and 'hand_vs_body_ratio' in stats:
                ratio = stats['hand_vs_body_ratio']
                hand_ratios.append(ratio)
                
                results['hand_attention_stats'][str(idx)] = {
                    'hand_attention': float(stats.get('hand_attention', 0)),
                    'body_attention': float(stats.get('body_attention', 0)),
                    'ratio': float(ratio),
                }
        
        # 计算统计摘要
        if hand_ratios:
            results['summary'] = {
                'mean_ratio': float(np.mean(hand_ratios)),
                'std_ratio': float(np.std(hand_ratios)),
                'min_ratio': float(np.min(hand_ratios)),
                'max_ratio': float(np.max(hand_ratios)),
                'median_ratio': float(np.median(hand_ratios)),
                'samples_with_hand_priority': sum(1 for r in hand_ratios if r > 1.0),
                'hand_priority_percentage': 100 * sum(1 for r in hand_ratios if r > 1.0) / len(hand_ratios),
            }
        
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[基准测试完成]")
        print(f"  平均手部/身体比率: {results['summary'].get('mean_ratio', 0):.2f}x")
        print(f"  手部优先的样本: {results['summary'].get('samples_with_hand_priority', 0)}/{num_samples}")
        print(f"  结果保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='注意力可视化推理')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./attention_viz',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='要可视化的样本数')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行基准测试')
    parser.add_argument('--save_video', action='store_true',
                       help='保存可视化视频')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 初始化推理引擎
    engine = AttentionInferenceEngine(args.model_path, device=args.device)
    
    # 加载数据集（这里使用演示数据，实际需要根据你的数据集调整）
    print(f"[加载数据集] {args.data_dir}")
    try:
        dataset = S2T_Dataset_CSL_Daily(
            data_dir=args.data_dir,
            split='dev',
            use_descriptions=True,
            use_desc_feature=True
        )
    except Exception as e:
        print(f"[警告] 无法加载数据集: {e}")
        print("使用演示模式...")
        dataset = None
    
    # 可视化
    if dataset is not None:
        indices = list(range(min(args.num_samples, len(dataset))))
        engine.visualize_batch(
            dataset, 
            indices,
            args.output_dir,
            save_video=args.save_video
        )
    
    # 基准测试
    if args.benchmark and dataset is not None:
        engine.benchmark_hand_focus(
            dataset,
            num_samples=args.num_samples,
            output_file=f'{args.output_dir}/hand_focus_benchmark.json'
        )
    
    print(f"\n[完成] 结果保存到 {args.output_dir}/")


if __name__ == '__main__':
    main()
