"""
直接在Python代码中使用注意力可视化 - 无需命令行

示例:
    from quick_attention_analysis import analyze_model
    
    # 最简单的使用
    analyze_model(checkpoint_path='model.pt', num_samples=10)
    
    # 或者更细致的控制
    from quick_attention_analysis import ModelAnalyzer
    analyzer = ModelAnalyzer('model.pt', device='cuda:0')
    analyzer.analyze_samples(10)
    analyzer.save_results('./my_results/')
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import Uni_Sign
from datasets import S2T_Dataset_CSL_Daily
from attention_visualization import AttentionVisualizer


class ModelAnalyzer:
    """简化版本的模型分析工具 - 直接用checkpoint"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0', 
                 dataset_name: str = 'CSL_Daily'):
        """
        加载模型
        
        参数:
            checkpoint_path: 模型文件路径
            device: GPU设备 ('cuda:0', 'cuda:1'等)
            dataset_name: 数据集名称
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.dataset_name = dataset_name
        self.model = None
        self.visualizer = AttentionVisualizer(device=device)
        self.stats_history = []
        
        self._load_model()
        print(f"✓ 模型已加载: {self.checkpoint_path}")
    
    def _load_model(self):
        """加载模型权重"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 构建完整的args对象
        class Args:
            pass
        
        args = Args()
        
        # 基础参数
        args.hidden_dim = 256
        args.dataset = self.dataset_name
        args.device = self.device
        args.label_smoothing = 0.1
        
        # 功能开关
        args.rgb_support = True
        args.use_descriptions = True
        args.use_desc_feature = True
        args.return_attention_weights = True  # 关键：启用注意力返回
        
        # 融合配置
        args.encoder_type = 'mt5'
        args.fusion_type = 'cross_attention'
        args.text_dropout_p = 0.0  # 推理不用dropout
        args.text_encoder_freeze = False
        
        # 创建模型实例
        self.model = Uni_Sign(args)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_samples(self, num_samples: int = 5, output_dir: str = './attention_results'):
        """
        分析样本并生成可视化
        
        参数:
            num_samples: 要分析的样本数
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n分析 {num_samples} 个样本...")
        
        # 加载数据集
        try:
            dataset = S2T_Dataset_CSL_Daily(
                split='dev',
                use_descriptions=True,
                use_desc_feature=True
            )
        except Exception as e:
            print(f"⚠ 无法加载数据集: {e}")
            return False
        
        all_stats = []
        
        for idx in range(min(num_samples, len(dataset))):
            print(f"  [{idx+1}/{num_samples}] 处理样本...", end='\r')
            
            try:
                sample = dataset[idx]
                
                # 准备数据
                src = {}
                for key, val in sample['src_input'].items():
                    if isinstance(val, torch.Tensor):
                        src[key] = val.unsqueeze(0).to(self.device)
                    else:
                        src[key] = val
                
                tgt = sample['tgt_input']
                
                # 推理
                with torch.no_grad():
                    output = self.model(src, tgt)
                
                # 提取注意力
                if 'fusion_attention_weights' not in output:
                    print(f"⚠ 样本{idx}: 未找到注意力权重")
                    continue
                
                attn = output['fusion_attention_weights']
                
                # 获取骨架
                skeleton = sample['src_input']['body'].numpy()
                if skeleton.ndim == 3:
                    skeleton = skeleton[0]  # 如果有batch维度
                
                # 生成热力图
                heatmap = self.visualizer.generate_attention_heatmap(
                    skeleton, attn[0], highlight_hands=True
                )
                
                # 保存热力图
                heatmap_path = output_dir / f'heatmap_{idx:04d}.png'
                cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
                
                # 提取统计
                stats = self.visualizer.extract_hand_attention(attn[0])
                all_stats.append(stats)
                
            except Exception as e:
                print(f"⚠ 样本{idx}处理失败: {e}")
                continue
        
        print()
        
        # 绘制统计图表
        if all_stats:
            stats_path = output_dir / 'statistics.png'
            self.visualizer.plot_attention_statistics(
                all_stats,
                video_name="analysis",
                save_path=str(stats_path)
            )
            
            # 打印总结
            avg_ratio = sum(s['hand_vs_body_ratio'] for s in all_stats) / len(all_stats)
            print(f"\n统计摘要:")
            print(f"  样本数: {len(all_stats)}")
            print(f"  平均手部/身体比率: {avg_ratio:.2f}x")
            print(f"  评级: {'优秀 (>1.5)' if avg_ratio > 1.5 else '良好 (1.2-1.5)' if avg_ratio > 1.2 else '一般 (1.0-1.2)' if avg_ratio > 1.0 else '不足 (<1.0)'}")
            
            self.stats_history.append({
                'num_samples': len(all_stats),
                'avg_ratio': avg_ratio,
                'stats': all_stats
            })
        
        return True
    
    def save_results(self, output_dir: str = './attention_results'):
        """结果已在analyze_samples中保存，这个函数提供额外的统计导出"""
        pass


# ============================================================================
# 快捷函数 - 一键分析
# ============================================================================

def analyze_model(checkpoint_path: str, num_samples: int = 5, 
                  output_dir: str = './attention_results',
                  device: str = 'cuda:0') -> Dict:
    """
    一键分析模型 - 最简单的使用方式
    
    参数:
        checkpoint_path: 模型文件路径
        num_samples: 分析样本数
        output_dir: 输出目录
        device: GPU设备
    
    返回:
        包含统计信息的字典
    
    示例:
        >> result = analyze_model('model.pt', 10)
        >> print(f"平均比率: {result['avg_ratio']:.2f}x")
    """
    analyzer = ModelAnalyzer(checkpoint_path, device=device)
    analyzer.analyze_samples(num_samples, output_dir)
    
    if analyzer.stats_history:
        history = analyzer.stats_history[-1]
        return {
            'num_samples': history['num_samples'],
            'avg_ratio': history['avg_ratio'],
            'status': '✓ 分析成功',
            'output_dir': output_dir
        }
    else:
        return {
            'status': '✗ 分析失败',
            'output_dir': output_dir
        }


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    print("""
╔═════════════════════════════════════════════════════════════╗
║         注意力可视化 - Python API 使用示例                   ║
╚═════════════════════════════════════════════════════════════╝

方式1: 最简单（一行代码）
────────────────────────
    from quick_attention_analysis import analyze_model
    result = analyze_model('checkpoint.pt', num_samples=10, 
                          output_dir='./results/')
    print(f"平均比率: {result['avg_ratio']:.2f}x")


方式2: 自定义分析
────────────────
    from quick_attention_analysis import ModelAnalyzer
    
    analyzer = ModelAnalyzer('checkpoint.pt', device='cuda:0')
    analyzer.analyze_samples(num_samples=20, output_dir='./results/')


方式3: 在Jupyter中使用
─────────────────────
    %run quick_attention_analysis.py
    
    analyzer = ModelAnalyzer('model.pt')
    analyzer.analyze_samples(5)


关键参数说明:
    checkpoint_path: 模型文件路径（绝对或相对路径）
    num_samples: 分析的样本数（更多样本=更准确，但更慢）
    device: 'cuda:0', 'cuda:1', 或 'cpu'
    output_dir: 热力图保存目录


输出说明:
    heatmap_*.png: 注意力热力图（越红=注意力越高）
    statistics.png: 统计对比图表
    
    关键指标 hand_vs_body_ratio:
        > 1.5x  : 优秀 ✓
        1.2-1.5 : 良好 ✓
        1.0-1.2 : 一般
        < 1.0   : 不足 ✗
    """)
