#!/usr/bin/env python
"""
快速可视化脚本 - 直接加载checkpoint并生成注意力热力图
==========================================================

使用方法（最简单的形式）:
    python run_attention_viz.py --model checkpoint.pt --samples 10 --out results/

支持的参数:
    --model CHECKPOINT_PATH    模型路径（必须）
    --samples N               样本数（默认5）
    --out OUTPUT_DIR          输出目录（默认./attention_results/）
    --device cuda:0           GPU设备（默认cuda:0）
    --data-dir DATA_DIR       数据目录（默认./data/）
"""

import torch
import argparse
import sys
from pathlib import Path
import traceback

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from inference_with_attention import AttentionInferenceEngine
from attention_visualization import AttentionVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='注意力可视化 - 加载checkpoint并分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析10个样本
  python run_attention_viz.py --model checkpoint.pt --samples 10

  # 自定义输出目录
  python run_attention_viz.py --model model.pt --out ./my_results/

  # 使用CPU
  python run_attention_viz.py --model model.pt --device cpu
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型检查点路径（必须）')
    parser.add_argument('--samples', type=int, default=5,
                       help='分析的样本数（默认5）')
    parser.add_argument('--out', type=str, default='./attention_results',
                       help='输出目录（默认./attention_results）')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备（默认cuda:0）')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='数据目录（默认./data）')
    parser.add_argument('--dataset', type=str, default='CSL_Daily',
                       help='数据集名称（默认CSL_Daily）')
    
    args = parser.parse_args()
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        return False
    
    print("="*70)
    print("注意力可视化分析")
    print("="*70)
    print(f"模型: {model_path.absolute()}")
    print(f"样本数: {args.samples}")
    print(f"输出目录: {args.out}")
    print(f"设备: {args.device}")
    print()
    
    try:
        # 初始化推理引擎
        print("[1/3] 加载模型...")
        engine = AttentionInferenceEngine(str(model_path), device=args.device)
        
        # 加载数据集
        print("[2/3] 加载数据集...")
        try:
            from datasets import S2T_Dataset
            dataset = S2T_Dataset(
                split='dev',
                use_descriptions=True,
                use_desc_feature=True
            )
            print(f"      数据集大小: {len(dataset)}")
        except Exception as e:
            print(f"[警告] 无法加载数据集: {e}")
            print("      使用演示模式（生成模拟数据）...")
            dataset = None
        
        # 生成可视化
        print("[3/3] 生成可视化...")
        if dataset is not None:
            indices = list(range(min(args.samples, len(dataset))))
            engine.visualize_batch(
                dataset,
                indices,
                args.out,
                save_video=False
            )
        else:
            print("      跳过（无可用数据集）")
        
        print()
        print("="*70)
        print("✓ 分析完成！")
        print(f"结果保存到: {Path(args.out).absolute()}")
        print("="*70)
        
        # 显示结果统计
        output_dir = Path(args.out)
        if output_dir.exists():
            heatmaps = list(output_dir.glob('heatmap_*.png'))
            skeletons = list(output_dir.glob('skeleton_*.png'))
            print(f"\n生成的文件:")
            print(f"  热力图: {len(heatmaps)} 张")
            print(f"  骨架图: {len(skeletons)} 张")
            if output_dir.glob('statistics.png'):
                print(f"  统计图: 1 张")
        
        return True
        
    except Exception as e:
        print(f"\n[错误] 分析失败:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
