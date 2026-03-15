"""
═════════════════════════════════════════════════════════════════════════════
                    注意力可视化 - 使用示例代码
═════════════════════════════════════════════════════════════════════════════

这个脚本展示了如何在你的代码中使用注意力可视化功能。
直接修改下面的配置，然后运行：python example_attention_analysis.py
"""

import sys
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# 配置部分 - 修改这里！
# ═════════════════════════════════════════════════════════════════════════════

# 你的模型检查点路径（必须）
CHECKPOINT_PATH = './outputs/checkpoint.pt'  # ← 改成你的模型路径

# 分析参数
NUM_SAMPLES = 10           # 分析多少个样本
OUTPUT_DIR = './attention_results'  # 输出目录
DEVICE = 'cuda:0'          # GPU设备 ('cuda:0', 'cuda:1', or 'cpu')


# ═════════════════════════════════════════════════════════════════════════════
# 使用方式A：最简单（推荐新手）
# ═════════════════════════════════════════════════════════════════════════════

def method_A_quickest():
    """最快的方式 - 一行命令"""
    from quick_attention_analysis import analyze_model
    
    print("\n方式A: 一行命令分析")
    print("-" * 70)
    
    result = analyze_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_samples=NUM_SAMPLES,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )
    
    print(f"\n结果: {result['status']}")
    if 'avg_ratio' in result:
        print(f"平均手部/身体比率: {result['avg_ratio']:.2f}x")
    print(f"输出目录: {result['output_dir']}")


# ═════════════════════════════════════════════════════════════════════════════
# 使用方式B：稍微复杂一点（推荐中等用户）
# ═════════════════════════════════════════════════════════════════════════════

def method_B_normal():
    """标准方式 - 有一些控制"""
    from quick_attention_analysis import ModelAnalyzer
    
    print("\n方式B: 标准使用方式")
    print("-" * 70)
    
    # 创建分析器
    analyzer = ModelAnalyzer(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE
    )
    
    # 分析样本
    analyzer.analyze_samples(
        num_samples=NUM_SAMPLES,
        output_dir=OUTPUT_DIR
    )
    
    # 查看统计结果
    if analyzer.stats_history:
        stats = analyzer.stats_history[-1]
        print(f"\n分析完成:")
        print(f"  样本数: {stats['num_samples']}")
        print(f"  平均比率: {stats['avg_ratio']:.2f}x")


# ═════════════════════════════════════════════════════════════════════════════
# 使用方式C：高级（推荐有经验的用户，或需要自定义的）
# ═════════════════════════════════════════════════════════════════════════════

def method_C_advanced():
    """高级方式 - 完全控制分析流程"""
    import torch
    import cv2
    from models import Uni_Sign
    from datasets import S2T_Dataset_CSL_Daily
    from attention_visualization import AttentionVisualizer
    
    print("\n方式C: 高级自定义分析")
    print("-" * 70)
    
    device = DEVICE
    
    # 1. 加载模型
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    class Args:
        hidden_dim = 256
        dataset = 'CSL_Daily'
        device = device
        label_smoothing = 0.1
        rgb_support = True
        use_descriptions = True
        use_desc_feature = True
        encoder_type = 'mt5'
        fusion_type = 'cross_attention'
        text_dropout_p = 0.0
        text_encoder_freeze = False
        return_attention_weights = True  # ← 关键！必须启用
    
    args = Args()
    model = Uni_Sign(args)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    print("✓ 模型加载完成")
    
    # 2. 加载数据集
    dataset = S2T_Dataset_CSL_Daily(split='dev', use_descriptions=True, use_desc_feature=True)
    print(f"✓ 数据集加载完成 (大小: {len(dataset)})")
    
    # 3. 初始化可视化工具
    visualizer = AttentionVisualizer(device=device)
    
    # 4. 分析指定的样本
    print(f"\n分析 {NUM_SAMPLES} 个样本...")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    for sample_idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[sample_idx]
        
        # 准备输入
        src = {k: v.unsqueeze(0).to(device) for k, v in sample['src_input'].items() 
               if isinstance(v, torch.Tensor)}
        tgt = sample['tgt_input']
        
        # 推理 - 关键：会返回注意力权重
        with torch.no_grad():
            output = model(src, tgt)
        
        # 获取注意力权重（这是新增的）
        if 'fusion_attention_weights' in output:
            attn = output['fusion_attention_weights']
            
            # 获取骨架数据
            skeleton = sample['src_input']['body'].numpy()
            
            # 生成热力图
            heatmap = visualizer.generate_attention_heatmap(
                skeleton, attn[0], highlight_hands=True
            )
            cv2.imwrite(
                str(output_dir / f'heatmap_{sample_idx:04d}.png'),
                cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            )
            
            # 提取统计
            stats = visualizer.extract_hand_attention(attn[0])
            all_stats.append(stats)
            
            print(f"  样本{sample_idx}: 手部/身体比率 = {stats['hand_vs_body_ratio']:.2f}x")
    
    # 5. 绘制汇总统计
    if all_stats:
        visualizer.plot_attention_statistics(
            all_stats,
            video_name="custom_analysis",
            save_path=str(output_dir / 'statistics.png')
        )
        
        avg_ratio = sum(s['hand_vs_body_ratio'] for s in all_stats) / len(all_stats)
        print(f"\n统计摘要:")
        print(f"  平均比率: {avg_ratio:.2f}x")
        print(f"  评价: {'优秀' if avg_ratio > 1.5 else '良好' if avg_ratio > 1.2 else '一般'}")


# ═════════════════════════════════════════════════════════════════════════════
# 入口点
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("注意力可视化 - 完整示例")
    print("="*70)
    
    # 检查模型文件
    if not Path(CHECKPOINT_PATH).exists():
        print(f"\n✗ 错误: 模型文件不存在")
        print(f"  期望路径: {Path(CHECKPOINT_PATH).absolute()}")
        print(f"\n  请修改此文件中的 CHECKPOINT_PATH 变量")
        return False
    
    try:
        # 根据用户需求选择分析方式
        # 取消注释想要的方式（默认是方式A）
        
        method_A_quickest()      # ← 最简单，推荐
        # method_B_normal()       # 标准方式
        # method_C_advanced()     # 高级方式
        
        print("\n" + "="*70)
        print("✓ 分析完成！")
        print(f"  结果保存到: {Path(OUTPUT_DIR).absolute()}")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    from pathlib import Path
    
    success = main()
    sys.exit(0 if success else 1)
