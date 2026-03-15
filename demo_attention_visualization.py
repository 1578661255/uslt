"""
注意力可视化完整演示脚本
========================

这个脚本展示了如何使用注意力可视化功能来分析模型是否正确关注手部细节。

使用方法：
    python demo_attention_visualization.py
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from attention_visualization import AttentionVisualizer


def demo_generated_data():
    """
    使用生成的模拟数据进行演示
    """
    print("\n" + "="*60)
    print("注意力可视化演示 - 生成的模拟数据")
    print("="*60)
    
    visualizer = AttentionVisualizer(device='cpu')
    
    # 1. 生成模拟骨架数据 (T, 133, 2)
    # 133个关键点，每个点有(x, y)坐标
    T = 20  # 时间步长
    num_keypoints = 133
    
    # 生成逼真的骨架数据：关键点在0-255范围内
    skeleton = np.random.rand(T, num_keypoints, 2) * 255
    
    # 确保手部关键点（91-132）更活跃（更靠近图像中心）
    skeleton[:, 91:112, :] = 120 + np.random.randn(T, 21, 2) * 30  # 左手
    skeleton[:, 112:133, :] = 120 + np.random.randn(T, 21, 2) * 30  # 右手
    
    print(f"[数据] 生成骨架数据: shape={skeleton.shape}")
    
    # 2. 生成模拟注意力权重
    # 情景1：模型正确关注手部
    attn_weights_good = torch.randn(8, T, T)  # (num_heads, T, T)
    
    # 让手部关键点的注意力权重更高
    for head in range(8):
        for t in range(T):
            # 身体关键点（0-16）的注意力较低
            attn_weights_good[head, :, :17] *= 0.5
            # 手部关键点（91-132）的注意力较高
            attn_weights_good[head, :, 91:] *= 2.0
    
    attn_weights_good = torch.nn.functional.softmax(attn_weights_good, dim=-1)
    
    # 情景2：模型未能正确关注手部（对比组）
    attn_weights_bad = torch.randn(8, T, T)
    # 不进行特殊处理，就是均匀分布
    attn_weights_bad = torch.nn.functional.softmax(attn_weights_bad, dim=-1)
    
    print(f"[数据] 生成注意力权重: shape={attn_weights_good.shape}")
    
    # 3. 提取注意力统计
    print("\n[分析1] 模型正确关注手部的情况")
    stats_good = visualizer.extract_hand_attention(attn_weights_good)
    print(f"  手部注意力: {stats_good['hand_attention']:.4f}")
    print(f"  身体注意力: {stats_good['body_attention']:.4f}")
    print(f"  手部/身体比率: {stats_good['hand_vs_body_ratio']:.2f}x")
    print(f"  → 评价: {'✓ 模型正确聚焦手部' if stats_good['hand_vs_body_ratio'] > 1.2 else '✗ 聚焦不够'}")
    
    print("\n[分析2] 模型未能正确关注手部的情况（对比）")
    stats_bad = visualizer.extract_hand_attention(attn_weights_bad)
    print(f"  手部注意力: {stats_bad['hand_attention']:.4f}")
    print(f"  身体注意力: {stats_bad['body_attention']:.4f}")
    print(f"  手部/身体比率: {stats_bad['hand_vs_body_ratio']:.2f}x")
    print(f"  → 评价: {'✓ 模型正确聚焦手部' if stats_bad['hand_vs_body_ratio'] > 1.2 else '✗ 聚焦不足（期望情况）'}")
    
    # 4. 生成热力图
    output_dir = Path('./attention_viz_demo')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n[可视化1] 生成好的模型的热力图...")
    import cv2
    heatmap_good = visualizer.generate_attention_heatmap(
        skeleton,
        attn_weights_good,
        frame_size=(800, 800),
        highlight_hands=True
    )
    heatmap_good_path = output_dir / 'heatmap_good_focus.png'
    cv2.imwrite(str(heatmap_good_path), cv2.cvtColor(heatmap_good, cv2.COLOR_RGB2BGR))
    print(f"  → 保存到: {heatmap_good_path}")
    
    print(f"[可视化2] 生成差的模型的热力图...")
    heatmap_bad = visualizer.generate_attention_heatmap(
        skeleton,
        attn_weights_bad,
        frame_size=(800, 800),
        highlight_hands=True
    )
    heatmap_bad_path = output_dir / 'heatmap_bad_focus.png'
    cv2.imwrite(str(heatmap_bad_path), cv2.cvtColor(heatmap_bad, cv2.COLOR_RGB2BGR))
    print(f"  → 保存到: {heatmap_bad_path}")
    
    # 5. 生成骨架可视化
    print(f"[可视化3] 生成好的模型的骨架可视化...")
    skeleton_good = visualizer.visualize_skeleton_with_attention(
        skeleton,
        attn_weights_good,
        frame_size=(800, 800)
    )
    skeleton_good_path = output_dir / 'skeleton_good_focus.png'
    cv2.imwrite(str(skeleton_good_path), cv2.cvtColor(skeleton_good, cv2.COLOR_RGB2BGR))
    print(f"  → 保存到: {skeleton_good_path}")
    
    print(f"[可视化4] 生成差的模型的骨架可视化...")
    skeleton_bad = visualizer.visualize_skeleton_with_attention(
        skeleton,
        attn_weights_bad,
        frame_size=(800, 800)
    )
    skeleton_bad_path = output_dir / 'skeleton_bad_focus.png'
    cv2.imwrite(str(skeleton_bad_path), cv2.cvtColor(skeleton_bad, cv2.COLOR_RGB2BGR))
    print(f"  → 保存到: {skeleton_bad_path}")
    
    # 6. 绘制统计对比图
    print(f"[可视化5] 生成统计对比图...")
    stats_list = [stats_good] * T  # 每一帧都有相同的统计（演示用）
    stats_path = output_dir / 'hand_focus_comparison.png'
    visualizer.plot_attention_statistics(
        stats_list,
        video_name="Demo (Good Focus)",
        save_path=str(stats_path)
    )
    print(f"  → 保存到: {stats_path}")
    
    # 7. 生成详细报告
    print(f"\n[报告] 注意力分析详细报告")
    print("-" * 60)
    print(f"{'指标':<20} {'好的模型':<15} {'差的模型':<15} {'差异':<15}")
    print("-" * 60)
    
    diff_hand = abs(stats_good['hand_attention'] - stats_bad['hand_attention'])
    print(f"{'手部注意力':<20} {stats_good['hand_attention']:>14.4f} {stats_bad['hand_attention']:>14.4f} {diff_hand:>14.4f}")
    
    diff_body = abs(stats_good['body_attention'] - stats_bad['body_attention'])
    print(f"{'身体注意力':<20} {stats_good['body_attention']:>14.4f} {stats_bad['body_attention']:>14.4f} {diff_body:>14.4f}")
    
    diff_ratio = stats_good['hand_vs_body_ratio'] - stats_bad['hand_vs_body_ratio']
    print(f"{'手部/身体比率':<20} {stats_good['hand_vs_body_ratio']:>14.2f}x {stats_bad['hand_vs_body_ratio']:>14.2f}x {diff_ratio:>14.2f}x")
    
    print("-" * 60)
    
    print(f"\n✓ 演示完成！所有结果已保存到 {output_dir}/")
    print(f"\n推荐查看的文件：")
    print(f"  1. heatmap_good_focus.png - 好的模型的热力图（蓝色框标注手部区域）")
    print(f"  2. heatmap_bad_focus.png - 差的模型的热力图（对比）")
    print(f"  3. skeleton_good_focus.png - 骨架+注意力融合可视化")
    print(f"  4. hand_focus_comparison.png - 统计数据对比图")


def demo_hand_regions_analysis():
    """
    演示对手部不同区域的注意力分析
    """
    print("\n" + "="*60)
    print("手部细粒度区域分析")
    print("="*60)
    
    visualizer = AttentionVisualizer()
    
    # 生成注意力权重
    T = 20
    attn_weights = torch.randn(8, T, T)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # 提取手部注意力
    stats = visualizer.extract_hand_attention(attn_weights, hand_side='both')
    
    print(f"\n[手部注意力统计]")
    print(f"  左手21个关键点的注意力分布:")
    left_hand_attn = stats.get('hand_keypoints_attn', np.array([]))[:21]
    if len(left_hand_attn) > 0:
        print(f"    平均: {left_hand_attn.mean():.4f}")
        print(f"    最大: {left_hand_attn.max():.4f} (关键点 {left_hand_attn.argmax()})")
        print(f"    最小: {left_hand_attn.min():.4f} (关键点 {left_hand_attn.argmin()})")
    
    print(f"\n  右手21个关键点的注意力分布:")
    right_hand_attn = stats.get('hand_keypoints_attn', np.array([]))[21:] if len(stats.get('hand_keypoints_attn', np.array([]))) > 21 else np.array([])
    if len(right_hand_attn) > 0:
        print(f"    平均: {right_hand_attn.mean():.4f}")
        print(f"    最大: {right_hand_attn.max():.4f} (关键点 {right_hand_attn.argmax()})")
        print(f"    最小: {right_hand_attn.min():.4f} (关键点 {right_hand_attn.argmin()})")
    
    print(f"\n[解释]")
    print(f"  - 如果右手的注意力明显高于左手，说明模型在处理主要手时表现更好")
    print(f"  - 指尖和掌心的注意力最高，表示模型正确聚焦于细粒度动作")
    print(f"  - 如果整体注意力分布较为均匀，可能表示模型没有正确区分对象")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("■ 注意力可视化完整演示")
    print("="*60)
    
    # 运行演示
    demo_generated_data()
    demo_hand_regions_analysis()
    
    print("\n" + "="*60)
    print("实际应用步骤：")
    print("="*60)
    print("""
1. 在 fine_tuning.py 中启用注意力返回：
   args.return_attention_weights = True
   
2. 使用推理脚本进行可视化：
   python inference_with_attention.py \\
       --model_path ./outputs/checkpoint.pt \\
       --num_samples 10 \\
       --benchmark
       
3. 查看生成的热力图和统计报告
   
4. 检查以下指标确认模型是否正确关注手部：
   ✓ hand_vs_body_ratio > 1.2 (手部注意力显著高于身体)
   ✓ 热力图中手部区域（蓝色框）颜色最深/最亮
   ✓ 左右手都有相似的注意力分布（对称性）
   ✓ 指尖和掌心区域的注意力最高
    """)
