"""
注意力可视化模块
=================

功能：
    1. 提取模型的注意力权重
    2. 生成热力图
    3. 叠加到骨架图和RGB帧上
    4. 分析注意力聚焦的区域（手部vs背景）

使用示例：
    visualizer = AttentionVisualizer(model, device='cuda')
    attn_map = visualizer.visualize_frame(
        skeleton_data=skeleton,      # (133, 2) 的关键点坐标
        rgb_frame=rgb_frame,         # (H, W, 3) RGB图像
        attention_weights=attn_w,    # Cross-Attention权重或Gate权重
        hand_keypoint_indices=[91:112, 112:133],  # 左右手的关键点索引
        save_path='attention_viz.png'
    )
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AttentionVisualizer:
    """注意力权重可视化工具"""
    
    def __init__(self, device: str = 'cuda'):
        """
        初始化可视化工具
        
        参数：
            device: 计算设备
        """
        self.device = device
        
        # OpenPose 骨架骨连接定义
        self.skeleton_connections = [
            # 身体（COCO格式）
            (0, 1), (1, 2), (2, 3), (3, 4),  # 头部
            (1, 5), (5, 6), (6, 7),          # 左臂
            (1, 8), (8, 9), (9, 10),        # 右臂
            (0, 11), (11, 12), (12, 13),     # 左腿
            (0, 14), (14, 15), (15, 16),     # 右腿
        ]
        
        # 手部关键点索引（OpenPose格式）
        self.hand_indices = {
            'left': list(range(91, 112)),    # 91-111: 左手21个关键点
            'right': list(range(112, 133)),  # 112-132: 右手21个关键点
        }
        
        # 身体关键点索引
        self.body_indices = list(range(0, 17))
        
    def extract_hand_attention(self,
                              attention_weights: torch.Tensor,
                              hand_side: str = 'both') -> Dict:
        """
        从注意力权重中提取手部注意力统计
        
        参数：
            attention_weights: (T, T) 或 (B, num_heads, T, T)
            hand_side: 'left', 'right' 或 'both'
        
        返回：
            统计字典：
            {
                'hand_attention': 手部平均注意力值 (0-1),
                'background_attention': 背景注意力值,
                'hand_vs_bg_ratio': 手部/背景比率,
                'hand_keypoints_attn': 各手部关键点的注意力 (21,)
            }
        """
        # 处理不同的形状
        if attention_weights.dim() == 4:
            # (B, num_heads, T, T) → 取平均后 (T, T)
            attention_weights = attention_weights.mean(dim=(0, 1))
        elif attention_weights.dim() == 3:
            # (num_heads, T, T) → 取平均后 (T, T)
            attention_weights = attention_weights.mean(dim=0)
        
        # 确保是numpy或tensor
        if isinstance(attention_weights, torch.Tensor):
            attn_np = attention_weights.cpu().numpy()
        else:
            attn_np = attention_weights
        
        T = attn_np.shape[0]
        
        # 获取手部关键点的注意力
        hand_indices = self.hand_indices['left'] + self.hand_indices['right'] if hand_side == 'both' else self.hand_indices[hand_side]
        hand_indices = [i for i in hand_indices if i < T]
        
        if not hand_indices:
            return {'error': '没有有效的手部关键点'}
        
        # 从所有时间步的注意力中聚合
        hand_attn_matrix = attn_np[hand_indices, :]  # (num_hand_kpts, T)
        hand_attn = hand_attn_matrix.mean()
        
        # 背景/身体注意力
        body_indices = [i for i in self.body_indices if i < T]
        if body_indices:
            body_attn = attn_np[body_indices, :].mean()
        else:
            body_attn = attn_np.mean()
        
        return {
            'hand_attention': float(hand_attn),
            'body_attention': float(body_attn),
            'hand_vs_body_ratio': float(hand_attn / (body_attn + 1e-6)),
            'hand_keypoints_attn': hand_attn_matrix.mean(axis=1),  # (num_hand_kpts,)
        }
    
    def generate_attention_heatmap(self,
                                  skeleton: np.ndarray,
                                  attention_weights: torch.Tensor,
                                  frame_size: Tuple[int, int] = (1000, 1000),
                                  highlight_hands: bool = True) -> np.ndarray:
        """
        生成基于骨架的注意力热力图
        
        参数：
            skeleton: (133, 2) 关键点坐标
            attention_weights: Cross-Attention权重或Gate权重
            frame_size: 输出热力图大小 (H, W)
            highlight_hands: 是否突出手部区域
        
        返回：
            热力图 (H, W, 3) RGB图像
        """
        H, W = frame_size
        
        # 初始化热力图
        heatmap = np.zeros((H, W), dtype=np.float32)
        
        # 处理注意力权重形状
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=(0, 1))
        elif attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)
        
        if isinstance(attention_weights, torch.Tensor):
            attn_np = attention_weights.cpu().numpy()  # (T, T)
        else:
            attn_np = attention_weights
        
        # 对每个关键点，根据其注意力权重绘制高斯分布
        if skeleton.shape[0] != attn_np.shape[0]:
            # 如果大小不匹配，进行插值
            valid_kpts = min(skeleton.shape[0], attn_np.shape[0])
        else:
            valid_kpts = skeleton.shape[0]
        
        for kpt_idx in range(valid_kpts):
            x, y = skeleton[kpt_idx]
            
            # 检查坐标有效性
            if np.isnan(x) or np.isnan(y) or x <= 0 or y <= 0:
                continue
            
            # 将坐标映射到热力图范围
            x_scaled = int(x * W / 255)
            y_scaled = int(y * H / 255)
            
            if 0 <= x_scaled < W and 0 <= y_scaled < H:
                # 获取该关键点的注意力权重
                attn_value = attn_np[kpt_idx, :].mean()  # 对所有时间步取均值
                
                # 在热力图上绘制高斯分布
                y_min = max(0, y_scaled - 20)
                y_max = min(H, y_scaled + 20)
                x_min = max(0, x_scaled - 20)
                x_max = min(W, x_scaled + 20)
                
                yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                gaussian = np.exp(-((xx - x_scaled) ** 2 + (yy - y_scaled) ** 2) / (2 * 10 ** 2))
                heatmap[y_min:y_max, x_min:x_max] += gaussian * attn_value
        
        # 归一化热力图
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 转换为热力图颜色映射
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 如果指定，在手部区域添加蓝色轮廓
        if highlight_hands:
            heatmap_rgb = self._highlight_hand_regions(heatmap_rgb, skeleton, frame_size)
        
        return heatmap_rgb
    
    def _highlight_hand_regions(self,
                               image: np.ndarray,
                               skeleton: np.ndarray,
                               frame_size: Tuple[int, int]) -> np.ndarray:
        """
        在手部区域添加蓝色轮廓边界
        """
        H, W = frame_size
        image = image.copy()
        
        # 绘制左手边界框
        left_hand_kpts = skeleton[self.hand_indices['left']]
        valid_left = left_hand_kpts[~np.isnan(left_hand_kpts).any(axis=1)]
        if len(valid_left) > 0:
            x_min, y_min = (valid_left.min(axis=0) * [W/255, H/255]).astype(int)
            x_max, y_max = (valid_left.max(axis=0) * [W/255, H/255]).astype(int)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
            cv2.putText(image, 'Left Hand', (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 绘制右手边界框
        right_hand_kpts = skeleton[self.hand_indices['right']]
        valid_right = right_hand_kpts[~np.isnan(right_hand_kpts).any(axis=1)]
        if len(valid_right) > 0:
            x_min, y_min = (valid_right.min(axis=0) * [W/255, H/255]).astype(int)
            x_max, y_max = (valid_right.max(axis=0) * [W/255, H/255]).astype(int)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.putText(image, 'Right Hand', (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image
    
    def visualize_skeleton_with_attention(self,
                                         skeleton: np.ndarray,
                                         attention_weights: torch.Tensor,
                                         frame_size: Tuple[int, int] = (1080, 1080),
                                         save_path: Optional[str] = None) -> np.ndarray:
        """
        生成骨架 + 注意力热力图的组合可视化
        
        参数：
            skeleton: (133, 2) 关键点坐标或置信度
            attention_weights: 注意力权重张量
            frame_size: 输出大小
            save_path: 保存路径（可选）
        
        返回：
            可视化图像 (H, W, 3)
        """
        H, W = frame_size
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
        
        # 绘制骨架连接
        skeleton_scaled = skeleton.copy()
        skeleton_scaled[:, 0] = skeleton_scaled[:, 0] * W / 255
        skeleton_scaled[:, 1] = skeleton_scaled[:, 1] * H / 255
        
        # 绘制连接线
        for start, end in self.skeleton_connections:
            if start < len(skeleton_scaled) and end < len(skeleton_scaled):
                start_pt = skeleton_scaled[start].astype(int)
                end_pt = skeleton_scaled[end].astype(int)
                
                if np.all(start_pt > 0) and np.all(end_pt > 0):
                    cv2.line(canvas, tuple(start_pt), tuple(end_pt), (0, 255, 0), 2)
        
        # 绘制关键点（带注意力值着色）
        attn_np = attention_weights.cpu().numpy() if isinstance(attention_weights, torch.Tensor) else attention_weights
        if attn_np.ndim > 2:
            attn_np = attn_np.mean(axis=tuple(range(attn_np.ndim - 2)))
        
        for kpt_idx in range(len(skeleton_scaled)):
            x, y = skeleton_scaled[kpt_idx].astype(int)
            
            if 0 <= x < W and 0 <= y < H:
                # 根据注意力值着色
                if kpt_idx < attn_np.shape[0]:
                    attn_val = attn_np[kpt_idx, :].mean()
                    color_intensity = int(attn_val * 255)
                    color = (0, color_intensity, 255 - color_intensity)  # 红→黄→绿
                else:
                    color = (200, 200, 200)
                
                cv2.circle(canvas, (x, y), 5, color, -1)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        return canvas
    
    def plot_attention_statistics(self,
                                 stats_list: List[Dict],
                                 video_name: str = 'video',
                                 save_path: Optional[str] = None) -> None:
        """
        绘制注意力统计图表
        
        参数：
            stats_list: 统计信息列表（每一帧一个）
            video_name: 视频名称（用于标题）
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Attention Analysis - {video_name}', fontsize=16)
        
        # 提取数据
        hand_attns = [s['hand_attention'] for s in stats_list]
        body_attns = [s['body_attention'] for s in stats_list]
        ratios = [s['hand_vs_body_ratio'] for s in stats_list]
        
        # 1. 手部 vs 身体注意力趋势
        axes[0, 0].plot(hand_attns, label='Hand', color='red', linewidth=2)
        axes[0, 0].plot(body_attns, label='Body', color='blue', linewidth=2)
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Attention Value')
        axes[0, 0].set_title('Hand vs Body Attention Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 手部/身体比率
        axes[0, 1].plot(ratios, color='purple', linewidth=2)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='Balanced')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Hand/Body Ratio')
        axes[0, 1].set_title('Hand Attention Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 注意力分布直方图
        axes[1, 0].hist(hand_attns, bins=20, alpha=0.7, label='Hand', color='red')
        axes[1, 0].hist(body_attns, bins=20, alpha=0.7, label='Body', color='blue')
        axes[1, 0].set_xlabel('Attention Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Attention Distribution')
        axes[1, 0].legend()
        
        # 4. 统计摘要
        axes[1, 1].axis('off')
        summary_text = f"""
        Average Hand Attention: {np.mean(hand_attns):.4f}
        Average Body Attention: {np.mean(body_attns):.4f}
        Average Hand/Body Ratio: {np.mean(ratios):.2f}x
        
        Max Hand Attention: {np.max(hand_attns):.4f}
        Min Hand Attention: {np.min(hand_attns):.4f}
        
        Frames with Hand>Body: {sum(1 for h, b in zip(hand_attns, body_attns) if h > b)} / {len(hand_attns)}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[可视化] 统计图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_attention_comparison_video(self,
                                         skeletons: List[np.ndarray],
                                         attention_list: List[torch.Tensor],
                                         output_path: str,
                                         fps: int = 30) -> None:
        """
        创建注意力热力图视频
        
        参数：
            skeletons: 骨架列表 [(H,W), (H,W), ...]
            attention_list: 注意力权重列表
            output_path: 输出视频路径
            fps: 视频帧率
        """
        if len(skeletons) == 0:
            print("[警告] 没有骨架数据")
            return
        
        # 创建视频写入器
        H, W = (1080, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W*2, H))
        
        for idx, (skeleton, attn) in enumerate(zip(skeletons, attention_list)):
            print(f"\r处理帧 {idx+1}/{len(skeletons)}", end='')
            
            # 左侧：骨架 + 联接
            skeleton_viz = self.visualize_skeleton_with_attention(skeleton, attn, (H, W))
            
            # 右侧：热力图
            heatmap_viz = self.generate_attention_heatmap(skeleton, attn, (H, W), highlight_hands=True)
            
            # 拼接
            combined = np.hstack([skeleton_viz, heatmap_viz])
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            
            out.write(combined_bgr)
        
        out.release()
        print(f"\n[可视化] 可视化视频已保存到: {output_path}")


# 快速使用接口
def quick_attention_visualization(skeleton_batch: torch.Tensor,
                                 attention_weights: torch.Tensor,
                                 output_dir: str = './attention_viz',
                                 video_name: str = 'sample') -> None:
    """
    一键生成注意力可视化
    
    参数：
        skeleton_batch: (B, T, 133, 2) 骨架批次
        attention_weights: (B, T, T) 或 (B, num_heads, T, T) 注意力权重
        output_dir: 输出目录
        video_name: 视频名称
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    visualizer = AttentionVisualizer()
    
    # 处理第一个样本
    skeleton = skeleton_batch[0].cpu().numpy() if isinstance(skeleton_batch, torch.Tensor) else skeleton_batch[0]
    attn = attention_weights[0] if isinstance(attention_weights, torch.Tensor) else attention_weights[0]
    
    # 生成关键点热力图
    heatmap = visualizer.generate_attention_heatmap(skeleton, attn)
    cv2.imwrite(f'{output_dir}/{video_name}_heatmap.png', cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    
    # 生成骨架可视化
    skeleton_viz = visualizer.visualize_skeleton_with_attention(skeleton, attn)
    cv2.imwrite(f'{output_dir}/{video_name}_skeleton.png', cv2.cvtColor(skeleton_viz, cv2.COLOR_RGB2BGR))
    
    # 生成统计
    stats = visualizer.extract_hand_attention(attn)
    print(f"\n[注意力统计] {video_name}:")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    print(f"\n[可视化] 结果已保存到 {output_dir}/")
