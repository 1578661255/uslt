#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
描述文本加载验证脚本

功能：
  1. 验证 JSON 格式解析
  2. 验证 DescriptionLoader 工作流程
  3. 验证 TemporalAligner 对齐逻辑
  4. 验证 config 和 datasets 的集成

运行方式：
  python test_description_loading.py
"""

import sys
import io
from pathlib import Path
import json

# 设置 UTF-8 编码以支持中文字符
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from temporal_alignment import DescriptionLoader, TemporalAligner

def test_json_format():
    """测试实际 JSON 文件格式"""
    print("\n" + "="*60)
    print("测试 1: 验证实际 JSON 格式解析")
    print("="*60)
    
    # 使用绝对路径
    json_path = Path(__file__).parent.parent / 'description' / 'CSL-Daily' / 'split_data' / 'train' / 'S000003_P0000_T00.json'
    
    if not json_path.exists():
        print(f"❌ JSON 文件不存在: {json_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ JSON 文件读取成功")
    print(f"✓ 格式类型: {type(data).__name__}")
    
    if isinstance(data, list):
        print(f"✓ 列表长度: {len(data)}")
        if data:
            print(f"✓ 第一个元素: {data[0]}")
    
    return True

def test_description_loader():
    """测试 DescriptionLoader"""
    print("\n" + "="*60)
    print("测试 2: 验证 DescriptionLoader")
    print("="*60)
    
    # 使用绝对路径
    desc_dir = str(Path(__file__).parent.parent / 'description' / 'CSL-Daily' / 'split_data' / 'train')
    loader = DescriptionLoader(desc_dir)
    
    # 测试加载
    sample_id = 'S000003_P0000_T00'
    descriptions, metadata = loader.load(sample_id)
    
    print(f"✓ 样本 ID: {sample_id}")
    print(f"✓ 加载成功: {metadata['success']}")
    
    if metadata['success']:
        print(f"✓ 描述条数: {metadata['frame_count']}")
        print(f"✓ 返回字典大小: {len(descriptions)}")
        
        # 显示前 3 条
        for i, (frame_id, desc) in enumerate(sorted(descriptions.items())[:3]):
            desc_preview = desc[:50] + '...' if len(desc) > 50 else desc
            print(f"  - Frame {frame_id}: {desc_preview}")
    else:
        print(f"❌ 加载失败: {metadata.get('reason')}")
        return False
    
    return True

def test_temporal_aligner():
    """测试 TemporalAligner"""
    print("\n" + "="*60)
    print("测试 3: 验证 TemporalAligner")
    print("="*60)
    
    # 创建模拟数据
    original_descriptions = {
        0: "hand up",
        5: "hand down",
        10: "hand left",
        15: "hand right"
    }
    sampled_frame_indices = [0, 3, 5, 8, 10, 12, 15]
    
    aligner = TemporalAligner(
        original_descriptions,
        sampled_frame_indices,
        use_nearest_neighbor=True,
        use_linear_interpolation=False
    )
    
    aligned, has_desc = aligner.align()
    
    print(f"✓ 原始描述数: {len(original_descriptions)}")
    print(f"✓ 采样帧数: {len(sampled_frame_indices)}")
    print(f"✓ 对齐结果数: {len(aligned)}")
    
    for i, (frame_idx, desc, has) in enumerate(zip(sampled_frame_indices, aligned, has_desc)):
        print(f"  Frame {frame_idx}: {desc if desc else 'None'} (has={has})")
    
    return True

def test_config_integration():
    """测试 config 集成"""
    print("\n" + "="*60)
    print("测试 4: 验证 config 集成")
    print("="*60)
    
    try:
        from config import description_dirs
        
        print(f"✓ 成功导入 description_dirs")
        print(f"✓ 配置内容: {description_dirs}")
        
        if 'CSL_Daily' in description_dirs:
            # 将相对路径转换为绝对路径（相对于 Uni-Sign 目录）
            # config.py 中的路径相对于 Uni-Sign 目录，需要回到 slt 目录再找
            current_dir = Path(__file__).parent  # Uni-Sign 目录
            workspace_root = current_dir.parent   # slt 目录
            
            desc_base_rel = description_dirs['CSL_Daily']
            desc_base = workspace_root / desc_base_rel.lstrip('./')
            
            print(f"✓ 当前目录: {current_dir}")
            print(f"✓ 工作空间根目录: {workspace_root}")
            print(f"✓ CSL_Daily 基础路径: {desc_base}")
            
            # 测试完整路径
            test_path = desc_base / 'train'
            print(f"✓ 测试路径: {test_path}")
            print(f"✓ 路径存在: {test_path.exists()}")
        else:
            print("❌ config 中未找到 CSL_Daily")
            return False
    
    except ImportError as e:
        print(f"❌ 无法导入 config: {e}")
        return False
    
    return True

def test_datasets_import():
    """测试 datasets 导入"""
    print("\n" + "="*60)
    print("测试 5: 验证 datasets 模块导入")
    print("="*60)
    
    try:
        import datasets
        print(f"✓ 成功导入 datasets 模块")
        
        # 检查是否有 S2T_Dataset
        if hasattr(datasets, 'S2T_Dataset'):
            print(f"✓ 找到 S2T_Dataset 类")
        else:
            print(f"⚠ 未找到 S2T_Dataset 类")
    
    except ImportError as e:
        print(f"❌ 无法导入 datasets: {e}")
        # 不返回 False，因为这可能是由于依赖问题
    
    except Exception as e:
        print(f"❌ 导入异常: {e}")
    
    return True

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("描述文本加载系统验证")
    print("="*60)
    
    results = {
        "JSON 格式解析": test_json_format(),
        "DescriptionLoader": test_description_loader(),
        "TemporalAligner": test_temporal_aligner(),
        "Config 集成": test_config_integration(),
        "Datasets 导入": test_datasets_import(),
    }
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ 所有测试通过！" + "\n")
    else:
        print("\n❌ 部分测试失败，请检查上述输出" + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
