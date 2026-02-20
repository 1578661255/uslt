#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本 - 检查 models.py 的参数修复

功能：
  验证 GatingFusion 和 LearnableMaskEmbedding 的初始化参数是否正确
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def check_models_initialization():
    """检查 models.py 中的参数初始化"""
    print("\n" + "="*70)
    print("验证 models.py 参数修复")
    print("="*70)
    
    try:
        with open('models.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查修复1：GatingFusion 参数
        if 'GatingFusion(\n                feature_dim=768,\n                gating_hidden_dim=512' in content:
            print("✓ GatingFusion 参数已修正")
            print("  - feature_dim=768")
            print("  - gating_hidden_dim=512")
        else:
            print("❌ GatingFusion 参数未正确修复")
            return False
        
        # 检查修复2：LearnableMaskEmbedding 参数
        if 'LearnableMaskEmbedding(hidden_dim=self.text_feature_dim)' in content:
            print("✓ LearnableMaskEmbedding 参数已修正")
            print("  - hidden_dim=self.text_feature_dim")
        else:
            print("❌ LearnableMaskEmbedding 参数未正确修复")
            return False
        
        # 移除错误的参数名
        if 'pose_feature_dim' not in content and 'text_feature_dim=self' not in content.split('GatingFusion')[1].split(')')[-0]:
            print("✓ 旧的错误参数已移除")
        else:
            print("⚠ 警告：可能仍存在旧的参数")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def check_text_fusion_modules():
    """检查 text_fusion_modules.py 中的类定义"""
    print("\n" + "="*70)
    print("验证 text_fusion_modules.py 中的类签名")
    print("="*70)
    
    try:
        with open('text_fusion_modules.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查1：GatingFusion 的定义
        if 'def __init__(self,\n                 feature_dim: int = 768,\n                 gating_hidden_dim: int = 512)' in content:
            print("✓ GatingFusion 类定义正确")
            print("  参数：feature_dim, gating_hidden_dim")
        else:
            print("❌ GatingFusion 类定义检查失败")
            return False
        
        # 检查2：LearnableMaskEmbedding 的定义
        if 'def __init__(self,\n                 hidden_dim: int = 768,' in content:
            print("✓ LearnableMaskEmbedding 类定义正确")
            print("  参数：hidden_dim, init_std")
        else:
            print("❌ LearnableMaskEmbedding 类定义检查失败")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def check_syntax():
    """检查 Python 语法"""
    print("\n" + "="*70)
    print("验证 Python 语法")
    print("="*70)
    
    try:
        import py_compile
        py_compile.compile('models.py', doraise=True)
        print("✓ models.py 语法检查通过")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ 语法错误: {e}")
        return False


def main():
    """主验证函数"""
    print("\n" + "="*70)
    print("Stage 3 多模态融合 - 参数修复验证")
    print("="*70)
    
    results = {
        "models.py 参数修复": check_models_initialization(),
        "text_fusion_modules.py 类定义": check_text_fusion_modules(),
        "Python 语法": check_syntax(),
    }
    
    print("\n" + "="*70)
    print("验证总结")
    print("="*70)
    
    for check_name, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {check_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("✓ 所有验证通过！修复成功")
        print("="*70)
        print("\n修复内容：")
        print("  1. GatingFusion 参数修正")
        print("     From: GatingFusion(pose_feature_dim=768, text_feature_dim=768, hidden_dim=768)")
        print("     To:   GatingFusion(feature_dim=768, gating_hidden_dim=512)")
        print("\n  2. LearnableMaskEmbedding 参数修正")
        print("     From: LearnableMaskEmbedding(dim=self.text_feature_dim)")
        print("     To:   LearnableMaskEmbedding(hidden_dim=self.text_feature_dim)")
        print("\n✓ 现在可以重新运行训练脚本")
        return 0
    else:
        print("\n❌ 部分验证失败，请查看上述输出")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
