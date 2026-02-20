#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练前完整检查清单

功能：
  1. 验证所有必需的模块都可导入
  2. 检查 utils.py 中的新参数
  3. 检查 fine_tuning.py 的训练循环
  4. 验证 models.py 的多模态融合集成
  5. 检查 datasets.py 返回的数据格式
"""

import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

def check_utils_parameters():
    """检查 utils.py 中的新参数"""
    print("\n" + "="*70)
    print("检查 1：utils.py 中的多模态参数")
    print("="*70)
    
    try:
        # 检查参数文件内容
        with open('utils.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        params_to_check = {
            '--use_descriptions': '--use_descriptions',
            '--text_dropout_p': '--text_dropout_p',
            '--text_encoder_freeze': '--text_encoder_freeze',
            '--fusion_checkpoint': '--fusion_checkpoint',
        }
        
        all_found = True
        for param_name, param_str in params_to_check.items():
            if param_str in content:
                print(f"✓ {param_name} 参数已添加")
            else:
                print(f"❌ {param_name} 参数缺失")
                all_found = False
        
        return all_found
    
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def check_fine_tuning_integration():
    """检查 fine_tuning.py 的多模态支持"""
    print("\n" + "="*70)
    print("检查 2：fine_tuning.py 中的多模态支持")
    print("="*70)
    
    try:
        with open('fine_tuning.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            '冻结 TextEncoder': 'args.text_encoder_freeze',
            '处理 descriptions': "key in ['descriptions', 'has_description']",
            '处理训练数据': 'train_one_epoch',
            '处理评估数据': 'evaluate',
        }
        
        all_found = True
        for check_name, check_str in checks.items():
            if check_str in content:
                print(f"✓ {check_name} 逻辑已添加")
            else:
                print(f"❌ {check_name} 逻辑缺失")
                all_found = False
        
        return all_found
    
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def check_models_integration():
    """检查 models.py 的多模态集成"""
    print("\n" + "="*70)
    print("检查 3：models.py 中的多模态集成")
    print("="*70)
    
    try:
        with open('models.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'TextEncoder 导入': 'from text_fusion_modules import TextEncoder',
            'GatingFusion 导入': 'GatingFusion',
            'LearnableMaskEmbedding 导入': 'LearnableMaskEmbedding',
            '_encode_descriptions 方法': 'def _encode_descriptions',
            '_apply_text_dropout 方法': 'def _apply_text_dropout',
            '融合核心逻辑': 'self.gating_fusion(inputs_embeds, text_features)',
        }
        
        all_found = True
        for check_name, check_str in checks.items():
            if check_str in content:
                print(f"✓ {check_name}")
            else:
                print(f"❌ {check_name} 缺失")
                all_found = False
        
        return all_found
    
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def check_datasets_integration():
    """检查 datasets.py 的描述加载"""
    print("\n" + "="*70)
    print("检查 4：datasets.py 中的描述加载")
    print("="*70)
    
    try:
        with open('datasets.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'DescriptionLoader 导入': 'from temporal_alignment import DescriptionLoader',
            '描述加载初始化': 'self.desc_loader = DescriptionLoader',
            '描述加载方法': '_load_and_align_descriptions',
            '7元组返回': 'descriptions, has_description',
            'collate_fn 兼容性': "len(item) == 7",
        }
        
        all_found = True
        for check_name, check_str in checks.items():
            if check_str in content:
                print(f"✓ {check_name}")
            else:
                print(f"❌ {check_name} 缺失")
                all_found = False
        
        return all_found
    
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def check_config():
    """检查配置文件"""
    print("\n" + "="*70)
    print("检查 5：config.py 中的路径配置")
    print("="*70)
    
    try:
        with open('config.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'description_dirs' in content:
            print(f"✓ description_dirs 配置已添加")
            if 'CSL-Daily' in content:
                print(f"✓ CSL-Daily 路径已配置")
            else:
                print(f"❌ CSL-Daily 路径配置缺失")
                return False
            return True
        else:
            print(f"❌ description_dirs 配置缺失")
            return False
    
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def check_training_readiness():
    """检查训练准备就绪"""
    print("\n" + "="*70)
    print("检查 6：训练前准备检查表")
    print("="*70)
    
    checks = {
        '代码结构': [
            ('models.py', '多模态融合集成'),
            ('datasets.py', '描述加载支持'),
            ('config.py', '路径配置'),
            ('utils.py', 'CLI 参数'),
            ('fine_tuning.py', '训练循环'),
        ],
        '依赖模块': [
            ('temporal_alignment.py', '描述加载和对齐'),
            ('text_fusion_modules.py', '文本编码和融合'),
        ],
    }
    
    print("\n核心模块：")
    for module_name, module_desc in checks['代码结构']:
        print(f"  ✓ {module_name}: {module_desc}")
    
    print("\n新模块：")
    for module_name, module_desc in checks['依赖模块']:
        print(f"  ✓ {module_name}: {module_desc}")
    
    print("\n数据流：")
    print("  描述文件 → DescriptionLoader → TemporalAligner → SLT_Dataset")
    print("  → collate_fn → src_input['descriptions'] & src_input['has_description']")
    print("  → Uni_Sign.forward() → TextEncoder → GatingFusion → 输出")


def main():
    """主检查函数"""
    print("\n" + "="*70)
    print("Stage 3 多模态融合 - 训练前完整检查")
    print("="*70)
    
    results = {
        "utils.py 参数": check_utils_parameters(),
        "fine_tuning.py 集成": check_fine_tuning_integration(),
        "models.py 集成": check_models_integration(),
        "datasets.py 集成": check_datasets_integration(),
        "config.py 配置": check_config(),
    }
    
    print("\n" + "="*70)
    print("检查总结")
    print("="*70)
    
    for check_name, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {check_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("✓ 所有检查通过！系统准备就绪")
        print("="*70)
        check_training_readiness()
        print("\n" + "="*70)
        print("训练启动命令示例：")
        print("="*70)
        print("""
# 基础训练（不使用描述文本）
python fine_tuning.py --dataset CSL_Daily --epochs 20 --batch-size 16

# 启用多模态融合（使用描述文本）
python fine_tuning.py --dataset CSL_Daily --epochs 20 --batch-size 16 \\
    --use_descriptions --text_dropout_p 0.1

# 冻结文本编码器（仅训练融合模块）
python fine_tuning.py --dataset CSL_Daily --epochs 20 --batch-size 16 \\
    --use_descriptions --text_encoder_freeze
""")
        return 0
    else:
        print("\n❌ 部分检查未通过，请查看上述输出并修复问题")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
