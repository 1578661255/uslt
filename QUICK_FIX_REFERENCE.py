"""
快速参考：参数修复前后对照

======================================================================
修复 1：GatingFusion 参数
======================================================================

❌ 错误代码（导致 TypeError）：
────────────────────────────────────────────────────────────────────
self.gating_fusion = GatingFusion(
    pose_feature_dim=768,                    # 不存在此参数！
    text_feature_dim=self.text_feature_dim,  # 不存在此参数！
    hidden_dim=768                           # 不存在此参数！
)

✓ 正确代码：
────────────────────────────────────────────────────────────────────
self.gating_fusion = GatingFusion(
    feature_dim=768,        # 与 mT5-base 输出维度一致
    gating_hidden_dim=512   # 融合模块内部 MLP 隐层维度
)

错误原因分析：
────────────────────────────────────────────────────────────────────
text_fusion_modules.py 中 GatingFusion 的实际定义：

    def __init__(self, feature_dim: int = 768, gating_hidden_dim: int = 512):
        ...

所以：
  • pose_feature_dim ──> feature_dim
  • text_feature_dim ──> 应该融合到 feature_dim 中
  • hidden_dim ──> gating_hidden_dim


======================================================================
修复 2：LearnableMaskEmbedding 参数
======================================================================

❌ 错误代码（导致 TypeError）：
────────────────────────────────────────────────────────────────────
self.mask_embedding = LearnableMaskEmbedding(dim=self.text_feature_dim)

✓ 正确代码：
────────────────────────────────────────────────────────────────────
self.mask_embedding = LearnableMaskEmbedding(hidden_dim=self.text_feature_dim)

错误原因分析：
────────────────────────────────────────────────────────────────────
text_fusion_modules.py 中 LearnableMaskEmbedding 的实际定义：

    def __init__(self, hidden_dim: int = 768, init_std: float = 0.01):
        ...

所以：
  • dim ──> hidden_dim
  • init_std 使用默认值 0.01


======================================================================
影响范围分析
======================================================================

受影响的代码行：
  • models.py 第 168-175 行（Uni_Sign.__init__ 方法）
  • 只有这两处初始化使用了错误的参数名

相关类定义：
  • GatingFusion（text_fusion_modules.py 第 150-160 行）
  • LearnableMaskEmbedding（text_fusion_modules.py 第 200-210 行）


======================================================================
验证步骤
======================================================================

1. 检查 models.py 第 169-175 行是否已修正
2. 运行 verify_fix.py 进行自动检查
3. 上传文件到服务器
4. 清理服务器缓存：find . -type d -name __pycache__ -exec rm -rf {} +
5. 重新运行训练脚本


======================================================================
常见错误处理
======================================================================

问题：修复后仍显示"unexpected keyword argument"
────────────────────────────────────────────────────────────────────
解决：
  1. 确认文件是否正确上传
  2. 删除服务器 __pycache__ 目录
  3. 检查是否还有其他的错误参数

问题：提示"module not found: text_fusion_modules"
────────────────────────────────────────────────────────────────────
解决：
  1. 确认 text_fusion_modules.py 在 models.py 同目录
  2. 检查 sys.path 配置
  3. 验证导入语句：from .text_fusion_modules import ...

问题：修复后出现其他类型错误
────────────────────────────────────────────────────────────────────
解决：
  1. 查看完整的错误堆栈跟踪
  2. 检查是否有其他参数不匹配
  3. 验证依赖库版本（transformers, torch 等）
"""

# 用于快速查找和参考的映射表
PARAMETER_MAPPING = {
    "GatingFusion": {
        "错误参数": ["pose_feature_dim", "text_feature_dim", "hidden_dim"],
        "正确参数": ["feature_dim", "gating_hidden_dim"],
        "默认值": {"feature_dim": 768, "gating_hidden_dim": 512},
        "文件": "text_fusion_modules.py",
        "行号": "150-160"
    },
    "LearnableMaskEmbedding": {
        "错误参数": ["dim"],
        "正确参数": ["hidden_dim"],
        "默认值": {"hidden_dim": 768, "init_std": 0.01},
        "文件": "text_fusion_modules.py",
        "行号": "200-210"
    }
}

# 修复状态记录
FIXES_APPLIED = [
    {
        "类名": "GatingFusion",
        "文件": "models.py",
        "行号": "169-171",
        "状态": "✓ 已修复",
        "修复时间": "FIX_TIMESTAMP",
    },
    {
        "类名": "LearnableMaskEmbedding",
        "文件": "models.py",
        "行号": "175",
        "状态": "✓ 已修复",
        "修复时间": "FIX_TIMESTAMP",
    }
]

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("参数映射表")
    print("="*70)
    for cls_name, info in PARAMETER_MAPPING.items():
        print(f"\n{cls_name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
