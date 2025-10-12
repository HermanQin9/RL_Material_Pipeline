#!/usr/bin/env python3
"""
对比两个流水线函数的节点使用 / Compare Node Usage in Two Pipeline Functions

清楚展示 run_pipeline() 和 run_pipeline_config() 的区别
Clearly shows the differences between run_pipeline() and run_pipeline_config()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_pipeline_functions():
    """分析两个流水线函数 / Analyze two pipeline functions"""
    print("🔍 对比分析 run_pipeline() vs run_pipeline_config()")
    print("=" * 80)
    
    # 旧版6节点映射
    old_pipeline_nodes = {
        'N0': {'class': 'DataFetchNode', 'purpose': '数据获取 / Data Fetch'},
        'N1': {'class': 'ImputeNode', 'purpose': '缺失值填充 / Imputation'},
        'N2': {'class': 'FeatureMatrixNode', 'purpose': '特征矩阵 / Feature Matrix'},
        'N3': {'class': 'FeatureSelectionNode', 'purpose': '特征选择 (旧) / Feature Selection (Old)'},
        'N4': {'class': 'ScalingNode', 'purpose': '缩放 (旧) / Scaling (Old)'},
        'N5': {'class': 'ModelTrainingNode', 'purpose': '模型训练 (旧) / Model Training (Old)'},
    }
    
    # 新版10节点映射
    new_pipeline_nodes = {
        'N0': {'class': 'DataFetchNode', 'purpose': '数据获取 / Data Fetch', 'position': '固定首位'},
        'N1': {'class': 'ImputeNode', 'purpose': '缺失值填充 / Imputation', 'position': '灵活'},
        'N2': {'class': 'FeatureMatrixNode', 'purpose': '特征矩阵 / Feature Matrix', 'position': '固定第二'},
        'N3': {'class': 'CleaningNode', 'purpose': '数据清洗 (新) / Cleaning (New)', 'position': '灵活'},
        'N4': {'class': 'GNNNode', 'purpose': '图神经网络 (新) / GNN (New)', 'position': '灵活'},
        'N5': {'class': 'KGNode', 'purpose': '知识图谱 (新) / Knowledge Graph (New)', 'position': '灵活'},
        'N6': {'class': 'SelectionNode', 'purpose': '特征选择 (新位置) / Selection (New Position)', 'position': '灵活'},
        'N7': {'class': 'ScalingNodeB', 'purpose': '缩放 (新位置) / Scaling (New Position)', 'position': '灵活'},
        'N8': {'class': 'ModelTrainingNodeB', 'purpose': '模型训练 (新位置) / Training (New Position)', 'position': '固定倒二'},
        'N9': {'class': 'EndNode', 'purpose': '终止节点 / End Node', 'position': '固定终点'},
    }
    
    print("\n📊 节点对比表 / Node Comparison Table")
    print("-" * 80)
    print(f"{'节点ID':<8} {'旧版(run_pipeline)':<35} {'新版(run_pipeline_config)':<35}")
    print("-" * 80)
    
    for node_id in ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']:
        old_info = old_pipeline_nodes.get(node_id, {})
        new_info = new_pipeline_nodes.get(node_id, {})
        
        old_text = f"{old_info.get('class', '❌ 不存在')}" if old_info else "❌ 不存在 / Not Exist"
        new_text = f"{new_info.get('class', '-')}" if new_info else "-"
        
        # 标记重要变化
        if node_id in ['N3', 'N4', 'N5']:
            marker = "⚠️ 改变"
        elif node_id in ['N6', 'N7', 'N8', 'N9']:
            marker = "🆕 新增"
        else:
            marker = "✅ 相同"
        
        print(f"{node_id:<8} {old_text:<35} {new_text:<35} {marker}")
    
    print("-" * 80)
    
    # 详细说明
    print("\n📝 关键差异说明 / Key Differences")
    print("-" * 80)
    
    print("\n🔴 旧版 run_pipeline() 特点:")
    print("  1. 只有 6 个节点 (N0-N5)")
    print("  2. 固定执行顺序: N0→N2→N1→N3→N4→N5")
    print("  3. N3=特征选择, N4=缩放, N5=模型训练")
    print("  4. 没有 GNN 和知识图谱节点")
    print("  5. 用于向后兼容")
    
    print("\n🟢 新版 run_pipeline_config() 特点:")
    print("  1. 有 10 个节点 (N0-N9)")
    print("  2. 灵活执行顺序: N0→N2→[flexible]→N8→N9")
    print("  3. N3=数据清洗, N4=GNN, N5=知识图谱")
    print("  4. N6=特征选择, N7=缩放, N8=模型训练, N9=终止")
    print("  5. 支持 PPO 强化学习优化")
    
    # 为什么停在N5
    print("\n❓ 为什么旧版在N5就停止了？")
    print("-" * 80)
    print("  答案: 因为旧架构设计时只规划了6个节点！")
    print("  ")
    print("  旧版设计思路:")
    print("    N0: 获取数据")
    print("    N2: 构建特征")
    print("    N1: 填充缺失")
    print("    N3: 选择特征")
    print("    N4: 缩放数据")
    print("    N5: 训练模型 → 完成！返回结果")
    print("  ")
    print("  新版10节点是后来为了支持PPO和更多功能而扩展的！")
    
    # 代码位置
    print("\n📂 代码位置 / Code Location")
    print("-" * 80)
    print(f"  旧版 run_pipeline():        pipeline.py 第 65-165 行")
    print(f"  新版 run_pipeline_config(): pipeline.py 第 168-325 行")
    
    # 使用建议
    print("\n💡 使用建议 / Usage Recommendations")
    print("-" * 80)
    print("  🔹 新项目: 使用 run_pipeline_config()")
    print("     - 支持更多节点")
    print("     - 灵活的节点组合")
    print("     - PPO强化学习优化")
    print("  ")
    print("  🔹 旧项目: 可以继续使用 run_pipeline()")
    print("     - 向后兼容")
    print("     - 简单快速")
    print("     - 无需修改旧代码")
    
    print("\n" + "=" * 80)


def show_code_examples():
    """展示代码示例 / Show code examples"""
    print("\n\n📝 代码示例对比 / Code Example Comparison")
    print("=" * 80)
    
    print("\n🔴 旧版 run_pipeline() 使用示例:")
    print("-" * 80)
    print("""
from pipeline import run_pipeline

result = run_pipeline(
    cache=True,
    impute_strategy='mean',      # N1
    selection_strategy='pca',     # N3 (旧版的特征选择)
    scaling_strategy='standard',  # N4 (旧版的缩放)
    model_strategy='rf',          # N5 (旧版的模型训练)
)
# 自动执行: N0→N2→N1→N3→N4→N5 (6个节点)
# 没有 N6, N7, N8, N9
    """)
    
    print("\n🟢 新版 run_pipeline_config() 使用示例:")
    print("-" * 80)
    print("""
from pipeline import run_pipeline_config

config = {
    'sequence': ['N0','N2','N1','N3','N4','N5','N6','N7','N8','N9'],
    'N1_method': 'median',      # 缺失值填充
    'N3_method': 'outlier',     # 数据清洗 (新)
    'N4_method': 'gat',         # GNN (新)
    'N5_method': 'entity',      # 知识图谱 (新)
    'N6_method': 'pca',         # 特征选择 (新位置)
    'N7_method': 'std',         # 缩放 (新位置)
    'N8_method': 'xgb',         # 模型训练 (新位置)
    'cache': True
}
result = run_pipeline_config(**config)
# 可以执行所有 10 个节点！
# PPO 可以控制中间节点的顺序和选择
    """)
    
    print("\n" + "=" * 80)


def main():
    """主函数 / Main function"""
    print("\n" + "🚀 开始分析 / Starting Analysis ".center(80, "="))
    
    analyze_pipeline_functions()
    show_code_examples()
    
    print("\n" + "🎯 总结 / Summary ".center(80, "="))
    print()
    print("❓ 为什么旧版 run_pipeline() 在 N5 就停了？")
    print()
    print("✅ 答案:")
    print("  1. 旧版只有 6 个节点设计 (N0-N5)")
    print("  2. N5 是旧架构的终点（模型训练完成）")
    print("  3. N6-N9 是新架构才有的节点")
    print("  4. 新版 run_pipeline_config() 实现了所有 10 个节点")
    print("  5. 两个函数是独立的流水线系统，用于不同目的")
    print()
    print("💡 建议: 新项目使用 run_pipeline_config() 以获得完整的 10 节点支持！")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
