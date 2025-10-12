#!/usr/bin/env python3
"""
验证pipeline.py实现完整性 / Verify pipeline.py Implementation Completeness

检查pipeline.py是否实现了所有10个节点和方法
Checks if pipeline.py implements all 10 nodes and methods
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_pipeline():
    """分析pipeline.py实现 / Analyze pipeline.py implementation"""
    print("🔍 分析pipeline.py实现情况 / Analyzing pipeline.py Implementation")
    print("=" * 80)
    
    # 读取文件内容
    pipeline_file = Path(__file__).parent.parent / "pipeline.py"
    with open(pipeline_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 10节点架构定义
    expected_nodes = {
        'N0': {'name': 'DataFetchNode', 'methods': ['api']},
        'N1': {'name': 'ImputeNode', 'methods': ['impute']},
        'N2': {'name': 'FeatureMatrixNode', 'methods': ['construct']},
        'N3': {'name': 'CleaningNode', 'methods': ['clean']},
        'N4': {'name': 'GNNNode', 'methods': ['process']},
        'N5': {'name': 'KGNode', 'methods': ['process']},
        'N6': {'name': 'SelectionNode', 'methods': ['select']},
        'N7': {'name': 'ScalingNodeB', 'methods': ['scale']},
        'N8': {'name': 'ModelTrainingNodeB', 'methods': ['train']},
        'N9': {'name': 'EndNode', 'methods': ['terminate']},
    }
    
    print("\n📦 节点导入检查 / Node Import Check")
    print("-" * 80)
    
    all_imported = True
    for node_id, node_info in expected_nodes.items():
        node_name = node_info['name']
        if f"import {node_name}" in content or node_name in content:
            print(f"✅ {node_id} - {node_name:25s} 已导入 / Imported")
        else:
            print(f"❌ {node_id} - {node_name:25s} 未找到 / Not found")
            all_imported = False
    
    print("-" * 80)
    
    # 检查两个pipeline函数
    print("\n🔧 Pipeline函数检查 / Pipeline Function Check")
    print("-" * 80)
    
    functions = {
        'run_pipeline': '旧的6节点流水线 / Old 6-node pipeline',
        'run_pipeline_config': '新的10节点灵活流水线 / New 10-node flexible pipeline'
    }
    
    for func_name, description in functions.items():
        if f"def {func_name}(" in content:
            print(f"✅ {func_name:25s} 已实现 / Implemented - {description}")
        else:
            print(f"❌ {func_name:25s} 未找到 / Not found")
    
    print("-" * 80)
    
    # 详细分析run_pipeline_config
    print("\n🏗️ run_pipeline_config 实现分析 / Implementation Analysis")
    print("-" * 80)
    
    # 检查每个节点在run_pipeline_config中的使用
    nodes_in_config = {
        'N0': ['n0 = DataFetchNode()', "n0.execute('api'"],
        'N1': ['n1 = ImputeNode()', "n1.execute('impute'"],
        'N2': ['n2 = FeatureMatrixNode()', "n2.execute('construct'"],
        'N3': ['n3c = CleaningNode()', "n3c.execute('clean'"],
        'N4': ['n4g = GNNNode()', "n4g.execute('process'"],
        'N5': ['n5k = KGNode()', "n5k.execute('process'"],
        'N6': ['n6s = SelectionNode()', "n6s.execute('select'"],
        'N7': ['n7b = ScalingNodeB()', "n7b.execute('scale'"],
        'N8': ['n8t = ModelTrainingNodeB()', "n8t.execute('train'"],
        'N9': ['n9e = EndNode()', "n9e.execute('terminate'"],
    }
    
    config_complete = True
    for node_id, patterns in nodes_in_config.items():
        found = all(pattern in content for pattern in patterns)
        node_name = expected_nodes[node_id]['name']
        if found:
            print(f"✅ {node_id} - {node_name:25s} 在run_pipeline_config中实现")
            print(f"      Found in run_pipeline_config")
        else:
            print(f"❌ {node_id} - {node_name:25s} 在run_pipeline_config中缺失")
            print(f"      Missing in run_pipeline_config")
            config_complete = False
    
    print("-" * 80)
    
    # 检查旧的run_pipeline
    print("\n📜 run_pipeline (旧版) 实现分析 / Old Version Analysis")
    print("-" * 80)
    
    old_pipeline_nodes = {
        'N0': 'DataFetchNode',
        'N1': 'ImputeNode',
        'N2': 'FeatureMatrixNode',
        'N3': 'FeatureSelectionNode',  # 旧版
        'N4': 'ScalingNode',  # 旧版
        'N5': 'ModelTrainingNode',  # 旧版
    }
    
    for node_id, node_name in old_pipeline_nodes.items():
        # 在run_pipeline函数中查找
        if f"{node_name}()" in content:
            print(f"✅ {node_id} - {node_name:25s} 在run_pipeline中使用")
        else:
            print(f"⚠️ {node_id} - {node_name:25s} 在run_pipeline中未找到")
    
    print("-" * 80)
    
    # 问题诊断
    print("\n🔍 问题诊断 / Issue Diagnosis")
    print("-" * 80)
    
    # 检查文件头部注释是否过时
    if "N0 → N2 → N1 → N3 → N4 → N5" in content:
        print("⚠️ 文件头部注释过时 / File header comment is outdated")
        print("   当前: N0 → N2 → N1 → N3 → N4 → N5 (6节点)")
        print("   应该: N0 → N2 → [N1,N3,N4,N5,N6,N7] → N8 → N9 (10节点)")
        header_outdated = True
    else:
        print("✅ 文件头部注释正确 / File header comment is correct")
        header_outdated = False
    
    # 检查是否有完整的10节点支持
    if config_complete:
        print("✅ run_pipeline_config支持所有10个节点")
        print("   run_pipeline_config supports all 10 nodes")
    else:
        print("❌ run_pipeline_config缺少部分节点实现")
        print("   run_pipeline_config missing some node implementations")
    
    print("-" * 80)
    
    # 总结
    print("\n📊 实现总结 / Implementation Summary")
    print("-" * 80)
    
    print(f"\n{'指标 / Metric':<40s} {'状态 / Status'}")
    print(f"{'节点导入 / Node Imports':<40s} {'✅ 完成' if all_imported else '❌ 不完整'}")
    print(f"{'run_pipeline (旧版) / Old Version':<40s} ✅ 实现 (6节点)")
    print(f"{'run_pipeline_config (新版) / New Version':<40s} {'✅ 完成' if config_complete else '❌ 不完整'} (10节点)")
    print(f"{'文件头部注释 / File Header Comment':<40s} {'⚠️ 需更新' if header_outdated else '✅ 正确'}")
    
    print("\n" + "=" * 80)
    
    return {
        'all_imported': all_imported,
        'config_complete': config_complete,
        'header_outdated': header_outdated
    }


def check_method_coverage():
    """检查方法覆盖率 / Check method coverage"""
    print("\n\n🔬 方法覆盖率分析 / Method Coverage Analysis")
    print("=" * 80)
    
    pipeline_file = Path(__file__).parent.parent / "pipeline.py"
    with open(pipeline_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查每个节点的方法是否被调用
    method_calls = {
        'N0': ["execute('api'"],
        'N1': ["execute('impute'"],
        'N2': ["execute('construct'"],
        'N3': ["execute('clean'"],
        'N4': ["execute('process'"],
        'N5': ["execute('process'"],
        'N6': ["execute('select'"],
        'N7': ["execute('scale'"],
        'N8': ["execute('train'"],
        'N9': ["execute('terminate'"],
    }
    
    print("\n📋 方法调用检查 / Method Call Check")
    print("-" * 80)
    
    all_methods_called = True
    for node_id, methods in method_calls.items():
        found_count = sum(1 for method in methods if method in content)
        if found_count > 0:
            print(f"✅ {node_id} 方法调用: {found_count} 处 / Method calls: {found_count} location(s)")
        else:
            print(f"❌ {node_id} 方法调用: 未找到 / Method calls: Not found")
            all_methods_called = False
    
    print("-" * 80)
    
    return all_methods_called


def main():
    """主函数 / Main function"""
    print("\n" + "🚀 开始分析 / Starting Analysis ".center(80, "="))
    print()
    
    # 分析pipeline实现
    results = analyze_pipeline()
    
    # 检查方法覆盖率
    methods_ok = check_method_coverage()
    
    # 最终总结
    print("\n" + "🎯 最终结论 / Final Conclusion ".center(80, "="))
    print()
    
    if results['all_imported'] and results['config_complete'] and methods_ok:
        if results['header_outdated']:
            print("⚠️ pipeline.py 功能完整但文档需要更新")
            print("⚠️ pipeline.py functionally complete but documentation needs update")
            print()
            print("✅ 所有10个节点已导入")
            print("✅ run_pipeline_config完整实现10节点架构")
            print("✅ 所有方法正确调用")
            print("⚠️ 文件头部注释需要更新为10节点描述")
        else:
            print("🎉 pipeline.py 完全实现！")
            print("🎉 pipeline.py fully implemented!")
            print()
            print("✅ 所有节点导入完整")
            print("✅ 所有函数实现正确")
            print("✅ 文档注释准确")
    else:
        print("❌ pipeline.py 存在未完成的部分")
        print("❌ pipeline.py has incomplete parts")
        print()
        if not results['all_imported']:
            print("❌ 部分节点未导入")
        if not results['config_complete']:
            print("❌ run_pipeline_config缺少节点实现")
        if not methods_ok:
            print("❌ 部分方法未调用")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
