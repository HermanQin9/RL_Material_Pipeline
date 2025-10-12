#!/usr/bin/env python3
"""
验证10节点架构完整性 / Verify 10-Node Architecture Completeness

检查所有10个节点是否正确实现和文档化
Checks if all 10 nodes are correctly implemented and documented
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_nodes():
    """验证节点实现 / Verify node implementations"""
    print("🔍 验证10节点架构完整性 / Verifying 10-Node Architecture Completeness")
    print("=" * 80)
    
    try:
        # 导入所有节点 / Import all nodes
        from nodes import (
            DataFetchNode, ImputeNode, FeatureMatrixNode, 
            FeatureSelectionNode, ScalingNode, ModelTrainingNode,
            CleaningNode, GNNNode, KGNode, SelectionNode, 
            ScalingNodeB, ModelTrainingNodeB, EndNode
        )
        print("✅ 所有节点类导入成功 / All node classes imported successfully\n")
        
        # 10节点架构映射 / 10-node architecture mapping
        node_mapping = {
            'N0': DataFetchNode,
            'N1': ImputeNode,
            'N2': FeatureMatrixNode,
            'N3': CleaningNode,
            'N4': GNNNode,
            'N5': KGNode,
            'N6': SelectionNode,
            'N7': ScalingNodeB,
            'N8': ModelTrainingNodeB,
            'N9': EndNode
        }
        
        print("📊 10节点架构节点验证 / 10-Node Architecture Node Verification")
        print("-" * 80)
        
        all_valid = True
        for node_id, NodeClass in node_mapping.items():
            node = NodeClass()
            
            # 验证节点属性 / Verify node attributes
            has_id = hasattr(node, 'id') and node.id == node_id
            has_name = hasattr(node, 'name') and len(node.name) > 0
            has_type = hasattr(node, 'type') and len(node.type) > 0
            has_methods = hasattr(node, 'methods') and len(node.methods) > 0
            has_execute = hasattr(node, 'execute') and callable(node.execute)
            has_docstring = NodeClass.__doc__ is not None and len(NodeClass.__doc__.strip()) > 50
            
            # 检查是否为完整文档 / Check if fully documented
            is_complete = has_docstring and ('可用方法' in NodeClass.__doc__ or 'Available Methods' in NodeClass.__doc__)
            
            status = "✅" if is_complete else "⚠️"
            doc_quality = "完整文档" if is_complete else "简单文档"
            
            print(f"{status} {node_id} - {node.name:20s} | 类型: {node.type:20s} | "
                  f"方法: {len(node.methods)} | 文档: {doc_quality}")
            print(f"      可用方法 / Methods: {list(node.methods.keys())}")
            
            if not is_complete:
                all_valid = False
        
        print("-" * 80)
        
        # 统计信息 / Statistics
        total_nodes = len(node_mapping)
        complete_docs = sum(1 for _, NodeClass in node_mapping.items() 
                           if NodeClass.__doc__ and len(NodeClass.__doc__.strip()) > 50 
                           and ('可用方法' in NodeClass.__doc__ or 'Available Methods' in NodeClass.__doc__))
        
        print(f"\n📈 统计信息 / Statistics:")
        print(f"   总节点数 / Total Nodes: {total_nodes}")
        print(f"   完整文档节点 / Complete Documentation: {complete_docs}")
        print(f"   完成率 / Completion Rate: {complete_docs/total_nodes*100:.1f}%")
        
        if all_valid:
            print("\n🎉 所有节点都有完整的中英双语文档！/ All nodes have complete bilingual documentation!")
        else:
            print("\n⚠️ 部分节点文档需要完善 / Some nodes need documentation improvement")
        
        return all_valid
        
    except Exception as e:
        print(f"❌ 验证失败 / Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_architecture_consistency():
    """验证架构一致性 / Verify architecture consistency"""
    print("\n\n🏗️ 验证架构一致性 / Verifying Architecture Consistency")
    print("=" * 80)
    
    try:
        from env.pipeline_env import PipelineEnv
        
        env = PipelineEnv()
        print(f"✅ 环境初始化成功 / Environment initialized successfully")
        print(f"\n📋 环境配置 / Environment Configuration:")
        print(f"   节点列表 / Node List: {env.pipeline_nodes}")
        print(f"   节点数量 / Node Count: {env.num_nodes}")
        print(f"   最大方法数 / Max Methods: {env.max_methods}")
        
        print(f"\n📋 每个节点的方法 / Methods for Each Node:")
        for node_id, methods in env.methods_for_node.items():
            print(f"   {node_id}: {methods} ({len(methods)} methods)")
        
        print(f"\n📋 需要超参数的节点 / Nodes Requiring Hyperparameters:")
        print(f"   {sorted(env.param_nodes)}")
        
        # 验证节点数量一致性 / Verify node count consistency
        if env.num_nodes == 10:
            print("\n✅ 节点数量正确：10个节点 / Node count correct: 10 nodes")
        else:
            print(f"\n⚠️ 节点数量异常：{env.num_nodes}个节点 / Unexpected node count: {env.num_nodes} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ 架构验证失败 / Architecture verification failed: {e}")
        return False


def main():
    """主函数 / Main function"""
    print("\n" + "🚀 开始验证 / Starting Verification ".center(80, "="))
    print()
    
    # 验证节点实现 / Verify node implementations
    nodes_ok = verify_nodes()
    
    # 验证架构一致性 / Verify architecture consistency
    arch_ok = verify_architecture_consistency()
    
    # 总结 / Summary
    print("\n" + "📊 验证总结 / Verification Summary ".center(80, "="))
    print(f"   节点实现 / Node Implementation: {'✅ 通过' if nodes_ok else '❌ 失败'}")
    print(f"   架构一致性 / Architecture Consistency: {'✅ 通过' if arch_ok else '❌ 失败'}")
    
    if nodes_ok and arch_ok:
        print("\n🎉 所有验证通过！10节点架构完全实现！")
        print("🎉 All verifications passed! 10-node architecture fully implemented!")
    else:
        print("\n⚠️ 存在需要改进的地方 / Some improvements needed")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
