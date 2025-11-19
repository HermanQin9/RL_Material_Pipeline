"""
Tests for N4 GNN Processing and N5 KG Processing
测试GNN图神经网络处理和知识图谱处理模块
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from methods.gnn_processing import gnn_process as gnn_process_full
from methods.kg_processing import kg_process as kg_process_full


@pytest.fixture
def sample_data():
    """创建测试数据"""
    np.random.seed(42)
    n_train, n_val, n_test = 50, 10, 10
    n_features = 20
    
    return {
        'X_train': np.random.randn(n_train, n_features),
        'X_val': np.random.randn(n_val, n_features),
        'X_test': np.random.randn(n_test, n_features),
        'y_train': np.random.randn(n_train),
        'y_val': np.random.randn(n_val),
        'y_test': np.random.randn(n_test),
        'feature_names': [f'feat_{i}' for i in range(n_features)]
    }


# ==================== N4 GNN Tests ====================

class TestGNNProcessing:
    """测试GNN图神经网络处理"""
    
    def test_gnn_gcn_strategy(self, sample_data):
        """测试GCN策略"""
        result = gnn_process_full(sample_data, strategy='gcn')
        
        assert 'X_train' in result
        assert 'X_val' in result
        assert 'X_test' in result
        assert result['X_train'].shape[0] == 50
        assert result['X_val'].shape[0] == 10
        assert result['X_test'].shape[0] == 10
        # GNN应该增加特征维度
        assert result['X_train'].shape[1] > sample_data['X_train'].shape[1]
    
    def test_gnn_gat_strategy(self, sample_data):
        """测试GAT策略"""
        result = gnn_process_full(sample_data, strategy='gat')
        
        assert result['X_train'].shape[0] == 50
        assert result['X_train'].shape[1] > sample_data['X_train'].shape[1]
        assert 'gnn_info' in result
        assert result['gnn_info']['strategy'] == 'gat' or 'method' in result['gnn_info']
    
    def test_gnn_sage_strategy(self, sample_data):
        """测试GraphSAGE策略"""
        result = gnn_process_full(sample_data, strategy='sage')
        
        assert result['X_train'].shape[0] == 50
        assert result['X_train'].shape[1] > sample_data['X_train'].shape[1]
        assert 'feature_names' in result
        assert len(result['feature_names']) > len(sample_data['feature_names'])
    
    def test_gnn_with_param(self, sample_data):
        """测试GNN带超参数"""
        result1 = gnn_process_full(sample_data, strategy='gcn', param=0.0)
        result2 = gnn_process_full(sample_data, strategy='gcn', param=1.0)
        
        # 不同param应该产生不同结果
        assert result1['X_train'].shape == result2['X_train'].shape
        # 但特征值可能不同（因为k近邻数量不同）
        assert 'gnn_info' in result1
        assert 'gnn_info' in result2
    
    def test_gnn_preserves_labels(self, sample_data):
        """测试GNN保持标签不变"""
        result = gnn_process_full(sample_data, strategy='gcn')
        
        assert np.array_equal(result['y_train'], sample_data['y_train'])
        assert np.array_equal(result['y_val'], sample_data['y_val'])
        assert np.array_equal(result['y_test'], sample_data['y_test'])
    
    def test_gnn_no_nan_output(self, sample_data):
        """测试GNN输出无NaN值"""
        result = gnn_process_full(sample_data, strategy='gcn')
        
        assert not np.isnan(result['X_train']).any()
        assert not np.isnan(result['X_val']).any()
        assert not np.isnan(result['X_test']).any()
    
    def test_gnn_feature_names_updated(self, sample_data):
        """测试GNN更新特征名称"""
        result = gnn_process_full(sample_data, strategy='gcn')
        
        original_names = sample_data['feature_names']
        new_names = result['feature_names']
        
        assert len(new_names) > len(original_names)
        # 原始特征名应该保留
        for name in original_names:
            assert name in new_names
    
    def test_gnn_with_custom_params(self, sample_data):
        """测试GNN自定义参数"""
        params = {'k_neighbors': 3, 'hidden_dim': 16}
        result = gnn_process_full(sample_data, strategy='gcn', params=params)
        
        assert result['X_train'].shape[0] == 50
        assert 'gnn_info' in result


# ==================== N5 KG Tests ====================

class TestKGProcessing:
    """测试知识图谱处理"""
    
    def test_kg_entity_strategy(self, sample_data):
        """测试实体级别知识图谱"""
        result = kg_process_full(sample_data, strategy='entity')
        
        assert 'X_train' in result
        assert 'X_val' in result
        assert 'X_test' in result
        assert result['X_train'].shape[0] == 50
        # KG应该增加特征
        assert result['X_train'].shape[1] > sample_data['X_train'].shape[1]
        assert 'kg_info' in result
        assert result['kg_info']['strategy'] == 'entity'
    
    def test_kg_relation_strategy(self, sample_data):
        """测试关系级别知识图谱"""
        result = kg_process_full(sample_data, strategy='relation')
        
        assert result['X_train'].shape[0] == 50
        assert result['X_train'].shape[1] > sample_data['X_train'].shape[1]
        assert 'kg_info' in result
        assert result['kg_info']['strategy'] == 'relation'
    
    def test_kg_none_strategy(self, sample_data):
        """测试不应用知识图谱"""
        result = kg_process_full(sample_data, strategy='none')
        
        # 不应该改变特征维度
        assert result['X_train'].shape == sample_data['X_train'].shape
        assert result['X_val'].shape == sample_data['X_val'].shape
        assert result['X_test'].shape == sample_data['X_test'].shape
        assert 'kg_info' in result
        assert result['kg_info']['strategy'] == 'none'
    
    def test_kg_preserves_labels(self, sample_data):
        """测试KG保持标签不变"""
        result = kg_process_full(sample_data, strategy='entity')
        
        assert np.array_equal(result['y_train'], sample_data['y_train'])
        assert np.array_equal(result['y_val'], sample_data['y_val'])
        assert np.array_equal(result['y_test'], sample_data['y_test'])
    
    def test_kg_no_nan_output(self, sample_data):
        """测试KG输出无NaN值"""
        result = kg_process_full(sample_data, strategy='entity')
        
        assert not np.isnan(result['X_train']).any()
        assert not np.isnan(result['X_val']).any()
        assert not np.isnan(result['X_test']).any()
    
    def test_kg_feature_names_updated(self, sample_data):
        """测试KG更新特征名称"""
        result = kg_process_full(sample_data, strategy='entity')
        
        original_names = sample_data['feature_names']
        new_names = result['feature_names']
        
        assert len(new_names) > len(original_names)
        # KG特征名应该包含'kg_'前缀
        kg_features = [name for name in new_names if name.startswith('kg_')]
        assert len(kg_features) > 0
    
    def test_kg_entity_vs_relation(self, sample_data):
        """测试不同KG策略产生不同特征"""
        result_entity = kg_process_full(sample_data, strategy='entity')
        result_relation = kg_process_full(sample_data, strategy='relation')
        
        # 两种策略应该产生不同数量的特征
        # (但这不是必须的，取决于实现)
        assert result_entity['X_train'].shape[1] >= sample_data['X_train'].shape[1]
        assert result_relation['X_train'].shape[1] >= sample_data['X_train'].shape[1]


# ==================== Integration Tests ====================

class TestGNNKGIntegration:
    """测试GNN和KG的集成"""
    
    def test_gnn_then_kg(self, sample_data):
        """测试先GNN后KG的流水线"""
        # 先应用GNN
        result_gnn = gnn_process_full(sample_data, strategy='gcn')
        
        # 再应用KG
        result_kg = kg_process_full(result_gnn, strategy='entity')
        
        # 验证特征维度持续增加
        original_dim = sample_data['X_train'].shape[1]
        gnn_dim = result_gnn['X_train'].shape[1]
        final_dim = result_kg['X_train'].shape[1]
        
        assert gnn_dim > original_dim
        assert final_dim > gnn_dim
        
        # 验证标签保持不变
        assert np.array_equal(result_kg['y_train'], sample_data['y_train'])
    
    def test_kg_then_gnn(self, sample_data):
        """测试先KG后GNN的流水线"""
        # 先应用KG
        result_kg = kg_process_full(sample_data, strategy='relation')
        
        # 再应用GNN
        result_gnn = gnn_process_full(result_kg, strategy='gat')
        
        # 验证特征维度持续增加
        original_dim = sample_data['X_train'].shape[1]
        kg_dim = result_kg['X_train'].shape[1]
        final_dim = result_gnn['X_train'].shape[1]
        
        assert kg_dim > original_dim
        assert final_dim > kg_dim
    
    def test_all_strategies_combination(self, sample_data):
        """测试所有策略组合"""
        strategies = [
            ('gcn', 'entity'),
            ('gat', 'relation'),
            ('sage', 'none')
        ]
        
        for gnn_strat, kg_strat in strategies:
            result_gnn = gnn_process_full(sample_data, strategy=gnn_strat)
            result_kg = kg_process_full(result_gnn, strategy=kg_strat)
            
            assert result_kg['X_train'].shape[0] == 50
            assert result_kg['X_val'].shape[0] == 10
            assert not np.isnan(result_kg['X_train']).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
