"""
GNN Processing Module for N4 Node
图神经网络处理模块，支持GCN、GAT、GraphSAGE等架构

This module provides complete GNN implementations for materials science feature extraction.
If PyTorch Geometric is not available, falls back to graph statistical features.
"""

from typing import Dict, Any, Optional
import numpy as np
import warnings


def gnn_process(data: Dict[str, Any], strategy: str = 'gcn', param: Optional[float] = None, params: Optional[dict] = None) -> Dict[str, Any]:
    """
    N4节点：图神经网络处理主入口
    
    Args:
        data: 包含X_train, X_val, X_test等的状态字典
        strategy: GNN架构选择 ('gcn', 'gat', 'sage')
        param: 可选的超参数 [0,1]
        params: 额外参数字典
    
    Returns:
        处理后的状态字典，包含新增的GNN特征
    """
    try:
        import torch
        import torch_geometric
        # If PyG is available, use full GNN implementation
        return _gnn_full_implementation(data, strategy, param, params)
    except ImportError:
        warnings.warn("PyTorch Geometric not available. Using graph statistical features fallback.")
        return _gnn_statistical_fallback(data, strategy, param)


def _gnn_full_implementation(data: Dict[str, Any], strategy: str, param: Optional[float], params: Optional[dict]) -> Dict[str, Any]:
    """
    完整的GNN实现（需要PyTorch Geometric）
    
    构建材料特征的k-NN图，应用GNN架构提取图特征
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    from sklearn.neighbors import kneighbors_graph
    
    params = params or {}
    
    # 提取数据
    X_train = np.array(data['X_train'])
    X_val = np.array(data['X_val'])
    X_test = np.array(data.get('X_test')) if data.get('X_test') is not None else None
    
    # 超参数映射：param [0,1] -> 实际参数
    k_neighbors = params.get('k_neighbors', 5)
    hidden_dim = params.get('hidden_dim', 32)
    num_layers = params.get('num_layers', 2)
    
    if param is not None:
        # param控制k近邻数量：param=0->k=3, param=1->k=10
        k_neighbors = int(3 + 7 * param)
    
    # 构建k-NN图 (基于特征相似度)
    def build_knn_graph(X, k):
        """构建k近邻图的边索引"""
        adj = kneighbors_graph(X, k, mode='connectivity', include_self=False)
        edge_index = np.array(adj.nonzero())
        return torch.LongTensor(edge_index)
    
    # 定义简化的GNN模型
    class SimpleGNN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, conv_type='gcn'):
            super().__init__()
            if conv_type == 'gcn':
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)
            elif conv_type == 'gat':
                self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
                self.conv2 = GATConv(hidden_channels, out_channels, heads=1)
            elif conv_type == 'sage':
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)
            else:
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)
        
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    
    # 应用GNN提取图特征
    def extract_gnn_features(X, edge_index, model):
        """使用GNN模型提取图嵌入特征"""
        x = torch.FloatTensor(X)
        model.eval()
        with torch.no_grad():
            embeddings = model(x, edge_index)
        return embeddings.numpy()
    
    # 为训练集构建图和模型
    edge_index_train = build_knn_graph(X_train, k_neighbors)
    in_dim = X_train.shape[1]
    out_dim = max(4, in_dim // 4)  # 输出维度为输入的1/4
    
    # 初始化GNN模型
    model = SimpleGNN(in_dim, hidden_dim, out_dim, conv_type=strategy)
    
    # 提取GNN特征（无训练，使用随机初始化权重作为特征提取器）
    gnn_features_train = extract_gnn_features(X_train, edge_index_train, model)
    
    # 对验证集和测试集使用相同的图构建方式
    edge_index_val = build_knn_graph(X_val, k_neighbors)
    gnn_features_val = extract_gnn_features(X_val, edge_index_val, model)
    
    gnn_features_test = None
    if X_test is not None and len(X_test) > 0:
        edge_index_test = build_knn_graph(X_test, k_neighbors)
        gnn_features_test = extract_gnn_features(X_test, edge_index_test, model)
    
    # 拼接原始特征和GNN特征
    X_train_aug = np.concatenate([X_train, gnn_features_train], axis=1)
    X_val_aug = np.concatenate([X_val, gnn_features_val], axis=1)
    X_test_aug = np.concatenate([X_test, gnn_features_test], axis=1) if gnn_features_test is not None else None
    
    # 更新特征名称
    feature_names = list(data.get('feature_names', []))
    gnn_feature_names = [f'gnn_{strategy}_{i}' for i in range(out_dim)]
    feature_names.extend(gnn_feature_names)
    
    return {
        'X_train': X_train_aug,
        'X_val': X_val_aug,
        'X_test': X_test_aug,
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test'),
        'feature_names': feature_names,
        'gnn_info': {
            'strategy': strategy,
            'k_neighbors': k_neighbors,
            'hidden_dim': hidden_dim,
            'output_dim': out_dim
        }
    }


def _gnn_statistical_fallback(data: Dict[str, Any], strategy: str, param: Optional[float]) -> Dict[str, Any]:
    """
    GNN的统计特征备用方案（当PyTorch Geometric不可用时）
    计算每个样本的统计特征作为图结构的近似
    """
    X_train = np.array(data['X_train']) if data.get('X_train') is not None else None
    X_val = np.array(data['X_val']) if data.get('X_val') is not None else None
    X_test = np.array(data.get('X_test')) if data.get('X_test') is not None else None
    
    if X_train is None:
        return data
    
    def compute_graph_stats(X):
        """计算图统计特征"""
        if X is None:
            return None
        
        # 基础统计特征
        mean_feat = np.nanmean(X, axis=1, keepdims=True)
        std_feat = np.nanstd(X, axis=1, keepdims=True)
        min_feat = np.nanmin(X, axis=1, keepdims=True)
        max_feat = np.nanmax(X, axis=1, keepdims=True)
        
        # 特征交互项（模拟图结构）
        range_feat = max_feat - min_feat
        cv_feat = std_feat / (mean_feat + 1e-8)  # 变异系数
        
        # 分位数特征
        q25_feat = np.nanpercentile(X, 25, axis=1, keepdims=True)
        q75_feat = np.nanpercentile(X, 75, axis=1, keepdims=True)
        iqr_feat = q75_feat - q25_feat
        
        # 偏度和峰度的简化版本
        centered = X - mean_feat
        skew_feat = np.nanmean(centered**3, axis=1, keepdims=True) / (std_feat**3 + 1e-8)
        kurt_feat = np.nanmean(centered**4, axis=1, keepdims=True) / (std_feat**4 + 1e-8)
        
        stats = np.concatenate([
            mean_feat, std_feat, min_feat, max_feat,
            range_feat, cv_feat, q25_feat, q75_feat, iqr_feat,
            skew_feat, kurt_feat
        ], axis=1)
        
        return np.concatenate([X, stats], axis=1)
    
    X_train_aug = compute_graph_stats(X_train)
    X_val_aug = compute_graph_stats(X_val)
    X_test_aug = compute_graph_stats(X_test)
    
    # 更新特征名称
    feature_names = list(data.get('feature_names', []))
    stat_names = [
        f'gnn_{strategy}_mean', f'gnn_{strategy}_std', 
        f'gnn_{strategy}_min', f'gnn_{strategy}_max',
        f'gnn_{strategy}_range', f'gnn_{strategy}_cv',
        f'gnn_{strategy}_q25', f'gnn_{strategy}_q75', f'gnn_{strategy}_iqr',
        f'gnn_{strategy}_skew', f'gnn_{strategy}_kurt'
    ]
    feature_names.extend(stat_names)
    
    return {
        'X_train': X_train_aug,
        'X_val': X_val_aug,
        'X_test': X_test_aug,
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test'),
        'feature_names': feature_names,
        'gnn_info': {
            'strategy': strategy,
            'method': 'statistical_fallback',
            'features_added': len(stat_names)
        }
    }


# 导出函数
__all__ = ['gnn_process']
