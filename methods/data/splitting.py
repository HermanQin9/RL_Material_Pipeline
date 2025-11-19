"""
数据分割工具 / Data Splitting Utilities

提供精确的in-distribution和out-of-distribution数据分割
Provides precise in-distribution and out-of-distribution data splitting
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def split_in_out_distribution(
    df: pd.DataFrame,
    n_in_dist: int = 300,
    n_out_dist: int = 100,
    strategy: str = 'element_based',
    target_prop: str = 'formation_energy_per_atom',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    将数据集精确分割为in-distribution和out-of-distribution
    Split dataset into exact in-distribution and out-of-distribution samples
    
    Parameters
    ----------
    df : pd.DataFrame
        完整数据集 / Full dataset
    n_in_dist : int, default=300
        in-distribution样本数 / Number of in-distribution samples
    n_out_dist : int, default=100
        out-of-distribution样本数 / Number of out-of-distribution samples  
    strategy : str, default='element_based'
        OOD分割策略 / OOD splitting strategy:
        - 'element_based': 基于特定元素（稀土/贵金属）
        - 'energy_based': 基于formation energy极值
        - 'random': 随机分割（用于测试）
    target_prop : str
        目标属性名称 / Target property name
    random_state : int
        随机种子 / Random seed
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (in_dist_df, out_dist_df): in-distribution和out-of-distribution数据集
    """
    np.random.seed(random_state)
    
    logger.info(f"Splitting {len(df)} samples into {n_in_dist} in-dist + {n_out_dist} out-dist")
    logger.info(f"Strategy: {strategy}")
    
    if len(df) < n_in_dist + n_out_dist:
        logger.warning(f"Dataset size {len(df)} < required {n_in_dist + n_out_dist}")
        logger.warning("Adjusting split ratios to fit available data")
        total = n_in_dist + n_out_dist
        n_in_dist = int(len(df) * n_in_dist / total)
        n_out_dist = len(df) - n_in_dist
    
    if strategy == 'element_based':
        in_dist_df, out_dist_df = _split_by_elements(
            df, n_in_dist, n_out_dist, random_state
        )
    elif strategy == 'energy_based':
        in_dist_df, out_dist_df = _split_by_energy(
            df, n_in_dist, n_out_dist, target_prop, random_state
        )
    elif strategy == 'random':
        in_dist_df, out_dist_df = _split_random(
            df, n_in_dist, n_out_dist, random_state
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    logger.info(f"Split complete: in-dist={len(in_dist_df)}, out-dist={len(out_dist_df)}")
    
    return in_dist_df, out_dist_df


def _split_by_elements(
    df: pd.DataFrame,
    n_in: int,
    n_out: int,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于元素类型分割：含特殊元素=OOD，常见元素=in-dist
    Split by element types: rare elements = OOD, common elements = in-dist
    
    OOD定义: 含稀土元素或贵金属
    OOD definition: Contains rare earth or noble metals
    """
    # 稀土元素和贵金属 / Rare earth and noble metals
    rare_elements = [
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',  # 稀土
        'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Sc', 'Y',  # 稀土
        'Pt', 'Au', 'Ag', 'Rh', 'Pd', 'Ir', 'Os', 'Ru'  # 贵金属
    ]
    
    def contains_rare_element(composition):
        """检查是否含稀土或贵金属"""
        if composition is None:
            return False
        try:
            elements = composition.as_dict().keys()
            return any(elem in rare_elements for elem in elements)
        except Exception:
            return False
    
    # 标记OOD样本
    df = df.copy()
    df['is_ood'] = df['composition'].apply(contains_rare_element)
    
    # 分离候选集
    ood_candidates = df[df['is_ood']].copy()
    in_dist_candidates = df[~df['is_ood']].copy()
    
    logger.info(f"Found {len(ood_candidates)} OOD candidates (rare elements)")
    logger.info(f"Found {len(in_dist_candidates)} in-dist candidates (common elements)")
    
    # 采样
    np.random.seed(random_state)
    
    # 如果OOD样本不足，从in-dist中随机选择补充
    if len(ood_candidates) < n_out:
        logger.warning(f"OOD candidates ({len(ood_candidates)}) < required ({n_out})")
        logger.warning("Supplementing with random in-dist samples")
        extra_needed = n_out - len(ood_candidates)
        extra_ood = in_dist_candidates.sample(n=extra_needed, random_state=random_state)
        ood_candidates = pd.concat([ood_candidates, extra_ood], ignore_index=True)
        in_dist_candidates = in_dist_candidates.drop(extra_ood.index).reset_index(drop=True)
    
    # 采样OOD
    out_dist_df = ood_candidates.sample(
        n=min(n_out, len(ood_candidates)),
        random_state=random_state
    ).reset_index(drop=True)
    
    # 采样in-dist
    in_dist_df = in_dist_candidates.sample(
        n=min(n_in, len(in_dist_candidates)),
        random_state=random_state
    ).reset_index(drop=True)
    
    # 移除标记列
    in_dist_df = in_dist_df.drop(columns=['is_ood'])
    out_dist_df = out_dist_df.drop(columns=['is_ood'])
    
    return in_dist_df, out_dist_df


def _split_by_energy(
    df: pd.DataFrame,
    n_in: int,
    n_out: int,
    target_prop: str,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于formation energy分布分割：极值=OOD，中间值=in-dist
    Split by formation energy: extreme values = OOD, central values = in-dist
    """
    df = df.copy()
    fe = df[target_prop]
    
    # 计算极值阈值（10%和90%分位数）
    q_low = fe.quantile(0.10)
    q_high = fe.quantile(0.90)
    
    logger.info(f"Formation energy range: [{fe.min():.3f}, {fe.max():.3f}]")
    logger.info(f"OOD thresholds: <{q_low:.3f} or >{q_high:.3f}")
    
    # 标记OOD（极端formation energy）
    df['is_ood'] = (fe < q_low) | (fe > q_high)
    
    ood_candidates = df[df['is_ood']].copy()
    in_dist_candidates = df[~df['is_ood']].copy()
    
    logger.info(f"Found {len(ood_candidates)} OOD candidates (extreme energy)")
    logger.info(f"Found {len(in_dist_candidates)} in-dist candidates (normal energy)")
    
    # 采样
    np.random.seed(random_state)
    
    if len(ood_candidates) < n_out:
        logger.warning(f"OOD candidates ({len(ood_candidates)}) < required ({n_out})")
        extra_needed = n_out - len(ood_candidates)
        extra_ood = in_dist_candidates.sample(n=extra_needed, random_state=random_state)
        ood_candidates = pd.concat([ood_candidates, extra_ood], ignore_index=True)
        in_dist_candidates = in_dist_candidates.drop(extra_ood.index).reset_index(drop=True)
    
    out_dist_df = ood_candidates.sample(
        n=min(n_out, len(ood_candidates)),
        random_state=random_state
    ).reset_index(drop=True)
    
    in_dist_df = in_dist_candidates.sample(
        n=min(n_in, len(in_dist_candidates)),
        random_state=random_state
    ).reset_index(drop=True)
    
    in_dist_df = in_dist_df.drop(columns=['is_ood'])
    out_dist_df = out_dist_df.drop(columns=['is_ood'])
    
    return in_dist_df, out_dist_df


def _split_random(
    df: pd.DataFrame,
    n_in: int,
    n_out: int,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    随机分割（用于测试baseline）
    Random split (for baseline testing)
    """
    np.random.seed(random_state)
    
    # 随机打乱
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 简单分割
    in_dist_df = df_shuffled.iloc[:n_in].reset_index(drop=True)
    out_dist_df = df_shuffled.iloc[n_in:n_in+n_out].reset_index(drop=True)
    
    logger.info("Random split performed (no semantic OOD)")
    
    return in_dist_df, out_dist_df


def validate_split(
    in_dist_df: pd.DataFrame,
    out_dist_df: pd.DataFrame,
    target_prop: str = 'formation_energy_per_atom'
) -> dict:
    """
    验证数据分割的质量
    Validate quality of data split
    
    Returns
    -------
    dict
        包含分割统计信息的字典 / Dictionary containing split statistics
    """
    stats = {
        'n_in_dist': len(in_dist_df),
        'n_out_dist': len(out_dist_df),
        'in_dist_fe_mean': float(in_dist_df[target_prop].mean()),
        'in_dist_fe_std': float(in_dist_df[target_prop].std()),
        'out_dist_fe_mean': float(out_dist_df[target_prop].mean()),
        'out_dist_fe_std': float(out_dist_df[target_prop].std()),
        'fe_mean_diff': abs(
            in_dist_df[target_prop].mean() - out_dist_df[target_prop].mean()
        ),
        'overlap_ids': len(
            set(in_dist_df.get('material_id', [])) & 
            set(out_dist_df.get('material_id', []))
        )
    }
    
    logger.info("Split validation:")
    logger.info(f"  In-dist: n={stats['n_in_dist']}, "
                f"FE={stats['in_dist_fe_mean']:.3f}±{stats['in_dist_fe_std']:.3f}")
    logger.info(f"  Out-dist: n={stats['n_out_dist']}, "
                f"FE={stats['out_dist_fe_mean']:.3f}±{stats['out_dist_fe_std']:.3f}")
    logger.info(f"  FE mean difference: {stats['fe_mean_diff']:.3f}")
    logger.info(f"  Overlapping IDs: {stats['overlap_ids']}")
    
    # 检查是否有重叠
    if stats['overlap_ids'] > 0:
        logger.error(f"ERROR: {stats['overlap_ids']} overlapping material_ids found!")
    
    return stats
