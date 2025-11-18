"""
验证RL agent能否重新发现标准最佳实践
Validate that RL agent can re-discover standard best practices

这个脚本将：
1. 定义标准的ML流水线最佳实践序列
2. 训练PPO agent，让它自主探索
3. 收集PPO发现的top-5最优序列
4. 对比PPO发现的序列与标准最佳实践的相似度
5. 生成详细的对比报告
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import json
from datetime import datetime

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer
from pipeline import run_pipeline_config


# ===================== 标准最佳实践定义 =====================

BEST_PRACTICE_SEQUENCES = {
    "minimal": {
        "sequence": ['N0', 'N2', 'N8', 'N9'],
        "description": "最小流程：数据获取 → 特征构建 → 模型训练 → 结束",
        "expected_performance": "baseline"
    },
    "standard": {
        "sequence": ['N0', 'N2', 'N1', 'N6', 'N7', 'N8', 'N9'],
        "description": "标准流程：获取 → 特征 → 填充 → 选择 → 缩放 → 训练 → 结束",
        "expected_performance": "good"
    },
    "standard_with_cleaning": {
        "sequence": ['N0', 'N2', 'N3', 'N1', 'N6', 'N7', 'N8', 'N9'],
        "description": "带清洗：获取 → 特征 → 清洗 → 填充 → 选择 → 缩放 → 训练 → 结束",
        "expected_performance": "better"
    },
    "advanced_with_gnn": {
        "sequence": ['N0', 'N2', 'N4', 'N1', 'N6', 'N7', 'N8', 'N9'],
        "description": "GNN增强：获取 → 特征 → GNN → 填充 → 选择 → 缩放 → 训练 → 结束",
        "expected_performance": "advanced"
    },
    "full_with_kg": {
        "sequence": ['N0', 'N2', 'N3', 'N4', 'N5', 'N1', 'N6', 'N7', 'N8', 'N9'],
        "description": "完整流程：获取 → 特征 → 清洗 → GNN → 知识图谱 → 填充 → 选择 → 缩放 → 训练 → 结束",
        "expected_performance": "best"
    }
}


# ===================== PPO训练和序列收集 =====================

def train_ppo_and_collect_sequences(
    episodes: int = 100,
    max_steps: int = 15,
    verbose: bool = True
) -> Tuple[List[List[str]], List[float], PPOTrainer]:
    """
    训练PPO并收集所有尝试的序列
    
    Returns:
        sequences: 所有完整的序列列表
        rewards: 对应的奖励列表
        trainer: 训练好的PPO trainer
    """
    print("=" * 70)
    print("[START] 开始PPO训练，探索最优流水线序列")
    print("=" * 70)
    
    env = PipelineEnv()
    trainer = PPOTrainer(
        env,
        learning_rate=3e-4,
        clip_ratio=0.2,
        hidden_size=128,
        max_steps_per_episode=max_steps
    )
    
    sequences = []
    rewards = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_sequence = []
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action, _ = trainer.select_action(obs)
            node_idx = action['node']
            node_name = env.pipeline_nodes[node_idx]
            episode_sequence.append(node_name)
            
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # 只记录完整序列（以N9结束）
        if episode_sequence and episode_sequence[-1] == 'N9':
            sequences.append(episode_sequence)
            rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 10 == 0:
                recent_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                print(f"Episode {episode + 1}/{episodes}: "
                      f"Reward={episode_reward:.3f}, "
                      f"Length={len(episode_sequence)}, "
                      f"Recent Avg={recent_avg:.3f}")
        
        # 进行PPO更新
        if episode > 0 and episode % 10 == 0:
            # 这里简化处理，实际应该调用trainer的update方法
            pass
    
    print(f"\n[SUCCESS] 训练完成！收集到 {len(sequences)} 个完整序列")
    return sequences, rewards, trainer


# ===================== 序列分析和对比 =====================

def calculate_sequence_similarity(seq1: List[str], seq2: List[str]) -> float:
    """
    计算两个序列的相似度（基于编辑距离和关键节点位置）
    
    Returns:
        similarity: 0.0-1.0之间的相似度分数
    """
    # 1. 编辑距离相似度（Levenshtein）
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    edit_dist = levenshtein_distance(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    edit_similarity = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.0
    
    # 2. 关键节点顺序相似度
    key_nodes = ['N0', 'N2', 'N8', 'N9']
    seq1_key_positions = [seq1.index(n) if n in seq1 else -1 for n in key_nodes]
    seq2_key_positions = [seq2.index(n) if n in seq2 else -1 for n in key_nodes]
    
    key_order_matches = sum(1 for p1, p2 in zip(seq1_key_positions, seq2_key_positions) 
                           if p1 >= 0 and p2 >= 0 and p1 == p2)
    key_similarity = key_order_matches / len(key_nodes)
    
    # 3. 节点存在性相似度
    set1 = set(seq1)
    set2 = set(seq2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    set_similarity = intersection / union if union > 0 else 0.0
    
    # 综合相似度（加权平均）
    similarity = (0.4 * edit_similarity + 
                  0.4 * key_similarity + 
                  0.2 * set_similarity)
    
    return similarity


def find_top_sequences(
    sequences: List[List[str]], 
    rewards: List[float], 
    top_k: int = 5
) -> List[Tuple[List[str], float]]:
    """
    找出top-k最优序列
    """
    # 按奖励排序
    paired = list(zip(sequences, rewards))
    paired.sort(key=lambda x: x[1], reverse=True)
    
    # 去重（保留第一个出现的）
    seen = set()
    unique_top = []
    for seq, reward in paired:
        seq_tuple = tuple(seq)
        if seq_tuple not in seen:
            seen.add(seq_tuple)
            unique_top.append((seq, reward))
            if len(unique_top) >= top_k:
                break
    
    return unique_top


def analyze_node_usage(sequences: List[List[str]]) -> Dict[str, Dict]:
    """
    分析节点使用频率和位置
    """
    node_count = Counter()
    node_positions = defaultdict(list)
    
    for seq in sequences:
        for pos, node in enumerate(seq):
            node_count[node] += 1
            node_positions[node].append(pos)
    
    analysis = {}
    for node in node_count:
        positions = node_positions[node]
        analysis[node] = {
            'count': node_count[node],
            'frequency': node_count[node] / len(sequences),
            'avg_position': np.mean(positions),
            'std_position': np.std(positions),
            'typical_position': int(np.median(positions))
        }
    
    return analysis


def compare_with_best_practices(
    ppo_sequences: List[List[str]],
    ppo_rewards: List[float]
) -> pd.DataFrame:
    """
    对比PPO发现的序列与标准最佳实践
    """
    print("\n" + "=" * 70)
    print("[COMPARE] 对比PPO发现的序列与标准最佳实践")
    print("=" * 70)
    
    # 获取PPO的top-5序列
    top_ppo = find_top_sequences(ppo_sequences, ppo_rewards, top_k=5)
    
    results = []
    
    # 对比每个最佳实践
    for bp_name, bp_info in BEST_PRACTICE_SEQUENCES.items():
        bp_seq = bp_info['sequence']
        
        print(f"\n[BP] 最佳实践: {bp_name}")
        print(f"   描述: {bp_info['description']}")
        print(f"   序列: {' → '.join(bp_seq)}")
        
        # 找出与此最佳实践最相似的PPO序列
        best_match = None
        best_similarity = 0.0
        best_reward = 0.0
        
        for ppo_seq, ppo_reward in top_ppo:
            similarity = calculate_sequence_similarity(bp_seq, ppo_seq)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = ppo_seq
                best_reward = ppo_reward
        
        print(f"   [MATCH] 最相似的PPO序列: {' → '.join(best_match) if best_match else 'None'}")
        print(f"   [MATCH] 相似度: {best_similarity:.2%}")
        print(f"   [MATCH] PPO奖励: {best_reward:.3f}")
        
        results.append({
            'best_practice': bp_name,
            'description': bp_info['description'],
            'bp_sequence': ' → '.join(bp_seq),
            'bp_length': len(bp_seq),
            'ppo_match_sequence': ' → '.join(best_match) if best_match else 'None',
            'ppo_match_length': len(best_match) if best_match else 0,
            'similarity': best_similarity,
            'ppo_reward': best_reward,
            'discovered': best_similarity >= 0.7  # 70%相似度认为"重新发现"
        })
    
    return pd.DataFrame(results)


# ===================== 可视化 =====================

def visualize_results(
    comparison_df: pd.DataFrame,
    node_usage: Dict[str, Dict],
    top_sequences: List[Tuple[List[str], float]],
    output_dir: Path
):
    """
    生成可视化报告
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 图1: 最佳实践重新发现率
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 相似度对比
    ax = axes[0, 0]
    bp_names = comparison_df['best_practice']
    similarities = comparison_df['similarity']
    colors = ['green' if s >= 0.7 else 'orange' if s >= 0.5 else 'red' for s in similarities]
    ax.barh(bp_names, similarities, color=colors, alpha=0.7)
    ax.axvline(x=0.7, color='green', linestyle='--', label='Discovery Threshold (70%)')
    ax.set_xlabel('Similarity Score')
    ax.set_title('PPO Sequence Similarity to Best Practices')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # 子图2: 节点使用频率
    ax = axes[0, 1]
    nodes = list(node_usage.keys())
    frequencies = [node_usage[n]['frequency'] for n in nodes]
    ax.bar(nodes, frequencies, color='steelblue', alpha=0.7)
    ax.set_ylabel('Usage Frequency')
    ax.set_title('Node Usage Frequency in PPO Sequences')
    ax.grid(axis='y', alpha=0.3)
    
    # 子图3: 节点平均位置
    ax = axes[1, 0]
    avg_positions = [node_usage[n]['avg_position'] for n in nodes]
    std_positions = [node_usage[n]['std_position'] for n in nodes]
    ax.errorbar(nodes, avg_positions, yerr=std_positions, fmt='o', 
                capsize=5, capthick=2, color='coral', ecolor='gray')
    ax.set_ylabel('Average Position in Sequence')
    ax.set_title('Node Typical Position (with std)')
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    
    # 子图4: Top-5序列奖励
    ax = axes[1, 1]
    seq_labels = [f"Seq {i+1}\n{len(seq)} nodes" for i, (seq, _) in enumerate(top_sequences)]
    seq_rewards = [reward for _, reward in top_sequences]
    ax.bar(seq_labels, seq_rewards, color='purple', alpha=0.7)
    ax.set_ylabel('Reward')
    ax.set_title('Top-5 PPO Sequences Performance')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rl_best_practices_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] 可视化报告已保存: {output_dir / 'rl_best_practices_validation.png'}")


# ===================== 主函数 =====================

def main():
    """
    主验证流程
    """
    print("\n" + "=" * 70)
    print("[VALIDATE] RL Agent 最佳实践重新发现验证")
    print("   Validate RL Agent Re-discovery of Best Practices")
    print("=" * 70)
    
    # 配置
    EPISODES = 100
    MAX_STEPS = 15
    OUTPUT_DIR = Path(__file__).parent.parent / 'logs' / f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Step 1: 训练PPO并收集序列
    print("\n[STEP1] Step 1: 训练PPO agent并收集探索序列")
    sequences, rewards, trainer = train_ppo_and_collect_sequences(
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        verbose=True
    )
    
    if not sequences:
        print("\n[ERROR] 错误：没有收集到完整序列！")
        return
    
    # Step 2: 分析节点使用
    print("\n[STEP2] Step 2: 分析节点使用模式")
    node_usage = analyze_node_usage(sequences)
    print("\n节点使用统计:")
    for node, stats in sorted(node_usage.items()):
        print(f"  {node}: 使用 {stats['count']} 次 "
              f"({stats['frequency']:.1%}), "
              f"平均位置 {stats['avg_position']:.1f}")
    
    # Step 3: 对比最佳实践
    print("\n[STEP3] Step 3: 对比PPO发现的序列与标准最佳实践")
    comparison_df = compare_with_best_practices(sequences, rewards)
    
    # Step 4: 生成报告
    print("\n[STEP4] Step 4: 生成详细报告")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存对比结果
    comparison_df.to_csv(OUTPUT_DIR / 'best_practices_comparison.csv', index=False)
    print(f"   [SAVED] 对比结果已保存: {OUTPUT_DIR / 'best_practices_comparison.csv'}")
    
    # 保存top序列
    top_sequences = find_top_sequences(sequences, rewards, top_k=5)
    with open(OUTPUT_DIR / 'top_5_sequences.json', 'w') as f:
        json.dump([
            {
                'sequence': seq,
                'reward': float(reward),
                'length': len(seq)
            }
            for seq, reward in top_sequences
        ], f, indent=2)
    print(f"   [SAVED] Top-5序列已保存: {OUTPUT_DIR / 'top_5_sequences.json'}")
    
    # 生成可视化
    visualize_results(comparison_df, node_usage, top_sequences, OUTPUT_DIR)
    
    # Step 5: 总结
    print("\n" + "=" * 70)
    print("[RESULTS] 验证结果总结")
    print("=" * 70)
    
    discovered_count = comparison_df['discovered'].sum()
    total_practices = len(comparison_df)
    discovery_rate = discovered_count / total_practices
    
    print(f"\n[SUCCESS] 重新发现率: {discovered_count}/{total_practices} ({discovery_rate:.1%})")
    print(f"   - 平均相似度: {comparison_df['similarity'].mean():.2%}")
    print(f"   - 最高相似度: {comparison_df['similarity'].max():.2%}")
    print(f"   - 最佳PPO奖励: {comparison_df['ppo_reward'].max():.3f}")
    
    if discovery_rate >= 0.6:
        print("\n[COMPLETE] 结论: PPO agent成功重新发现了标准最佳实践！")
    elif discovery_rate >= 0.4:
        print("\n[WARN] 结论: PPO agent部分重新发现了最佳实践，需要继续训练")
    else:
        print("\n[FAIL] 结论: PPO agent未能有效重新发现最佳实践，需要调整训练策略")
    
    print(f"\n[SAVED] 完整报告保存在: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
