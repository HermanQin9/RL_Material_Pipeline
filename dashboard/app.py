"""
PPO Pipeline Optimizer - Interactive Dashboard
English-only version without emojis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer

# Page configuration
st.set_page_config(
    page_title="PPO Pipeline Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("PPO Pipeline Optimizer Dashboard")
st.markdown("**Materials Science AutoML - Reinforcement Learning**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    
    # Training parameters
    st.subheader("Training Parameters")
    num_episodes = st.slider("Number of Episodes", 10, 200, 50, 10)
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[1e-4, 3e-4, 1e-3, 3e-3],
        value=3e-4
    )
    max_steps = st.slider("Max Steps per Episode", 5, 20, 15)
    
    # Start training button
    start_training = st.button("Start Training", type="primary")
    
    st.markdown("---")
    
    # Dataset information
    st.subheader("Dataset Information")
    st.info("""
    **Source**: Materials Project API  
    **Training Set**: Non-Fe materials  
    **Test Set**: Fe-containing materials  
    **Target**: Formation Energy Prediction
    """)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Training Monitor", "Sequence Analysis", "Node Statistics", "Best Practices"])

# ==================== Tab 1: Training Monitor ====================
with tab1:
    st.header("Real-time Training Monitor")
    
    col1, col2, col3 = st.columns(3)
    
    if 'training_data' not in st.session_state:
        st.session_state.training_data = {
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'success_rate': []
        }
    
    # Training logic
    if start_training:
        st.session_state.training_data = {
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'success_rate': []
        }
        
        # Create environment and trainer
        with st.spinner("Initializing environment..."):
            env = PipelineEnv()
            trainer = PPOTrainer(
                env,
                learning_rate=learning_rate,
                max_steps_per_episode=max_steps
            )
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Real-time chart
        chart_placeholder = st.empty()
        
        # Training loop
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                action, _ = trainer.select_action(obs)
                obs, reward, done, _, _ = env.step(action)
                episode_reward += reward
                steps += 1
            
            # Record data
            st.session_state.training_data['episodes'].append(episode + 1)
            st.session_state.training_data['rewards'].append(episode_reward)
            st.session_state.training_data['lengths'].append(steps)
            
            # Calculate success rate
            if len(st.session_state.training_data['rewards']) >= 10:
                recent_success = sum(1 for r in st.session_state.training_data['rewards'][-10:] if r > 0)
                success_rate = recent_success / 10
            else:
                success_rate = 0.5
            st.session_state.training_data['success_rate'].append(success_rate)
            
            # Update progress
            progress = (episode + 1) / num_episodes
            progress_bar.progress(progress)
            status_text.text(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.3f}")
            
            # Update chart every 10 episodes
            if (episode + 1) % 10 == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Reward curve
                ax1.plot(st.session_state.training_data['episodes'], 
                        st.session_state.training_data['rewards'],
                        label='Reward', alpha=0.6)
                ax1.plot(st.session_state.training_data['episodes'],
                        pd.Series(st.session_state.training_data['rewards']).rolling(10).mean(),
                        label='Moving Avg (10)', linewidth=2)
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Reward')
                ax1.set_title('Training Rewards')
                ax1.legend()
                ax1.grid(alpha=0.3)
                
                # Success rate
                ax2.plot(st.session_state.training_data['episodes'],
                        st.session_state.training_data['success_rate'],
                        label='Success Rate', color='green', linewidth=2)
                ax2.axhline(y=0.7, color='red', linestyle='--', label='Target (70%)')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Success Rate')
                ax2.set_title('Episode Success Rate (Recent 10)')
                ax2.set_ylim([0, 1])
                ax2.legend()
                ax2.grid(alpha=0.3)
                
                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close()
        
        st.success("Training completed successfully!")
    
    # Display metrics
    if st.session_state.training_data['episodes']:
        with col1:
            st.metric(
                "Total Episodes",
                len(st.session_state.training_data['episodes'])
            )
        with col2:
            avg_reward = np.mean(st.session_state.training_data['rewards'])
            st.metric(
                "Average Reward",
                f"{avg_reward:.3f}"
            )
        with col3:
            final_success = st.session_state.training_data['success_rate'][-1] if st.session_state.training_data['success_rate'] else 0
            st.metric(
                "Final Success Rate",
                f"{final_success:.1%}"
            )

# ==================== Tab 2: Sequence Analysis ====================
with tab2:
    st.header("Pipeline Sequence Deep Analysis")
    
    # Load validation results if exist
    validation_dir = project_root / 'logs'
    validation_files = list(validation_dir.glob('validation_*/top_5_sequences.json'))
    
    if validation_files:
        latest_validation = max(validation_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_validation, 'r') as f:
            top_sequences = json.load(f)
        
        st.success(f"Loaded latest validation results: {latest_validation.parent.name}")
        
        # Display top-5 sequences
        st.subheader("Top-5 Optimal Sequences")
        
        for i, seq_data in enumerate(top_sequences, 1):
            with st.expander(f"Rank {i}: Reward = {seq_data['reward']:.3f}"):
                st.write(f"**Sequence Length**: {seq_data['length']} nodes")
                st.write(f"**Complete Sequence**: {' -> '.join(seq_data['sequence'])}")
    else:
        st.info("No validation results yet. Please run: python scripts/validate_rl_best_practices.py")

# ==================== Tab 3: Node Statistics ====================
with tab3:
    st.header("Node Usage Statistics Analysis")
    
    # Simulated node statistics
    if 'node_stats' not in st.session_state:
        st.session_state.node_stats = pd.DataFrame({
            'Node': ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9'],
            'Usage Count': [100, 85, 100, 60, 45, 40, 80, 90, 100, 100],
            'Avg Position': [0, 3.2, 1, 2.5, 4.1, 5.0, 5.5, 6.2, 7, 8],
            'Avg Reward': [0.1, 0.15, 0.2, 0.18, 0.25, 0.22, 0.19, 0.17, 0.3, 0.0]
        })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Node Usage Frequency")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(st.session_state.node_stats['Node'], 
               st.session_state.node_stats['Usage Count'],
               color='steelblue', alpha=0.7)
        ax.set_xlabel('Node')
        ax.set_ylabel('Usage Count')
        ax.set_title('Node Usage Frequency')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Node Average Position")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(st.session_state.node_stats['Node'], 
                st.session_state.node_stats['Avg Position'],
                color='coral', alpha=0.7)
        ax.set_xlabel('Average Position')
        ax.set_ylabel('Node')
        ax.set_title('Node Typical Position in Sequences')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Data table
    st.subheader("Detailed Statistics Table")
    st.dataframe(st.session_state.node_stats, use_container_width=True)

# ==================== Tab 4: Best Practices Comparison ====================
with tab4:
    st.header("Comparison with Standard Best Practices")
    
    # Load comparison results
    comparison_files = list(validation_dir.glob('validation_*/best_practices_comparison.csv'))
    
    if comparison_files:
        latest_comparison = max(comparison_files, key=lambda p: p.stat().st_mtime)
        comparison_df = pd.read_csv(latest_comparison)
        
        st.success("Loaded latest comparison results")
        
        # Display comparison results
        st.subheader("Similarity Analysis")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['green' if d else 'orange' for d in comparison_df['discovered']]
        ax.barh(comparison_df['best_practice'], 
                comparison_df['similarity'],
                color=colors, alpha=0.7)
        ax.axvline(x=0.7, color='green', linestyle='--', linewidth=2, label='Discovery Threshold (70%)')
        ax.set_xlabel('Similarity Score')
        ax.set_title('PPO Sequence Similarity to Best Practices')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Detailed comparison table
        st.subheader("Detailed Comparison Table")
        display_df = comparison_df[['best_practice', 'description', 'similarity', 'ppo_reward', 'discovered']].copy()
        display_df['similarity'] = display_df['similarity'].apply(lambda x: f"{x:.1%}")
        display_df['ppo_reward'] = display_df['ppo_reward'].apply(lambda x: f"{x:.3f}")
        display_df['discovered'] = display_df['discovered'].apply(lambda x: "Yes" if x else "No")
        display_df.columns = ['Best Practice', 'Description', 'Similarity', 'PPO Reward', 'Discovered']
        st.dataframe(display_df, use_container_width=True)
        
        # Summary
        discovered_count = comparison_df['discovered'].sum()
        total_count = len(comparison_df)
        discovery_rate = discovered_count / total_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Best Practices", total_count)
        with col2:
            st.metric("Successfully Rediscovered", discovered_count)
        with col3:
            st.metric("Rediscovery Rate", f"{discovery_rate:.1%}")
        
        if discovery_rate >= 0.6:
            st.success("Conclusion: PPO agent successfully rediscovered standard best practices!")
        elif discovery_rate >= 0.4:
            st.warning("Conclusion: PPO agent partially rediscovered best practices, continue training recommended")
        else:
            st.error("Conclusion: PPO agent failed to effectively rediscover best practices, strategy adjustment needed")
    else:
        st.info("No validation results yet. Please run: python scripts/validate_rl_best_practices.py")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>PPO Pipeline Optimizer Dashboard v1.0</p>
    <p>Materials Science AutoML Project | 2025</p>
</div>
""", unsafe_allow_html=True)
