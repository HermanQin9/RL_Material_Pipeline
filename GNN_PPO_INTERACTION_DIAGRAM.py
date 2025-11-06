#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN + PPO / GNN + PPO Interaction Architecture Diagram
 / Demonstrates how the complete system works
"""

DIAGRAM_CONTENT = """



 GNN + PPO / Complete System Architecture 





 1: / Layer 1: Input Data


Materials Project (4000+ )

 
 / Raw Data 
 CIF 
 (matminer) 

 () 

 



 2: / Layer 2: Data Preprocessing



 N0 - N2 - 
 (DataFetch) (FeatureMatrix) 

 : ID : 
 : + : [n×m] 
 + 




 3: PPO+GNN / Layer 3: PPO+GNN Core Processing



 PPO 
 (Reinforcement Learning) 

 
 (Policy Net) 
 - : 
 - : 

 

 (Value Net) 
 - : 
 - : 

 

 / Decision
 {method, param}

 
 N4 - GNN 
 (GNN Processing) 

 PPO: 
 method {0:GCN, 1:GAT, 2:SAGE} 
 param [0,1] dim {8,16,32} 

 : 
 1 
 : 
 : [, , ]
 : <5.0Å 

 2 GNN 
 GCN: Simple, Fast 
 GAT: Accurate, Slow 
 GraphSAGE: Scalable, Medium 

 3 
 [8/16/32] 

 4 
 [8/16/32] 

 : 
 [n × (m + dim)] 
 (ms/) 
 GNN () 

 

 / Other Nodes 

 N1 - (Imputation) 
 N3 - (Cleaning) 
 N5 - (Knowledge Graph) 
 N6 - (Feature Selection) 
 N7 - (Scaling) 

 PPO 
 (PPO flexibly arranges order/methods) 

 

 N8 - 
 (Model Training) 

 : 
 Random Forest (RF) 
 Gradient Boosting (GBR) 
 XGBoost (XGB) 
 CatBoost (CAT) 

 : + 
 R² () 
 MAE () 

 

 

 (Performance Evaluation) 

 
 (R², MAE, RMSE) 

 

 : 
 reward = (α × ΔR²) - (β × Δtime) 
 = - 

 



 4: PPO / Layer 4: PPO Learning & Feedback



 PPO 
 (Proximal Policy Optimization Loop) 

 : 

 

 

 : 
 GAE () 
 advantage = reward + γ*V(s') - V(s) 

 
 (>0) 
 (0) 
 (<0) 

 
 : 
 PPO 
 L = min(rt * A, clip(rt, 1-ε, 1+ε)*A)
 rt = / 
 A = 
 ε = clip (0.2) 

 : 

 

 epoch 
 (clip ratio) 

 : 

 

 

 

 PPO: 
 (Policy Entropy) 
 (Average Reward) 
 (Value Loss) 
 (Policy Loss) 
 (Clip Fraction) 

 
 : 

 

 




 / Complete Flow Diagram


 / Initialization

 

 N0/N2 N1/N3/N5/N6/N7

 

 PPO (Episode) 

 Episode = 1 
 to MaxEpisodes 
 (s) 

 
 (Step) (Trajectory) 
 Step = 0 to MaxSteps 

 PPO a 
 action = { 
 'node': N4 (GNN) 
 'method': 0/1/2 GNN 
 'param': 0.0~1.0 
 } GNN 

 / Env.step(action) 
 - GNN 
 - (N8) 
 - s', r, done 

 (s, a, r, s', d) 

 done=True? 

 

 
 PPO (Update Network) 
 Episode 
 1. (GAE) 
 2. 
 3. 
 4. 

 
 MaxEpisodes? Episode 
 (MaxEpisodes) 

 

 
 / Training Complete

 

 (Learned Policy) 
 - GCN/GAT/SAGE? 
 - ? 
 - ? 

 (Final Evaluation) 
 - 
 - 
 - 




 Episode / Detailed Example: One Complete Episode


 EPISODE 42 


 : 
 : [4000, 25] (25matminer) 
 R²: 0.820 (GNN) 
 : 0% () 
 : [4000, 0.820, 0, GPU, ...] 


 STEP 1: GNN 

 : [4000, 0.820, 0, ...] 

 : P(GCN)=0.2, P(GAT)=0.6, P(SAGE)=0.2 
 () 
 PPO: method=1 (GAT) 0.6 

 : param=0.75 

 : dim = 32 (0.75[0.67-1.0]) 

 : 
 action = { 
 'node': 4, # N4 - GNN Node 
 'method': 1, # GAT (Graph Attention) 
 'param': 0.75 # Maps to 32-dim output 
 } 


 GNN (N4) 

 : [4000 × 25] + 

 : 82ms (GAT + 32) 

 1: (4000, ~40) 
 1000+/ 

 2: GAT (32) 
 3 
 8 (multi-head attention) 
 : ~50k 
 adam 

 3: 
 : 4000 
 : [1, 32] 
 : [4000, 32] GAT 

 4: 
 : [4000, 25] 
 GNN: [4000, 32] 
 : concatenate [4000, 57] 

 : 
 enhanced_data = { 
 'X_train': [3000, 57] # (75%) 
 'X_val': [1000, 57] # (25%) 
 'feature_names': [25 + 32] = 57 
 'gnn_info': { 
 'method': 'GAT', 
 'output_dim': 32, 
 'processing_time_ms': 82, 
 'graph_stats': { 
 'n_graphs': 4000, 
 'avg_nodes': 38.5, 
 'avg_edges': 156.2 
 } 
 } 
 } 


 (N8) 

 : [3000, 57] + 

 PPO: XGBoost () 
 : 100 
 : 7 
 : 0.1 

 : XGBoost.fit(X_train[3000,57], y_train) 

 : 
 y_pred = model.predict(X_val[1000,57]) 
 R² = 0.858 0.820! 
 MAE = 0.082 eV/atom 
 RMSE = 0.115 eV/atom 

 : 
 models/formation_energy_xgb_gat32.joblib 



 
 (Before): R² = 0.820 
 (After): R² = 0.858 
 (Improvement): ΔR² = 0.038 3.8% 

 (Cost): 
 GNN: 82ms 
 : 20ms (GNN) 
 : 62ms per sample 

 : 
 α = 1.0 () 
 β = 0.01 () 

 reward = α * ΔR² - β * Δtime 
 = 1.0 * 0.038 - 0.01 * 62 
 = 0.038 - 0.62 
 = -0.582 

 : 

 GAT + 32"" 
 PPO: 

 : reward = -0.582 


 PPO 

 : 
 state: [4000, 0.820, 0, ...] 
 action: {method:1, param:0.75} 
 reward: -0.582 
 next_state: [4000, 0.858, 1, ...] 
 done: False 

 (GAE Buffer) 

 : 
 (GAT+32) (-0.582) 

 (GCNSAGE, 16) 

 : 
 P(GAT|) 
 P(GCN|) 
 P(SAGE|) 


 / Final Result for Episode 42:
 GNN: GAT ()
 : 32 ()
 R²: +0.038 (+3.8%)
 : +62ms
 : -0.582 ()
 PPO: 
 : GAT+32



 / Comparison: Different Configurations


4000GNN:


 R² (ms) 

 GNN - - 0% 0 0.0 
 GCN-8 GCN 8 +2.1% 35 +0.145 
 GCN-16 GCN 16 +2.8% 45 +0.183 
 GCN-32 GCN 32 +3.2% 55 +0.182 
 GAT-8 GAT 8 +2.9% 50 +0.169 
 GAT-16 GAT 16 +3.8% 65 +0.172 
 GAT-32 GAT 32 +3.8% 82 -0.582 
 SAGE-8 SAGE 8 +1.9% 25 +0.119 
 SAGE-16 SAGE 16 +2.4% 35 +0.157 
 SAGE-32 SAGE 32 +3.1% 45 +0.180 


PPO:
 :
 1: GCN-16 (+0.183 ) 
 2: GCN-8 (+0.145 ) 
 3: GAT-16 (+0.172 ) 
 : GAT-32 (-0.582 ) 

"""

if __name__ == '__main__':
 print(DIAGRAM_CONTENT)
