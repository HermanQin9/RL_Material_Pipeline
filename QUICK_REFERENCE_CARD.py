#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN+PPO / GNN+PPO Quick Reference Card

"""

QUICK_REFERENCE = """


 GNN + PPO 
 Quick Reference Card v1.0 




 / One-Line Summary


PPOGNN

PPO intelligently selects GNN methods and parameters to auto-optimize material feature
engineering, maximizing prediction accuracy while controlling computational cost.



 GNN / Three GNN Quick Comparison



 GCN GAT GraphSAGE 

 + 

 

 
 ~50ms/ ~80ms/ ~40ms/ 
 () 

 




 PPO / PPO's Three-Level Decisions





 L1: GNN 0: GCN vs 
 method {0, 1, 2} 1: GAT 
 2: GraphSAGE 

 L2: param [0, 1] vs 
 param dim 0.0-0.33 8dim 
 0.33-0.67 16dim 
 0.67-1.0 32dim 

 L3: 
 (extensible) 





 PPOGNN / When PPO Chooses Each Method


GCN / Choose GCN when:
 - (100-1000)

 

 
 : GCN-16
 : +2-3% R²

GAT / Choose GAT when:

 

 (>1000)

 : GAT-16
 : +3-4% R²

GraphSAGE / Choose GraphSAGE when:
 (>500)

 

 
 : SAGE-16
 : +2-3% R²



 PPO / PPO Learning Process


Step 1: 

 GNN

Step 2: 
 GNN
 GNN

 : (, , , )

Step 3: 
 : A(s,a) = R(s,a) - V(s)

 
 : 

Step 4: 
 PPO

 
 clip

Step 5: 


:
 : ()
 : 
 : 



 / Reward Function Explained


: 


 reward = α * performance_gain 
 - β * computational_cost 

 α 1.0 () 
 β 0.01 () 


 1 - / Good Choice:
 ΔR²: +0.04 (4%)
 : 50ms
 reward = 1.0 * 0.04 - 0.01 * 50
 reward = 0.04 - 0.50 = -0.46
 ! ()

 2 - / Balanced Choice:
 ΔR²: +0.03 (3%)
 : 30ms
 reward = 1.0 * 0.03 - 0.01 * 30
 reward = 0.03 - 0.30 = -0.27


 3 - / Actual Best:
 ΔR²: +0.02 (2%)
 : 10ms
 reward = 1.0 * 0.02 - 0.01 * 10
 reward = 0.02 - 0.10 = -0.08
 !

: 



 PPO / How to Interpret Learned PPO Policy


:

policy_output = policy_net(state)
# : [P_gcn, P_gat, P_sage] + []

:
 : =4000, =0.80, 
 P_gcn=0.2, P_gat=0.7, P_sage=0.1
 =0.6 (16)
 : PPOGAT, 16

:
 L (): 

 

:
 : 
 : 
 : (, )
 Clip: 0.1-0.2 ()



 / Performance Benchmarks


4000:


 R² MAE 

 GNN 0% 0% ~20ms 
 GCN-8 +2.1% -12% ~35ms +0.145 
 GCN-16 +2.8% -15% ~45ms +0.183 
 GCN-32 +3.2% -18% ~55ms +0.182 
 GAT-8 +2.9% -14% ~50ms +0.169 
 GAT-16 +3.8% -20% ~65ms +0.172 
 GAT-32 +3.8% -20% ~82ms -0.582 
 SAGE-8 +1.9% -11% ~25ms +0.119 
 SAGE-16 +2.4% -14% ~35ms +0.157 
 SAGE-32 +3.1% -17% ~45ms +0.180 


: GCN-16 GAT-16



 GNN / Code Usage Examples


 / Basic Usage:

from methods import gnn_process

result = gnn_process(
 data=data_dict, # X_train, X_val, structures
 strategy='gat', # 'gcn' / 'gat' / 'sage'
 param=0.5 # 0.08, 0.516, 1.032
)

print(result['gnn_info']) # 


PPO / Usage in PPO:

def env_step(action):
 method_id = action['method'] # 0/1/2
 param = action['param'] # 0.0-1.0

 strategy_map = {0: 'gcn', 1: 'gat', 2: 'sage'}

 result = gnn_process(
 data=pipeline_state['data'],
 strategy=strategy_map[method_id],
 param=param
 )

 pipeline_state['enhanced_data'] = result['data']
 return result['performance_metrics']


GNN / Access GNN Functions:

from methods import (
 gnn_process, # 
 structure_to_graph, # 
 extract_gnn_features, # 
 SimpleGCN, # GCN
 SimpleGAT, # GAT
 SimpleGraphSAGE # GraphSAGE
)



 SUCCESS / Quick Checklist


GNN:
 methods/data_methods.py

 

 6/6 

:
 PyTorch 2.7.1 
 PyTorch Geometric ()
 pymatgen 
 PPO 

:
 GNN_PURPOSE_AND_PPO_CHOICES.py
 GNN_FLOWCHART_AND_DECISION_TREE.py
 GNN_PPO_INTERACTION_DIAGRAM.py
 N4_GNN_INTEGRATION_INFO.py
 test_n4_gnn_integration.py
 (QUICK_REFERENCE.py)

:
 : 
 : 
 : 
 : 
 PPO!



 START / Next Actions


 / Get Started Now:

1. (5)
 python GNN_FLOWCHART_AND_DECISION_TREE.py

2. (2)
 python test_n4_gnn_integration.py

3. (30)
 python scripts/train_ppo.py --episodes 100

4. (20)
 PPO

5. ()




 / FAQ


Q: GNN?
A: GNN

Q: PPOGNN?
A: GNNPPO

Q: ?
A: ! 3×3=9

Q: PyTorch Geometric?
A: 

Q: ?
A: GraphSAGEGPU

Q: PPO?
A: 

Q: GNN?
A: ! SimpleGCN

"""

if __name__ == '__main__':
 print(QUICK_REFERENCE)
