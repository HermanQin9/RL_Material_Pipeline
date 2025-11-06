#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN + PPO / GNN + PPO Decision Tree and Flow Diagrams
"""

CONTENT = """


 . GNN - GNN? 


 PROBLEM:


 (matminer):
 (Elemental composition)
 (Crystal density)
 (Symmetry)
 (Lattice parameters)

ERROR :
 (Spatial atomic interactions) GNN!
 (Crystal topology) GNN!
 (Local chemical environment) GNN!

 SOLUTION:


GNN:

 GNN 

 Crystal Graph GNN Model Enhanced
 Structure with (GCN/GAT/ Feature
 nodes & GraphSAGE) Matrix
 edges 
 +GNN

 



 . GNN 



 1: GCN (Graph Convolutional Network) - 


:
 ():
 1. 
 2. (, Mean Aggregation)
 3. 
 4. 
 5. 

 GCN 

 h_v #1 
 h_u #2 MEAN W*feat + ReLU h'_v 
 ... #3 b 

 

:
 (~50ms/)

 


PPOGCN:
 : 
 : 
 : 
 : method=0 (GCN)



 2: GAT (Graph Attention Network) - 


:
 ():
 1. (0-1)
 2. 
 3. 
 4. 


 
 h_v #1 α=0.8 ×feature 
 h_u #2 α=0.1 ×feature SUM ReLU h'_v 
 ... #3 α=0.1 ×feature 

 

:
 (+4%)

 
 (~80ms/)


PPOGAT:
 : 
 : 
 : 
 : method=1 (GAT)



 3: GraphSAGE - 


:
 ():
 1. K
 2. K
 3. 
 4. 
 5. 

 +

 h_v #1() MEAN 
 h_u #2() CONCAT W*feat h'_v 
 ... #3() 

 

:
 (~40ms/)

 

 

PPOGraphSAGE:
 : (1000+)
 : 
 : 
 : method=2 (GraphSAGE)



 . PPO / PPO Decision Space 


PPO / PPO Tunable Parameters:


 1: GNN (action['method']) 

 0 GCN | | 
 1 GAT | | 
 2 GraphSAGE | | 



 2: (action['param']) 

 param [0.00-0.33) dim=8 | | | 
 param [0.33-0.67) dim=16 | | | 
 param [0.67-1.00] dim=32 | | | 



 () 3: 

 3.0~10.0 Å 
 mean/sum/max 
 1~20 GraphSAGE


:
 : 3 () × 3 () = 9 
 : 9 × = !



 . PPO / PPO Decision Flow 



 

 (Observation) 

 R² 

 

 

 PPO 
 (Policy Network: State Action Probs) 

 

 GNN / Choose GNN Method 

 (If): 
 R² GAT () 
 GCN () 
 SAGE () 

 

 / Choose Dim 

 (If): 
 32 () 
 16 () 
 8 () 

 

 GNN 

 GNN 

 

 

 / Compute Reward
 R² 
 R² 

 = α*-β*

 

 PPO 
 (Update using GAE) 

 
 / Next Episode



 . PPO / PPO Decision Examples 


1: / Case 1: Abundant Data, Accuracy Priority


: 4000

 (Observation):
 : 4000 ()
 R²: 0.85 ()
 : GPU ()
 : 

PPO:
 1 : GAT ()
 : 
 : R² +0.04 ()

 2 : 32 ()
 : 
 : +30ms

:
 action = {
 'node': 4,
 'method': 1, # GAT
 'param': 0.85 # 32
 }

:
 R²: 0.88-0.89 (0.85)
 : ~80ms/
 : ()


2: / Case 2: Sparse Data, Time Pressure


: 200

 (Observation):
 : 200 ()
 R²: 0.80 ()
 : CPU only ()
 : 

PPO:
 1 : GCN ()
 : 
 : ~50ms/

 2 : 8 ()
 : 
 : 

:
 action = {
 'node': 4,
 'method': 0, # GCN
 'param': 0.15 # 8
 }

:
 R²: 0.82 (0.80, )
 : ~50ms/ ()
 : (+)


3: / Case 3: Very Large Crystal Structures


: 1000+

 (Observation):
 : 1000+ ()
 R²: 0.82 ()
 : 
 : 

PPO:
 1 : GraphSAGE ()
 : 
 : ~40ms/ ()

 2 : 16 ()
 : 
 : 

:
 action = {
 'node': 4,
 'method': 2, # GraphSAGE
 'param': 0.50 # 16
 }

:
 R²: 0.84 (0.82)
 : ~40ms/ ()
 : 
 : ()



 . GNNPPO / Summary: GNN + PPO Synergy 


GNN:

 

 +3-4%


PPO:

 GNN

 


:

 PPO GNN PPO

 

 ...

 (Optimal Configuration)


 / Key Numbers:

 :
 GNN: 3
 : 3
 : 9 ()

 :
 : +2-4% (R²)
 : -10-22% (MAE)
 : 40-80ms/ (GNN)

 :
 Level 1: GNN ()
 Level 2: ()
 Level 3: ()

"""

if __name__ == '__main__':
 print(CONTENT)
