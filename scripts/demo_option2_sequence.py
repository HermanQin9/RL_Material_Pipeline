"""
Demo: Run a valid Option 2 sequence through the environment and pipeline
"""
import os, sys
import logging
# Ensure project root is on sys.path when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.pipeline_env import PipelineEnv

def main():
 # Reduce logging noise for a clean demo output
 logging.getLogger().setLevel(logging.WARNING)
 logging.getLogger('pipeline').setLevel(logging.WARNING)
 logging.getLogger('methods.data_methods').setLevel(logging.WARNING)
 env = PipelineEnv()
 obs = env.reset()

 # Sequence: N0 -> N2 -> [N1, N3, N6, N7] -> N8 -> N9 (skip N4,N5 for now)
 # Step 0: N0 (forced by mask)
 action = {'node': env.pipeline_nodes.index('N0'), 'method': 0, 'params': [0.0]}
 obs, r, d, _, _ = env.step(action)

 # Step 1: N2 (forced by mask)
 action = {'node': env.pipeline_nodes.index('N2'), 'method': 0, 'params': [0.0]}
 obs, r, d, _, _ = env.step(action)

 # Middle: N1 mean, N3 outlier, N6 variance, N7 std
 seq_mid = [
 ('N1', 0, 0.3), # mean with any param (ignored)
 ('N3', 0, 0.5), # outlier IQR
 ('N6', 0, 0.01), # variance
 ('N7', 0, 0.0), # std
 ]
 for nid, midx, p in seq_mid:
 action = {
 'node': env.pipeline_nodes.index(nid),
 'method': midx,
 'params': [float(p)]
 }
 obs, r, d, _, _ = env.step(action)

 # N8 train rf (use param=0.0 -> fewer trees for speed)
 action = {'node': env.pipeline_nodes.index('N8'), 'method': 0, 'params': [0.0]}
 obs, r, d, _, _ = env.step(action)

 # Inspect built config before N9
 print("Config before N9:", env.pipeline_config)

 # N9 terminate -> should compute reward
 action = {'node': env.pipeline_nodes.index('N9'), 'method': 0, 'params': [0.0]}
 obs, r, d, _, metrics = env.step(action)
 print('Done:', d, 'Reward:', r)
 print('Metrics keys:', list(metrics.keys()))

if __name__ == '__main__':
 main()
