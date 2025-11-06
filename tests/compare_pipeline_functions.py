#!/usr/bin/env python3
"""
 / Compare Node Usage in Two Pipeline Functions

 run_pipeline() run_pipeline_config() 
Clearly shows the differences between run_pipeline() and run_pipeline_config()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_pipeline_functions():
 """ / Analyze two pipeline functions"""
 print(" run_pipeline() vs run_pipeline_config()")
 print("=" * 80)

 # 6
 old_pipeline_nodes = {
 'N0': {'class': 'DataFetchNode', 'purpose': ' / Data Fetch'},
 'N1': {'class': 'ImputeNode', 'purpose': ' / Imputation'},
 'N2': {'class': 'FeatureMatrixNode', 'purpose': ' / Feature Matrix'},
 'N3': {'class': 'FeatureSelectionNode', 'purpose': ' () / Feature Selection (Old)'},
 'N4': {'class': 'ScalingNode', 'purpose': ' () / Scaling (Old)'},
 'N5': {'class': 'ModelTrainingNode', 'purpose': ' () / Model Training (Old)'},
 }

 # 10
 new_pipeline_nodes = {
 'N0': {'class': 'DataFetchNode', 'purpose': ' / Data Fetch', 'position': ''},
 'N1': {'class': 'ImputeNode', 'purpose': ' / Imputation', 'position': ''},
 'N2': {'class': 'FeatureMatrixNode', 'purpose': ' / Feature Matrix', 'position': ''},
 'N3': {'class': 'CleaningNode', 'purpose': ' () / Cleaning (New)', 'position': ''},
 'N4': {'class': 'GNNNode', 'purpose': ' () / GNN (New)', 'position': ''},
 'N5': {'class': 'KGNode', 'purpose': ' () / Knowledge Graph (New)', 'position': ''},
 'N6': {'class': 'SelectionNode', 'purpose': ' () / Selection (New Position)', 'position': ''},
 'N7': {'class': 'ScalingNodeB', 'purpose': ' () / Scaling (New Position)', 'position': ''},
 'N8': {'class': 'ModelTrainingNodeB', 'purpose': ' () / Training (New Position)', 'position': ''},
 'N9': {'class': 'EndNode', 'purpose': ' / End Node', 'position': ''},
 }

 print("\n / Node Comparison Table")
 print("-" * 80)
 print(f"{'ID':<8} {'(run_pipeline)':<35} {'(run_pipeline_config)':<35}")
 print("-" * 80)

 for node_id in ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']:
 old_info = old_pipeline_nodes.get(node_id, {})
 new_info = new_pipeline_nodes.get(node_id, {})

 old_text = f"{old_info.get('class', 'ERROR ')}" if old_info else "ERROR / Not Exist"
 new_text = f"{new_info.get('class', '-')}" if new_info else "-"

 # 
 if node_id in ['N3', 'N4', 'N5']:
 marker = "WARNING "
 elif node_id in ['N6', 'N7', 'N8', 'N9']:
 marker = " "
 else:
 marker = "SUCCESS "

 print(f"{node_id:<8} {old_text:<35} {new_text:<35} {marker}")

 print("-" * 80)

 # 
 print("\n / Key Differences")
 print("-" * 80)

 print("\n run_pipeline() :")
 print(" 1. 6 (N0-N5)")
 print(" 2. : N0N2N1N3N4N5")
 print(" 3. N3=, N4=, N5=")
 print(" 4. GNN ")
 print(" 5. ")

 print("\n🟢 run_pipeline_config() :")
 print(" 1. 10 (N0-N9)")
 print(" 2. : N0N2[flexible]N8N9")
 print(" 3. N3=, N4=GNN, N5=")
 print(" 4. N6=, N7=, N8=, N9=")
 print(" 5. PPO ")

 # N5
 print("\n N5")
 print("-" * 80)
 print(" : 6")
 print(" ")
 print(" :")
 print(" N0: ")
 print(" N2: ")
 print(" N1: ")
 print(" N3: ")
 print(" N4: ")
 print(" N5: ")
 print(" ")
 print(" 10PPO")

 # 
 print("\n / Code Location")
 print("-" * 80)
 print(f" run_pipeline(): pipeline.py 65-165 ")
 print(f" run_pipeline_config(): pipeline.py 168-325 ")

 # 
 print("\n / Usage Recommendations")
 print("-" * 80)
 print(" : run_pipeline_config()")
 print(" - ")
 print(" - ")
 print(" - PPO")
 print(" ")
 print(" : run_pipeline()")
 print(" - ")
 print(" - ")
 print(" - ")

 print("\n" + "=" * 80)


def show_code_examples():
 """ / Show code examples"""
 print("\n\n / Code Example Comparison")
 print("=" * 80)

 print("\n run_pipeline() :")
 print("-" * 80)
 print("""
from pipeline import run_pipeline

result = run_pipeline(
 cache=True,
 impute_strategy='mean', # N1
 selection_strategy='pca', # N3 ()
 scaling_strategy='standard', # N4 ()
 model_strategy='rf', # N5 ()
)
# : N0N2N1N3N4N5 (6)
# N6, N7, N8, N9
 """)

 print("\n🟢 run_pipeline_config() :")
 print("-" * 80)
 print("""
from pipeline import run_pipeline_config

config = {
 'sequence': ['N0','N2','N1','N3','N4','N5','N6','N7','N8','N9'],
 'N1_method': 'median', # 
 'N3_method': 'outlier', # ()
 'N4_method': 'gat', # GNN ()
 'N5_method': 'entity', # ()
 'N6_method': 'pca', # ()
 'N7_method': 'std', # ()
 'N8_method': 'xgb', # ()
 'cache': True
}
result = run_pipeline_config(**config)
# 10 
# PPO 
 """)

 print("\n" + "=" * 80)


def main():
 """ / Main function"""
 print("\n" + "START / Starting Analysis ".center(80, "="))

 analyze_pipeline_functions()
 show_code_examples()

 print("\n" + " / Summary ".center(80, "="))
 print()
 print(" run_pipeline() N5 ")
 print()
 print("SUCCESS :")
 print(" 1. 6 (N0-N5)")
 print(" 2. N5 ")
 print(" 3. N6-N9 ")
 print(" 4. run_pipeline_config() 10 ")
 print(" 5. ")
 print()
 print(" : run_pipeline_config() 10 ")
 print()
 print("=" * 80)


if __name__ == "__main__":
 main()
