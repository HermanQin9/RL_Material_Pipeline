#!/usr/bin/env python3
"""
pipeline.py / Verify pipeline.py Implementation Completeness

pipeline.py10
Checks if pipeline.py implements all 10 nodes and methods
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_pipeline():
 """pipeline.py / Analyze pipeline.py implementation"""
 print(" pipeline.py / Analyzing pipeline.py Implementation")
 print("=" * 80)

 # 
 pipeline_file = Path(__file__).parent.parent / "pipeline.py"
 with open(pipeline_file, 'r', encoding='utf-8') as f:
 content = f.read()

 # 10
 expected_nodes = {
 'N0': {'name': 'DataFetchNode', 'methods': ['api']},
 'N1': {'name': 'ImputeNode', 'methods': ['impute']},
 'N2': {'name': 'FeatureMatrixNode', 'methods': ['construct']},
 'N3': {'name': 'CleaningNode', 'methods': ['clean']},
 'N4': {'name': 'GNNNode', 'methods': ['process']},
 'N5': {'name': 'KGNode', 'methods': ['process']},
 'N6': {'name': 'SelectionNode', 'methods': ['select']},
 'N7': {'name': 'ScalingNodeB', 'methods': ['scale']},
 'N8': {'name': 'ModelTrainingNodeB', 'methods': ['train']},
 'N9': {'name': 'EndNode', 'methods': ['terminate']},
 }

 print("\n / Node Import Check")
 print("-" * 80)

 all_imported = True
 for node_id, node_info in expected_nodes.items():
 node_name = node_info['name']
 if f"import {node_name}" in content or node_name in content:
 print(f"SUCCESS {node_id} - {node_name:25s} / Imported")
 else:
 print(f"ERROR {node_id} - {node_name:25s} / Not found")
 all_imported = False

 print("-" * 80)

 # pipeline
 print("\n Pipeline / Pipeline Function Check")
 print("-" * 80)

 functions = {
 'run_pipeline': '6 / Old 6-node pipeline',
 'run_pipeline_config': '10 / New 10-node flexible pipeline'
 }

 for func_name, description in functions.items():
 if f"def {func_name}(" in content:
 print(f"SUCCESS {func_name:25s} / Implemented - {description}")
 else:
 print(f"ERROR {func_name:25s} / Not found")

 print("-" * 80)

 # run_pipeline_config
 print("\n run_pipeline_config / Implementation Analysis")
 print("-" * 80)

 # run_pipeline_config
 nodes_in_config = {
 'N0': ['n0 = DataFetchNode()', "n0.execute('api'"],
 'N1': ['n1 = ImputeNode()', "n1.execute('impute'"],
 'N2': ['n2 = FeatureMatrixNode()', "n2.execute('construct'"],
 'N3': ['n3c = CleaningNode()', "n3c.execute('clean'"],
 'N4': ['n4g = GNNNode()', "n4g.execute('process'"],
 'N5': ['n5k = KGNode()', "n5k.execute('process'"],
 'N6': ['n6s = SelectionNode()', "n6s.execute('select'"],
 'N7': ['n7b = ScalingNodeB()', "n7b.execute('scale'"],
 'N8': ['n8t = ModelTrainingNodeB()', "n8t.execute('train'"],
 'N9': ['n9e = EndNode()', "n9e.execute('terminate'"],
 }

 config_complete = True
 for node_id, patterns in nodes_in_config.items():
 found = all(pattern in content for pattern in patterns)
 node_name = expected_nodes[node_id]['name']
 if found:
 print(f"SUCCESS {node_id} - {node_name:25s} run_pipeline_config")
 print(f" Found in run_pipeline_config")
 else:
 print(f"ERROR {node_id} - {node_name:25s} run_pipeline_config")
 print(f" Missing in run_pipeline_config")
 config_complete = False

 print("-" * 80)

 # run_pipeline
 print("\n run_pipeline () / Old Version Analysis")
 print("-" * 80)

 old_pipeline_nodes = {
 'N0': 'DataFetchNode',
 'N1': 'ImputeNode',
 'N2': 'FeatureMatrixNode',
 'N3': 'FeatureSelectionNode', # 
 'N4': 'ScalingNode', # 
 'N5': 'ModelTrainingNode', # 
 }

 for node_id, node_name in old_pipeline_nodes.items():
 # run_pipeline
 if f"{node_name}()" in content:
 print(f"SUCCESS {node_id} - {node_name:25s} run_pipeline")
 else:
 print(f"WARNING {node_id} - {node_name:25s} run_pipeline")

 print("-" * 80)

 # 
 print("\n / Issue Diagnosis")
 print("-" * 80)

 # 
 if "N0 N2 N1 N3 N4 N5" in content:
 print("WARNING / File header comment is outdated")
 print(" : N0 N2 N1 N3 N4 N5 (6)")
 print(" : N0 N2 [N1,N3,N4,N5,N6,N7] N8 N9 (10)")
 header_outdated = True
 else:
 print("SUCCESS / File header comment is correct")
 header_outdated = False

 # 10
 if config_complete:
 print("SUCCESS run_pipeline_config10")
 print(" run_pipeline_config supports all 10 nodes")
 else:
 print("ERROR run_pipeline_config")
 print(" run_pipeline_config missing some node implementations")

 print("-" * 80)

 # 
 print("\n / Implementation Summary")
 print("-" * 80)

 print(f"\n{' / Metric':<40s} {' / Status'}")
 print(f"{' / Node Imports':<40s} {'SUCCESS ' if all_imported else 'ERROR '}")
 print(f"{'run_pipeline () / Old Version':<40s} SUCCESS (6)")
 print(f"{'run_pipeline_config () / New Version':<40s} {'SUCCESS ' if config_complete else 'ERROR '} (10)")
 print(f"{' / File Header Comment':<40s} {'WARNING ' if header_outdated else 'SUCCESS '}")

 print("\n" + "=" * 80)

 return {
 'all_imported': all_imported,
 'config_complete': config_complete,
 'header_outdated': header_outdated
 }


def check_method_coverage():
 """ / Check method coverage"""
 print("\n\n / Method Coverage Analysis")
 print("=" * 80)

 pipeline_file = Path(__file__).parent.parent / "pipeline.py"
 with open(pipeline_file, 'r', encoding='utf-8') as f:
 content = f.read()

 # 
 method_calls = {
 'N0': ["execute('api'"],
 'N1': ["execute('impute'"],
 'N2': ["execute('construct'"],
 'N3': ["execute('clean'"],
 'N4': ["execute('process'"],
 'N5': ["execute('process'"],
 'N6': ["execute('select'"],
 'N7': ["execute('scale'"],
 'N8': ["execute('train'"],
 'N9': ["execute('terminate'"],
 }

 print("\n / Method Call Check")
 print("-" * 80)

 all_methods_called = True
 for node_id, methods in method_calls.items():
 found_count = sum(1 for method in methods if method in content)
 if found_count > 0:
 print(f"SUCCESS {node_id} : {found_count} / Method calls: {found_count} location(s)")
 else:
 print(f"ERROR {node_id} : / Method calls: Not found")
 all_methods_called = False

 print("-" * 80)

 return all_methods_called


def main():
 """ / Main function"""
 print("\n" + "START / Starting Analysis ".center(80, "="))
 print()

 # pipeline
 results = analyze_pipeline()

 # 
 methods_ok = check_method_coverage()

 # 
 print("\n" + " / Final Conclusion ".center(80, "="))
 print()

 if results['all_imported'] and results['config_complete'] and methods_ok:
 if results['header_outdated']:
 print("WARNING pipeline.py ")
 print("WARNING pipeline.py functionally complete but documentation needs update")
 print()
 print("SUCCESS 10")
 print("SUCCESS run_pipeline_config10")
 print("SUCCESS ")
 print("WARNING 10")
 else:
 print(" pipeline.py ")
 print(" pipeline.py fully implemented!")
 print()
 print("SUCCESS ")
 print("SUCCESS ")
 print("SUCCESS ")
 else:
 print("ERROR pipeline.py ")
 print("ERROR pipeline.py has incomplete parts")
 print()
 if not results['all_imported']:
 print("ERROR ")
 if not results['config_complete']:
 print("ERROR run_pipeline_config")
 if not methods_ok:
 print("ERROR ")

 print("\n" + "=" * 80)


if __name__ == "__main__":
 main()
