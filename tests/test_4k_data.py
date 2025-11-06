#!/usr/bin/env python3
"""
4K
Test 4K Dataset Fetching and Processing
"""
import os
import sys
import time
from pathlib import Path

# 4K
os.environ['PIPELINE_TEST'] = '0'

# 
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_4k_data_fetch():
 """4K"""
 print(" 4K")
 print(" Testing 4K Dataset Fetching")
 print("=" * 60)

 try:
 # 
 from config import TEST_MODE, N_TOTAL, CACHE_FILE, API_KEY
 print(f" :")
 print(f" TEST_MODE: {TEST_MODE}")
 print(f" N_TOTAL: {N_TOTAL}")
 print(f" CACHE_FILE: {CACHE_FILE}")
 print(f" API_KEY: {'' if API_KEY else ''}")
 print()

 if not API_KEY:
 print("ERROR API_KEY ")
 return False

 # 
 from methods.data_methods import fetch_data

 # 
 cache_path = Path(CACHE_FILE)
 if cache_path.exists():
 print(f" : {cache_path}")
 print(f" : {cache_path.stat().st_size / (1024*1024):.1f} MB")

 # 
 print(" ...")
 try:
 df = fetch_data(cache=True)
 print(f"SUCCESS : {df.shape}")
 print(f" : {list(df.columns)}")
 print(f" 3:")
 print(df[['material_id', 'formula_pretty', 'formation_energy_per_atom']].head(3))
 return True
 except Exception as e:
 print(f"WARNING : {e}")
 print(" ...")
 cache_path.unlink()

 # 
 print(" API4K...")
 start_time = time.time()

 df = fetch_data(cache=False)

 fetch_time = time.time() - start_time
 print(f"SUCCESS !")
 print(f" : {df.shape}")
 print(f" : {fetch_time/60:.1f} ")
 print(f" : {list(df.columns)}")
 print()

 # 
 print(" :")
 print(f" : {df.isnull().sum().sum()}")

 if 'formation_energy_per_atom' in df.columns:
 target_values = df['formation_energy_per_atom']
 print(f" :")
 print(f" : {target_values.mean():.3f}")
 print(f" : {target_values.std():.3f}")
 print(f" : {target_values.min():.3f} ~ {target_values.max():.3f}")

 if 'structure' in df.columns:
 valid_structures = df['structure'].notna().sum()
 print(f" : {valid_structures} / {len(df)}")

 print(f"\n 5:")
 display_cols = ['material_id', 'formula_pretty', 'formation_energy_per_atom']
 available_cols = [col for col in display_cols if col in df.columns]
 print(df[available_cols].head())

 return True

 except Exception as e:
 print(f"ERROR : {e}")
 import traceback
 traceback.print_exc()
 return False

def test_4k_featurization():
 """4K"""
 print("\n" + "=" * 60)
 print(" 4K")
 print(" Testing 4K Data Featurization")
 print("=" * 60)

 try:
 from methods.data_methods import fetch_data, featurize_data

 # 
 print(" ...")
 df = fetch_data(cache=True)
 print(f"SUCCESS : {df.shape}")

 # 
 print(" ...")
 start_time = time.time()

 df_feat = featurize_data(df)

 feat_time = time.time() - start_time
 print(f"SUCCESS !")
 print(f" : {feat_time/60:.1f} ")
 print(f" : {df_feat.shape}")
 print(f" : {df_feat.shape[1] - df.shape[1]}")

 # 
 print(" :")
 numeric_cols = df_feat.select_dtypes(include=['number']).columns
 print(f" : {len(numeric_cols)}")
 print(f" : {df_feat.isnull().sum().sum()}")

 # 
 exclude_cols = ['material_id', 'structure', 'elements', 'formula_pretty', 'composition', 'formation_energy_per_atom']
 feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
 print(f" : {len(feature_cols)}")
 if feature_cols:
 print(f" 10: {feature_cols[:10]}")

 return True

 except Exception as e:
 print(f"ERROR : {e}")
 import traceback
 traceback.print_exc()
 return False

def test_4k_pipeline():
 """4K"""
 print("\n" + "=" * 60)
 print("START 4K")
 print("START Testing Complete 4K Data Pipeline")
 print("=" * 60)

 try:
 from methods.data_methods import fetch_and_featurize

 print(" ...")
 start_time = time.time()

 result = fetch_and_featurize(cache=True)

 pipeline_time = time.time() - start_time
 print(f"SUCCESS !")
 print(f" : {pipeline_time/60:.1f} ")

 # 
 print(" :")
 print(f" : {list(result.keys())}")

 if 'train_data' in result and result['train_data'] is not None:
 train_shape = result['train_data'].shape
 print(f" : {train_shape}")

 if 'test_data' in result and result['test_data'] is not None:
 test_shape = result['test_data'].shape
 print(f" : {test_shape}")

 if 'full_data' in result and result['full_data'] is not None:
 full_shape = result['full_data'].shape
 print(f" : {full_shape}")

 return True

 except Exception as e:
 print(f"ERROR : {e}")
 import traceback
 traceback.print_exc()
 return False

def main():
 """"""
 print(" 4K")
 print(" Complete 4K Dataset Testing")
 print("=" * 60)

 test_results = []

 # 1: 
 print(" 1: ")
 result1 = test_4k_data_fetch()
 test_results.append(("", result1))

 if result1:
 # 2: 
 print("\n 2: ")
 result2 = test_4k_featurization()
 test_results.append(("", result2))

 if result2:
 # 3: 
 print("\n 3: ")
 result3 = test_4k_pipeline()
 test_results.append(("", result3))

 # 
 print("\n" + "=" * 60)
 print(" ")
 print(" Test Results Summary")
 print("=" * 60)

 for test_name, result in test_results:
 status = "SUCCESS " if result else "ERROR "
 print(f" {test_name}: {status}")

 all_passed = all(result for _, result in test_results)

 if all_passed:
 print("\n ! 4K")
 print(" All tests passed! 4K dataset is ready")
 print(" PPO:")
 print(" $env:PIPELINE_TEST=\"0\"; python train_ppo_4k.py")
 else:
 print("\nWARNING ")
 print("WARNING Some tests failed, please check error messages")

 return all_passed

if __name__ == "__main__":
 success = main()
 sys.exit(0 if success else 1)
