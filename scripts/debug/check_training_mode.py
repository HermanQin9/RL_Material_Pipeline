#!/usr/bin/env python3
"""

Check training mode configuration
"""
import os
import sys
from pathlib import Path

# 
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import TEST_MODE, N_TOTAL, CACHE_FILE

def check_training_mode():
 print('=== / Training Mode Configuration ===')
 print(f'PIPELINE_TEST: {os.getenv("PIPELINE_TEST", "")}')
 print(f'TEST_MODE: {TEST_MODE}')
 print(f' / Dataset Size: {N_TOTAL:,} ')
 print(f' / Cache File: {CACHE_FILE}')

 if not TEST_MODE:
 print('\nSTART ! / Successfully switched to training mode!')
 print(f' - : {N_TOTAL:,} / Large dataset: {N_TOTAL:,} material samples')
 return True
 else:
 print('\nWARNING / Warning: Still in test mode')
 return False

if __name__ == "__main__":
 check_training_mode()
