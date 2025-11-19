#!/usr/bin/env python3
"""
4K
Command-line entry point for 4K dataset generation and validation.
"""
from __future__ import annotations

import os
from methods.data import generate_4k_data_safe, test_4k_data_loading

# 4K
os.environ["PIPELINE_TEST"] = "0"


def main() -> bool:
 """Generate the 4K dataset and validate cache integrity."""
 print(" 4K")
 print(" 4K Dataset Generation and Validation")

 success = generate_4k_data_safe()
 if not success:
 print("ERROR 4K")
 return False

 load_ok, _ = test_4k_data_loading()
 if load_ok:
 print("\n 4K!")
 print(" 4K Dataset Generation and Validation Complete!")
 print(" 4KPPO")
 return True

 print("\nWARNING ")
 return False


if __name__ == "__main__": # pragma: no cover - CLI entry
 try:
 main()
 except Exception as exc: # pragma: no cover
 print(f"ERROR : {exc}")
 raise
