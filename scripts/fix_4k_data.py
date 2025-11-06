#!/usr/bin/env python3
"""
4K
Command-line entry point for regenerating the safe 4K dataset cache.
"""
from __future__ import annotations

from methods.data import fix_4k_data_generation


def main() -> bool:
 """Regenerate the 4K cache applying stricter validation rules."""
 print(" 4K")
 print(" Fixing 4K Dataset Generation")

 success = fix_4k_data_generation()
 if success:
 print("\nSUCCESS 4KPPO")
 return True

 print("\nERROR ")
 return False


if __name__ == "__main__": # pragma: no cover - CLI entry
 try:
 main()
 except Exception as exc: # pragma: no cover
 print(f"ERROR : {exc}")
 raise
