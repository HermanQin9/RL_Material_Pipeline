#!/usr/bin/env python3
"""
4K数据集修复脚本（命令行入口）
Command-line entry point for regenerating the safe 4K dataset cache.
"""
from __future__ import annotations

from methods.data import fix_4k_data_generation


def main() -> bool:
    """Regenerate the 4K cache applying stricter validation rules."""
    print("🔧 修复4K数据集生成")
    print("🔧 Fixing 4K Dataset Generation")

    success = fix_4k_data_generation()
    if success:
        print("\n✅ 现在可以尝试使用4K数据集进行PPO训练")
        return True

    print("\n❌ 修复失败，请检查错误信息")
    return False


if __name__ == "__main__":  # pragma: no cover - CLI entry
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"❌ 修复过程中出现错误: {exc}")
        raise
