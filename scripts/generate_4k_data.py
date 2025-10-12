#!/usr/bin/env python3
"""
4K数据集生成和验证脚本（命令行入口）
Command-line entry point for 4K dataset generation and validation.
"""
from __future__ import annotations

import os
from methods.data import generate_4k_data_safe, test_4k_data_loading

# 强制设置为4K模式
os.environ["PIPELINE_TEST"] = "0"


def main() -> bool:
    """Generate the 4K dataset and validate cache integrity."""
    print("🎯 4K数据集生成和验证")
    print("🎯 4K Dataset Generation and Validation")

    success = generate_4k_data_safe()
    if not success:
        print("❌ 4K数据集生成失败")
        return False

    load_ok, _ = test_4k_data_loading()
    if load_ok:
        print("\n🎉 4K数据集生成和验证完成!")
        print("🎉 4K Dataset Generation and Validation Complete!")
        print("📁 现在可以使用4K数据集进行PPO训练")
        return True

    print("\n⚠️ 数据生成成功但加载测试失败")
    return False


if __name__ == "__main__":  # pragma: no cover - CLI entry
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"❌ 程序执行出错: {exc}")
        raise
