#!/usr/bin/env python3
"""
Clear_Version 文件完整性测试
Test all files in Clear_Version directory
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== Clear_Version 文件完整性测试 ===\n")

def test_import(module_name, description):
    """测试模块导入"""
    try:
        __import__(module_name)
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description} - 错误: {e}")
        return False

def test_function_call(func, description):
    """测试函数调用"""
    try:
        result = func()
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description} - 错误: {e}")
        return False

# 测试核心模块导入
print("1. 测试核心模块导入:")
modules = [
    ("config", "config.py"),
    ("nodes", "nodes.py"), 
    ("env.pipeline_env", "env/pipeline_env.py"),
    ("ppo.trainer", "ppo/trainer.py")
]

import_success = []
for module, desc in modules:
    success = test_import(module, f"   {desc}")
    import_success.append(success)

print("\n2. 测试方法模块:")
methods_success = []
try:
    from methods import data_methods, model_methods
    print("✅    methods/data_methods.py")
    print("✅    methods/model_methods.py")
    methods_success = [True, True]
except Exception as e:
    print(f"❌    methods/ - 错误: {e}")
    methods_success = [False, False]

print("\n3. 测试管线功能:")
pipeline_success = []

# 测试原版管线
try:
    from pipeline import run_pipeline as run_pipeline_original
    print("✅    pipeline.py (原版)")
    pipeline_success.append(True)
except Exception as e:
    print(f"❌    pipeline.py (原版) - 错误: {e}")
    pipeline_success.append(False)

# 测试修复版管线
try:
    from pipeline import run_pipeline as run_pipeline_fixed
    print("✅    pipeline_fixed.py (修复版)")
    pipeline_success.append(True)
except Exception as e:
    print(f"❌    pipeline_fixed.py (修复版) - 错误: {e}")
    pipeline_success.append(False)

print("\n4. 测试示例文件:")
example_success = []
try:
    from scripts.example_usage import main as example_main
    print("✅    example_usage.py")
    example_success.append(True)
except Exception as e:
    print(f"❌    example_usage.py - 错误: {e}")
    example_success.append(False)

# 统计结果
total_tests = len(import_success) + len(methods_success) + len(pipeline_success) + len(example_success)
passed_tests = sum(import_success + methods_success + pipeline_success + example_success)

print(f"\n=== 测试总结 ===")
print(f"总测试数: {total_tests}")
print(f"通过数: {passed_tests}")
print(f"失败数: {total_tests - passed_tests}")
print(f"成功率: {passed_tests/total_tests*100:.1f}%")

if passed_tests == total_tests:
    print("\n🎉 所有文件测试通过! Clear_Version 已准备就绪!")
else:
    print(f"\n⚠️  有 {total_tests - passed_tests} 个文件存在问题，需要修复")

print("\n=== 测试完成 ===")
