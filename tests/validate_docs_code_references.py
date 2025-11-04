"""
验证文档中的代码引用是否准确 / Validate Code References in Documentation

此脚本验证文档中提到的代码行数和内容是否与实际代码匹配。
This script validates that code line numbers and content mentioned in the documentation
match the actual code.
"""

import os
import sys
from pathlib import Path


def read_file_lines(filepath, start_line, end_line):
    """读取文件的指定行 / Read specific lines from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[start_line-1:end_line]
    except Exception as e:
        return None


def validate_code_reference(description, filepath, line_range, expected_content_keywords):
    """验证代码引用 / Validate code reference"""
    print(f"\n验证 / Validating: {description}")
    print(f"  文件 / File: {filepath}")
    print(f"  行数 / Lines: {line_range}")
    
    full_path = Path(__file__).parent.parent / filepath
    if not full_path.exists():
        print(f"  ❌ 文件不存在 / File does not exist")
        return False
    
    start, end = line_range
    lines = read_file_lines(full_path, start, end)
    
    if lines is None:
        print(f"  ❌ 无法读取文件 / Cannot read file")
        return False
    
    content = ''.join(lines)
    
    # 检查关键词
    all_found = True
    for keyword in expected_content_keywords:
        if keyword in content:
            print(f"  ✓ 找到关键词 / Found keyword: '{keyword}'")
        else:
            print(f"  ❌ 未找到关键词 / Keyword not found: '{keyword}'")
            all_found = False
    
    return all_found


def main():
    """主测试函数 / Main test function"""
    print("="*80)
    print("验证文档中的代码引用 / Validating Code References in Documentation")
    print("="*80)
    
    # 定义要验证的代码引用
    references = [
        {
            'description': '节点-方法映射定义 / Node-Method Mapping',
            'file': 'env/pipeline_env.py',
            'lines': (38, 44),
            'keywords': ['methods_for_node', 'N1', 'mean', 'median', 'knn', 'N5', 'rf', 'gbr']
        },
        {
            'description': 'PPO策略网络节点头 / PPO Policy Node Head',
            'file': 'ppo/policy.py',
            'lines': (34, 38),
            'keywords': ['node_head', 'Linear', '6']
        },
        {
            'description': 'PPO策略网络方法头 / PPO Policy Method Head',
            'file': 'ppo/policy.py',
            'lines': (40, 45),
            'keywords': ['method_head', 'Linear', '10']
        },
        {
            'description': '方法选择核心逻辑 / Method Selection Core Logic',
            'file': 'ppo/trainer.py',
            'lines': (105, 113),
            'keywords': ['node_idx', 'methods_for_node', 'num_methods', 'method_logits_masked', 'Categorical']
        },
        {
            'description': '动作验证函数 / Action Validation Function',
            'file': 'env/pipeline_env.py',
            'lines': (159, 196),
            'keywords': ['select_node', 'node_action', 'return False', 'current_step']
        },
        {
            'description': '动作掩码计算 / Action Mask Computation',
            'file': 'env/pipeline_env.py',
            'lines': (136, 157),
            'keywords': ['_compute_action_mask', 'current_step', 'mask', 'node_visited']
        },
    ]
    
    passed = 0
    failed = 0
    
    for ref in references:
        try:
            result = validate_code_reference(
                ref['description'],
                ref['file'],
                ref['lines'],
                ref['keywords']
            )
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ 验证出错 / Validation error: {str(e)}")
            failed += 1
    
    print("\n" + "="*80)
    print("验证总结 / Validation Summary")
    print("="*80)
    print(f"✅ 通过 / Passed: {passed}")
    print(f"❌ 失败 / Failed: {failed}")
    print(f"总计 / Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 所有代码引用验证通过！")
        print("🎉 All code references validated successfully!")
    else:
        print(f"\n⚠️  有 {failed} 个验证失败。")
        print(f"⚠️  {failed} validation(s) failed.")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
