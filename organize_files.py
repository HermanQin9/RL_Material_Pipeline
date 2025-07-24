#!/usr/bin/env python3
"""
项目文件整理脚本
Project File Organization Script
"""
import os
import shutil
from pathlib import Path

def organize_project_files():
    """整理项目文件结构"""
    print("🗂️ 开始整理项目文件结构")
    print("🗂️ Starting Project File Organization")
    print("=" * 60)
    
    # 定义文件分类
    file_movements = {
        # 测试和验证脚本 -> tests/
        "test_4k_data.py": "tests/test_4k_data.py",
        "test_ppo_simple.py": "tests/test_ppo_simple.py",
        "validate_ppo_training.py": "tests/validate_ppo_training.py",
        "extended_ppo_validation.py": "tests/extended_ppo_validation.py",
        "simplified_ppo_validation.py": "tests/simplified_ppo_validation.py",
        
        # PPO训练脚本 -> scripts/
        "train_ppo_4k.py": "scripts/train_ppo_4k.py",
        "train_ppo_safe.py": "scripts/train_ppo_safe.py",
        "main.py": "scripts/main.py",
        "run.py": "scripts/run.py",
        
        # 数据生成和修复脚本 -> scripts/
        "generate_4k_data.py": "scripts/generate_4k_data.py",
        "fix_4k_data.py": "scripts/fix_4k_data.py",
        
        # 分析脚本 -> scripts/analysis/
        "analyze_ppo_results.py": "scripts/analysis/analyze_ppo_results.py",
        "reward_analysis.py": "scripts/analysis/reward_analysis.py",
        
        # 检查和调试脚本 -> scripts/debug/
        "check_training_mode.py": "scripts/debug/check_training_mode.py",
        
        # 文档 -> docs/
        "PPO_VALIDATION_REPORT.md": "docs/PPO_VALIDATION_REPORT.md",
    }
    
    # 创建必要的目录
    directories_to_create = [
        "scripts/analysis",
        "scripts/debug",
        "tests",
        "docs"
    ]
    
    for directory in directories_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 确保目录存在: {directory}")
    
    # 移动文件
    moved_files = []
    skipped_files = []
    
    for source, destination in file_movements.items():
        if Path(source).exists():
            try:
                # 如果目标文件已存在，先备份
                if Path(destination).exists():
                    backup_name = f"{destination}.backup"
                    shutil.move(destination, backup_name)
                    print(f"📦 备份现有文件: {destination} -> {backup_name}")
                
                # 移动文件
                shutil.move(source, destination)
                moved_files.append((source, destination))
                print(f"📂 移动: {source} -> {destination}")
                
            except Exception as e:
                print(f"❌ 移动失败 {source}: {e}")
                skipped_files.append((source, str(e)))
        else:
            print(f"⚠️ 文件不存在: {source}")
    
    # 报告结果
    print(f"\n📊 整理结果:")
    print(f"  成功移动: {len(moved_files)} 个文件")
    print(f"  跳过文件: {len(skipped_files)} 个文件")
    
    if moved_files:
        print(f"\n✅ 成功移动的文件:")
        for source, dest in moved_files:
            print(f"    {source} -> {dest}")
    
    if skipped_files:
        print(f"\n⚠️ 跳过的文件:")
        for source, error in skipped_files:
            print(f"    {source}: {error}")
    
    return moved_files

def update_import_statements():
    """更新import语句"""
    print(f"\n🔧 更新import语句...")
    
    # 需要更新import的文件和对应的更新规则
    import_updates = {
        # scripts/train_ppo_4k.py 中可能需要更新的import
        "scripts/train_ppo_4k.py": {
            "from test_4k_data import": "from tests.test_4k_data import",
            "import test_4k_data": "import tests.test_4k_data",
        },
        # scripts/analysis/ 中的文件可能需要更新
        "scripts/analysis/analyze_ppo_results.py": {
            "sys.path.append('.')": "sys.path.append('../..')",
        },
        "scripts/analysis/reward_analysis.py": {
            "sys.path.append('.')": "sys.path.append('../..')",
        },
        # tests/ 中的文件可能需要更新
        "tests/test_4k_data.py": {
            "sys.path.append('.')": "sys.path.append('..')",
        }
    }
    
    for file_path, replacements in import_updates.items():
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                modified = False
                for old_import, new_import in replacements.items():
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        modified = True
                        print(f"  📝 更新 {file_path}: {old_import} -> {new_import}")
                
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ 保存更新: {file_path}")
                
            except Exception as e:
                print(f"  ❌ 更新失败 {file_path}: {e}")

def clean_up_empty_dirs():
    """清理空目录"""
    print(f"\n🧹 清理空目录...")
    
    # 检查并删除__pycache__目录
    pycache_dirs = list(Path('.').rglob('__pycache__'))
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"  🗑️ 删除缓存目录: {pycache_dir}")
        except Exception as e:
            print(f"  ⚠️ 无法删除 {pycache_dir}: {e}")

def create_readme_in_subdirs():
    """在子目录中创建README文件"""
    print(f"\n📝 创建子目录README文件...")
    
    readme_contents = {
        "scripts/analysis/README.md": """# Analysis Scripts

This directory contains analysis and visualization scripts for PPO training results.

## Files

- `analyze_ppo_results.py`: Main PPO training results analysis and visualization
- `reward_analysis.py`: Detailed reward function analysis and improvement suggestions

## Usage

Run from the project root directory:

```bash
python scripts/analysis/analyze_ppo_results.py
python scripts/analysis/reward_analysis.py
```
""",
        
        "scripts/debug/README.md": """# Debug Scripts

This directory contains debugging and diagnostic scripts.

## Files

- `check_training_mode.py`: Check current training configuration and mode

## Usage

Run from the project root directory:

```bash
python scripts/debug/check_training_mode.py
```
""",
        
        "tests/README.md": """# Test Scripts

This directory contains test scripts for validating different components of the system.

## Files

- `test_4k_data.py`: Comprehensive 4K dataset testing
- `test_ppo_simple.py`: Simple PPO testing
- `validate_ppo_training.py`: PPO training validation
- `extended_ppo_validation.py`: Extended validation suite
- `simplified_ppo_validation.py`: Simplified validation

## Usage

Run tests from the project root directory:

```bash
python tests/test_4k_data.py
python tests/validate_ppo_training.py
```
"""
    }
    
    for readme_path, content in readme_contents.items():
        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  📄 创建: {readme_path}")
        except Exception as e:
            print(f"  ❌ 创建失败 {readme_path}: {e}")

def main():
    """主函数"""
    print("🎯 项目文件整理")
    print("🎯 Project File Organization")
    print("=" * 60)
    
    # 执行整理步骤
    moved_files = organize_project_files()
    update_import_statements()
    clean_up_empty_dirs()
    create_readme_in_subdirs()
    
    print(f"\n" + "=" * 60)
    print("🎉 项目整理完成!")
    print("🎉 Project Organization Complete!")
    print("=" * 60)
    
    print(f"📋 整理总结:")
    print(f"  - 移动了 {len(moved_files)} 个文件到适当位置")
    print(f"  - 更新了相关的import语句")
    print(f"  - 清理了缓存目录")
    print(f"  - 创建了子目录说明文档")
    
    print(f"\n💡 下一步:")
    print(f"  1. 检查移动后的文件是否正常工作")
    print(f"  2. 更新主README.md文件")
    print(f"  3. 测试所有脚本的执行")
    
    return True

if __name__ == "__main__":
    main()
