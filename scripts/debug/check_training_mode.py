#!/usr/bin/env python3
"""
检查训练模式配置
Check training mode configuration
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import TEST_MODE, N_TOTAL, CACHE_FILE

def check_training_mode():
    print('=== 训练模式配置验证 / Training Mode Configuration ===')
    print(f'PIPELINE_TEST环境变量: {os.getenv("PIPELINE_TEST", "未设置")}')
    print(f'TEST_MODE: {TEST_MODE}')
    print(f'数据集大小 / Dataset Size: {N_TOTAL:,} 样本')
    print(f'缓存文件 / Cache File: {CACHE_FILE}')
    
    if not TEST_MODE:
        print('\n🚀 成功切换到训练模式! / Successfully switched to training mode!')
        print(f'  - 大数据集: {N_TOTAL:,} 材料样本 / Large dataset: {N_TOTAL:,} material samples')
        return True
    else:
        print('\n⚠️ 警告：仍在测试模式 / Warning: Still in test mode')
        return False

if __name__ == "__main__":
    check_training_mode()
