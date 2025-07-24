#!/usr/bin/env python3
"""
修复4K数据集的生成问题
Fix 4K Dataset Generation Issues
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from mp_api.client import MPRester

# Add project root to path
sys.path.append('..')
from config import API_KEY, TARGET_PROP

def fix_4k_data_generation():
    """
    重新生成安全的4K数据集，避免featurization错误
    Regenerate safe 4K dataset avoiding featurization errors
    """
    print("🔧 修复4K数据集生成")
    print("🔧 Fixing 4K Dataset Generation")
    print("=" * 60)
    
    # 配置参数
    N_TARGET = 4000
    BATCH_SIZE = 100
    cache_file = Path("data/processed/mp_data_cache_4k.pkl")
    
    print(f"📊 目标样本数: {N_TARGET}")
    print(f"📊 Target samples: {N_TARGET}")
    print(f"🔗 API Key: {'已设置' if API_KEY else '未设置'}")
    print(f"🔗 API Key: {'Set' if API_KEY else 'Not set'}")
    print(f"💾 缓存文件: {cache_file}")
    print(f"💾 Cache file: {cache_file}")
    print()
    
    if not API_KEY:
        print("❌ 错误: API_KEY 未设置")
        print("❌ Error: API_KEY not set")
        return False
    
    try:
        # 获取数据
        print("📥 获取材料数据...")
        print("📥 Fetching material data...")
        
        all_data = []
        with MPRester(API_KEY) as mpr:
            # 分批获取数据
            docs_iter = mpr.materials.summary.search(
                fields=["material_id", "structure", "elements", "formula_pretty", TARGET_PROP],
                chunk_size=BATCH_SIZE,
                num_chunks=(N_TARGET // BATCH_SIZE) + 2  # 多获取一些以防过滤后不够
            )
            
            for docs in tqdm(docs_iter, desc="获取MP数据"):
                if not docs:
                    continue
                    
                docs = docs if isinstance(docs, list) else [docs]
                
                for doc in docs:
                    # 安全地提取数据
                    try:
                        # 检查必需字段
                        if not hasattr(doc, 'material_id') or not hasattr(doc, 'structure'):
                            continue
                        if not hasattr(doc, TARGET_PROP) or getattr(doc, TARGET_PROP) is None:
                            continue
                            
                        material_id = getattr(doc, 'material_id', None)
                        structure = getattr(doc, 'structure', None)
                        elements = getattr(doc, 'elements', None)
                        formula_pretty = getattr(doc, 'formula_pretty', None)
                        formation_energy = getattr(doc, TARGET_PROP, None)
                        
                        # 验证关键字段
                        if not all([material_id, structure, formation_energy is not None]):
                            continue
                        
                        # 验证结构对象
                        if not hasattr(structure, 'composition'):
                            continue
                            
                        composition = structure.composition
                        if composition is None:
                            continue
                        
                        # 检查所有元素都有atomic_radius属性
                        skip_this = False
                        for element in composition.elements:
                            if not hasattr(element, 'atomic_radius') or element.atomic_radius is None:
                                skip_this = True
                                break
                        
                        if skip_this:
                            continue
                        
                        # 添加到列表
                        all_data.append({
                            'material_id': material_id,
                            'structure': structure,
                            'elements': elements,
                            'formula_pretty': formula_pretty,
                            TARGET_PROP: formation_energy,
                            'composition': composition
                        })
                        
                        # 检查是否已经获得足够的数据
                        if len(all_data) >= N_TARGET:
                            break
                            
                    except Exception as e:
                        # 跳过有问题的条目
                        continue
                
                if len(all_data) >= N_TARGET:
                    break
        
        print(f"✅ 成功获取 {len(all_data)} 个有效样本")
        print(f"✅ Successfully fetched {len(all_data)} valid samples")
        
        if len(all_data) < N_TARGET:
            print(f"⚠️ 获取的样本数 ({len(all_data)}) 少于目标 ({N_TARGET})")
            print(f"⚠️ Fetched samples ({len(all_data)}) less than target ({N_TARGET})")
        
        # 截取到目标数量
        all_data = all_data[:N_TARGET]
        
        # 转换为DataFrame
        print("🔄 转换为DataFrame...")
        print("🔄 Converting to DataFrame...")
        df = pd.DataFrame(all_data)
        
        # 基本数据验证
        print("🔍 数据验证...")
        print("🔍 Data validation...")
        print(f"  数据形状: {df.shape}")
        print(f"  Data shape: {df.shape}")
        print(f"  列名: {list(df.columns)}")
        print(f"  Column names: {list(df.columns)}")
        print(f"  缺失值数量: {df.isnull().sum().sum()}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
        
        # 目标变量统计
        target_values = df[TARGET_PROP].values
        print(f"  目标变量统计:")
        print(f"  Target variable statistics:")
        print(f"    均值: {np.mean(target_values):.3f}")
        print(f"    Mean: {np.mean(target_values):.3f}")
        print(f"    标准差: {np.std(target_values):.3f}")
        print(f"    Std: {np.std(target_values):.3f}")
        print(f"    范围: {np.min(target_values):.3f} ~ {np.max(target_values):.3f}")
        print(f"    Range: {np.min(target_values):.3f} ~ {np.max(target_values):.3f}")
        
        # 保存缓存文件
        print(f"💾 保存缓存文件...")
        print(f"💾 Saving cache file...")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        # 验证文件
        file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 缓存文件已保存: {cache_file}")
        print(f"✅ Cache file saved: {cache_file}")
        print(f"  文件大小: {file_size:.1f} MB")
        print(f"  File size: {file_size:.1f} MB")
        
        # 测试加载
        print("🧪 测试文件加载...")
        print("🧪 Testing file loading...")
        with open(cache_file, 'rb') as f:
            test_df = pickle.load(f)
        print(f"✅ 加载成功，形状: {test_df.shape}")
        print(f"✅ Loading successful, shape: {test_df.shape}")
        
        print()
        print("🎉 4K数据集修复完成!")
        print("🎉 4K Dataset Fix Complete!")
        print(f"📁 缓存文件位置: {cache_file}")
        print(f"📁 Cache file location: {cache_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 4K数据集修复失败: {e}")
        print(f"❌ 4K Dataset fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_4k_data_generation()
    if success:
        print("\n✅ 现在可以尝试使用4K数据集进行PPO训练")
        print("✅ Now you can try PPO training with 4K dataset")
    else:
        print("\n❌ 修复失败，请检查错误信息")
        print("❌ Fix failed, please check error messages")
