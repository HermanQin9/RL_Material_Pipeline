#!/usr/bin/env python3
"""
4K数据集生成和验证脚本
4K Dataset Generation and Validation Script
"""
import os
import sys
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# 强制设置为4K模式
os.environ['PIPELINE_TEST'] = '0'

sys.path.append('..')
from config import N_TOTAL, BATCH_SIZE, CACHE_FILE, API_KEY, TARGET_PROP
from mp_api.client import MPRester

def get_value(d, key, default=None):
    """安全获取属性值"""
    try:
        if hasattr(d, key):
            return getattr(d, key, default)
        elif isinstance(d, dict):
            return d.get(key, default)
        else:
            return default
    except:
        return default

def generate_4k_data_safe():
    """
    安全地生成4K数据集，包含错误处理
    Safely generate 4K dataset with error handling
    """
    print("🚀 开始生成4K材料数据集")
    print("🚀 Starting 4K Material Dataset Generation")
    print("=" * 70)
    print(f"📊 目标配置:")
    print(f"  - 目标样本数: {N_TOTAL:,}")
    print(f"  - 批处理大小: {BATCH_SIZE}")
    print(f"  - 缓存文件: {CACHE_FILE}")
    print(f"  - API密钥: {'已设置' if API_KEY else '未设置'}")
    print("=" * 70)
    
    if not API_KEY:
        print("❌ 错误: 未找到Materials Project API密钥")
        return False
    
    dfs = []
    fetched = 0
    error_count = 0
    start_time = time.time()
    
    try:
        print(f"🔗 连接Materials Project API...")
        with MPRester(API_KEY) as mpr:
            print(f"✅ API连接成功")
            
            # 计算需要的批次数
            num_chunks = (N_TOTAL // BATCH_SIZE) + 2  # 额外获取一些以防数据不足
            print(f"📦 预计批次数: {num_chunks}")
            
            print(f"📥 开始获取数据...")
            docs_iter = mpr.materials.summary.search(
                fields=["material_id", "structure", "elements", "formula_pretty", TARGET_PROP],
                chunk_size=BATCH_SIZE,
                num_chunks=num_chunks,
            )
            
            batch_count = 0
            for docs in tqdm(docs_iter, desc="获取MP数据", total=num_chunks):
                batch_count += 1
                try:
                    # 确保docs是列表
                    if not isinstance(docs, list):
                        docs = [docs]
                    
                    # 过滤有效数据
                    valid_docs = []
                    for d in docs:
                        try:
                            # 检查必需字段
                            target_value = get_value(d, TARGET_PROP)
                            structure = get_value(d, "structure")
                            
                            if target_value is not None and structure is not None:
                                valid_docs.append(d)
                        except Exception as e:
                            error_count += 1
                            continue
                    
                    if not valid_docs:
                        print(f"⚠️ 批次 {batch_count} 无有效数据")
                        continue
                    
                    # 创建DataFrame
                    batch_data = []
                    for d in valid_docs:
                        try:
                            # 安全提取composition
                            structure = get_value(d, "structure")
                            composition = None
                            if structure and hasattr(structure, 'composition'):
                                composition = structure.composition
                            
                            row_data = {
                                "material_id": get_value(d, "material_id"),
                                "structure": structure,
                                "elements": get_value(d, "elements"),
                                "formula_pretty": get_value(d, "formula_pretty"),
                                TARGET_PROP: get_value(d, TARGET_PROP),
                                "composition": composition
                            }
                            batch_data.append(row_data)
                            
                        except Exception as e:
                            error_count += 1
                            continue
                    
                    if batch_data:
                        df_batch = pd.DataFrame(batch_data)
                        # 删除structure为None的行
                        df_batch = df_batch.dropna(subset=["structure"]).reset_index(drop=True)
                        
                        if len(df_batch) > 0:
                            dfs.append(df_batch)
                            fetched += len(df_batch)
                            
                            if batch_count % 5 == 0:
                                print(f"📊 已获取 {fetched:,} / {N_TOTAL:,} 样本 (批次 {batch_count})")
                    
                    # 检查是否已达到目标
                    if fetched >= N_TOTAL:
                        print(f"✅ 已达到目标样本数: {fetched:,}")
                        break
                        
                except Exception as e:
                    print(f"❌ 批次 {batch_count} 处理错误: {str(e)[:100]}")
                    error_count += 1
                    continue
            
    except Exception as e:
        print(f"❌ API连接或数据获取失败: {e}")
        return False
    
    # 合并数据
    if not dfs:
        print("❌ 未获取到任何有效数据")
        return False
    
    print(f"\n📊 合并数据...")
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 截取到目标大小
    if len(full_df) > N_TOTAL:
        full_df = full_df.iloc[:N_TOTAL].reset_index(drop=True)
    
    actual_size = len(full_df)
    elapsed_time = time.time() - start_time
    
    print(f"✅ 数据获取完成!")
    print(f"  实际样本数: {actual_size:,}")
    print(f"  目标样本数: {N_TOTAL:,}")
    print(f"  完成率: {actual_size/N_TOTAL*100:.1f}%")
    print(f"  错误数量: {error_count}")
    print(f"  总耗时: {elapsed_time/60:.1f} 分钟")
    
    # 验证数据质量
    print(f"\n🔍 数据质量检查...")
    print(f"  列数: {len(full_df.columns)}")
    print(f"  列名: {list(full_df.columns)}")
    print(f"  缺失值: {full_df.isnull().sum().sum()}")
    
    # 检查目标变量
    target_stats = full_df[TARGET_PROP].describe()
    print(f"  目标变量 ({TARGET_PROP}):")
    print(f"    均值: {target_stats['mean']:.3f}")
    print(f"    标准差: {target_stats['std']:.3f}")
    print(f"    范围: {target_stats['min']:.3f} ~ {target_stats['max']:.3f}")
    
    # 保存缓存
    print(f"\n💾 保存缓存文件...")
    try:
        cache_path = Path(CACHE_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为简单的DataFrame格式（避免复杂的dict结构）
        with open(cache_path, "wb") as f:
            pickle.dump(full_df, f)
        
        print(f"✅ 缓存已保存: {cache_path}")
        print(f"  文件大小: {cache_path.stat().st_size / (1024*1024):.1f} MB")
        
        # 验证缓存文件
        print(f"🔍 验证缓存文件...")
        with open(cache_path, "rb") as f:
            test_load = pickle.load(f)
        
        if isinstance(test_load, pd.DataFrame) and len(test_load) == actual_size:
            print(f"✅ 缓存文件验证成功")
            return True
        else:
            print(f"❌ 缓存文件验证失败")
            return False
            
    except Exception as e:
        print(f"❌ 缓存保存失败: {e}")
        return False

def test_4k_data_loading():
    """
    测试4K数据加载
    Test 4K data loading
    """
    print(f"\n🧪 测试4K数据加载...")
    
    try:
        # 测试通过data_methods加载
        from methods.data_methods import fetch_data
        
        start_time = time.time()
        df = fetch_data(cache=True)
        load_time = time.time() - start_time
        
        print(f"✅ 数据加载成功!")
        print(f"  加载时间: {load_time:.1f} 秒")
        print(f"  数据形状: {df.shape}")
        print(f"  内存使用: {df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
        
        # 检查数据内容
        print(f"  前5行预览:")
        print(df.head()[['material_id', 'formula_pretty', TARGET_PROP]])
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 4K数据集生成和验证")
    print("🎯 4K Dataset Generation and Validation")
    
    try:
        # 生成4K数据
        success = generate_4k_data_safe()
        
        if success:
            # 测试数据加载
            test_success = test_4k_data_loading()
            
            if test_success:
                print(f"\n🎉 4K数据集生成和验证完成!")
                print(f"🎉 4K Dataset Generation and Validation Complete!")
                print(f"📁 现在可以使用4K数据集进行PPO训练")
            else:
                print(f"\n⚠️ 数据生成成功但加载测试失败")
        else:
            print(f"\n❌ 4K数据集生成失败")
            
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
