#!/usr/bin/env python3
"""
测试4K数据集获取和处理
Test 4K Dataset Fetching and Processing
"""
import os
import sys
import time
from pathlib import Path

# 设置环境变量为4K模式
os.environ['PIPELINE_TEST'] = '0'

# 添加项目根目录到路径
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_4k_data_fetch():
    """测试4K数据获取"""
    print("🎯 测试4K数据集获取")
    print("🎯 Testing 4K Dataset Fetching")
    print("=" * 60)
    
    try:
        # 导入配置
        from config import TEST_MODE, N_TOTAL, CACHE_FILE, API_KEY
        print(f"📊 配置检查:")
        print(f"  TEST_MODE: {TEST_MODE}")
        print(f"  N_TOTAL: {N_TOTAL}")
        print(f"  CACHE_FILE: {CACHE_FILE}")
        print(f"  API_KEY: {'已设置' if API_KEY else '未设置'}")
        print()
        
        if not API_KEY:
            print("❌ API_KEY 未设置，无法继续测试")
            return False
        
        # 导入数据方法
        from methods.data_methods import fetch_data
        
        # 检查缓存文件是否存在
        cache_path = Path(CACHE_FILE)
        if cache_path.exists():
            print(f"📁 发现现有缓存文件: {cache_path}")
            print(f"   文件大小: {cache_path.stat().st_size / (1024*1024):.1f} MB")
            
            # 尝试加载现有缓存
            print("🔄 测试加载现有缓存...")
            try:
                df = fetch_data(cache=True)
                print(f"✅ 缓存加载成功: {df.shape}")
                print(f"   列名: {list(df.columns)}")
                print(f"   前3行预览:")
                print(df[['material_id', 'formula_pretty', 'formation_energy_per_atom']].head(3))
                return True
            except Exception as e:
                print(f"⚠️ 缓存加载失败: {e}")
                print("🔄 删除损坏的缓存，重新获取...")
                cache_path.unlink()
        
        # 重新获取数据
        print("📥 从API重新获取4K数据...")
        start_time = time.time()
        
        df = fetch_data(cache=False)
        
        fetch_time = time.time() - start_time
        print(f"✅ 数据获取成功!")
        print(f"   数据形状: {df.shape}")
        print(f"   获取时间: {fetch_time/60:.1f} 分钟")
        print(f"   列名: {list(df.columns)}")
        print()
        
        # 数据质量检查
        print("🔍 数据质量检查:")
        print(f"   缺失值数量: {df.isnull().sum().sum()}")
        
        if 'formation_energy_per_atom' in df.columns:
            target_values = df['formation_energy_per_atom']
            print(f"   目标变量统计:")
            print(f"     均值: {target_values.mean():.3f}")
            print(f"     标准差: {target_values.std():.3f}")
            print(f"     范围: {target_values.min():.3f} ~ {target_values.max():.3f}")
        
        if 'structure' in df.columns:
            valid_structures = df['structure'].notna().sum()
            print(f"   有效结构数量: {valid_structures} / {len(df)}")
        
        print(f"\n📊 前5行数据预览:")
        display_cols = ['material_id', 'formula_pretty', 'formation_energy_per_atom']
        available_cols = [col for col in display_cols if col in df.columns]
        print(df[available_cols].head())
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_4k_featurization():
    """测试4K数据特征化"""
    print("\n" + "=" * 60)
    print("🔧 测试4K数据特征化")
    print("🔧 Testing 4K Data Featurization")
    print("=" * 60)
    
    try:
        from methods.data_methods import fetch_data, featurize_data
        
        # 获取原始数据
        print("📥 加载原始数据...")
        df = fetch_data(cache=True)
        print(f"✅ 原始数据加载成功: {df.shape}")
        
        # 特征化处理
        print("🔧 开始特征化处理...")
        start_time = time.time()
        
        df_feat = featurize_data(df)
        
        feat_time = time.time() - start_time
        print(f"✅ 特征化完成!")
        print(f"   特征化时间: {feat_time/60:.1f} 分钟")
        print(f"   特征化后形状: {df_feat.shape}")
        print(f"   新增特征数: {df_feat.shape[1] - df.shape[1]}")
        
        # 检查特征化结果
        print("🔍 特征化结果检查:")
        numeric_cols = df_feat.select_dtypes(include=['number']).columns
        print(f"   数值型列数: {len(numeric_cols)}")
        print(f"   总缺失值: {df_feat.isnull().sum().sum()}")
        
        # 显示一些特征列名
        exclude_cols = ['material_id', 'structure', 'elements', 'formula_pretty', 'composition', 'formation_energy_per_atom']
        feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
        print(f"   特征列数量: {len(feature_cols)}")
        if feature_cols:
            print(f"   前10个特征列: {feature_cols[:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_4k_pipeline():
    """测试完整4K数据流水线"""
    print("\n" + "=" * 60)
    print("🚀 测试完整4K数据流水线")
    print("🚀 Testing Complete 4K Data Pipeline")
    print("=" * 60)
    
    try:
        from methods.data_methods import fetch_and_featurize
        
        print("🔧 运行完整数据流水线...")
        start_time = time.time()
        
        result = fetch_and_featurize(cache=True)
        
        pipeline_time = time.time() - start_time
        print(f"✅ 流水线执行完成!")
        print(f"   执行时间: {pipeline_time/60:.1f} 分钟")
        
        # 检查输出结果
        print("🔍 流水线输出检查:")
        print(f"   输出键: {list(result.keys())}")
        
        if 'train_data' in result and result['train_data'] is not None:
            train_shape = result['train_data'].shape
            print(f"   训练数据形状: {train_shape}")
        
        if 'test_data' in result and result['test_data'] is not None:
            test_shape = result['test_data'].shape
            print(f"   测试数据形状: {test_shape}")
        
        if 'full_data' in result and result['full_data'] is not None:
            full_shape = result['full_data'].shape
            print(f"   完整数据形状: {full_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎯 4K数据集完整测试")
    print("🎯 Complete 4K Dataset Testing")
    print("=" * 60)
    
    test_results = []
    
    # 测试1: 数据获取
    print("📋 测试1: 数据获取")
    result1 = test_4k_data_fetch()
    test_results.append(("数据获取", result1))
    
    if result1:
        # 测试2: 特征化
        print("\n📋 测试2: 数据特征化")
        result2 = test_4k_featurization()
        test_results.append(("数据特征化", result2))
        
        if result2:
            # 测试3: 完整流水线
            print("\n📋 测试3: 完整流水线")
            result3 = test_4k_pipeline()
            test_results.append(("完整流水线", result3))
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("📊 Test Results Summary")
    print("=" * 60)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    all_passed = all(result for _, result in test_results)
    
    if all_passed:
        print("\n🎉 所有测试通过! 4K数据集准备就绪")
        print("🎉 All tests passed! 4K dataset is ready")
        print("💡 现在可以运行PPO训练:")
        print("   $env:PIPELINE_TEST=\"0\"; python train_ppo_4k.py")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")
        print("⚠️ Some tests failed, please check error messages")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
