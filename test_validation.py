#!/usr/bin/env python3
"""
快速验证测试脚本 - 只测试数据验证部分
"""
import sys
import os
sys.path.append('.')

import fix_numpy_float

def test_data_validation():
    """测试数据验证功能"""
    print("=" * 60)
    print("TESTING DATA VALIDATION")
    print("=" * 60)
    
    try:
        from data_streams.data_validator import DataStreamValidator
        print("✓ DataStreamValidator imported successfully")
        
        # 创建验证器
        validator = DataStreamValidator()
        print("✓ Validator created successfully")
        
        # 测试单个数据流验证
        print("\nTesting single data stream validation...")
        result = validator.validate_data_stream(
            generator_type='SINE1',
            imbalance_ratio=[5, 5],
            n_samples=500,  # 更小的样本数用于测试
            drift_start=200,
            drift_end=300,
            seed=42
        )
        
        print(f"✓ Single validation completed")
        print(f"  Total samples: {result['total_samples']}")
        print(f"  Overall quality: {result['overall_quality']}")
        
        # 测试可视化
        print("\nTesting visualization...")
        df = result['data_frame']
        expected_ratio = [5, 5]
        
        validator.visualizer.generate_validation_report(
            df=df,
            expected_ratio=expected_ratio,
            generator_type='SINE1',
            save_path='test_validation_output'
        )
        
        print("✓ Visualization test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_validation():
    """测试完整验证流程"""
    print("\n" + "=" * 60)
    print("TESTING FULL VALIDATION WORKFLOW")
    print("=" * 60)
    
    try:
        from data_streams.data_validator import DataStreamValidator
        
        validator = DataStreamValidator()
        
        # 只测试一个配置来快速验证
        print("Testing single configuration...")
        
        # 创建单个配置的测试
        configurations = [('SINE1', [5, 5], 'Test_Group_A1')]
        
        for generator_type, imbalance_ratio, group_name in configurations:
            print(f"\nValidating {group_name}: {generator_type} with ratio {imbalance_ratio}")
            
            result = validator.validate_data_stream(
                generator_type=generator_type,
                imbalance_ratio=imbalance_ratio,
                n_samples=500,  # 更小的样本数
                drift_start=200,
                drift_end=300
            )
            
            result['group_name'] = group_name
            
            # 测试可视化保存
            df = result['data_frame']
            save_dir = f"test_output/{group_name}"
            
            validator.visualizer.generate_validation_report(
                df=df,
                expected_ratio=imbalance_ratio,
                generator_type=generator_type,
                save_path=save_dir
            )
            
            print(f"✓ {group_name} completed successfully")
        
        print("\n✓ Full validation workflow test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Full validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("QUICK VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        ("Data Validation", test_data_validation),
        ("Full Validation Workflow", test_full_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n>>> Running {test_name} test...")
            result = test_func()
            results.append((test_name, result))
            
            if not result:
                print(f"\n⚠ Test '{test_name}' failed. Stopping here.")
                break
                
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
            break
    
    # 总结
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    if all_passed:
        print("\n🎉 验证测试通过！现在可以运行完整实验了：")
        print("  python main.py")
    else:
        print("\n💥 验证测试失败。请检查上面的错误信息。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)