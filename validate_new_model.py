# validate_new_model.py
import sys
sys.path.append('.')
import fix_numpy_float

from models.quadcdd_network import create_quadcdd_model
from models.classifier import generate_accuracy_sequence
from data_streams.synthetic_generator import DataStreamFactory
import numpy as np
import os

def validate_new_model():
    print("🔍 Validating new QuadCDD model...")
    
    # 检查新模型是否存在
    model_path = 'models/pretrained_models/quadcdd_pretrained_v2.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run retrain_quadcdd.py first")
        return False
    
    # 加载模型
    trainer = create_quadcdd_model()
    trainer.load_checkpoint(model_path)
    print(f"✅ Model loaded from: {model_path}")
    
    # 生成测试数据
    generator = DataStreamFactory.create_generator(
        'SINE1',
        n_samples=500,
        drift_start=200,
        drift_end=300,
        drift_severity=0.5,
        drift_type='gradual',
        imbalance_ratio=[6, 4],
        random_state=42
    )
    
    data_stream = generator.generate_stream()
    feature_cols = [col for col in data_stream.columns if col.startswith('x')]
    X = data_stream[feature_cols].values
    y = data_stream['y'].values
    
    # 生成准确率序列
    accuracy_seq = generate_accuracy_sequence(X, y)
    print(f"✅ Generated accuracy sequence: {len(accuracy_seq)} points")
    
    # 测试模型预测
    result = trainer.predict(accuracy_seq)
    print(f"✅ Model prediction completed")
    
    # 验证预测结果
    print(f"Predicted quadruple:")
    print(f"  Ds (drift start): {result['Ds']:.3f}")
    print(f"  De (drift end): {result['De']:.3f}")
    print(f"  Dv (drift severity): {result['Dv']:.3f}")
    print(f"  Dt (drift type): {result['Dt']:.3f}")
    
    # 验证预测合理性
    expected_ds = 200 / 500  # 0.4
    expected_de = 300 / 500  # 0.6
    
    ds_error = abs(result['Ds'] - expected_ds)
    de_error = abs(result['De'] - expected_de)
    
    print(f"\nPrediction accuracy:")
    print(f"  Ds error: {ds_error:.3f} (expected ~0.4)")
    print(f"  De error: {de_error:.3f} (expected ~0.6)")
    
    if ds_error < 0.2 and de_error < 0.2:
        print("✅ Model prediction accuracy: GOOD")
        return True
    else:
        print("⚠️  Model prediction accuracy: NEEDS IMPROVEMENT")
        return False

if __name__ == "__main__":
    success = validate_new_model()
    if success:
        print("\n🎉 New model validation passed!")
    else:
        print("\n⚠️  Model may need further training or parameter adjustment")