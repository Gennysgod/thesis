# retrain_quadcdd.py
import sys
sys.path.append('.')
import fix_numpy_float

from experiments.quadcdd_trainer import QuadCDDPretrainer, create_pretraining_config
import json
import os

def retrain_quadcdd_model():
    print("🚀 Starting QuadCDD retraining with improved components...")
    
    # 创建配置
    config = create_pretraining_config()
    
    # 可以调整配置以快速测试
    config.update({
        'total_streams': 2000,  # 减少用于快速测试，生产环境建议4000
        'epochs': 30,           # 减少用于快速测试，生产环境建议50
        'batch_size': 32,
    })
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # 创建预训练器
    pretrainer = QuadCDDPretrainer(config)
    
    # 模型保存路径
    model_path = 'models/pretrained_models/quadcdd_pretrained_v2.pth'
    
    # 运行完整的预训练流程
    print("\n📊 Starting pretraining pipeline...")
    summary = pretrainer.run_complete_pretraining(model_path)
    
    print("\n✅ Pretraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training data size: {summary['training_data_size']}")
    print(f"Validation loss: {summary['evaluation_results']['validation_loss']:.4f}")
    print(f"Overall MAE: {summary['evaluation_results']['overall_mae']:.4f}")
    
    # 保存详细摘要
    summary_path = model_path.replace('.pth', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return model_path

if __name__ == "__main__":
    model_path = retrain_quadcdd_model()
    print(f"\n🎉 New QuadCDD model ready at: {model_path}")