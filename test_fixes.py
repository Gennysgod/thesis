# update_config.py
import sys
sys.path.append('.')
import json
import os

def update_experiment_config():
    print("🔧 Updating experiment configuration...")
    
    # 更新实验配置文件
    config_files = [
        'experiments/experiment_config.py',
    ]
    
    new_model_path = 'models/pretrained_models/quadcdd_pretrained_v2.pth'
    
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 更新模型路径
            content = content.replace(
                'quadcdd_pretrained.pth',
                'quadcdd_pretrained_v2.pth'
            )
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Updated {config_file}")
    
    print("✅ Configuration update completed")

if __name__ == "__main__":
    update_experiment_config()