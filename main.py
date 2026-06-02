import yaml
from box import Box  # Dynamic dict access (allows dot notation like config.engine_type)
from engines.bert_engine import run_bert_training
from engines.dpo_engine import run_dpo_training
from engines.grpo_engine import run_grpo_training
from device import device_select

def main(config_path: str):
    # 1. Load the YAML file
    with open(config_path, "r") as f:
        user_yaml = yaml.safe_load(f)
    
    # Convert dict to Box so you can use dot notation (config.engine_type)
    config = Box(user_yaml)
    
    print(f"🚀 Initializing project: {config.project_name}")

    device = device_select()

    # 2. Trigger the correct model pipeline using a simple conditional check
    if config.project_name.lower() == "bert":
        run_bert_training(device,config)
        
    elif config.project_name.lower() == "dpo":
        run_dpo_training(device,config)
        
    elif config.project_name.lower() == "grpo":
        run_grpo_training(device,config)
        
    else:
        raise ValueError(f"Unknown engine_type: {config.project_name}. Check your YAML file.")
if __name__ == "__main__":
    # Example: python src/main.py --config configs/bert_classification.yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to your configuration YAML file")
    args = parser.parse_args()
    
    main(args.config)