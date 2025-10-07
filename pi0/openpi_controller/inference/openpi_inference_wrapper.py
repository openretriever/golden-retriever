#!/usr/bin/env python3
"""
OpenPI Inference Wrapper

This script runs in the OpenPI environment and provides a simple interface
to load a policy and run inference. It communicates via JSON files to avoid
dependency conflicts with the main retriever environment.
"""

import json
import sys
import os
import argparse
import pathlib
import numpy as np

def load_policy(checkpoint_dir: str, config_name: str, default_prompt: str = None):
    """Load OpenPI policy in the OpenPI environment"""
    try:
        # Add OpenPI to path
        sys.path.insert(0, 'src')
        
        # Import OpenPI modules
        from openpi.policies import policy_config
        from openpi.training import config as _config
        
        print(f"🤖 Loading OpenPI model from {checkpoint_dir}")
        train_config = _config.get_config(config_name)
        
        policy = policy_config.create_trained_policy(
            train_config=train_config,
            checkpoint_dir=checkpoint_dir,
            default_prompt=default_prompt
        )
        
        print(f"✅ OpenPI policy loaded successfully")
        return policy
        
    except Exception as e:
        print(f"❌ Failed to load OpenPI policy: {e}")
        raise

def run_inference(policy, obs_dict: dict) -> dict:
    """Run inference and return results"""
    try:
        result = policy.infer(obs_dict)
        return {
            "success": True,
            "actions": result["actions"].tolist(),  # Convert numpy to list for JSON
            "state": result.get("state", None),
            "timing": result.get("policy_timing", {})
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="OpenPI Inference Wrapper")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory")
    parser.add_argument("--config-name", required=True, help="Config name")
    parser.add_argument("--input-file", required=True, help="Input observation JSON file")
    parser.add_argument("--output-file", required=True, help="Output actions JSON file")
    parser.add_argument("--default-prompt", help="Default prompt")
    
    args = parser.parse_args()
    
    try:
        # Load policy
        policy = load_policy(args.checkpoint_dir, args.config_name, args.default_prompt)
        
        # Read observation
        with open(args.input_file, 'r') as f:
            obs_data = json.load(f)
        
        # Convert lists back to numpy arrays for images
        for key in obs_data:
            if key.endswith('_image') or key.endswith('/image'):
                obs_data[key] = np.array(obs_data[key], dtype=np.uint8)
            elif key.endswith('state') or key.endswith('/state'):
                obs_data[key] = np.array(obs_data[key], dtype=np.float32)
        
        # Run inference
        result = run_inference(policy, obs_data)
        
        # Write result
        with open(args.output_file, 'w') as f:
            json.dump(result, f)
            
        print(f"✅ Inference completed, result written to {args.output_file}")
        
    except Exception as e:
        # Write error result
        error_result = {"success": False, "error": str(e)}
        with open(args.output_file, 'w') as f:
            json.dump(error_result, f)
        print(f"❌ Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
