#!/usr/bin/env python3
"""
Preprocessing script: Cache posterior samples for all models.

This runs MCMC once on all models and saves the posterior samples to disk,
so the evaluation script can load them quickly without re-running inference.

Usage:
    python cache_model_posteriors.py \
        --results experiment_results_anant/results_hurricane_eal_counties_qwen25_coder_n250.json \
        --task tasks/hurricane_eal_counties.json \
        --output cached_posteriors/hurricane_qwen.pkl \
        --target median_loss
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from llb.mcmc_log import run_inference


def cache_posteriors(results_file, task_file, output_file, target):
    """Run MCMC once on all models and cache the posterior samples."""
    
    # Load data and results
    with open(task_file) as f:
        task = json.load(f)
    data_train = task["data"]
    
    with open(results_file) as f:
        result = json.load(f)
    
    model_codes = result["full_result"]["model_codes"]
    n_valid = result["metrics"]["n_models_valid_final"]
    n_valid = min(n_valid, len(model_codes))
    
    print(f"Caching posterior samples for {n_valid} models...")
    print(f"This will take ~{n_valid * 0.5:.0f} minutes at 30 sec/model")
    print()
    
    cached_data = {
        "model_posteriors": [],
        "target": target,
        "data_train": data_train,
        "n_models": n_valid
    }
    
    for i in range(n_valid):
        code = model_codes[i]
        
        print(f"[{i+1}/{n_valid}] Running inference on model {i}...", end=" ", flush=True)
        
        try:
            infer_out = run_inference(
                code=code,
                data=data_train,
                targets=[target],
                num_warmup=500,
                num_samples=1000,
                rng_seed=42 + i
            )
            
            # Extract only what we need
            model_data = {
                "model_index": i,
                "posterior_samples": {k: v.tolist() for k, v in infer_out["samples"].items()},
                "code": code
            }
            
            cached_data["model_posteriors"].append(model_data)
            print("✓")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    # Save to disk
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(cached_data, f)
    
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"\n✓ Cached {len(cached_data['model_posteriors'])} models to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Cache model posterior samples')
    parser.add_argument('--results', required=True, help='Path to results JSON')
    parser.add_argument('--task', required=True, help='Path to task JSON')
    parser.add_argument('--output', required=True, help='Output pickle file')
    parser.add_argument('--target', default='median_loss', help='Target variable')
    
    args = parser.parse_args()
    
    cache_posteriors(args.results, args.task, args.output, args.target)


if __name__ == '__main__':
    main()