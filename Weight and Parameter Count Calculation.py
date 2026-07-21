#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Analyzer - Calculate model parameter count and FLOPs
Read the best_model.pth file in the same folder, calculate model parameter count and FLOPs, and save results to a txt file
"""

import torch
import torch.nn as nn
import os
import sys
from collections import OrderedDict
import numpy as np
from datetime import datetime

def load_model_state_dict(pth_path):
    """Load model state dictionary"""
    print(f"Loading model file: {pth_path}")

    try:
        # Check if file exists
        if not os.path.exists(pth_path):
            print(f"Error: File not found: {pth_path}")
            return None

        # Check file size
        file_size = os.path.getsize(pth_path)
        print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

        if file_size == 0:
            print("Error: File is empty")
            return None

        # Load model state dictionary
        state_dict = torch.load(pth_path, map_location='cpu')
        print(f"Successfully loaded model, contains {len(state_dict)} parameter groups")

        return state_dict

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def calculate_parameters(state_dict):
    """Calculate model parameter count"""
    print("\n=== Calculate Model Parameters ===")

    total_params = 0
    param_details = []

    for name, param in state_dict.items():
        param_count = param.numel()
        total_params += param_count

        param_details.append({
            'name': name,
            'shape': list(param.shape),
            'params': param_count,
            'dtype': str(param.dtype)
        })

    # Sort by parameter count
    param_details.sort(key=lambda x: x['params'], reverse=True)

    print(f"Total parameter count: {total_params:,}")
    print(f"Number of parameter groups: {len(param_details)}")

    # Print top 10 largest parameter groups
    print(f"\nTop 10 largest parameter groups:")
    for i, info in enumerate(param_details[:10]):
        print(f"{i+1:2d}. {info['name']:30s} {str(info['shape']):20s} {info['params']:10,}")
    
    return total_params, param_details

def calculate_flops(state_dict, input_shape=(1, 96, 7)):
    """Calculate model FLOPs"""
    print(f"\n=== Calculate Model FLOPs ===")
    print(f"Input shape: {input_shape}")

    total_flops = 0
    flops_details = []

    # Iterate over all parameters to estimate FLOPs
    for name, param in state_dict.items():
        param_shape = param.shape
        flops = 0
        layer_type = "Unknown"

        # Linear layer
        if len(param_shape) == 2 and 'weight' in name:
            layer_type = "Linear"
            flops = param_shape[0] * param_shape[1]

        # Convolutional layer (Conv1d)
        elif len(param_shape) == 3 and 'weight' in name:
            layer_type = "Conv1d"
            # Conv1d FLOPs = output feature map size * kernel parameter count
            output_size = input_shape[1]  # Sequence length
            kernel_params = param_shape[0] * param_shape[1] * param_shape[2]
            flops = output_size * kernel_params

        # Batch Normalization
        elif 'bn' in name.lower() or 'norm' in name.lower():
            layer_type = "BatchNorm"
            if len(param_shape) == 1:
                flops = 2 * param_shape[0]  # mean + var calculation

        # Attention mechanism related parameters
        elif 'attn' in name.lower() or 'attention' in name.lower():
            layer_type = "Attention"
            if 'weight' in name and len(param_shape) == 2:
                flops = param_shape[0] * param_shape[1]

        # Other weight parameters
        elif 'weight' in name:
            layer_type = "Other"
            flops = np.prod(param_shape)

        if flops > 0:
            total_flops += flops
            flops_details.append({
                'name': name,
                'type': layer_type,
                'shape': param_shape,
                'flops': flops
            })

    # Sort by FLOPs
    flops_details.sort(key=lambda x: x['flops'], reverse=True)

    print(f"Total FLOPs: {total_flops:,}")
    print(f"Total FLOPs (scientific notation): {total_flops:.2e}")

    # Print top 10 largest FLOPs contributors
    print(f"\nTop 10 largest FLOPs contributors:")
    for i, item in enumerate(flops_details[:10]):
        print(f"{i+1:2d}. {item['name']:30s} {item['type']:10s} {str(item['shape']):15s} {item['flops']:10,}")
    
    return total_flops, flops_details

def format_number(num):
    """Format number for display"""
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.0f}"

def save_results_to_file(pth_path, total_params, param_details, total_flops, flops_details, output_file):
    """Save results to txt file"""
    print(f"\n=== Save Results to File ===")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Model Analysis Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model file: {pth_path}\n")
            f.write(f"Total parameter count: {total_params:,}\n")
            f.write(f"Total parameter count (formatted): {format_number(total_params)}\n")
            f.write(f"Number of parameter groups: {len(param_details)}\n")
            f.write(f"Total FLOPs: {total_flops:,}\n")
            f.write(f"Total FLOPs (formatted): {format_number(total_flops)}\n\n")

            f.write("Parameter Details:\n")
            f.write("-" * 50 + "\n")
            for info in param_details:
                f.write(f"{info['name']:30s} {str(info['shape']):20s} {info['params']:10,}\n")

            f.write(f"\nFLOPs Details:\n")
            f.write("-" * 50 + "\n")
            for item in flops_details:
                f.write(f"{item['name']:30s} {item['type']:10s} {str(item['shape']):15s} {item['flops']:10,}\n")

        print(f"Results saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("Model Analyzer - Calculate Model Parameter Count and FLOPs")
    print("=" * 60)

    # Set file paths
    pth_path = "best_model.pth"
    output_file = "model_analysis_results.txt"

    # Check current directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Check if pth file exists
    if not os.path.exists(pth_path):
        print(f"Error: {pth_path} not found in current directory")
        print("Please ensure best_model.pth is in the current directory")
        return

    # Load model
    state_dict = load_model_state_dict(pth_path)
    if state_dict is None:
        return

    # Calculate parameter count
    total_params, param_details = calculate_parameters(state_dict)

    # Calculate FLOPs
    total_flops, flops_details = calculate_flops(state_dict)

    # Print summary
    print(f"\n=== Analysis Summary ===")
    print(f"Model file: {pth_path}")
    print(f"Total parameter count: {total_params:,} ({format_number(total_params)})")
    print(f"Total FLOPs: {total_flops:,} ({format_number(total_flops)})")
    print(f"Number of parameter groups: {len(param_details)}")

    # Save results to file
    success = save_results_to_file(pth_path, total_params, param_details,
                                 total_flops, flops_details, output_file)

    if success:
        print(f"\nAnalysis complete! Results saved to {output_file}")
    else:
        print("\nAnalysis complete, but an error occurred while saving the file")

if __name__ == "__main__":
    main()
