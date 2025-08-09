"""
Calculate and visualize the number of parameters for each model architecture
with both single-neuron and dual-neuron output configurations.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from models import create_model

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    """Count all parameters in a model (trainable and non-trainable)"""
    return sum(p.numel() for p in model.parameters())

def calculate_parameter_counts():
    """Calculate parameter counts for all model architectures with both output configurations"""
    os.makedirs('paper/figures', exist_ok=True)
    architectures = ['small_cnn', 'resnet50', 'vit']
    
    param_counts = {
        'Architecture': [],
        'Single Neuron': [],
        'Dual Neuron': []
    }
    
    for arch in architectures:
        single_neuron_model = create_model(arch, output_neurons=1)
        dual_neuron_model = create_model(arch, output_neurons=2)
        
        single_params = count_parameters(single_neuron_model)
        dual_params = count_parameters(dual_neuron_model)
        
        single_all_params = count_all_parameters(single_neuron_model)
        dual_all_params = count_all_parameters(dual_neuron_model)
        
        param_counts['Architecture'].append(arch)
        param_counts['Single Neuron'].append(single_all_params) 
        param_counts['Dual Neuron'].append(dual_all_params) 
        
        print(f"{arch.upper()} Architecture:")
        print(f"  Single Neuron: {single_params:,} trainable parameters (Total: {single_all_params:,})")
        print(f"  Dual Neuron: {dual_params:,} trainable parameters (Total: {dual_all_params:,})")
        print(f"  Difference: {dual_params - single_params:,} parameters")
        print(f"  Percentage Increase: {((dual_params - single_params) / single_params) * 100:.2f}%")
        
        print()
    
    return param_counts

def visualize_parameter_counts(param_counts, output_dir='paper/figures'):
    """Create visualizations of model parameter counts"""
    os.makedirs(output_dir, exist_ok=True)
    
    architectures = param_counts['Architecture']
    single_neuron_params = param_counts['Single Neuron']
    dual_neuron_params = param_counts['Dual Neuron']
    
    differences = [d - s for s, d in zip(single_neuron_params, dual_neuron_params)]
    percentages = [((d - s) / s) * 100 for s, d in zip(single_neuron_params, dual_neuron_params)]
    
    display_names = [arch.replace('_', ' ').upper() for arch in architectures]
    
    plt.figure(figsize=(12, 8.5))
    x = np.arange(len(architectures))
    width = 0.35
    
    single_in_millions = [s / 1_000_000 for s in single_neuron_params]
    dual_in_millions = [d / 1_000_000 for d in dual_neuron_params]
    
    fig, ax = plt.subplots(figsize=(12, 8.5))
    rects1 = ax.bar(x - width/2, single_in_millions, width, label='Single Neuron')
    rects2 = ax.bar(x + width/2, dual_in_millions, width, label='Dual Neuron')
    
    ax.set_xlabel('Model Architecture', fontsize=12)
    ax.set_ylabel('Parameters (millions)', fontsize=12)
    ax.set_title('Total Number of Parameters by Model Architecture and Output Configuration', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    
    ax.legend(loc='upper left', fontsize=10)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}M',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_count_comparison.png", dpi=300)
    
    plt.figure(figsize=(12, 8.5))
    plt.bar(display_names, percentages, color='teal')
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Parameter Increase (%)', fontsize=12)
    plt.title('Percentage Increase in Parameters: Dual Neuron vs. Single Neuron', fontsize=14)
    
    for i, v in enumerate(percentages):
        plt.text(i, v + 0.1, f'{v:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_increase_percentage.png", dpi=300)
    
    print("Parameter count visualizations saved to:", output_dir)

if __name__ == "__main__":
    param_counts = calculate_parameter_counts()
    
    visualize_parameter_counts(param_counts)
