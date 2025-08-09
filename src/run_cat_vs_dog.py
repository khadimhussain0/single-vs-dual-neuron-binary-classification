"""
Vision Transformer (ViT) experiment for cat vs dog binary classification.
This script runs only the cat vs dog experiment with reduced batch size.
"""

import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import gc

from train import run_experiment

# Define CIFAR-10 class names for reference
CIFAR10_CLASSES = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def run_experiment_with_memory_cleanup(epochs=15, batch_size=16):
    """
    Run ViT experiment for cat vs dog with memory cleanup and smaller batch size.
    
    Args:
        epochs: Number of epochs to train for
        batch_size: Batch size for training (reduced to save memory)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Force CUDA memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Test pair: cat vs dog
    class_a, class_b = 3, 5  # cat vs dog
    
    # ViT requires images of at least 224x224
    img_size = (224, 224, 3)
    
    print(f"\n{'='*80}\nRunning ViT experiment: {CIFAR10_CLASSES[class_a]} vs {CIFAR10_CLASSES[class_b]}\n{'='*80}\n")
    
    result = run_experiment(
        model_name='vit',
        dataset=(class_a, class_b),
        batch_size=batch_size,
        epochs=epochs,
        img_size=img_size
    )
    
    single_metrics = result['single_neuron']['test_metrics']
    dual_metrics = result['dual_neuron']['test_metrics']
    
    summary = {
        'Class_A': CIFAR10_CLASSES[class_a],
        'Class_B': CIFAR10_CLASSES[class_b],
        'Single_Accuracy': single_metrics['accuracy'],
        'Dual_Accuracy': dual_metrics['accuracy'],
        'Accuracy_Diff': dual_metrics['accuracy'] - single_metrics['accuracy'],
        'Single_F1': single_metrics['f1'],
        'Dual_F1': dual_metrics['f1'],
        'F1_Diff': dual_metrics['f1'] - single_metrics['f1'],
        'Single_AUC': result['single_neuron']['roc']['auc'],
        'Dual_AUC': result['dual_neuron']['roc']['auc'],
        'AUC_Diff': result['dual_neuron']['roc']['auc'] - result['single_neuron']['roc']['auc'],
        'Single_Convergence_Epochs': len(result['single_neuron']['history']['train_loss']),
        'Dual_Convergence_Epochs': len(result['dual_neuron']['history']['train_loss']),
        'Report_Path': result['comparison_path']
    }
    
    results_df = pd.DataFrame([summary])
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/vit_cat_vs_dog_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, 'vit_cat_vs_dog_experiment.csv'), index=False)
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ViT experiment for cat vs dog binary classification')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (use smaller value to save memory)')
    
    args = parser.parse_args()
    
    run_experiment_with_memory_cleanup(epochs=args.epochs, batch_size=args.batch_size)
