"""
Vision Transformer (ViT) experiments for binary classification research.
This script runs experiments using ViT as the backbone architecture
to compare single-neuron vs. dual-neuron output layers for binary classification.
"""

import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

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

def run_all_experiments(epochs=15, batch_size=32, test_pairs=None):
    """
    Run ViT experiments for the research paper.
    
    Args:
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        test_pairs: List of class pairs to test. If None, uses default pairs.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    if test_pairs is None:
        test_pairs = [
            (0, 1),  # airplane vs. automobile
            (3, 5),  # cat vs. dog
        ]
    
    # ViT requires images of at least 224x224
    img_size = (224, 224, 3)
    
    all_results = []
    for class_a, class_b in test_pairs:
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
        
        all_results.append(summary)
    
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/vit_summary_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, 'vit_experiments_summary.csv'), index=False)
    
    create_consolidated_report(results_df, results_dir)
    
    return results_df

def create_consolidated_report(results_df, output_dir):
    """
    Create a consolidated markdown report of all experiments.
    
    Args:
        results_df: DataFrame with all experiment results
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, 'vit_consolidated_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Vision Transformer (ViT) Binary Classification Research: Single vs. Dual Neuron Output\n\n")
        f.write("## ViT Experiments Summary\n\n")
        
        # Overall statistics
        single_wins = sum(results_df['Accuracy_Diff'] < 0)
        dual_wins = sum(results_df['Accuracy_Diff'] > 0)
        ties = sum(results_df['Accuracy_Diff'] == 0)
        
        f.write("### Overall Performance\n\n")
        f.write(f"* Total experiments: {len(results_df)}\n")
        f.write(f"* Single neuron wins: {single_wins} ({single_wins/len(results_df)*100:.1f}%)\n")
        f.write(f"* Dual neuron wins: {dual_wins} ({dual_wins/len(results_df)*100:.1f}%)\n")
        f.write(f"* Ties: {ties} ({ties/len(results_df)*100:.1f}%)\n\n")
        
        avg_acc_diff = results_df['Accuracy_Diff'].mean()
        avg_f1_diff = results_df['F1_Diff'].mean()
        avg_auc_diff = results_df['AUC_Diff'].mean()
        
        f.write("### Average Metric Differences (Dual - Single)\n\n")
        f.write(f"* Accuracy: {avg_acc_diff:.4f}\n")
        f.write(f"* F1 Score: {avg_f1_diff:.4f}\n")
        f.write(f"* AUC: {avg_auc_diff:.4f}\n\n")
        
        f.write("### Detailed Results\n\n")
        f.write("| Class Pair | Single Accuracy | Dual Accuracy | Diff | Single F1 | Dual F1 | Diff | Single AUC | Dual AUC | Diff | Winner |\n")
        f.write("|------------|----------------|---------------|------|-----------|---------|------|------------|----------|------|--------|\n")
        
        for _, row in results_df.iterrows():
            class_pair = f"{row['Class_A']} vs {row['Class_B']}"
            
            metrics_diff = [row['Accuracy_Diff'], row['F1_Diff'], row['AUC_Diff']]
            positive_diffs = sum(diff > 0 for diff in metrics_diff)
            negative_diffs = sum(diff < 0 for diff in metrics_diff)
            
            if positive_diffs > negative_diffs:
                winner = "Dual"
            elif negative_diffs > positive_diffs:
                winner = "Single"
            else:
                winner = "Tie"
            
            f.write(f"| {class_pair} | {row['Single_Accuracy']:.4f} | {row['Dual_Accuracy']:.4f} | {row['Accuracy_Diff']:.4f} | ")
            f.write(f"{row['Single_F1']:.4f} | {row['Dual_F1']:.4f} | {row['F1_Diff']:.4f} | ")
            f.write(f"{row['Single_AUC']:.4f} | {row['Dual_AUC']:.4f} | {row['AUC_Diff']:.4f} | {winner} |\n")
        
        f.write("\n### Convergence Analysis\n\n")
        f.write("| Class Pair | Single Epochs | Dual Epochs | Faster Convergence |\n")
        f.write("|------------|---------------|-------------|--------------------|\n")
        
        for _, row in results_df.iterrows():
            class_pair = f"{row['Class_A']} vs {row['Class_B']}"
            single_epochs = row['Single_Convergence_Epochs']
            dual_epochs = row['Dual_Convergence_Epochs']
            
            if single_epochs < dual_epochs:
                faster = "Single"
            elif dual_epochs < single_epochs:
                faster = "Dual"
            else:
                faster = "Tie"
                
            f.write(f"| {class_pair} | {single_epochs} | {dual_epochs} | {faster} |\n")
        
        f.write("\n## ViT-Specific Findings\n\n")
        
        if avg_acc_diff > 0 and avg_f1_diff > 0 and avg_auc_diff > 0:
            better_approach = "dual-neuron"
        elif avg_acc_diff < 0 and avg_f1_diff < 0 and avg_auc_diff < 0:
            better_approach = "single-neuron"
        else:
            better_approach = "mixed, with no clear winner"
            
        f.write(f"1. For ViT, the **{better_approach}** approach performed better overall.\n")
        
        single_faster = sum(results_df['Single_Convergence_Epochs'] < results_df['Dual_Convergence_Epochs'])
        dual_faster = sum(results_df['Single_Convergence_Epochs'] > results_df['Dual_Convergence_Epochs'])
        
        if single_faster > dual_faster:
            convergence_pattern = "single-neuron models generally converged faster"
        elif dual_faster > single_faster:
            convergence_pattern = "dual-neuron models generally converged faster"
        else:
            convergence_pattern = "both approaches showed similar convergence patterns"
            
        f.write(f"2. In terms of training dynamics, {convergence_pattern}.\n")
 
        print(f"Consolidated report created: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ViT experiments for binary classification research')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--pair', type=str, help='Specific class pair to test (format: "0,1")')
    
    args = parser.parse_args()
    
    if args.pair:
        class_a, class_b = map(int, args.pair.split(','))
        test_pairs = [(class_a, class_b)]
        run_all_experiments(epochs=args.epochs, batch_size=args.batch_size, test_pairs=test_pairs)
    else:
        run_all_experiments(epochs=args.epochs, batch_size=args.batch_size)
