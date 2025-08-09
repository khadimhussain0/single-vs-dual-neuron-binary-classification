"""
Training script for binary classification models with different output layer configurations.
This script trains and evaluates models with single-neuron and two-neuron output layers using PyTorch.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support, accuracy_score

from data_loader import load_cifar10_binary, create_data_loaders
from models import create_model, get_loss_function

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train_model(model_name, output_neurons, dataset, batch_size=32, epochs=50, img_size=(32, 32, 3), learning_rate=0.001):
    """
    Train and evaluate a model with specified output layer configuration using PyTorch.
    
    Args:
        model_name: Name of the model architecture ('small_cnn', 'vgg16', 'resnet50')
        output_neurons: Number of neurons in output layer (1 or 2)
        dataset: Tuple of (class_a, class_b) for binary classification
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        img_size: Input image dimensions
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary with training history and evaluation metrics
    """
    class_a, class_b = dataset
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{model_name}_{output_neurons}neuron_{class_a}vs{class_b}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset, val_dataset, test_dataset = load_cifar10_binary(
        class_a=class_a, 
        class_b=class_b,
        img_size=img_size[:2]
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=batch_size
    )
    
    model = create_model(
        model_name=model_name,
        input_channels=3,
        output_neurons=output_neurons,
        pretrained=True
    ).to(device)
    
    criterion = get_loss_function(output_neurons)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        patience=10,
        verbose=True
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lrs': []
    }
    
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    print(f"Training {model_name} with {output_neurons} output neuron(s)...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            
            if output_neurons == 1:
                labels = labels.view(-1, 1).to(device)
            else:
                # For 2 neurons with CrossEntropyLoss, we need class indices (0 or 1)
                labels = labels.long().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            if output_neurons == 1:
                loss = criterion(outputs, labels)
            else:
                # For CrossEntropyLoss with 2 neurons, we use class indices
                loss = criterion(outputs, labels.squeeze())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            if output_neurons == 1:
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == labels).sum().item()
            else:
                predicted = torch.argmax(outputs, dim=1)
                train_correct += (predicted == labels.squeeze()).sum().item()
            
            train_total += labels.size(0)
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                
                if output_neurons == 1:
                    labels = labels.view(-1, 1).to(device)
                else:
                    labels = labels.long().to(device)
                
                outputs = model(inputs)
                
                if output_neurons == 1:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels.squeeze())
                
                val_loss += loss.item() * inputs.size(0)
                
                if output_neurons == 1:
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                else:
                    predicted = torch.argmax(outputs, dim=1)
                    val_correct += (predicted == labels.squeeze()).sum().item()
                
                val_total += labels.size(0)
                
                val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
        
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['lrs'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        scheduler.step(epoch_val_loss)
        
        early_stopping(epoch_val_loss, model, best_model_path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    print("Evaluating on test set...")
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            all_labels.append(labels.cpu().numpy())
            
            outputs = model(inputs)
            
            if output_neurons == 1:
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
            else:
                probs = outputs[:, 1].cpu().numpy()  # Probability of class 1
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_predictions.append(preds)
            all_probabilities.append(probs)
    
    y_test = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    y_pred_prob = np.concatenate(all_probabilities)
    
    if output_neurons == 1 and y_pred.ndim > 1:
        y_pred = y_pred.reshape(-1)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    test_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc
    }
    
    results = {
        'model_name': model_name,
        'output_neurons': output_neurons,
        'classes': (class_a, class_b),
        'history': history,
        'test_metrics': test_metrics,
        'classification_report': report,
        'confusion_matrix': cm,
        'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}
    }
    
    plot_training_history(history, output_dir)
    
    plot_roc_curve(fpr, tpr, roc_auc, output_dir)
    
    plot_confusion_matrix(cm, output_dir)
    
    save_results_to_csv(results, output_dir)
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    return results

def plot_training_history(history, output_dir):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Training history dictionary with metrics
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history['lrs'])
    plt.title('Learning Rate')
    plt.ylabel('Learning Rate')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, output_dir):
    """
    Plot ROC curve with enhanced styling.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: Area under ROC curve
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, color='#FF8C00', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='#192841', lw=2, linestyle='--', label='Random guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.legend(loc='lower right', fontsize=12, frameon=True, facecolor='white', edgecolor='#CCCCCC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(cm, output_dir):
    """
    Plot confusion matrix using seaborn for enhanced visualization.
    
    Args:
        cm: Confusion matrix
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar=True, square=True, linewidths=.5,
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    
    plt.yticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'], rotation=0, fontsize=12)
    plt.xticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'], fontsize=12)
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.figtext(0.01, 0.01, 
               f"Accuracy: {accuracy:.3f}\n"
               f"Precision: {precision:.3f}\n"
               f"Recall: {recall:.3f}\n"
               f"F1 Score: {f1:.3f}", 
               fontsize=12, 
               bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def save_results_to_csv(results, output_dir):
    """
    Save key results to CSV file.
    
    Args:
        results: Dictionary with results
        output_dir: Directory to save results
    """
    metrics = results['test_metrics']
    report = results['classification_report']
    
    data = {
        'Model': results['model_name'],
        'Output_Neurons': results['output_neurons'],
        'Classes': f"{results['classes'][0]}vs{results['classes'][1]}",
        'Test_Accuracy': metrics['accuracy'],
        'Test_Precision': metrics['precision'],
        'Test_Recall': metrics['recall'],
        'F1_Score': metrics['f1'],
        'ROC_AUC': results['roc']['auc']
    }
    
    df = pd.DataFrame([data])
    
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    df.to_csv(csv_path, index=False)
    
    md_path = os.path.join(output_dir, 'results_summary.md')
    with open(md_path, 'w') as f:
        f.write(f"# Experiment Results: {results['model_name']} with {results['output_neurons']} Output Neuron(s)\n\n")
        f.write(f"## Dataset: Class {results['classes'][0]} vs Class {results['classes'][1]}\n\n")
        f.write("### Test Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("| ------ | ----- |\n")
        for key, value in metrics.items():
            f.write(f"| {key.replace('_', ' ').title()} | {value:.4f} |\n")
        f.write(f"| ROC AUC | {results['roc']['auc']:.4f} |\n\n")
        
        f.write("### Classification Report\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("| ----- | --------- | ------ | -------- | ------- |\n")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                f.write(f"| {class_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {metrics['support']} |\n")

def run_experiment(model_name, dataset, batch_size=32, epochs=30, img_size=(32, 32, 3), learning_rate=0.001):
    """
    Run experiments for both single-neuron and two-neuron configurations.
    
    Args:
        model_name: Name of the model architecture ('small_cnn', 'vgg16', 'resnet50')
        dataset: Tuple of (class_a, class_b) for binary classification
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        img_size: Input image dimensions
        learning_rate: Learning rate for optimizer
    
    Returns:
        Dictionary with results for both configurations
    """
    print(f"\n{'='*80}\nRunning experiment: {model_name} on classes {dataset[0]} vs {dataset[1]}\n{'='*80}\n")
    
    # Train single-neuron model
    print(f"\n{'*'*50}\nTraining with SINGLE output neuron\n{'*'*50}\n")
    results_single = train_model(
        model_name=model_name,
        output_neurons=1,
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        img_size=img_size,
        learning_rate=learning_rate
    )
    
    # Train two-neuron model
    print(f"\n{'*'*50}\nTraining with TWO output neurons\n{'*'*50}\n")
    results_dual = train_model(
        model_name=model_name,
        output_neurons=2,
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        img_size=img_size,
        learning_rate=learning_rate
    )
    
    # Compare results
    print(f"\n{'#'*50}\nRESULTS COMPARISON\n{'#'*50}\n")
    print(f"Model: {model_name}, Classes: {dataset[0]} vs {dataset[1]}")
    print(f"Single neuron accuracy: {results_single['test_metrics']['accuracy']:.4f}")
    print(f"Dual neuron accuracy: {results_dual['test_metrics']['accuracy']:.4f}")
    print(f"Difference (dual - single): {results_dual['test_metrics']['accuracy'] - results_single['test_metrics']['accuracy']:.4f}")
    print(f"\nSingle neuron F1: {results_single['test_metrics']['f1']:.4f}")
    print(f"Dual neuron F1: {results_dual['test_metrics']['f1']:.4f}")
    print(f"Difference (dual - single): {results_dual['test_metrics']['f1'] - results_single['test_metrics']['f1']:.4f}")
    print(f"\nSingle neuron AUC: {results_single['roc']['auc']:.4f}")
    print(f"Dual neuron AUC: {results_dual['roc']['auc']:.4f}")
    print(f"Difference (dual - single): {results_dual['roc']['auc'] - results_single['roc']['auc']:.4f}")
    
    # Create a consolidated markdown report for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"results/comparison_{model_name}_{dataset[0]}vs{dataset[1]}_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# Single vs. Dual Neuron Binary Classification Comparison\n\n")
        f.write(f"## Experiment Details\n\n")
        f.write(f"- **Model Architecture:** {model_name}\n")
        f.write(f"- **Classification Task:** Class {dataset[0]} vs Class {dataset[1]}\n")
        f.write(f"- **Batch Size:** {batch_size}\n")
        f.write(f"- **Max Epochs:** {epochs}\n")
        f.write(f"- **Learning Rate:** {learning_rate}\n\n")
        
        f.write(f"## Performance Comparison\n\n")
        f.write("| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |\n")
        f.write("| ------ | ------------- | ---------- | -------------------------- |\n")
        s_metrics = results_single['test_metrics']
        d_metrics = results_dual['test_metrics']
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            diff = d_metrics[metric] - s_metrics[metric]
            f.write(f"| {metric.title()} | {s_metrics[metric]:.4f} | {d_metrics[metric]:.4f} | {diff:.4f} |\n")
        s_auc = results_single['roc']['auc']
        d_auc = results_dual['roc']['auc']
        auc_diff = d_auc - s_auc
        f.write(f"| ROC AUC | {s_auc:.4f} | {d_auc:.4f} | {auc_diff:.4f} |\n\n")
        
        f.write(f"## Key Findings\n\n")
        # Determine which approach performed better overall
        metrics_compared = ['accuracy', 'precision', 'recall', 'f1']
        diff_sum = sum(d_metrics[m] - s_metrics[m] for m in metrics_compared) + auc_diff
        better_approach = "dual neuron" if diff_sum > 0 else "single neuron"
        
        f.write(f"- The **{better_approach}** approach performed better overall for this experiment.\n")
        f.write(f"- Largest difference observed in **{max(metrics_compared + ['auc'], key=lambda m: abs(d_metrics[m] - s_metrics[m] if m != 'auc' else d_auc - s_auc))}** metric.\n")
        
        # Training dynamics
        s_epochs = len(results_single['history']['train_loss'])
        d_epochs = len(results_dual['history']['train_loss'])
        faster_convergence = "Single" if s_epochs < d_epochs else "Dual" if d_epochs < s_epochs else "Both approaches"
        f.write(f"- **{faster_convergence}** {'neuron' if faster_convergence != 'Both approaches' else 'neurons'} converged faster ({min(s_epochs, d_epochs)} vs {max(s_epochs, d_epochs)} epochs).\n")
        
    print(f"\nComparison report saved to: {report_path}")
    
    return {
        'single_neuron': results_single,
        'dual_neuron': results_dual,
        'comparison_path': report_path
    }

if __name__ == "__main__":
    print("Starting binary classification research experiments...")
    
    run_experiment(
        model_name='small_cnn',
        dataset=(0, 1),
        batch_size=64,
        epochs=20
    )


