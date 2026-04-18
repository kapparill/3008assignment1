
# Visualization Script for CIFAR-10 Training Metrics
# Creates plots from training history to analyze model performance
# Created with the help of Claude, model Haiku


import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

class TrainingVisualizer:
    """Visualize training metrics from training history"""
    
    def __init__(self):
        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
    
    def add_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc):
        """Add metrics for one epoch"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['test_loss'].append(test_loss)
        self.history['test_acc'].append(test_acc)
    
    def save_history(self, filename='training_history.json'):
        """Save training history to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {filename}")
    
    def load_history(self, filename='training_history.json'):
        """Load training history from JSON file"""
        with open(filename, 'r') as f:
            self.history = json.load(f)
        print(f"History loaded from {filename}")
    
    def plot_loss(self, save_path='loss_plot.png'):
        """Plot training vs test loss over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epochs'], self.history['train_loss'], 
                 marker='o', label='Train Loss', linewidth=2)
        plt.plot(self.history['epochs'], self.history['test_loss'], 
                 marker='s', label='Test Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training vs Test Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Loss plot saved to {save_path}")
        plt.close()
    
    def plot_accuracy(self, save_path='accuracy_plot.png'):
        """Plot training vs test accuracy over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epochs'], self.history['train_acc'], 
                 marker='o', label='Train Accuracy', linewidth=2)
        plt.plot(self.history['epochs'], self.history['test_acc'], 
                 marker='s', label='Test Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training vs Test Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Accuracy plot saved to {save_path}")
        plt.close()
    
    def plot_combined(self, save_path='combined_metrics.png'):
        """Plot all metrics in a 2x2 grid"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Train Loss
        axes[0, 0].plot(self.history['epochs'], self.history['train_loss'], 
                       marker='o', color='blue', linewidth=2)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test Loss
        axes[0, 1].plot(self.history['epochs'], self.history['test_loss'], 
                       marker='s', color='orange', linewidth=2)
        axes[0, 1].set_ylabel('Loss', fontsize=11)
        axes[0, 1].set_title('Test Loss', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Train Accuracy
        axes[1, 0].plot(self.history['epochs'], self.history['train_acc'], 
                       marker='o', color='green', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Accuracy (%)', fontsize=11)
        axes[1, 0].set_title('Training Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Test Accuracy
        axes[1, 1].plot(self.history['epochs'], self.history['test_acc'], 
                       marker='s', color='red', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Accuracy (%)', fontsize=11)
        axes[1, 1].set_title('Test Accuracy', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Combined metrics plot saved to {save_path}")
        plt.close()
    
    def plot_overfit_analysis(self, save_path='overfit_analysis.png'):
        """Analyze overfitting by plotting train-test gap"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss gap (higher = more overfitting)
        loss_gap = np.array(self.history['test_loss']) - np.array(self.history['train_loss'])
        axes[0].bar(self.history['epochs'], loss_gap, color='red', alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Test Loss - Train Loss', fontsize=11)
        axes[0].set_title('Loss Gap (Overfitting Indicator)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Accuracy gap (lower = less overfitting)
        acc_gap = np.array(self.history['train_acc']) - np.array(self.history['test_acc'])
        axes[1].bar(self.history['epochs'], acc_gap, color='orange', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Train Accuracy - Test Accuracy (%)', fontsize=11)
        axes[1].set_title('Accuracy Gap (Overfitting Indicator)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Overfit analysis plot saved to {save_path}")
        plt.close()
    
    def plot_all(self):
        """Create all plots"""
        self.plot_loss()
        self.plot_accuracy()
        self.plot_combined()
        self.plot_overfit_analysis()
        print("\nAll plots created successfully!")
    
    def print_summary(self):
        """Print training summary statistics"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Epochs: {len(self.history['epochs'])}")
        print(f"\nBest Test Accuracy: {max(self.history['test_acc']):.2f}%")
        print(f"Final Test Accuracy: {self.history['test_acc'][-1]:.2f}%")
        print(f"Best Test Loss: {min(self.history['test_loss']):.4f}")
        print(f"Final Test Loss: {self.history['test_loss'][-1]:.4f}")
        print(f"\nTrain Accuracy Improvement: {self.history['train_acc'][-1] - self.history['train_acc'][0]:.2f}%")
        print(f"Test Accuracy Improvement: {self.history['test_acc'][-1] - self.history['test_acc'][0]:.2f}%")
        
        # Overfitting analysis
        final_loss_gap = self.history['test_loss'][-1] - self.history['train_loss'][-1]
        final_acc_gap = self.history['train_acc'][-1] - self.history['test_acc'][-1]
        print(f"\nFinal Loss Gap (Test - Train): {final_loss_gap:.4f}")
        print(f"Final Accuracy Gap (Train - Test): {final_acc_gap:.2f}%")
        
        if final_acc_gap > 5:
            print("!  Significant overfitting detected!")
        elif final_acc_gap > 2:
            print("!  Some overfitting present (normal)")
        else:
            print("✓ Good generalization (low overfitting)")
        print("="*60 + "\n")