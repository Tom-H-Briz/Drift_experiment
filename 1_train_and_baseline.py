"""
Script 1: Train MNIST classifier and extract baseline activations.

Outputs:
    - model.pth: Trained model weights
    - baseline_activations.pkl: Activations from clean test images
    - baseline_stats.json: Baseline statistics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
from tqdm import tqdm

from model import SimpleCNN
from utils import (
    load_mnist_data,
    extract_activations,
    compute_simple_stats,
    compute_accuracy
)


def train_model(model, train_loader, device='cpu', epochs=10):
    """Train the CNN model."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={running_loss/total:.4f}, Accuracy={epoch_acc:.2f}%")
    
    return model


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading MNIST data...")
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    print("Creating model...")
    model = SimpleCNN()
    layer_names = model.get_layer_names()
    print(f"Will monitor layers: {layer_names}")
    
    model = train_model(model, train_loader, device=device, epochs=10)
    
    print("\nEvaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    if test_accuracy < 95.0:
        print("WARNING: Model accuracy below 95%. Consider training longer.")
    
    print("\nSaving model...")
    torch.save(model.state_dict(), 'model.pth')
    print("Saved: model.pth")
    
    print("\nExtracting baseline activations from clean test images...")
    baseline_activations, labels, predictions = extract_activations(
        model, test_loader, layer_names, device=device, max_samples=1000
    )
    
    baseline_accuracy = compute_accuracy(predictions, labels)
    print(f"Baseline accuracy on 1000 samples: {baseline_accuracy*100:.2f}%")
    
    print("\nComputing baseline statistics...")
    baseline_stats = {
        'accuracy': float(baseline_accuracy),
        'layer_stats': {}
    }
    
    for layer_name in layer_names:
        acts = baseline_activations[layer_name]
        stats = compute_simple_stats(acts)
        baseline_stats['layer_stats'][layer_name] = {
            'mean': float(stats['mean']),
            'variance': float(stats['variance'])
        }
        print(f"{layer_name}: mean={stats['mean']:.4f}, variance={stats['variance']:.4f}")
    
    print("\nSaving baseline activations...")
    with open('baseline_activations.pkl', 'wb') as f:
        pickle.dump({
            'activations': baseline_activations,
            'labels': labels,
            'predictions': predictions
        }, f)
    print("Saved: baseline_activations.pkl")
    
    print("Saving baseline statistics...")
    with open('baseline_stats.json', 'w') as f:
        json.dump(baseline_stats, f, indent=2)
    print("Saved: baseline_stats.json")
    
    print("\n✓ Training and baseline extraction complete!")
    print(f"  Model accuracy: {test_accuracy:.2f}%")
    print(f"  Baseline activations saved for {len(labels)} samples")
    print(f"  Ready for drift experiment (run 2_drift_experiment.py)")


if __name__ == '__main__':
    main()
    