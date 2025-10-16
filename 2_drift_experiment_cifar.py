"""
Script 2 (CIFAR): Apply drift and measure detection performance on CIFAR-10.

This is the CIFAR-10 version of 2_drift_experiment.py

Inputs:
    - model_cifar.pth (from Colab training)
    - baseline_activations_cifar.pkl
    - baseline_stats_cifar.json

Outputs:
    - drift_results_cifar.pkl: All measurements across drift levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import pickle
import json
import numpy as np
from tqdm import tqdm

from utils import (
    extract_activations,
    compute_mmd,
    compute_simple_stats,
    compute_accuracy
)


# ============================================================================
# CONFIGURATION: Toggle drift types to test
# ============================================================================
DRIFT_CONFIG = {
    'blur': True,
    'noise': False,           # Start with noise
    'rotation': False,
    'brightness': False,
    'contrast': False,
    'pixelate': False,
}

# Drift intensity levels for each type (adjusted for CIFAR)
DRIFT_LEVELS = {
    'blur': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    'noise': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],      # Lower for color images
    'rotation': [0, 5, 10, 15, 20, 30, 45],
    'brightness': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    'contrast': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
    'pixelate': [32, 28, 24, 20, 16, 12, 8],             # 32 = original size
}
# ============================================================================


def get_resnet18_model():
    """Create ResNet18 for CIFAR-10 (must match training config)."""
    model = models.resnet18(weights=None)
    
    # Same modifications as training
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model


def get_layer_names():
    """Return names of layers to monitor in ResNet18."""
    return ['layer3', 'fc']


def load_cifar10_test_data(batch_size=1000):
    """Load CIFAR-10 test data WITHOUT normalization for drift application."""
    
    # No normalization - we'll apply drift to raw images
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                    transform=test_transform)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


def normalize_cifar(images):
    """Apply CIFAR-10 normalization (needed after drift application)."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    return (images - mean) / std


def apply_blur(images, sigma, kernel_size=3):
    """Apply Gaussian blur to images."""
    if sigma == 0:
        return images
    
    blur_transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    blurred = torch.stack([blur_transform(img) for img in images])
    return blurred


def apply_noise(images, noise_level):
    """Apply Gaussian noise to images."""
    if noise_level == 0:
        return images
    
    noise = torch.randn_like(images) * noise_level
    noisy = torch.clamp(images + noise, 0, 1)
    return noisy


def apply_rotation(images, degrees):
    """Apply rotation to images."""
    if degrees == 0:
        return images
    
    rotate_transform = transforms.RandomRotation(degrees=(degrees, degrees))
    rotated = torch.stack([rotate_transform(img) for img in images])
    return rotated


def apply_brightness(images, factor):
    """Apply brightness adjustment to images."""
    if factor == 0:
        return images
    
    adjusted = torch.clamp(images + factor, 0, 1)
    return adjusted


def apply_contrast(images, factor):
    """Apply contrast adjustment to images."""
    if factor == 0:
        return images
    
    mean = images.mean(dim=(2, 3), keepdim=True)
    adjusted = torch.clamp(mean + (1 + factor) * (images - mean), 0, 1)
    return adjusted


def apply_pixelate(images, target_size):
    """Apply pixelation by downsampling and upsampling."""
    if target_size == 32:
        return images
    
    downsampled = F.interpolate(images, size=(target_size, target_size), mode='nearest')
    pixelated = F.interpolate(downsampled, size=(32, 32), mode='nearest')
    return pixelated


def compute_input_mmd(baseline_images, drifted_images):
    """Compute MMD between baseline and drifted input images."""
    baseline_flat = baseline_images.reshape(baseline_images.size(0), -1).numpy()
    drifted_flat = drifted_images.reshape(drifted_images.size(0), -1).numpy()
    
    return compute_mmd(baseline_flat, drifted_flat)


def run_drift_sweep(model, baseline_activations, test_images, test_labels, 
                    drift_type, drift_levels, layer_names, device='cpu'):
    """Run drift experiment across multiple drift levels."""
    
    drift_functions = {
        'blur': apply_blur,
        'noise': apply_noise,
        'rotation': apply_rotation,
        'brightness': apply_brightness,
        'contrast': apply_contrast,
        'pixelate': apply_pixelate,
    }
    
    apply_drift = drift_functions[drift_type]
    
    results = {
        'drift_type': drift_type,
        'drift_levels': drift_levels,
        'accuracy': [],
        'input_mmd': [],
        'layer_mmd': {name: [] for name in layer_names},
        'layer_stats': {name: {'mean': [], 'variance': []} for name in layer_names}
    }
    
    print(f"Running {drift_type} drift sweep...")
    for level in tqdm(drift_levels, desc=f"{drift_type.capitalize()} levels"):
        # Apply drift
        drifted_images = apply_drift(test_images, level)
        
        # Compute input MMD
        input_mmd = compute_input_mmd(test_images, drifted_images)
        results['input_mmd'].append(input_mmd)
        
        # Normalize for model (CIFAR normalization)
        normalized_images = normalize_cifar(drifted_images)
        
        # Create dataloader
        drifted_dataset = TensorDataset(normalized_images, test_labels)
        drifted_loader = DataLoader(drifted_dataset, batch_size=64, shuffle=False)
        
        # Extract activations
        drifted_activations, labels, predictions = extract_activations(
            model, drifted_loader, layer_names, device=device, max_samples=1000
        )
        
        accuracy = compute_accuracy(predictions, labels)
        results['accuracy'].append(accuracy)
        
        # Compute MMD and stats for each layer
        for layer_name in layer_names:
            baseline_acts = baseline_activations[layer_name]
            drifted_acts = drifted_activations[layer_name]
            
            layer_mmd = compute_mmd(baseline_acts, drifted_acts)
            results['layer_mmd'][layer_name].append(layer_mmd)
            
            stats = compute_simple_stats(drifted_acts)
            results['layer_stats'][layer_name]['mean'].append(stats['mean'])
            results['layer_stats'][layer_name]['variance'].append(stats['variance'])
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check which drift types are enabled
    enabled_drifts = [drift_type for drift_type, enabled in DRIFT_CONFIG.items() if enabled]
    
    if not enabled_drifts:
        print("ERROR: No drift types enabled! Set at least one to True in DRIFT_CONFIG.")
        return
    
    print(f"\nEnabled drift types: {', '.join(enabled_drifts)}")
    
    # Load model
    print("\nLoading trained CIFAR-10 model...")
    model = get_resnet18_model()
    model.load_state_dict(torch.load('model_cifar.pth', map_location=device))
    model.eval()
    layer_names = get_layer_names()
    print(f"Monitoring layers: {layer_names}")
    
    # Load baseline activations
    print("Loading baseline activations...")
    with open('baseline_activations_cifar.pkl', 'rb') as f:
        baseline_data = pickle.load(f)
    baseline_activations = baseline_data['activations']
    print(f"Loaded baseline activations for {len(baseline_data['labels'])} samples")
    
    with open('baseline_stats_cifar.json', 'r') as f:
        baseline_stats = json.load(f)
    print(f"Baseline accuracy: {baseline_stats['accuracy']*100:.2f}%")
    
    # Load test data (unnormalized for drift application)
    print("Loading CIFAR-10 test data...")
    test_loader = load_cifar10_test_data(batch_size=1000)
    
    test_images, test_labels = next(iter(test_loader))
    print(f"Loaded {len(test_images)} test images")
    
    # Run experiments for each enabled drift type
    all_results = {}
    
    for drift_type in enabled_drifts:
        print(f"\n{'='*60}")
        print(f"Testing {drift_type.upper()} drift")
        print(f"{'='*60}")
        
        drift_levels = DRIFT_LEVELS[drift_type]
        print(f"Drift levels: {drift_levels}")
        
        results = run_drift_sweep(
            model=model,
            baseline_activations=baseline_activations,
            test_images=test_images,
            test_labels=test_labels,
            drift_type=drift_type,
            drift_levels=drift_levels,
            layer_names=layer_names,
            device=device
        )
        
        results['baseline_stats'] = baseline_stats
        all_results[drift_type] = results
        
        # Print summary
        print(f"\n{drift_type.upper()} DRIFT SUMMARY")
        print("-"*60)
        print(f"{'Level':<10} {'Accuracy':<12} {'Input MMD':<12} {layer_names[0]+' MMD':<12} {layer_names[1]+' MMD'}")
        print("-"*60)
        for i, level in enumerate(drift_levels):
            print(f"{level:<10} {results['accuracy'][i]*100:<12.2f} "
                  f"{results['input_mmd'][i]:<12.4f} "
                  f"{results['layer_mmd'][layer_names[0]][i]:<12.4f} "
                  f"{results['layer_mmd'][layer_names[1]][i]:<12.4f}")
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    
    if len(enabled_drifts) == 1:
        save_data = all_results[enabled_drifts[0]]
        filename = 'drift_results_cifar.pkl'
    else:
        save_data = {
            'multiple_drift_types': True,
            'results': all_results
        }
        filename = 'drift_results_cifar_multi.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {filename}")
    
    print(f"\n{'='*60}")
    print("✓ CIFAR-10 drift experiment complete!")
    print(f"{'='*60}")
    print(f"  Tested {len(enabled_drifts)} drift type(s): {', '.join(enabled_drifts)}")
    print(f"  Results saved to: {filename}")
    print(f"  Next: Run 3_visualize_results.py (update to load {filename})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()