"""
Script 2: Apply drift and measure detection performance.

Inputs:
    - model.pth
    - baseline_activations.pkl
    - baseline_stats.json

Outputs:
    - drift_results.pkl: All measurements across drift levels
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from model import SimpleCNN
from utils import (
    load_mnist_data,
    extract_activations,
    compute_mmd,
    compute_simple_stats,
    compute_accuracy
)


# ============================================================================
# CONFIGURATION: Toggle drift types to test
# ============================================================================
DRIFT_CONFIG = {
    'blur': True,           # Gaussian blur
    'noise': False,          # Gaussian noise (default: True)
    'rotation': False,       # Random rotation
    'brightness': False,     # Brightness adjustment
    'contrast': False,       # Contrast adjustment
    'pixelate': False,       # Pixelation/downsampling
}

# Drift intensity levels for each type
DRIFT_LEVELS = {
    'blur': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],              # sigma values
    'noise': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],             # noise std
    'rotation': [0, 5, 10, 15, 20, 30, 45],                   # degrees
    'brightness': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5],        # brightness factor
    'contrast': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5],          # contrast factor
    'pixelate': [28, 24, 20, 16, 12, 8, 4],                   # image size (pixels)
}
# ============================================================================


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
    if target_size == 28:
        return images
    
    downsampled = F.interpolate(images, size=(target_size, target_size), mode='nearest')
    pixelated = F.interpolate(downsampled, size=(28, 28), mode='nearest')
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
        drifted_images = apply_drift(test_images, level)
        
        input_mmd = compute_input_mmd(test_images, drifted_images)
        results['input_mmd'].append(input_mmd)
        
        drifted_dataset = TensorDataset(drifted_images, test_labels)
        drifted_loader = DataLoader(drifted_dataset, batch_size=64, shuffle=False)
        
        drifted_activations, labels, predictions = extract_activations(
            model, drifted_loader, layer_names, device=device, max_samples=1000
        )
        
        accuracy = compute_accuracy(predictions, labels)
        results['accuracy'].append(accuracy)
        
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
    
    print("\nLoading trained model...")
    model = SimpleCNN()
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()
    layer_names = model.get_layer_names()
    print(f"Monitoring layers: {layer_names}")
    
    print("Loading baseline activations...")
    with open('baseline_activations.pkl', 'rb') as f:
        baseline_data = pickle.load(f)
    baseline_activations = baseline_data['activations']
    print(f"Loaded baseline activations for {len(baseline_data['labels'])} samples")
    
    with open('baseline_stats.json', 'r') as f:
        baseline_stats = json.load(f)
    print(f"Baseline accuracy: {baseline_stats['accuracy']*100:.2f}%")
    
    print("Loading test data...")
    _, test_loader = load_mnist_data(batch_size=1000)
    
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
        
        # Print summary for this drift type
        print(f"\n{drift_type.upper()} DRIFT SUMMARY")
        print("-"*60)
        print(f"{'Level':<10} {'Accuracy':<12} {'Input MMD':<12} {layer_names[0]+' MMD':<12} {layer_names[1]+' MMD'}")
        print("-"*60)
        for i, level in enumerate(drift_levels):
            print(f"{level:<10} {results['accuracy'][i]*100:<12.2f} "
                  f"{results['input_mmd'][i]:<12.4f} "
                  f"{results['layer_mmd'][layer_names[0]][i]:<12.4f} "
                  f"{results['layer_mmd'][layer_names[1]][i]:<12.4f}")
    
    # Save all results
    print(f"\n{'='*60}")
    print("Saving results...")
    
    if len(enabled_drifts) == 1:
        # Single drift type - save as before
        save_data = all_results[enabled_drifts[0]]
        filename = 'drift_results.pkl'
    else:
        # Multiple drift types - save all
        save_data = {
            'multiple_drift_types': True,
            'results': all_results
        }
        filename = 'drift_results_multi.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved: {filename}")
    
    print(f"\n{'='*60}")
    print("✓ Drift experiment complete!")
    print(f"{'='*60}")
    print(f"  Tested {len(enabled_drifts)} drift type(s): {', '.join(enabled_drifts)}")
    print(f"  Results saved to: {filename}")
    print(f"  Next: Run 3_visualize_results.py to see plots")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()