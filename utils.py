"""
Shared utility functions for drift detection experiment.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def load_mnist_data(batch_size=64):
    """
    Load MNIST dataset.
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def extract_activations(model, dataloader, layer_names, device='cpu', max_samples=1000):
    """
    Extract activations from specified layers.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with images
        layer_names: List of layer names to extract (e.g., ['conv3', 'fc2'])
        device: Device to run on
        max_samples: Maximum number of samples to process
        
    Returns:
        activations: Dict mapping layer names to numpy arrays of activations
        labels: Numpy array of labels
        predictions: Numpy array of predicted labels
    """
    model.eval()
    model.to(device)
    
    activations = {name: [] for name in layer_names}
    all_labels = []
    all_predictions = []
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name].append(output.detach().cpu())
        return hook
    
    hooks = []
    for name in layer_names:
        layer = dict(model.named_modules())[name]
        hooks.append(layer.register_forward_hook(get_activation(name)))
    
    samples_processed = 0
    with torch.no_grad():
        for images, labels in dataloader:
            if samples_processed >= max_samples:
                break
                
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            
            all_labels.append(labels.numpy())
            all_predictions.append(predictions.cpu().numpy())
            
            samples_processed += images.size(0)
    
    for hook in hooks:
        hook.remove()
    
    result_activations = {}
    for name in layer_names:
        acts = torch.cat(activations[name], dim=0)
        acts = acts.reshape(acts.size(0), -1)
        result_activations[name] = acts.numpy()[:max_samples]
    
    all_labels = np.concatenate(all_labels)[:max_samples]
    all_predictions = np.concatenate(all_predictions)[:max_samples]
    
    return result_activations, all_labels, all_predictions


def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """
    Compute Maximum Mean Discrepancy between two distributions.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        Y: numpy array of shape (m_samples, n_features)
        kernel: Kernel type ('rbf' supported)
        gamma: RBF kernel bandwidth (if None, uses 1/n_features)
        
    Returns:
        mmd: float, MMD value
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    
    mmd_squared = XX.mean() + YY.mean() - 2 * XY.mean()
    
    return np.sqrt(max(mmd_squared, 0))


def compute_simple_stats(activations):
    """
    Compute simple statistics for activations.
    
    Args:
        activations: numpy array of shape (n_samples, n_features)
        
    Returns:
        stats: dict with 'mean' and 'variance'
    """
    return {
        'mean': np.mean(activations),
        'variance': np.var(activations)
    }


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.
    
    Args:
        predictions: numpy array of predicted labels
        labels: numpy array of true labels
        
    Returns:
        accuracy: float between 0 and 1
    """
    return np.mean(predictions == labels)


def calibrate_mmd_threshold(mmd_values, accuracy_values, method='percentile', percentile=95):
    """
    Calibrate MMD threshold based on accuracy degradation.
    
    Args:
        mmd_values: List or array of MMD values at different drift levels
        accuracy_values: Corresponding accuracy values (0-1 range)
        method: 'percentile', 'accuracy_drop', or 'fit_curve'
        percentile: For 'percentile' method, which percentile to use
        
    Returns:
        threshold_info: Dict with calibration information
    """
    mmd_values = np.array(mmd_values)
    accuracy_values = np.array(accuracy_values)
    
    result = {
        'method': method,
        'mmd_values': mmd_values,
        'accuracy_values': accuracy_values
    }
    
    if method == 'percentile':
        # Use percentile of MMD values
        threshold = np.percentile(mmd_values, percentile)
        result['threshold'] = threshold
        result['percentile'] = percentile
        
    elif method == 'accuracy_drop':
        # Find MMD where accuracy drops below threshold (e.g., 95%)
        accuracy_threshold = 0.95
        idx = np.where(accuracy_values < accuracy_threshold)[0]
        if len(idx) > 0:
            threshold = mmd_values[idx[0]]
        else:
            threshold = mmd_values[-1]  # Use max if never drops
        result['threshold'] = threshold
        result['accuracy_threshold'] = accuracy_threshold
        
    elif method == 'fit_curve':
        # Fit polynomial to predict accuracy from MMD
        from numpy.polynomial import Polynomial
        
        # Fit 2nd degree polynomial
        p = Polynomial.fit(mmd_values, accuracy_values, deg=2)
        result['poly_coeffs'] = p.convert().coef
        result['predict_accuracy'] = lambda mmd: p(mmd)
        
        # Suggest threshold at 5% accuracy drop
        baseline_acc = accuracy_values[0]
        target_acc = baseline_acc - 0.05
        
        # Find MMD that gives target accuracy (approximate)
        test_mmds = np.linspace(mmd_values.min(), mmd_values.max(), 100)
        pred_accs = p(test_mmds)
        idx = np.argmin(np.abs(pred_accs - target_acc))
        result['threshold_5pct_drop'] = test_mmds[idx]
        result['baseline_accuracy'] = baseline_acc
        
    return result


def interpret_mmd_value(mmd_value, calibration_info):
    """
    Interpret an MMD value using calibration information.
    
    Args:
        mmd_value: Observed MMD value
        calibration_info: Dict from calibrate_mmd_threshold
        
    Returns:
        interpretation: Dict with interpretation details
    """
    result = {
        'mmd_value': mmd_value,
        'drift_detected': False,
        'severity': 'none'
    }
    
    if calibration_info['method'] in ['percentile', 'accuracy_drop']:
        threshold = calibration_info['threshold']
        result['threshold'] = threshold
        result['drift_detected'] = mmd_value > threshold
        
        # Estimate severity
        if mmd_value <= threshold:
            result['severity'] = 'none'
        elif mmd_value <= threshold * 1.5:
            result['severity'] = 'mild'
        elif mmd_value <= threshold * 2.0:
            result['severity'] = 'moderate'
        else:
            result['severity'] = 'severe'
            
    elif calibration_info['method'] == 'fit_curve':
        # FIX: Reconstruct predict_fn from poly_coeffs if it's missing (due to pickling)
        if 'predict_accuracy' not in calibration_info:
            poly_coeffs = calibration_info['poly_coeffs']
            p = np.poly1d(poly_coeffs) # np.poly1d is a NumPy function
            predict_fn = lambda mmd: p(mmd)
        else:
            # Should not happen in this setup, but kept for robustness
            predict_fn = calibration_info['predict_accuracy']
            
        # Predict accuracy from MMD
        predicted_accuracy = float(predict_fn(mmd_value))
        baseline_accuracy = calibration_info['baseline_accuracy']
        
        # ... rest of the code
        
        result['predicted_accuracy'] = predicted_accuracy
        result['baseline_accuracy'] = baseline_accuracy
        result['accuracy_drop'] = baseline_accuracy - predicted_accuracy
        result['accuracy_drop_percent'] = (baseline_accuracy - predicted_accuracy) * 100
        
        # Determine if drift based on 5% drop threshold
        result['drift_detected'] = mmd_value > calibration_info.get('threshold_5pct_drop', float('inf'))
        
        # Severity based on accuracy drop
        drop_pct = result['accuracy_drop_percent']
        if drop_pct < 2:
            result['severity'] = 'none'
        elif drop_pct < 5:
            result['severity'] = 'mild'
        elif drop_pct < 10:
            result['severity'] = 'moderate'
        else:
            result['severity'] = 'severe'
    
    return result