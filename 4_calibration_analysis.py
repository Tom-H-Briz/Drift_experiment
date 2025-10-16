"""
Script 4: Calibrate MMD thresholds for drift detection.

Inputs:
    - drift_results.pkl

Outputs:
    - calibration_curves.png: Visualization of MMD vs accuracy
    - calibration_info.pkl: Calibration parameters for deployment
    - deployment_guide.txt: How to use calibration in production
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import calibrate_mmd_threshold, interpret_mmd_value


def plot_calibration_curves(results, output_file='calibration_curves.png'):
    """
    Plot MMD vs Accuracy for each detection method with fitted curves.
    """
    drift_levels = results.get('noise_levels', results.get('blur_levels', []))
    drift_label = 'Noise Level' if 'noise_levels' in results else 'Blur Level'
    accuracy = np.array(results['accuracy']) * 100
    input_mmd = np.array(results['input_mmd'])
    
    layer_names = list(results['layer_mmd'].keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Input Space MMD
    ax = axes[0]
    ax.scatter(input_mmd, accuracy, s=100, alpha=0.7, color='blue', edgecolors='black')
    
    # Fit curve
    calibration = calibrate_mmd_threshold(input_mmd, accuracy/100, method='fit_curve')
    if 'predict_accuracy' in calibration:
        mmd_range = np.linspace(input_mmd.min(), input_mmd.max(), 100)
        pred_acc = calibration['predict_accuracy'](mmd_range) * 100
        ax.plot(mmd_range, pred_acc, 'r--', linewidth=2, label='Fitted Curve')
        
        # Mark threshold
        if 'threshold_5pct_drop' in calibration:
            thresh = calibration['threshold_5pct_drop']
            ax.axvline(thresh, color='red', linestyle=':', linewidth=2, label=f'5% Drop Threshold')
    
    ax.set_xlabel('Input Space MMD', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Input Space: MMD vs Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([70, 105])
    
    # Plot 2 & 3: Activation layers
    for i, layer_name in enumerate(layer_names):
        ax = axes[i+1]
        layer_mmd = np.array(results['layer_mmd'][layer_name])
        
        ax.scatter(layer_mmd, accuracy, s=100, alpha=0.7, 
                  color=['red', 'green'][i], edgecolors='black')
        
        # Fit curve
        calibration = calibrate_mmd_threshold(layer_mmd, accuracy/100, method='fit_curve')
        if 'predict_accuracy' in calibration:
            mmd_range = np.linspace(layer_mmd.min(), layer_mmd.max(), 100)
            pred_acc = calibration['predict_accuracy'](mmd_range) * 100
            ax.plot(mmd_range, pred_acc, 'k--', linewidth=2, label='Fitted Curve')
            
            # Mark threshold
            if 'threshold_5pct_drop' in calibration:
                thresh = calibration['threshold_5pct_drop']
                ax.axvline(thresh, color='red', linestyle=':', linewidth=2, 
                          label=f'5% Drop Threshold')
        
        ax.set_xlabel(f'{layer_name} MMD', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{layer_name}: MMD vs Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([70, 105])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def generate_calibration_info(results, output_file='calibration_info.pkl'):
    """
    Generate calibration parameters for deployment.
    
    Fixes:
    1. Initializes 'calibration_data' to resolve NameError.
    2. Defines 'accuracy', 'input_mmd', and 'layer_names' to resolve NameErrors.
    3. Removes the un-picklable 'predict_accuracy' lambda function before saving.
    """
    
    # 1. Define variables from the results dictionary
    accuracy = np.array(results['accuracy'])
    input_mmd = np.array(results['input_mmd'])
    layer_names = list(results['layer_mmd'].keys())
    
    # 2. Initialize the dictionary
    calibration_data = {}
    
    # Calibrate input space
    calib_input = calibrate_mmd_threshold(
        input_mmd, accuracy, method='fit_curve'
    )
    # 3. Remove the un-picklable lambda function
    if 'predict_accuracy' in calib_input:
        del calib_input['predict_accuracy'] 
        
    calibration_data['input_space'] = calib_input
    
    # Calibrate each layer
    for layer_name in layer_names:
        layer_mmd = np.array(results['layer_mmd'][layer_name])
        calib_layer = calibrate_mmd_threshold(
            layer_mmd, accuracy, method='fit_curve'
        )
        # 3. Remove the un-picklable lambda function
        if 'predict_accuracy' in calib_layer:
            del calib_layer['predict_accuracy']
            
        calibration_data[layer_name] = calib_layer
    
    # Save
    with open(output_file, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print(f"Saved: {output_file}")
    return calibration_data


def print_calibration_summary(calibration_data):
    """
    Print summary of calibration results.
    """
    print("\n" + "="*80)
    print("CALIBRATION SUMMARY")
    print("="*80)
    print("\nThresholds for 5% Accuracy Drop:")
    print("-"*80)
    
    for method_name, calib in calibration_data.items():
        if 'threshold_5pct_drop' in calib:
            threshold = calib['threshold_5pct_drop']
            baseline_acc = calib['baseline_accuracy']
            print(f"{method_name:<20} MMD Threshold: {threshold:.4f}  "
                  f"(Baseline Acc: {baseline_acc*100:.2f}%)")
    
    print("\n" + "="*80)
    print("EXAMPLE INTERPRETATIONS")
    print("="*80)
    
    # Show examples for fc2 (decision layer)
    if 'fc2' in calibration_data:
        fc2_calib = calibration_data['fc2']
        
        print("\nFC2 Layer (Decision Layer) Examples:")
        print("-"*80)
        
        test_mmds = [0.01, 0.03, 0.05, 0.07]
        for test_mmd in test_mmds:
            interp = interpret_mmd_value(test_mmd, fc2_calib)
            print(f"\nMMD = {test_mmd:.4f}:")
            if 'predicted_accuracy' in interp:
                print(f"  Predicted Accuracy: {interp['predicted_accuracy']*100:.2f}%")
                print(f"  Accuracy Drop: {interp['accuracy_drop_percent']:.2f}%")
            print(f"  Drift Detected: {interp['drift_detected']}")
            print(f"  Severity: {interp['severity']}")


def generate_deployment_guide(calibration_data, output_file='deployment_guide.txt'):
    """
    Generate human-readable deployment guide.
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEPLOYMENT GUIDE: Using Calibrated MMD Thresholds\n")
        f.write("="*80 + "\n\n")
        
        f.write("## Overview\n\n")
        f.write("This guide explains how to use the calibrated MMD thresholds to monitor\n")
        f.write("model performance in production without requiring labels.\n\n")
        
        f.write("## Quick Start\n\n")
        f.write("1. Extract activations from incoming data (same layers as training)\n")
        f.write("2. Compute MMD between baseline and current activations\n")
        f.write("3. Use calibration curve to predict accuracy\n")
        f.write("4. Alert if predicted accuracy drops below threshold\n\n")
        
        f.write("="*80 + "\n")
        f.write("RECOMMENDED APPROACH: FC2 Layer (Decision Layer)\n")
        f.write("="*80 + "\n\n")
        
        if 'fc2' in calibration_data:
            fc2_calib = calibration_data['fc2']
            
            if 'threshold_5pct_drop' in fc2_calib:
                f.write(f"Alert Threshold: MMD > {fc2_calib['threshold_5pct_drop']:.4f}\n")
                f.write(f"Baseline Accuracy: {fc2_calib['baseline_accuracy']*100:.2f}%\n\n")
            
            f.write("Alert Levels:\n")
            f.write("-" * 80 + "\n")
            f.write("Severity   | Accuracy Drop | Action\n")
            f.write("-" * 80 + "\n")
            f.write("None       | < 2%          | Continue normal operation\n")
            f.write("Mild       | 2-5%          | Increase monitoring frequency\n")
            f.write("Moderate   | 5-10%         | Flag cases for human review\n")
            f.write("Severe     | > 10%         | Stop automated predictions\n")
            f.write("-" * 80 + "\n\n")
        
        f.write("## Why FC2 Layer?\n\n")
        f.write("Based on experiments:\n")
        f.write("- Input Space & Conv3: Max out immediately (high false positives)\n")
        f.write("- FC2 (Decision Layer): Gradual increase tracking accuracy degradation\n")
        f.write("- FC2 provides progressive warning signal, not binary alert\n\n")
        
        f.write("## Example Usage\n\n")
        f.write("```python\n")
        f.write("from utils import interpret_mmd_value\n")
        f.write("import pickle\n\n")
        f.write("# Load calibration\n")
        f.write("with open('calibration_info.pkl', 'rb') as f:\n")
        f.write("    calibration = pickle.load(f)\n\n")
        f.write("# In production\n")
        f.write("current_mmd = compute_mmd(baseline_acts, current_acts)\n")
        f.write("result = interpret_mmd_value(current_mmd, calibration['fc2'])\n\n")
        f.write("if result['drift_detected']:\n")
        f.write("    print(f\"Alert! Predicted accuracy: {result['predicted_accuracy']*100:.1f}%\")\n")
        f.write("    print(f\"Severity: {result['severity']}\")\n")
        f.write("```\n\n")
        
        f.write("="*80 + "\n")
        f.write("COMPARISON: All Methods\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Method':<20} {'Threshold (5% drop)':<25} {'Baseline Acc'}\n")
        f.write("-"*80 + "\n")
        
        for method_name, calib in calibration_data.items():
            if 'threshold_5pct_drop' in calib:
                thresh = calib['threshold_5pct_drop']
                baseline = calib['baseline_accuracy'] * 100
                f.write(f"{method_name:<20} {thresh:<25.4f} {baseline:.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Saved: {output_file}")


def main():
    print("Loading drift experiment results...")
    with open('drift_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    drift_type = 'noise' if 'noise_levels' in results else 'blur'
    print(f"Loaded {drift_type} drift experiment results")
    
    print("\nGenerating calibration curves...")
    plot_calibration_curves(results)
    
    print("\nCalibrating MMD thresholds...")
    calibration_data = generate_calibration_info(results)
    
    print_calibration_summary(calibration_data)
    
    print("\nGenerating deployment guide...")
    generate_deployment_guide(calibration_data)
    
    print("\n" + "="*80)
    print("✓ Calibration complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - calibration_curves.png (MMD vs Accuracy for all methods)")
    print("  - calibration_info.pkl (Calibration parameters for code)")
    print("  - deployment_guide.txt (Human-readable deployment instructions)")
    print("\nKey Finding:")
    print("  FC2 (decision layer) provides best progressive drift signal.")
    print("  Use FC2 MMD threshold for production monitoring.")
    print("="*80)


if __name__ == '__main__':
    main()