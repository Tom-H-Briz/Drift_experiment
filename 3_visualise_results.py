"""
Script 3: Visualize drift experiment results.

Inputs:
    - drift_results.pkl

Outputs:
    - drift_comparison.png: Main visualization
    - correlation_analysis.txt: Statistical analysis
    - layer_comparison.png: Layer-wise comparison
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def normalize_to_01(values):
    """Normalize values to 0-1 range for comparison."""
    values = np.array(values)
    min_val = values.min()
    max_val = values.max()
    if max_val - min_val == 0:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def plot_drift_curves(results, output_file='drift_comparison.png'):
    """Create main visualization comparing all detection methods."""
    # Detect drift type
    drift_type = results.get('drift_type', 'unknown')
    
    # Legacy support for old format
    if 'noise_levels' in results:
        drift_levels = results['noise_levels']
        drift_label = 'Noise Level'
    elif 'blur_levels' in results:
        drift_levels = results['blur_levels']
        drift_label = 'Blur Level (σ)'
    else:
        drift_levels = results['drift_levels']
        drift_label = f"{drift_type.capitalize()} Level"
    
    accuracy = np.array(results['accuracy']) * 100
    
    input_mmd_norm = normalize_to_01(results['input_mmd'])
    
    layer_names = list(results['layer_mmd'].keys())
    layer_mmds_norm = {
        name: normalize_to_01(results['layer_mmd'][name]) 
        for name in layer_names
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(drift_levels, accuracy, 'o-', linewidth=2, markersize=8, 
             color='black', label='Accuracy (Ground Truth)')
    ax1.set_xlabel(drift_label, fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Performance Under Drift', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 105])
    
    ax2.plot(drift_levels, input_mmd_norm, 's-', linewidth=2, markersize=8,
             label='Input Space MMD', color='blue')
    
    colors = ['red', 'green']
    markers = ['o', '^']
    for i, name in enumerate(layer_names):
        ax2.plot(drift_levels, layer_mmds_norm[name], f'{markers[i]}-', 
                linewidth=2, markersize=8, label=f'{name} MMD', color=colors[i])
    
    ax2.set_xlabel(drift_label, fontsize=12)
    ax2.set_ylabel('Normalized MMD (0-1)', fontsize=12)
    ax2.set_title('Drift Detection Methods (Normalized)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def compute_correlations(results, output_file='correlation_analysis.txt'):
    """Compute correlation between accuracy and each detection method."""
    accuracy = np.array(results['accuracy'])
    input_mmd = np.array(results['input_mmd'])
    
    layer_names = list(results['layer_mmd'].keys())
    
    correlations = {}
    correlations['input_mmd'] = pearsonr(accuracy, input_mmd)
    
    for name in layer_names:
        layer_mmd = np.array(results['layer_mmd'][name])
        correlations[name] = pearsonr(accuracy, layer_mmd)
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CORRELATION ANALYSIS: Accuracy vs Detection Methods\n")
        f.write("="*60 + "\n\n")
        f.write("Higher |correlation| = better tracking of accuracy changes\n\n")
        
        f.write(f"{'Method':<20} {'Correlation':<15} {'P-value'}\n")
        f.write("-"*60 + "\n")
        
        for method, (corr, pval) in correlations.items():
            f.write(f"{method:<20} {corr:>14.4f} {pval:>14.2e}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("="*60 + "\n")
        
        best_method = max(correlations.items(), key=lambda x: abs(x[1][0]))
        f.write(f"\nStrongest correlation: {best_method[0]} (r = {best_method[1][0]:.4f})\n")
        
        input_corr = abs(correlations['input_mmd'][0])
        activation_corrs = {name: abs(correlations[name][0]) for name in layer_names}
        best_activation = max(activation_corrs.items(), key=lambda x: x[1])
        
        f.write(f"\nInput space MMD correlation: {input_corr:.4f}\n")
        f.write(f"Best activation MMD correlation: {best_activation[0]} = {best_activation[1]:.4f}\n")
        
        if best_activation[1] > input_corr:
            improvement = (best_activation[1] - input_corr) / input_corr * 100
            f.write(f"\n✓ Activation monitoring shows {improvement:.1f}% stronger correlation!\n")
        else:
            f.write(f"\n✗ Input space MMD has stronger correlation.\n")
    
    print(f"Saved: {output_file}")
    
    print("\n" + "="*60)
    print("CORRELATION SUMMARY")
    print("="*60)
    for method, (corr, pval) in correlations.items():
        print(f"{method:<20} r = {corr:>7.4f}  (p = {pval:.2e})")


def plot_layer_comparison(results, output_file='layer_comparison.png'):
    """Compare behavior of different layers."""
    # Detect drift type
    drift_type = results.get('drift_type', 'unknown')
    
    # Legacy support
    if 'noise_levels' in results:
        drift_levels = results['noise_levels']
        drift_label = 'Noise Level'
    elif 'blur_levels' in results:
        drift_levels = results['blur_levels']
        drift_label = 'Blur Level (σ)'
    else:
        drift_levels = results['drift_levels']
        drift_label = f"{drift_type.capitalize()} Level"
    
    layer_names = list(results['layer_mmd'].keys())
    
    fig, axes = plt.subplots(len(layer_names), 2, figsize=(12, 5*len(layer_names)))
    
    if len(layer_names) == 1:
        axes = axes.reshape(1, -1)
    
    for i, layer_name in enumerate(layer_names):
        layer_mmd = results['layer_mmd'][layer_name]
        axes[i, 0].plot(drift_levels, layer_mmd, 'o-', linewidth=2, markersize=8)
        axes[i, 0].set_xlabel(drift_label, fontsize=11)
        axes[i, 0].set_ylabel('MMD', fontsize=11)
        axes[i, 0].set_title(f'{layer_name} - MMD vs Drift', fontsize=12, fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)
        
        mean_vals = results['layer_stats'][layer_name]['mean']
        var_vals = results['layer_stats'][layer_name]['variance']
        
        ax_right = axes[i, 1]
        ax_var = ax_right.twinx()
        
        line1 = ax_right.plot(drift_levels, mean_vals, 'o-', linewidth=2, 
                             markersize=8, color='blue', label='Mean')
        line2 = ax_var.plot(drift_levels, var_vals, 's-', linewidth=2, 
                           markersize=8, color='red', label='Variance')
        
        ax_right.set_xlabel(drift_label, fontsize=11)
        ax_right.set_ylabel('Mean Activation', fontsize=11, color='blue')
        ax_var.set_ylabel('Activation Variance', fontsize=11, color='red')
        ax_right.set_title(f'{layer_name} - Statistics', fontsize=12, fontweight='bold')
        ax_right.tick_params(axis='y', labelcolor='blue')
        ax_var.tick_params(axis='y', labelcolor='red')
        ax_right.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_right.legend(lines, labels, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def print_summary_table(results):
    """Print a nice summary table to console."""
    # Detect drift type
    drift_type = results.get('drift_type', 'unknown')
    
    # Legacy support
    if 'noise_levels' in results:
        drift_levels = results['noise_levels']
        drift_label = 'Noise'
    elif 'blur_levels' in results:
        drift_levels = results['blur_levels']
        drift_label = 'Blur σ'
    else:
        drift_levels = results['drift_levels']
        drift_label = drift_type.capitalize()
    
    accuracy = np.array(results['accuracy']) * 100
    input_mmd = results['input_mmd']
    layer_names = list(results['layer_mmd'].keys())
    
    print("\n" + "="*80)
    print("DETAILED RESULTS TABLE")
    print("="*80)
    
    header = f"{drift_label:<10} {'Accuracy':<12} {'Input MMD':<12}"
    for name in layer_names:
        header += f" {name+' MMD':<12}"
    print(header)
    print("-"*80)
    
    for i, level in enumerate(drift_levels):
        row = f"{level:<10.1f} {accuracy[i]:<12.2f} {input_mmd[i]:<12.4f}"
        for name in layer_names:
            row += f" {results['layer_mmd'][name][i]:<12.4f}"
        print(row)
    
    print("="*80)


def main():
    print("Loading drift experiment results...")
    with open('drift_results_cifar.pkl', 'rb') as f:
    #drift_results.pkl
        results = pickle.load(f)
    
    # Detect drift type
    drift_type = results.get('drift_type', 'unknown')
    
    # Legacy support
    if 'noise_levels' in results:
        drift_levels = results['noise_levels']
    elif 'blur_levels' in results:
        drift_levels = results['blur_levels']
    else:
        drift_levels = results['drift_levels']
    
    layer_names = list(results['layer_mmd'].keys())
    print(f"Loaded results for {len(drift_levels)} {drift_type} drift levels")
    print(f"Monitored layers: {layer_names}")
    
    print_summary_table(results)
    
    print("\nGenerating visualizations...")
    
    print("1. Main drift comparison plot...")
    plot_drift_curves(results, output_file='drift_comparison.png')
    
    print("2. Layer-wise comparison plot...")
    plot_layer_comparison(results, output_file='layer_comparison.png')
    
    print("3. Computing correlation analysis...")
    compute_correlations(results, output_file='correlation_analysis.txt')
    
    print("\n" + "="*80)
    print("✓ Visualization complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - drift_comparison.png (main results)")
    print("  - layer_comparison.png (detailed layer analysis)")
    print("  - correlation_analysis.txt (statistical analysis)")
    print("\nNext steps:")
    print("  1. Review drift_comparison.png - does activation MMD track accuracy better?")
    print("  2. Check correlation_analysis.txt - which method correlates best?")
    print("  3. Examine layer_comparison.png - which layer is most informative?")
    print("="*80)


if __name__ == '__main__':
    main()