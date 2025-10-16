# Activation-Based Drift Detection

Exploring whether monitoring neural network activations detects concept drift more effectively than input space monitoring.

## Research Questions

1. Does monitoring model activations at different layers detect drift differently than input space monitoring?
2. Do any methods correlate more closely with actual model performance degradation?

## Key Findings

- **Drift-type dependency**: Different layers respond to different drift types
- **Final layer (fc)** tracks performance well for noise-based drift
- **Input space** detects global transformations (brightness) earlier
- **Two-tier system**: Input (sensitive) + FC (specific) provides robust monitoring

## Quick Start

### MNIST Experiments
```bash
pip install -r requirements.txt
cd mnist/
python 1_train_and_baseline.py
python 2_drift_experiment.py
python 3_visualize_results.py
```

### CIFAR-10 Experiments
Train on Google Colab (GPU recommended), then run drift experiments locally.

## Drift Types Tested

- Gaussian Noise
- Gaussian Blur  
- Rotation
- Brightness

## Results

See `docs/experiment_writeup.pdf` for detailed analysis.

## Citation

Inspired by: Komorniczak & Ksieniewicz, "Unsupervised Concept Drift Detection Based on Parallel Activations of Neural Network" (2024)

## License

MIT
```

### 2. **requirements.txt**
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
tqdm>=4.65.0
```

### 3. **.gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
myenv/

# Data and models (too large for git)
*.pth
*.pkl
*.json
data/MNIST/
data/CIFAR10/
data/cifar-10-batches-py/

# Results
*.png
results/*.pkl
results/*.txt

# System
.DS_Store
.idea/
.vscode/

# Jupyter
.ipynb_checkpoints/

# Keep folder structure
!.gitkeep
