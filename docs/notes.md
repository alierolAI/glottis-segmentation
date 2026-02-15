# Notes

## What this repo contains
- `notebooks/`: Original Kaggle notebook used during development
- `src/`: Minimal reproducible scripts for nnU-Net v2 training, inference and evaluation
- `configs/`: Path presets for Kaggle and local runs
- `docs/`: Report and method notes

## Key idea (calibration-independent metric)
Direct pixel area changes with camera distance/zoom. We compute:
- `A`: area in pixels from predicted mask
- `L`: dominant axis length estimated via PCA on mask foreground pixels
- scale-invariant metric: `A / L^2`

## Running order (Kaggle)
1. Prepare Test50 to 3D shape: `src/prepare_test50_fixed3d.py`
2. Train: `src/train_nnunet.py`
3. Predict: `src/predict.py`
4. Evaluate: `src/evaluate_boundary_test50.py`

## Data split used in this project
- Training: 6,000 randomly sampled image–mask pairs from BAGLS
- Evaluation: Test50 = 50 random samples taken from the BAGLS test split
