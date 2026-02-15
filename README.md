# Glottis Segmentation (nnU-Net v2) + Calibration-Independent Area Measurement

This project segments the glottis region from endoscopic images and computes a scale-invariant,
calibration-independent glottal area metric.

## Problem
Pixel-based area measurements change with camera–tissue distance and zoom, making direct
comparison unreliable when pixel-to-mm calibration is unavailable.

## Method
Pipeline:
1) Glottis segmentation with nnU-Net v2 (trained on the BAGLS dataset)
2) Pixel area computation (A) from the predicted binary mask
3) Principal Component Analysis (PCA) on mask foreground pixels to estimate the dominant anatomical axis length (L)
4) Scale-invariant normalized area metric: A / L²

## Dataset & Splits
- Original dataset: BAGLS (Benchmark for Automatic Glottis Segmentation)
- Training set: a randomly sampled subset of 6,000 image–mask pairs extracted from the original dataset.
- Test set (Test50): 50 randomly selected samples from the dataset's test split (used for reporting metrics).
> Note: The dataset itself is not included in this repository. This repo contains code and instructions to reproduce the pipeline.


## Results (from report)
- Test set metrics reported for 50 samples and for positive-only samples (A_gt > 0)

## Repository Structure
- `notebooks/` : Kaggle notebooks (training / inference / evaluation)
- `src/`       : reusable scripts (dataset prep, inference, evaluation)
- `docs/`      : report and method notes
- `assets/`    : figures used in README

## How to Run (planned)
- Add environment setup + scripts after importing Kaggle code

## Citation
If you use this repository, please cite the BAGLS dataset and nnU-Net:
- Isensee et al., nnU-Net (Nature Methods, 2021)
- Gómez et al., BAGLS dataset (Kaggle, 2020)

## Data availability
Due to dataset size/licensing, images and masks are not pushed to GitHub. Use the Kaggle dataset link to download and place files locally.
