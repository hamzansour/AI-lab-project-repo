# Final Project Report - Two Dataset Comparison

## Overview
This report compares **two datasets** using **two different algorithms**.

## Datasets
- **Digits**: Handwritten digit images (8x8), 10 classes.
  - Samples: 1797
  - Features: 64
  - Classes: 10
- **Wine**: Wine cultivar classification using 13 chemical features.
  - Samples: 178
  - Features: 13
  - Classes: 3

## Model Comparison Results
| Dataset | Model | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| Digits | Logistic Regression | 0.9722 | 0.9719 |
| Digits | Random Forest | 0.9694 | 0.9689 |
| Wine | Logistic Regression | 0.9722 | 0.9710 |
| Wine | Random Forest | 1.0000 | 1.0000 |

## Findings
- **Digits** best model: **Logistic Regression** (accuracy **0.9722**, macro F1 **0.9719**).
- **Wine** best model: **Random Forest** (accuracy **1.0000**, macro F1 **1.0000**).

## Presentation Talking Points
- Why comparing multiple datasets provides a stronger evaluation.
- Trade-offs between linear (Logistic Regression) and ensemble (Random Forest) models.
- Which dataset was easier/harder for each model and why.