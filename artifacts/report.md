# Final Project Report - MNIST Classification

## Dataset
- Selected dataset: MNIST (from Kaggle CSV format, with offline demo fallback available).
- Task: Multi-class classification (digits 0-9).

## Compared Algorithms
1. Logistic Regression (SAGA)
2. Random Forest Classifier

## Results
- **random_forest** accuracy: **0.9875**
- **logistic_regression** accuracy: **0.9792**

## Findings
- Best model in this run: **random_forest** with accuracy **0.9875**.
- Random Forest usually performs strongly on non-linear boundaries.
- Logistic Regression is fast and a good baseline.

## Presentation Talking Points
- Why MNIST is a standard benchmark.
- Why we compared a linear model vs. a tree ensemble.
- Trade-offs between interpretability, speed, and accuracy.

## Output Files
- `artifacts/metrics.json`
- `artifacts/accuracy_comparison.png`
- `artifacts/confusion_matrix_logistic_regression.png`
- `artifacts/confusion_matrix_random_forest.png`