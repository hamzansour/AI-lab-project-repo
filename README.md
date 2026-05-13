# AI Lab Final Project (Ready-to-Present)

This repository now contains a complete final-project implementation using the **MNIST dataset** (one of the allowed options in your prompt).

## What is included

- Dataset option selected: **MNIST classification**
- Comparison of **two algorithms**:
  - Logistic Regression
  - Random Forest
- Automatic generation of:
  - Metrics JSON
  - Accuracy comparison plot
  - Confusion matrices
  - Written report (`artifacts/report.md`)

## Project structure

- `src/mnist_project.py`: end-to-end training, evaluation, and report generation
- `requirements.txt`: Python dependencies
- `artifacts/`: generated outputs for your report/presentation

## Quick start

1. Create and activate a virtual environment
2. Install dependencies
3. Run the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/mnist_project.py
```

## Dataset usage

The script supports two sources:

1. **Kaggle MNIST CSV** (preferred for your class prompt)
   - Put `mnist_train.csv` in `data/`
2. **Offline demo fallback**
   - Run with `--use-demo-data` to use sklearn digits data if you need a quick local demo

## What to present

- Problem: image digit classification (0–9)
- Why these two algorithms were chosen
- Accuracy comparison (from `artifacts/accuracy_comparison.png`)
- Confusion matrix insights
- Final conclusion from `artifacts/report.md`

## Optional run controls

```bash
python src/mnist_project.py --sample-size 20000 --test-size 0.2
```

Use `--sample-size` to speed up training if needed.

If you want a fully offline run:

```bash
python src/mnist_project.py --use-demo-data --sample-size 1500
```
