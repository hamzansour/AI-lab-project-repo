from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ACCURACY_PLOT_PADDING = 0.05
LOGISTIC_SOLVER = "lbfgs"
LOGISTIC_MAX_ITER = 1000
DEFAULT_SAMPLE_SIZE = 15000


def load_mnist(
    data_dir: Path, test_size: float, random_state: int, sample_size: int | None, use_demo: bool
) -> tuple[Any, Any, Any, Any]:
    """Load MNIST-style data and return stratified train/test splits.

    Args:
        data_dir: Directory expected to contain mnist_train.csv.
        test_size: Fraction of examples reserved for testing.
        random_state: Random seed for deterministic splitting.
        sample_size: Optional cap on number of rows used.
        use_demo: If True and MNIST CSV is missing, use sklearn digits data.

    Returns:
        Tuple of X_train, X_test, y_train, y_test.
    """
    train_csv = data_dir / "mnist_train.csv"

    if train_csv.exists():
        df = pd.read_csv(train_csv)
        y = df["label"].astype(int)
        x = df.drop(columns=["label"]).astype(np.float32) / 255.0
    elif use_demo:
        from sklearn.datasets import load_digits

        digits = load_digits()
        x = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(int)
    else:
        raise FileNotFoundError(
            "Could not find data/mnist_train.csv. Download the MNIST CSV dataset from the provided "
            "Kaggle link and place it in the data/ directory, or run with --use-demo-data."
        )

    if sample_size is not None and sample_size > 0:
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:sample_size]
            y = y.iloc[:sample_size]
        else:
            x = x[:sample_size]
            y = y[:sample_size]

    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def train_models(x_train: Any, y_train: Any) -> dict[str, Any]:
    """Train configured classification models and return fitted estimators by name."""
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=LOGISTIC_MAX_ITER,
            solver=LOGISTIC_SOLVER,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        ),
    }

    fitted = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted[name] = model

    return fitted


def evaluate_models(models: dict[str, Any], x_test: Any, y_test: Any) -> dict[str, Any]:
    """Evaluate trained models and return accuracy, class report, and confusion matrix."""
    results = {}
    for name, model in models.items():
        predictions = model.predict(x_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        cm = confusion_matrix(y_test, predictions)
        results[name] = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
    return results


def save_plots(results: dict[str, Any], output_dir: Path) -> None:
    """Create and save accuracy comparison and per-model confusion matrix plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_df = pd.DataFrame(
        {"model": list(results.keys()), "accuracy": [v["accuracy"] for v in results.values()]}
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=accuracy_df, x="model", y="accuracy", hue="model", legend=False)
    min_accuracy = float(accuracy_df["accuracy"].min())
    lower_bound = max(0.0, min_accuracy - ACCURACY_PLOT_PADDING)
    plt.ylim(lower_bound, 1.0)
    plt.title("MNIST Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=200)
    plt.close()

    for model_name, model_result in results.items():
        cm = np.array(model_result["confusion_matrix"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=200)
        plt.close()


def save_report(results: dict[str, Any], output_dir: Path) -> None:
    """Write a concise markdown report summarizing model comparison results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models_by_accuracy = sorted(results.items(), key=lambda item: item[1]["accuracy"], reverse=True)
    best_model, best_metrics = models_by_accuracy[0]

    lines = [
        "# Final Project Report - MNIST Classification",
        "",
        "## Dataset",
        "- Selected dataset: MNIST (from Kaggle CSV format, with offline demo fallback available).",
        "- Task: Multi-class classification (digits 0-9).",
        "",
        "## Compared Algorithms",
        f"1. Logistic Regression ({LOGISTIC_SOLVER})",
        "2. Random Forest Classifier",
        "",
        "## Results",
    ]

    for model_name, metrics in models_by_accuracy:
        lines.append(f"- **{model_name}** accuracy: **{metrics['accuracy']:.4f}**")

    lines.extend(
        [
            "",
            "## Findings",
            f"- Best model in this run: **{best_model}** with accuracy **{best_metrics['accuracy']:.4f}**.",
            "- Random Forest usually performs strongly on non-linear boundaries.",
            "- Logistic Regression is fast and a good baseline.",
            "",
            "## Presentation Talking Points",
            "- Why MNIST is a standard benchmark.",
            "- Why we compared a linear model vs. a tree ensemble.",
            "- Trade-offs between interpretability, speed, and accuracy.",
            "",
            "## Output Files",
            "- `artifacts/metrics.json`",
            "- `artifacts/accuracy_comparison.png`",
            "- `artifacts/confusion_matrix_logistic_regression.png`",
            "- `artifacts/confusion_matrix_random_forest.png`",
        ]
    )

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run the full CLI pipeline: load data, train, evaluate, and save artifacts."""
    parser = argparse.ArgumentParser(description="MNIST final project pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--use-demo-data", action="store_true")

    args = parser.parse_args()

    x_train, x_test, y_train, y_test = load_mnist(
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_size=args.sample_size,
        use_demo=args.use_demo_data,
    )

    models = train_models(x_train, y_train)
    results = evaluate_models(models, x_test, y_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_plots(results, args.output_dir)
    save_report(results, args.output_dir)

    print("Training complete. Best result:")
    best = max(results.items(), key=lambda item: item[1]["accuracy"])
    print(f"{best[0]} -> accuracy: {best[1]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
