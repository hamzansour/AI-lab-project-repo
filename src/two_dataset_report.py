from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
LOGISTIC_MAX_ITER = 2000


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    loader: Any
    description: str


def get_datasets() -> list[DatasetConfig]:
    return [
        DatasetConfig(
            name="Digits",
            loader=load_digits,
            description="Handwritten digit images (8x8), 10 classes.",
        ),
        DatasetConfig(
            name="Wine",
            loader=load_wine,
            description="Wine cultivar classification using 13 chemical features.",
        ),
    ]


def get_models() -> dict[str, Any]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=LOGISTIC_MAX_ITER,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_dataset(dataset: DatasetConfig) -> dict[str, Any]:
    data = dataset.loader()
    x = data.data.astype(np.float32)
    y = data.target.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = {}
    for model_name, model in get_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[model_name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        }

    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "samples": int(x.shape[0]),
        "features": int(x.shape[1]),
        "classes": int(len(np.unique(y))),
        "results": results,
    }


def build_report(evaluations: list[dict[str, Any]]) -> str:
    lines = [
        "# Final Project Report - Two Dataset Comparison",
        "",
        "## Overview",
        "This report compares **two datasets** using **two different algorithms**.",
        "",
        "## Datasets",
    ]

    for evaluation in evaluations:
        lines.extend(
            [
                f"- **{evaluation['dataset']}**: {evaluation['description']}",
                f"  - Samples: {evaluation['samples']}",
                f"  - Features: {evaluation['features']}",
                f"  - Classes: {evaluation['classes']}",
            ]
        )

    lines.extend(["", "## Model Comparison Results"])

    rows = []
    for evaluation in evaluations:
        for model_name, metrics in evaluation["results"].items():
            rows.append(
                {
                    "Dataset": evaluation["dataset"],
                    "Model": model_name,
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "Macro F1": f"{metrics['macro_f1']:.4f}",
                }
            )

    headers = ["Dataset", "Model", "Accuracy", "Macro F1"]
    lines.append(f"| {' | '.join(headers)} |")
    lines.append(f"| {' | '.join(['---'] * len(headers))} |")
    for row in rows:
        lines.append(
            f"| {row['Dataset']} | {row['Model']} | {row['Accuracy']} | {row['Macro F1']} |"
        )

    lines.extend(["", "## Findings"])

    for evaluation in evaluations:
        best_result = max(evaluation["results"].items(), key=lambda item: item[1]["accuracy"])
        lines.append(
            f"- **{evaluation['dataset']}** best model: **{best_result[0]}** "
            f"(accuracy **{best_result[1]['accuracy']:.4f}**, macro F1 **{best_result[1]['macro_f1']:.4f}**)."
        )

    lines.extend(
        [
            "",
            "## Presentation Talking Points",
            "- Why comparing multiple datasets provides a stronger evaluation.",
            "- Trade-offs between linear (Logistic Regression) and ensemble (Random Forest) models.",
            "- Which dataset was easier/harder for each model and why.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a two-dataset comparison report")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report_two_datasets.md"),
        help="Path to write the markdown report",
    )

    args = parser.parse_args()

    evaluations = [evaluate_dataset(dataset) for dataset in get_datasets()]
    report = build_report(evaluations)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
