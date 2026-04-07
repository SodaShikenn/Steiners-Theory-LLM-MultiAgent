"""
Correlation Analysis and Results Summarization

Computes accuracy rates and correlation matrices between Steiner's metrics
and task correctness, reproducing Tables 2-6 from the paper.

Metrics analyzed:
  - is_correct (正誤): Whether MA got the correct answer
  - choosing_wrong_ideas (誤選択): Wrong idea selected from others
  - generating_wrong_ideas (誤生成): Wrong ideas generated
  - same_idea (同意見): Repeated same idea across agents
  - correction (修正): Wrong ideas corrected by other agents
  - novel_idea (新案): Novel ideas not in single-agent response

Usage:
    python analyze_results.py --metrics ../results/evaluation/bbh_cooperative_metrics.json
    python analyze_results.py --all_dir ../results/evaluation/
"""

import json
import os
import argparse
import numpy as np

# Column names matching the paper's Japanese labels
METRIC_NAMES = {
    "is_correct": "正誤",
    "choosing_wrong_ideas": "誤選択",
    "generating_wrong_ideas": "誤生成",
    "same_idea": "同意見",
    "correction": "修正",
    "novel_idea": "新案",
}

METRIC_KEYS = list(METRIC_NAMES.keys())


def load_metrics(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy(metrics: list) -> float:
    """Compute accuracy (percentage of correct answers)."""
    if not metrics:
        return 0.0
    correct = sum(1 for m in metrics if m.get("is_correct", 0) == 1)
    return (correct / len(metrics)) * 100


def compute_correlation_matrix(metrics: list) -> tuple:
    """Compute Pearson correlation matrix between all metric pairs.

    Returns:
        (correlation_matrix, metric_keys) where correlation_matrix is a numpy array
    """
    n = len(metrics)
    if n < 3:
        print("WARNING: Too few data points for meaningful correlation")
        return None, METRIC_KEYS

    data = np.zeros((n, len(METRIC_KEYS)))
    for i, m in enumerate(metrics):
        for j, key in enumerate(METRIC_KEYS):
            data[i, j] = m.get(key, 0)

    corr_matrix = np.corrcoef(data.T)
    return corr_matrix, METRIC_KEYS


def print_accuracy(metrics: list, label: str):
    acc = compute_accuracy(metrics)
    print(f"{label}: {acc:.1f}% ({sum(1 for m in metrics if m.get('is_correct', 0) == 1)}/{len(metrics)})")


def print_correlation_matrix(corr_matrix, keys, label: str):
    """Pretty-print a correlation matrix matching the paper's table format."""
    print(f"\n{'=' * 60}")
    print(f"Correlation Matrix: {label}")
    print(f"{'=' * 60}")

    jp_names = [METRIC_NAMES[k] for k in keys]
    header = f"{'':>8}" + "".join(f"{name:>8}" for name in jp_names)
    print(header)
    print("-" * len(header))

    for i, key in enumerate(keys):
        row = f"{jp_names[i]:>8}"
        for j in range(len(keys)):
            val = corr_matrix[i, j]
            row += f"{val:>8.2f}"
        print(row)


def analyze_single(filepath: str):
    """Analyze a single metrics JSON file."""
    metrics = load_metrics(filepath)
    label = os.path.splitext(os.path.basename(filepath))[0]

    print_accuracy(metrics, label)

    corr_matrix, keys = compute_correlation_matrix(metrics)
    if corr_matrix is not None:
        print_correlation_matrix(corr_matrix, keys, label)

    # Print metric means
    print(f"\nMetric means (n={len(metrics)}):")
    for key in METRIC_KEYS:
        values = [m.get(key, 0) for m in metrics]
        mean_val = np.mean(values)
        print(f"  {METRIC_NAMES[key]} ({key}): {mean_val:.3f}")

    return metrics, corr_matrix


def analyze_all(eval_dir: str):
    """Analyze all metrics files in a directory."""
    json_files = sorted([f for f in os.listdir(eval_dir) if f.endswith(".json")])

    if not json_files:
        print(f"No JSON metrics files found in {eval_dir}")
        return

    print("=" * 60)
    print("ACCURACY SUMMARY (Table 2 from paper)")
    print("=" * 60)

    for filename in json_files:
        filepath = os.path.join(eval_dir, filename)
        metrics = load_metrics(filepath)
        label = os.path.splitext(filename)[0].replace("_metrics", "")
        print_accuracy(metrics, label)

    # Print correlation matrices (Tables 3-6)
    for filename in json_files:
        filepath = os.path.join(eval_dir, filename)
        metrics = load_metrics(filepath)
        label = os.path.splitext(filename)[0].replace("_metrics", "")
        corr_matrix, keys = compute_correlation_matrix(metrics)
        if corr_matrix is not None:
            print_correlation_matrix(corr_matrix, keys, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Steiner's metrics and compute correlations")
    parser.add_argument("--metrics", type=str, help="Path to a single metrics JSON file")
    parser.add_argument("--all_dir", type=str, help="Directory containing all metrics JSON files")
    args = parser.parse_args()

    if args.metrics:
        analyze_single(args.metrics)
    elif args.all_dir:
        analyze_all(args.all_dir)
    else:
        parser.print_help()
