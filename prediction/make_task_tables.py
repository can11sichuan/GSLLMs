#!/usr/bin/env python3
"""Create deterministic feature and prediction task tables for MAF0.02 GP."""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def write_table(rows, path):
    frame = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    print(f"[INFO] wrote {len(frame)} tasks: {path}")


RF_SETTINGS = [
    ("default_sqrt_leaf1", "sqrt", 1),
    ("regularized_sqrt_leaf5", "sqrt", 5),
    ("regularized_sqrt_leaf10", "sqrt", 10),
    ("fixed71_leaf5", "71", 5),
    ("fixed200_leaf5", "200", 5),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    root = Path(cfg["root"])
    task_dir = root / "tasks"
    raw = root / "raw" / "raw_selected_maf0p02.npz"
    features = [("raw", raw)]
    feature_rows = []
    task_id = 0
    for model in cfg["embedding_models"]:
        for window in cfg["windows"]:
            label = f"{model}_w{int(window)}"
            out = root / "features" / f"{model}_pca{int(cfg['pca_components'])}_flatten_w{int(window)}.npz"
            features.append((label, out))
            feature_rows.append({
                "task_id": task_id,
                "model": model,
                "window": int(window),
                "feature": label,
                "genotype_npz": cfg["genotype_embedding_npz"][model][int(window)],
                "out_features": str(out),
                "out_pca": str(root / "pca" / f"{model}_pca{int(cfg['pca_components'])}_flatten_w{int(window)}.joblib"),
                "out_meta": str(root / "meta" / f"{model}_pca{int(cfg['pca_components'])}_flatten_w{int(window)}.json"),
            })
            task_id += 1
    write_table(feature_rows, task_dir / "feature_tasks.tsv")

    prediction_rows = []
    task_id = 0
    for feature, feature_npz in features:
        for trait in cfg["traits"]:
            fold = Path(cfg["canonical_folds"]) / f"{trait}.folds.tsv"
            for method in cfg["methods"]:
                prediction_rows.append({
                    "task_id": task_id,
                    "feature": feature,
                    "trait": trait,
                    "method": method,
                    "feature_npz": str(feature_npz),
                    "folds": str(fold),
                    "out_oof": str(root / "predictions" / feature / f"{trait}.{method}.oof.tsv"),
                    "out_metrics": str(root / "metrics" / feature / f"{trait}.{method}.metrics.tsv"),
                })
                task_id += 1
    write_table(prediction_rows, task_dir / "prediction_tasks.tsv")

    rf_rows = []
    w500_features = [("raw", raw)] + [
        (
            f"{model}_w500",
            root / "features" / f"{model}_pca{int(cfg['pca_components'])}_flatten_w500.npz",
        )
        for model in cfg["embedding_models"]
    ]
    for setting, max_features, min_samples_leaf in RF_SETTINGS:
        for feature, feature_npz in w500_features:
            for trait in cfg["traits"]:
                rf_rows.append({
                    "task_id": len(rf_rows),
                    "setting": setting,
                    "max_features": max_features,
                    "min_samples_leaf": min_samples_leaf,
                    "feature": feature,
                    "feature_npz": str(feature_npz),
                    "trait": trait,
                    "folds": str(Path(cfg["canonical_folds"]) / f"{trait}.folds.tsv"),
                    "out_oof": str(root / "rf_sensitivity" / "oof" / setting / feature / f"{trait}.oof.tsv"),
                    "out_metrics": str(root / "rf_sensitivity" / "metrics" / setting / feature / f"{trait}.metrics.tsv"),
                })
    write_table(rf_rows, task_dir / "rf_sensitivity_tasks.tsv")


if __name__ == "__main__":
    main()
