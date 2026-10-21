#!/usr/bin/env python3
"""Validate and summarize the paired MAF0.02 raw-versus-embedding GP results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_source(feature):
    if feature == "raw":
        return "raw", 0
    model, window = feature.rsplit("_w", 1)
    return model, int(window)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    root = Path(cfg["root"])
    models = [str(value) for value in cfg["embedding_models"]]
    windows = [int(value) for value in cfg["windows"]]
    features = ["raw"] + [f"{model}_w{window}" for model in models for window in windows]
    rows = []
    for feature in features:
        for trait in cfg["traits"]:
            for method in cfg["methods"]:
                path = root / "metrics" / feature / f"{trait}.{method}.metrics.tsv"
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"missing metric: {path}")
                frame = pd.read_csv(path, sep="\t")
                if len(frame) != 1:
                    raise RuntimeError(f"expected one metric row: {path}")
                row = frame.iloc[0].to_dict()
                if (str(row["feature"]), str(row["trait"]), str(row["method"])) != (feature, trait, method):
                    raise RuntimeError(f"metric metadata mismatch: {path}")
                for column in ("pearson_r", "r2", "rmse", "mae"):
                    if not np.isfinite(float(row[column])):
                        raise RuntimeError(f"non-finite {column}: {path}")
                source, window = parse_source(feature)
                row.update({"source": source, "window": window, "trait_label": trait.removesuffix("_BLUP")})
                rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(["method", "trait", "source", "window"])
    raw = metrics[metrics["source"] == "raw"][["trait", "method", "pearson_r"]].rename(columns={"pearson_r": "raw_pearson_r"})
    embedded = metrics[metrics["source"] != "raw"].merge(raw, on=["trait", "method"], validate="many_to_one")
    embedded["delta_pearson_r"] = embedded["pearson_r"] - embedded["raw_pearson_r"]
    trait_method = embedded.groupby(["method", "trait", "trait_label"], as_index=False).agg(
        mean_embedding_delta=("delta_pearson_r", "mean"),
        median_embedding_delta=("delta_pearson_r", "median"),
        positive_settings=("delta_pearson_r", lambda x: int((x > 0).sum())),
        n_settings=("delta_pearson_r", "size"),
    )
    summary = trait_method.groupby("method", as_index=False).agg(
        mean_delta_pearson_r=("mean_embedding_delta", "mean"),
        median_delta_pearson_r=("mean_embedding_delta", "median"),
        positive_traits=("mean_embedding_delta", lambda x: int((x > 0).sum())),
        n_traits=("trait", "size"),
    )
    out = root / "summary"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "all_metrics.tsv", sep="\t", index=False)
    embedded.to_csv(tables / "paired_embedding_vs_raw.tsv", sep="\t", index=False)
    trait_method.to_csv(tables / "trait_method_mean_delta.tsv", sep="\t", index=False)
    summary.to_csv(tables / "method_summary.tsv", sep="\t", index=False)
    lines = ["# Soybean2795 BLUP MAF >= 0.02 prediction summary", "", "All raw and embedding features use the same selected SNPs and fixed phenotype folds.", "", "| Method | Mean embedding - raw Pearson r | Positive traits |", "|---|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.method} | {row.mean_delta_pearson_r:+.6f} | {row.positive_traits}/{row.n_traits} |")
    lines.extend([
        "",
        "The combined GP RF row reflects the RF setting used by this chain (currently the archived `max_features=200` coordinate-count control). For the separate embedding-uplift estimand, use the completed `sqrt(p)` RF diagnostics: raw has 5,000 scalar SNP features (about 71 candidates per split), while each 8D embedding has 40,000 scalar coordinates (200 candidates per split).",
    ])
    (out / "prediction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
