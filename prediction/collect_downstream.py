#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import yaml
except Exception as exc:
    raise SystemExit("PyYAML is required. Run in a conda env with yaml installed.") from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


EMBEDDING_MODELS = ("evo2", "dnabert2", "plantcad", "agront")
METRIC_COLUMNS = ["pearson_r", "r2", "rmse", "mae", "feature_ncol", "n"]


def read_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def natural_chr_key(x):
    s = str(x).replace("chr", "").replace("Chr", "")
    try:
        return int(s)
    except Exception:
        return s


def as_float_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_feature_name(feature):
    raw = str(feature)
    f = raw.lower()
    if f in {"snp", "raw", "raw_dosage", "raw_selected", "selected_5k"} or f.startswith("raw_"):
        return {
            "feature_class": "raw",
            "embedding_model": "raw",
            "window": 0,
            "pca_components": np.nan,
            "aggregation": "",
            "canonical_feature": "raw",
        }

    model = None
    for m in EMBEDDING_MODELS:
        if f.startswith(m):
            model = m
            break
    win_match = re.search(r"(?:^|_)w(\d+)(?:_|$)", f)
    pca_match = re.search(r"_pca(\d+)", f)
    agg = ""
    for candidate in ("flatten", "mean_std", "mean"):
        if candidate in f:
            agg = candidate
            break
    if model and win_match:
        win = int(win_match.group(1))
        return {
            "feature_class": "embedding",
            "embedding_model": model,
            "window": win,
            "pca_components": int(pca_match.group(1)) if pca_match else np.nan,
            "aggregation": agg,
            "canonical_feature": f"{model}_w{win}",
        }
    return {
        "feature_class": "other",
        "embedding_model": model or "unknown",
        "window": np.nan,
        "pca_components": int(pca_match.group(1)) if pca_match else np.nan,
        "aggregation": agg,
        "canonical_feature": raw,
    }


def parse_metric_path(path):
    base = os.path.basename(path)
    stem = base
    if stem.endswith(".metrics.tsv"):
        stem = stem[: -len(".metrics.tsv")]
    parts = stem.split(".")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def read_metrics_file(path):
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return df
    trait_from_path, method_from_path = parse_metric_path(path)
    if "trait" not in df.columns or df["trait"].isna().all():
        df["trait"] = trait_from_path
    if "method" not in df.columns or df["method"].isna().all():
        df["method"] = method_from_path
    if "feature" not in df.columns:
        parent = os.path.basename(os.path.dirname(path))
        df["feature"] = parent
    return df


def collect_one_run(run_name, run_cfg):
    pieces = []
    summary = str(run_cfg.get("summary", "") or "")
    if summary and os.path.exists(summary):
        df = pd.read_csv(summary, sep="\t")
        df["metric_source"] = summary
        pieces.append(df)

    pattern = str(run_cfg.get("metrics_glob", "") or "")
    files = sorted(glob.glob(pattern, recursive=True))
    for path in files:
        try:
            df = read_metrics_file(path)
            if not df.empty:
                df["metric_source"] = path
                pieces.append(df)
        except Exception as exc:
            print(f"[WARN] failed to read metric file {path}: {exc}", file=sys.stderr)

    if not pieces:
        print(f"[WARN] no metric tables found for run {run_name}", file=sys.stderr)
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True, sort=False)
    out["source_run"] = run_name
    out["expected_embedding"] = str(run_cfg.get("expected_embedding", run_name))

    parsed = out["feature"].map(parse_feature_name).apply(pd.Series)
    out = pd.concat([out, parsed], axis=1)
    expected = str(run_cfg.get("expected_embedding", run_name)).lower()
    keep = (out["feature_class"] == "raw") | (out["embedding_model"] == expected)
    removed = int((~keep).sum())
    if removed:
        print(f"[INFO] filtered {removed} off-target feature rows from {run_name}", file=sys.stderr)
    out = out.loc[keep].copy()
    out = as_float_columns(out, METRIC_COLUMNS)
    out["method"] = out["method"].astype(str)
    out["trait"] = out["trait"].astype(str)
    out = out.drop_duplicates(["source_run", "trait", "feature", "method"], keep="last")
    return out


def collect_gp_metrics(cfg):
    runs = cfg.get("gp_runs", {})
    pieces = []
    for run_name, run_cfg in runs.items():
        df = collect_one_run(run_name, run_cfg)
        if not df.empty:
            pieces.append(df)
    if not pieces:
        raise RuntimeError("No GP metrics were collected.")
    gp = pd.concat(pieces, ignore_index=True, sort=False)
    gp = gp.sort_values(["trait", "method", "source_run", "feature_class", "window", "feature"])
    return gp


def compute_delta_vs_raw(gp):
    raw = gp.loc[gp["feature_class"] == "raw", ["source_run", "trait", "method", "pearson_r", "r2", "rmse", "mae"]].copy()
    raw = raw.rename(
        columns={
            "pearson_r": "raw_pearson_r",
            "r2": "raw_r2",
            "rmse": "raw_rmse",
            "mae": "raw_mae",
        }
    )
    merged = gp.merge(raw, on=["source_run", "trait", "method"], how="left")
    merged["delta_pearson_r_vs_raw"] = merged["pearson_r"] - merged["raw_pearson_r"]
    merged["delta_r2_vs_raw"] = merged["r2"] - merged["raw_r2"]
    merged["delta_rmse_vs_raw"] = merged["rmse"] - merged["raw_rmse"]
    merged["relative_rmse_reduction_vs_raw"] = (merged["raw_rmse"] - merged["rmse"]) / merged["raw_rmse"]
    return merged


def read_h2(cfg):
    h2_cfg = cfg.get("heritability", {})
    path = str(h2_cfg.get("gain_vs_snp") or h2_cfg.get("by_feature_trait") or "")
    if not path or not os.path.exists(path):
        print("[WARN] no heritability table found", file=sys.stderr)
        return pd.DataFrame()
    h2 = pd.read_csv(path, sep="\t")
    parsed = h2["feature"].map(parse_feature_name).apply(pd.Series)
    h2 = pd.concat([h2, parsed], axis=1)
    for col in ["h2", "h2_snp", "delta_h2_vs_snp", "relative_gain_vs_snp", "sigma_g2", "sigma_e2"]:
        if col in h2.columns:
            h2[col] = pd.to_numeric(h2[col], errors="coerce")
    h2["trait"] = h2["trait"].astype(str)
    return h2


def join_gp_h2(gp_delta, h2):
    if h2.empty:
        return gp_delta.copy()
    keep_cols = [
        "trait",
        "canonical_feature",
        "h2",
        "h2_snp",
        "delta_h2_vs_snp",
        "relative_gain_vs_snp",
        "sigma_g2",
        "sigma_e2",
        "converged",
        "bound_hit",
    ]
    keep_cols = [c for c in keep_cols if c in h2.columns]
    h2_small = h2[keep_cols].drop_duplicates(["trait", "canonical_feature"], keep="last")
    return gp_delta.merge(h2_small, on=["trait", "canonical_feature"], how="left")


def summarize_best(gp_h2):
    emb = gp_h2.loc[gp_h2["feature_class"] == "embedding"].copy()
    if emb.empty:
        return pd.DataFrame()
    sort_cols = ["trait", "method", "delta_pearson_r_vs_raw", "pearson_r"]
    emb = emb.sort_values(sort_cols, ascending=[True, True, False, False])
    best = emb.groupby(["trait", "method"], as_index=False).head(1).copy()
    best = best.sort_values(["trait", "method"])
    return best


def summarize_trait_response(gp_h2, h2):
    rows = []
    traits = sorted(gp_h2["trait"].dropna().astype(str).unique())
    for trait in traits:
        sub = gp_h2[(gp_h2["trait"] == trait) & (gp_h2["feature_class"] == "embedding")].copy()
        if sub.empty:
            continue
        best = sub.sort_values(["delta_pearson_r_vs_raw", "pearson_r"], ascending=[False, False]).head(1)
        h2_snp = np.nan
        best_h2_delta = np.nan
        if not h2.empty:
            raw_h2 = h2[(h2["trait"] == trait) & (h2["canonical_feature"] == "raw")]
            if not raw_h2.empty:
                h2_snp = float(raw_h2["h2"].iloc[0])
            emb_h2 = h2[(h2["trait"] == trait) & (h2["feature_class"] == "embedding")]
            if not emb_h2.empty and "delta_h2_vs_snp" in emb_h2.columns:
                best_h2_delta = float(emb_h2["delta_h2_vs_snp"].max())
        best_delta = float(best["delta_pearson_r_vs_raw"].iloc[0]) if not best.empty else np.nan
        best_r = float(best["pearson_r"].iloc[0]) if not best.empty else np.nan
        h2_band = "unknown"
        if np.isfinite(h2_snp):
            if h2_snp < 0.3:
                h2_band = "low"
            elif h2_snp < 0.6:
                h2_band = "medium"
            else:
                h2_band = "high"
        if np.isfinite(best_delta) and best_delta >= 0.02:
            response = "strong_prediction_gain"
        elif np.isfinite(best_delta) and best_delta >= 0.005:
            response = "modest_prediction_gain"
        elif np.isfinite(best_h2_delta) and best_h2_delta >= 0.01:
            response = "h2_gain_only"
        else:
            response = "little_gain"
        rows.append(
            {
                "trait": trait,
                "h2_snp": h2_snp,
                "h2_band": h2_band,
                "best_delta_pearson_r_vs_raw": best_delta,
                "best_embedding_pearson_r": best_r,
                "best_delta_h2_vs_snp": best_h2_delta,
                "response_class": response,
                "best_source_run": best["source_run"].iloc[0] if not best.empty else "",
                "best_feature": best["feature"].iloc[0] if not best.empty else "",
                "best_method": best["method"].iloc[0] if not best.empty else "",
                "positive_embedding_results": int((sub["delta_pearson_r_vs_raw"] > 0).sum()),
                "tested_embedding_results": int(sub["delta_pearson_r_vs_raw"].notna().sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["response_class", "best_delta_pearson_r_vs_raw"], ascending=[True, False])
    return out


def build_feature_dimension_table(cfg, gp):
    rows = []
    seen = set()
    for run_name, run_cfg in cfg.get("gp_runs", {}).items():
        raw_path = str(run_cfg.get("raw_feature_npz", "") or "")
        if raw_path and raw_path not in seen:
            rows.append({"source_run": run_name, "feature_label": "raw", "feature_npz": raw_path})
            seen.add(raw_path)
        for label, path in (run_cfg.get("feature_npz") or {}).items():
            if path and path not in seen:
                rows.append({"source_run": run_name, "feature_label": label, "feature_npz": path})
                seen.add(path)
    dims = pd.DataFrame(rows)
    if dims.empty:
        return dims
    parsed = dims["feature_label"].map(parse_feature_name).apply(pd.Series)
    dims = pd.concat([dims, parsed], axis=1)
    ncol = (
        gp.groupby("canonical_feature")["feature_ncol"]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
        .reset_index()
        .rename(columns={"feature_ncol": "feature_ncol_from_metrics"})
    )
    dims = dims.merge(ncol, on="canonical_feature", how="left")
    dims["file_exists"] = dims["feature_npz"].map(os.path.exists)
    dims["file_size_gb"] = dims["feature_npz"].map(lambda p: os.path.getsize(p) / 1e9 if os.path.exists(p) else np.nan)
    return dims


def ordered_features(cfg):
    feats = ["raw"]
    for model in EMBEDDING_MODELS:
        for win in cfg.get("windows", [200, 500, 1000]):
            feats.append(f"{model}_w{int(win)}")
    return feats


def save_heatmaps(gp_h2, cfg, plot_dir):
    traits = list(cfg.get("traits") or sorted(gp_h2["trait"].unique()))
    feats = ordered_features(cfg)
    methods = list(cfg.get("plots", {}).get("methods", sorted(gp_h2["method"].unique())))
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="paper")
    for method in methods:
        sub = gp_h2[gp_h2["method"] == method].copy()
        if sub.empty:
            continue
        mat = sub.groupby(["trait", "canonical_feature"])["pearson_r"].max().unstack()
        mat = mat.reindex(index=traits, columns=[f for f in feats if f in mat.columns])
        if mat.empty:
            continue
        fig_w = max(8.0, 0.55 * mat.shape[1] + 2.5)
        fig_h = max(6.0, 0.25 * mat.shape[0] + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        if HAS_SEABORN:
            sns.heatmap(mat, ax=ax, cmap="viridis", linewidths=0.15, linecolor="white", cbar_kws={"label": "Pearson r"})
        else:
            im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax, label="Pearson r")
            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_xticklabels(mat.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_yticklabels(mat.index)
        ax.set_title(f"Prediction accuracy: {method}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"heatmap_pearson_{method}.png"), dpi=300)
        fig.savefig(os.path.join(plot_dir, f"heatmap_pearson_{method}.pdf"))
        plt.close(fig)


def save_delta_boxplot(gp_h2, plot_dir):
    emb = gp_h2[gp_h2["feature_class"] == "embedding"].copy()
    emb = emb[np.isfinite(emb["delta_pearson_r_vs_raw"])]
    if emb.empty:
        return
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.boxplot(data=emb, x="embedding_model", y="delta_pearson_r_vs_raw", hue="method", ax=ax, fliersize=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Delta Pearson r vs raw SNP")
        ax.set_xlabel("")
        ax.set_title("Embedding gain over raw SNP by predictor")
        ax.legend(title="Method", ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "delta_pearson_boxplot.png"), dpi=300)
        fig.savefig(os.path.join(plot_dir, "delta_pearson_boxplot.pdf"))
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = [g["delta_pearson_r_vs_raw"].dropna().values for _, g in emb.groupby("embedding_model")]
        labels = [k for k, _ in emb.groupby("embedding_model")]
        ax.boxplot(groups, labels=labels)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Delta Pearson r vs raw SNP")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "delta_pearson_boxplot.png"), dpi=300)
        plt.close(fig)


def save_h2_gain_plot(response, plot_dir):
    if response.empty:
        return
    df = response.copy()
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    colors = {
        "strong_prediction_gain": "#D55E00",
        "modest_prediction_gain": "#0072B2",
        "h2_gain_only": "#009E73",
        "little_gain": "#666666",
    }
    for klass, sub in df.groupby("response_class"):
        ax.scatter(
            sub["h2_snp"],
            sub["best_delta_pearson_r_vs_raw"],
            s=42,
            alpha=0.85,
            label=klass,
            color=colors.get(klass, "#666666"),
            edgecolor="white",
            linewidth=0.5,
        )
        for _, row in sub.nlargest(3, "best_delta_pearson_r_vs_raw").iterrows():
            if np.isfinite(row["best_delta_pearson_r_vs_raw"]):
                ax.text(row["h2_snp"], row["best_delta_pearson_r_vs_raw"], str(row["trait"]), fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Raw SNP h2")
    ax.set_ylabel("Best delta Pearson r vs raw SNP")
    ax.set_title("Trait architecture vs embedding gain")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "h2_vs_embedding_gain.png"), dpi=300)
    fig.savefig(os.path.join(plot_dir, "h2_vs_embedding_gain.pdf"))
    plt.close(fig)


def load_oof_table(run_name, run_cfg, cache):
    if run_name in cache:
        return cache[run_name]
    oof = str(run_cfg.get("oof", "") or "")
    if oof and os.path.exists(oof):
        usecols = ["sample", "trait", "y_true", "fold", "y_pred", "feature", "method", "feature_ncol"]
        df = pd.read_csv(oof, sep="\t", usecols=lambda c: c in usecols)
        cache[run_name] = df
        return df
    cache[run_name] = pd.DataFrame()
    return cache[run_name]


def load_individual_oof(run_cfg, feature, trait, method):
    root = str(run_cfg.get("root", "") or "")
    if not root:
        return pd.DataFrame()
    path = os.path.join(root, "oof", str(feature), f"{trait}.{method}.oof.tsv")
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t")
    return pd.DataFrame()


def find_oof_for_metric(row, cfg, cache):
    run_name = str(row["source_run"])
    run_cfg = cfg.get("gp_runs", {}).get(run_name, {})
    df = load_oof_table(run_name, run_cfg, cache)
    if not df.empty:
        sub = df[
            (df["trait"].astype(str) == str(row["trait"]))
            & (df["method"].astype(str) == str(row["method"]))
            & (df["feature"].astype(str) == str(row["feature"]))
        ].copy()
        if not sub.empty:
            return sub
    return load_individual_oof(run_cfg, row["feature"], row["trait"], row["method"])


def save_prediction_scatters(gp_h2, best, cfg, plot_dir):
    traits = list(cfg.get("plots", {}).get("prediction_traits", []))
    methods = list(
        cfg.get("plots", {}).get("prediction_methods", [])
        or cfg.get("plots", {}).get("methods", [])
        or [cfg.get("primary_method", "reml_gblup")]
    )
    if not traits or best.empty:
        return
    cache = {}
    for trait in traits:
        for method in methods:
            method = str(method)
            b = best[(best["trait"] == trait) & (best["method"] == method)].copy()
            if b.empty:
                continue
            b = b.sort_values("delta_pearson_r_vs_raw", ascending=False).iloc[0]
            raw = gp_h2[
                (gp_h2["source_run"] == b["source_run"])
                & (gp_h2["trait"] == trait)
                & (gp_h2["method"] == b["method"])
                & (gp_h2["feature_class"] == "raw")
            ]
            if raw.empty:
                continue
            raw = raw.iloc[0]
            raw_oof = find_oof_for_metric(raw, cfg, cache)
            emb_oof = find_oof_for_metric(b, cfg, cache)
            if raw_oof.empty or emb_oof.empty:
                continue
            panels = [(raw, raw_oof, "Raw SNP"), (b, emb_oof, str(b["canonical_feature"]))]
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=False, sharey=False)
            for ax, (metric_row, oof, label) in zip(axes, panels):
                y = pd.to_numeric(oof["y_true"], errors="coerce")
                p = pd.to_numeric(oof["y_pred"], errors="coerce")
                keep = y.notna() & p.notna()
                y = y[keep]
                p = p[keep]
                if y.empty or p.empty:
                    continue
                ax.scatter(y, p, s=12, alpha=0.45, color="#0072B2", edgecolor="none")
                lo = float(np.nanmin([y.min(), p.min()]))
                hi = float(np.nanmax([y.max(), p.max()]))
                pad = (hi - lo) * 0.04 if hi > lo else 1.0
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.8)
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_title(f"{label}\nr={float(metric_row['pearson_r']):.3f}, RMSE={float(metric_row['rmse']):.3g}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")
            fig.suptitle(f"{trait}: observed vs OOF predicted ({b['method']})")
            fig.tight_layout()
            outbase = os.path.join(plot_dir, f"prediction_scatter_{trait}_{b['method']}")
            fig.savefig(outbase + ".png", dpi=300)
            fig.savefig(outbase + ".pdf")
            plt.close(fig)

def write_report(cfg, gp_h2, best, response, outdir):
    path = os.path.join(outdir, "Maize1404_downstream_summary.md")
    lines = []
    lines.append("# Maize1404 downstream summary")
    lines.append("")
    lines.append("## Publishability judgement")
    lines.append("")
    lines.append(
        "The current design is publishable if the paper is framed as a fair, multi-foundation-model benchmark plus trait-architecture analysis, not merely as another SNP-window embedding test."
    )
    lines.append(
        "The strongest claims should come from same SNPs, same windows, same samples, same folds, same predictors, larger panels, and multi-crop or multi-environment extension."
    )
    lines.append("")
    if not response.empty:
        lines.append("## Trait response classes")
        lines.append("")
        top = response.sort_values("best_delta_pearson_r_vs_raw", ascending=False).head(10)
        lines.append(top.to_markdown(index=False))
        lines.append("")
    if not best.empty:
        lines.append("## Best embedding per trait and method")
        lines.append("")
        cols = [
            "trait",
            "method",
            "source_run",
            "canonical_feature",
            "pearson_r",
            "raw_pearson_r",
            "delta_pearson_r_vs_raw",
            "rmse",
        ]
        cols = [c for c in cols if c in best.columns]
        lines.append(best[cols].sort_values(["method", "delta_pearson_r_vs_raw"], ascending=[True, False]).head(40).to_markdown(index=False))
        lines.append("")
    lines.append("## Recommended paper angle")
    lines.append("")
    lines.append("1. Benchmark four genomic language/foundation models under a locked evaluation protocol.")
    lines.append("2. Link embedding gains to trait genetic architecture using SNP h2, embedding-kernel h2, and prediction gain.")
    lines.append("3. Add interpretable weight-shift figures showing which SNP regions become more or less influential after embedding.")
    lines.append("4. Strengthen with multi-kernel or stacking only if evaluated fold-wise without leakage.")
    lines.append("5. For soybean and peanut, prioritize multi-environment and GxE figures; that is likely the most defensible novelty.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description="Collect GP, h2, and plotting outputs for SNP-window embedding downstream analysis.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-prediction-plots", action="store_true")
    args = ap.parse_args()

    cfg = read_config(args.config)
    outdir = ensure_dir(str(cfg.get("outdir", "analysis_outputs/Maize1404")))
    table_dir = ensure_dir(os.path.join(outdir, "tables"))
    plot_dir = ensure_dir(os.path.join(outdir, "plots"))

    gp = collect_gp_metrics(cfg)
    gp_delta = compute_delta_vs_raw(gp)
    h2 = read_h2(cfg)
    gp_h2 = join_gp_h2(gp_delta, h2)
    best = summarize_best(gp_h2)
    response = summarize_trait_response(gp_h2, h2)
    dims = build_feature_dimension_table(cfg, gp)

    gp.to_csv(os.path.join(table_dir, "all_gp_metrics.canonical.tsv"), sep="\t", index=False)
    gp_delta.to_csv(os.path.join(table_dir, "gp_delta_vs_raw.tsv"), sep="\t", index=False)
    gp_h2.to_csv(os.path.join(table_dir, "gp_h2_join.tsv"), sep="\t", index=False)
    best.to_csv(os.path.join(table_dir, "best_embedding_by_trait_method.tsv"), sep="\t", index=False)
    response.to_csv(os.path.join(table_dir, "trait_embedding_response.tsv"), sep="\t", index=False)
    dims.to_csv(os.path.join(table_dir, "feature_dimension_storage.tsv"), sep="\t", index=False)
    if not h2.empty:
        h2.to_csv(os.path.join(table_dir, "heritability.canonical.tsv"), sep="\t", index=False)

    save_heatmaps(gp_h2, cfg, plot_dir)
    save_delta_boxplot(gp_h2, plot_dir)
    save_h2_gain_plot(response, plot_dir)
    if not args.skip_prediction_plots:
        save_prediction_scatters(gp_h2, best, cfg, plot_dir)
    report = write_report(cfg, gp_h2, best, response, outdir)

    print(f"[INFO] wrote tables to {table_dir}")
    print(f"[INFO] wrote plots to {plot_dir}")
    print(f"[INFO] wrote report to {report}")


if __name__ == "__main__":
    main()
