#!/usr/bin/env python3
"""Perturb SNPs promoted by embedding reweighting for Figure 4 version 2."""

from __future__ import annotations

import argparse
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metric_values(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    keep = np.isfinite(y) & np.isfinite(pred)
    y = y[keep]
    pred = pred[keep]
    pearson = np.nan
    if len(y) >= 3 and np.std(y) > 0 and np.std(pred) > 0:
        pearson = float(np.corrcoef(y, pred)[0, 1])
    return {
        "n": int(len(y)),
        "pearson_r": pearson,
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
        "rmse": float(math.sqrt(mean_squared_error(y, pred))) if len(y) else np.nan,
        "mae": float(mean_absolute_error(y, pred)) if len(y) else np.nan,
    }


def string_array(z, keys):
    for key in keys:
        if key in z.files:
            return np.asarray(z[key]).astype(str)
    return None


def load_feature_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float32)
    samples = np.asarray(z["samples"]).astype(str)
    snp_ids = string_array(z, ["snp_id", "snp_ids", "variant_id", "feature_names"])
    if snp_ids is None:
        raise RuntimeError(f"missing SNP ids in {path}")
    return X, samples, snp_ids


def align_xy(X_all, samples_all, folds_path: Path):
    folds = pd.read_csv(folds_path, sep="\t")
    folds["sample"] = folds["sample"].astype(str)
    sample_to_idx = {sample: idx for idx, sample in enumerate(samples_all)}
    idx = np.asarray([sample_to_idx[sample] for sample in folds["sample"]], dtype=np.int64)
    y = pd.to_numeric(folds["value"], errors="coerce").to_numpy(float)
    keep = np.isfinite(y)
    return X_all[idx[keep], :], y[keep], folds.loc[keep].reset_index(drop=True)


def impute_train_test(Xtr, Xte):
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)
    finite = np.isfinite(Xtr)
    n = finite.sum(axis=0).astype(np.float32)
    sums = np.where(finite, Xtr, 0.0).sum(axis=0)
    means = np.divide(sums, n, out=np.zeros_like(sums, dtype=np.float32), where=n > 0)
    Xtr_imp = np.where(finite, Xtr, means).astype(np.float32)
    Xte_imp = np.where(np.isfinite(Xte), Xte, means).astype(np.float32)
    return Xtr_imp, Xte_imp, means.astype(np.float32)


def load_effect(path: Path, snp_ids) -> pd.DataFrame:
    usecols = ["snp_id", "effect", "chr", "bp"]
    frame = pd.read_csv(path, sep="\t", compression="infer", usecols=lambda c: c in usecols)
    frame["snp_id"] = frame["snp_id"].astype(str)
    frame["effect"] = pd.to_numeric(frame["effect"], errors="coerce")
    frame = frame.drop_duplicates("snp_id").set_index("snp_id")
    aligned = frame.reindex(pd.Index(np.asarray(snp_ids).astype(str), name="snp_id")).reset_index()
    if aligned["effect"].isna().any():
        raise RuntimeError(f"effect table does not align to feature SNPs: {path}")
    return aligned


def random_chrom_matched(chroms, selected, rng):
    selected = np.asarray(selected, dtype=np.int64)
    if selected.size == 0:
        return selected
    chroms = np.asarray(chroms).astype(str)
    selected_set = set(int(x) for x in selected)
    out = []
    for chrom in pd.Series(chroms[selected]).value_counts().index:
        need = int(np.sum(chroms[selected] == chrom))
        pool = np.where(chroms == chrom)[0]
        pool = np.asarray([int(x) for x in pool if int(x) not in selected_set], dtype=np.int64)
        if pool.size >= need:
            out.extend(rng.choice(pool, size=need, replace=False).tolist())
        else:
            fallback = np.where(chroms == chrom)[0]
            out.extend(rng.choice(fallback, size=need, replace=fallback.size < need).tolist())
    return np.asarray(out, dtype=np.int64)


def mask_columns(window_idx, n_components):
    window_idx = np.asarray(window_idx, dtype=np.int64)
    if window_idx.size == 0:
        return np.array([], dtype=np.int64)
    return (window_idx[:, None] * int(n_components) + np.arange(int(n_components), dtype=np.int64)).ravel()


def build_evaluations(rank_values, raw_abs, chroms, fractions, random_reps, seed, exclude_raw_top1=True, top_strategy="reweighting_top", random_strategy="reweighting_random_chrom_matched"):
    n_windows = len(rank_values)
    raw_top1_n = max(1, int(math.ceil(0.01 * n_windows)))
    raw_top1 = set(np.argsort(-raw_abs)[:raw_top1_n].tolist())
    order = [idx for idx in np.argsort(-rank_values).tolist() if (idx not in raw_top1 or not exclude_raw_top1)]
    rng = np.random.default_rng(seed)
    evals = [{"strategy": "baseline", "fraction": 0.0, "replicate": 0, "window_idx": np.array([], dtype=np.int64)}]
    for frac in fractions:
        k = max(1, int(math.ceil(float(frac) * n_windows)))
        selected = np.asarray(order[:k], dtype=np.int64)
        evals.append({"strategy": top_strategy, "fraction": float(frac), "replicate": 0, "window_idx": selected})
        for rep in range(1, int(random_reps) + 1):
            evals.append(
                {
                    "strategy": random_strategy,
                    "fraction": float(frac),
                    "replicate": rep,
                    "window_idx": random_chrom_matched(chroms, selected, rng),
                }
            )
    return evals


def job_table_from_existing(root: Path) -> pd.DataFrame:
    metrics_path = root / "tables" / "perturbation_all" / "embedding_perturbation_all_metrics.tsv.gz"
    usecols = [
        "task_id",
        "trait",
        "method",
        "feature",
        "effect_feature_label",
        "embedding_model",
        "window",
        "feature_npz",
        "folds",
        "original_embedding_pearson_r",
        "original_raw_pearson_r",
        "original_delta_pearson_r_vs_raw",
    ]
    frame = pd.read_csv(metrics_path, sep="\t", compression="infer", usecols=usecols)
    jobs = frame.drop_duplicates(["trait", "method", "embedding_model", "window"]).copy()
    jobs = jobs[jobs["method"].astype(str).eq("rf") & pd.to_numeric(jobs["window"], errors="coerce").eq(200)].copy()
    return jobs.sort_values(["trait", "embedding_model"], kind="stable").reset_index(drop=True)


def effect_paths(root: Path, trait: str, method: str, model: str, window: int):
    base = root / "method_effects_all" / trait / method
    label = f"{model}_w{int(window)}"
    return base / f"{trait}.{method}.{label}.effects.tsv.gz", base / f"{trait}.{method}.raw.effects.tsv.gz"


def safe_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def shard_path(shard_dir: Path, job: pd.Series, out_name: str) -> Path:
    name = ".".join(
        [
            out_name,
            safe_token(job["trait"]),
            safe_token(job["method"]),
            safe_token(job["embedding_model"]),
            f"w{int(job['window'])}",
            "tsv.gz",
        ]
    )
    return shard_dir / name


def run_job(root: Path, job: pd.Series, args) -> pd.DataFrame:
    trait = str(job["trait"])
    method = str(job["method"])
    model = str(job["embedding_model"])
    window = int(job["window"])
    feature_npz = Path(str(job["feature_npz"]))
    folds_path = Path(str(job["folds"]))
    effect_path, raw_path = effect_paths(root, trait, method, model, window)

    X_all, samples_all, snp_ids = load_feature_npz(feature_npz)
    X, y, folds = align_xy(X_all, samples_all, folds_path)
    n_windows = len(snp_ids)
    if X.shape[1] % n_windows != 0:
        raise RuntimeError(f"feature columns are not windows x components: {feature_npz}")
    n_components = X.shape[1] // n_windows

    embed = load_effect(effect_path, snp_ids)
    raw = load_effect(raw_path, snp_ids)
    embed_abs = np.abs(embed["effect"].to_numpy(float))
    raw_abs = np.abs(raw["effect"].to_numpy(float))
    embed_share = embed_abs / max(float(embed_abs.sum()), 1e-300)
    raw_share = raw_abs / max(float(raw_abs.sum()), 1e-300)
    weight_gain = embed_share - raw_share
    chroms = embed["chr"].fillna("unknown").astype(str).to_numpy()

    seed = int(args.seed) + int(pd.util.hash_pandas_object(pd.Series([trait, model, window])).sum() % 1_000_000)
    if args.selection_mode == "embedding":
        rank_values = embed_abs
        top_strategy = "embedding_top"
        random_strategy = "random_chrom_matched"
        exclude_raw_top1 = False
    else:
        rank_values = weight_gain
        top_strategy = "reweighting_top"
        random_strategy = "reweighting_random_chrom_matched"
        exclude_raw_top1 = args.exclude_raw_top1
    evals = build_evaluations(rank_values, raw_abs, chroms, args.fractions, args.random_reps, seed, exclude_raw_top1, top_strategy, random_strategy)
    pred_by_eval = {idx: np.full(len(y), np.nan, dtype=float) for idx in range(len(evals))}

    for fold in sorted(int(x) for x in folds["fold"].unique()):
        tr = np.where(folds["fold"].astype(int).to_numpy() != fold)[0]
        te = np.where(folds["fold"].astype(int).to_numpy() == fold)[0]
        Xtr, Xte, means = impute_train_test(X[tr, :], X[te, :])
        model_fit = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_features=args.rf_max_features,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=args.threads,
            random_state=int(args.seed) + fold,
        )
        model_fit.fit(Xtr, y[tr])
        for idx, ev in enumerate(evals):
            cols = mask_columns(ev["window_idx"], n_components)
            if cols.size:
                Xmask = Xte.copy()
                Xmask[:, cols] = means[cols]
                pred = model_fit.predict(Xmask)
            else:
                pred = model_fit.predict(Xte)
            pred_by_eval[idx][te] = pred
        print(f"[INFO] {trait} {model} fold={fold} done", flush=True)

    baseline = metric_values(y, pred_by_eval[0])
    baseline_r = baseline["pearson_r"]
    rows = []
    for idx, ev in enumerate(evals):
        met = metric_values(y, pred_by_eval[idx])
        rows.append(
            {
                "task_id": job.get("task_id", np.nan),
                "trait": trait,
                "method": method,
                "feature": str(job["feature"]),
                "effect_feature_label": f"{model}_w{window}",
                "embedding_model": model,
                "window": window,
                "strategy": ev["strategy"],
                "fraction": float(ev["fraction"]),
                "replicate": int(ev["replicate"]),
                "n_masked_windows": int(len(ev["window_idx"])),
                "baseline_pearson_r": baseline_r,
                "pearson_drop_vs_baseline": float(baseline_r - met["pearson_r"]) if np.isfinite(baseline_r) and np.isfinite(met["pearson_r"]) else np.nan,
                "original_embedding_pearson_r": pd.to_numeric(job.get("original_embedding_pearson_r", np.nan), errors="coerce"),
                "original_raw_pearson_r": pd.to_numeric(job.get("original_raw_pearson_r", np.nan), errors="coerce"),
                "original_delta_pearson_r_vs_raw": pd.to_numeric(job.get("original_delta_pearson_r_vs_raw", np.nan), errors="coerce"),
                **met,
            }
        )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["trait", "method", "embedding_model", "window", "strategy", "fraction"]
    return (
        metrics[metrics["strategy"].ne("baseline")]
        .groupby(keys, as_index=False)
        .agg(
            mean_pearson_drop=("pearson_drop_vs_baseline", "mean"),
            q025_pearson_drop=("pearson_drop_vs_baseline", lambda x: float(np.quantile(pd.to_numeric(x, errors="coerce").dropna(), 0.025))),
            q975_pearson_drop=("pearson_drop_vs_baseline", lambda x: float(np.quantile(pd.to_numeric(x, errors="coerce").dropna(), 0.975))),
            n_replicates=("pearson_drop_vs_baseline", "count"),
            original_delta_pearson_r_vs_raw=("original_delta_pearson_r_vs_raw", "first"),
        )
    )


def combine_shards(shard_dir: Path, metrics_path: Path, summary_path: Path) -> int:
    shards = sorted(shard_dir.glob("*.tsv.gz"))
    if not shards:
        print(f"[WARN] no completed shards under {shard_dir}", flush=True)
        return 0
    metrics = pd.concat((pd.read_csv(path, sep="\t", compression="infer") for path in shards), ignore_index=True)
    metrics.to_csv(metrics_path, sep="\t", index=False, compression="gzip")
    summarize(metrics).to_csv(summary_path, sep="\t", index=False)
    print(f"[INFO] combined {len(shards)} shards into {metrics_path}", flush=True)
    print(f"[INFO] wrote {summary_path}", flush=True)
    return len(shards)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-name", default="reweighting_snp_perturbation")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--random-reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--rf-max-features", default="sqrt")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--selection-mode", choices=["reweighting", "embedding"], default="reweighting")
    parser.add_argument("--exclude-raw-top1", action="store_true", default=True)
    parser.add_argument("--include-raw-top1", dest="exclude_raw_top1", action="store_false")
    parser.add_argument("--job-index", type=int, nargs="+", help="Run only zero-based job indices from the job table.")
    parser.add_argument("--max-jobs", type=int, help="Run at most this many incomplete jobs.")
    parser.add_argument("--force", action="store_true", help="Recompute completed job shards.")
    parser.add_argument("--combine-only", action="store_true", help="Only combine existing shards into metrics and summary tables.")
    args = parser.parse_args()

    outdir = args.root / "tables" / "perturbation_all"
    outdir.mkdir(parents=True, exist_ok=True)
    shard_dir = outdir / f"{args.out_name}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / f"{args.out_name}_metrics.tsv.gz"
    summary_path = outdir / f"{args.out_name}_summary.tsv"
    jobs = job_table_from_existing(args.root)
    jobs.to_csv(outdir / f"{args.out_name}_jobs.tsv", sep="\t", index=False)

    if args.combine_only:
        combine_shards(shard_dir, metrics_path, summary_path)
        return

    if args.job_index:
        wanted = set(int(x) for x in args.job_index)
        jobs = jobs.iloc[[idx for idx in range(len(jobs)) if idx in wanted]].copy()

    completed_now = 0
    for job_number, (_, job) in enumerate(jobs.iterrows(), start=1):
        path = shard_path(shard_dir, job, args.out_name)
        if path.exists() and not args.force:
            print(f"[INFO] skipping completed shard {path.name}", flush=True)
            continue
        if args.max_jobs is not None and completed_now >= int(args.max_jobs):
            print(f"[INFO] reached --max-jobs {args.max_jobs}", flush=True)
            break
        print(f"[INFO] running reweighting perturbation {job['trait']} {job['embedding_model']} w{job['window']}", flush=True)
        result = run_job(args.root, job, args)
        result.to_csv(path, sep="\t", index=False, compression="gzip")
        completed_now += 1
        print(f"[INFO] wrote shard {job_number}/{len(jobs)}: {path}", flush=True)

    combine_shards(shard_dir, metrics_path, summary_path)


if __name__ == "__main__":
    main()
