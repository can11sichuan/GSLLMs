#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import math
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)


def pearsonr_safe(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    keep = np.isfinite(y) & np.isfinite(p)
    y = y[keep]
    p = p[keep]

    if len(y) < 3:
        return np.nan
    if np.nanstd(y) == 0 or np.nanstd(p) == 0:
        return np.nan

    return float(np.corrcoef(y, p)[0, 1])


def metrics_dict(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    keep = np.isfinite(y) & np.isfinite(pred)
    y = y[keep]
    pred = pred[keep]

    if len(y) == 0:
        return {
            "n": 0,
            "pearson_r": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }

    return {
        "n": int(len(y)),
        "pearson_r": pearsonr_safe(y, pred),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
    }


def touch_outputs_newer_than_inputs(outputs, inputs, plus_seconds=10.0):
    input_mtimes = []
    for f in inputs:
        try:
            if f and os.path.exists(f):
                input_mtimes.append(os.path.getmtime(f))
        except Exception:
            pass

    target_mtime = max([time.time()] + input_mtimes) + float(plus_seconds)

    for f in outputs:
        if not os.path.exists(f):
            raise RuntimeError(f"expected output was not created: {f}")
        os.utime(f, (target_mtime, target_mtime))
        print(f"[INFO] touched output mtime: {f}", file=sys.stderr)


def load_feature_matrix(path):
    z = np.load(path, allow_pickle=True)

    if "X" not in z.files or "samples" not in z.files:
        raise RuntimeError(f"feature NPZ must contain X and samples: {path}")

    X = z["X"].astype(np.float32)
    samples = np.asarray(z["samples"]).astype(str)

    if X.shape[0] != len(samples):
        raise RuntimeError(
            f"X row number != samples length in {path}: "
            f"X={X.shape}, samples={len(samples)}"
        )

    return X, samples


def align_xy(feature_npz, folds_path):
    X_all, samples_all = load_feature_matrix(feature_npz)
    sample_to_idx = {s: i for i, s in enumerate(samples_all)}

    folds = pd.read_csv(folds_path, sep="\t")
    required = {"sample", "trait", "value", "fold"}
    missing_cols = required - set(folds.columns)
    if missing_cols:
        raise RuntimeError(f"fold file missing columns {missing_cols}: {folds_path}")

    folds["sample"] = folds["sample"].astype(str)

    missing = [s for s in folds["sample"] if s not in sample_to_idx]
    if missing:
        raise RuntimeError(
            f"{len(missing)} fold samples missing in feature matrix. "
            f"First 10: {missing[:10]}"
        )

    idx = np.asarray([sample_to_idx[s] for s in folds["sample"]], dtype=np.int64)

    X = X_all[idx, :]
    y = pd.to_numeric(folds["value"], errors="coerce").values.astype(float)

    keep = np.isfinite(y)
    folds = folds.loc[keep].reset_index(drop=True)
    X = X[keep, :]
    y = y[keep]

    if len(y) < 5:
        raise RuntimeError(f"too few usable samples after alignment: n={len(y)}")

    return X, y, folds


def impute_nonfinite_by_train_mean(Xtr, Xte):
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)

    finite_tr = np.isfinite(Xtr)

    col_sum = np.where(finite_tr, Xtr, 0.0).sum(axis=0)
    col_n = finite_tr.sum(axis=0)

    col_mean = np.zeros(Xtr.shape[1], dtype=np.float32)
    ok = col_n > 0
    col_mean[ok] = col_sum[ok] / col_n[ok]

    Xtr_imp = np.where(np.isfinite(Xtr), Xtr, col_mean)
    Xte_imp = np.where(np.isfinite(Xte), Xte, col_mean)

    return Xtr_imp.astype(np.float32), Xte_imp.astype(np.float32)


def build_kinship_from_features(Xtr, Xte, normalize_k=True):
    Xtr, Xte = impute_nonfinite_by_train_mean(Xtr, Xte)

    scaler = StandardScaler(with_mean=True, with_std=True)

    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    p = max(1, Xtr_s.shape[1])

    Ktr = (Xtr_s @ Xtr_s.T).astype(np.float64) / float(p)
    Kte = (Xte_s @ Xtr_s.T).astype(np.float64) / float(p)

    Ktr = 0.5 * (Ktr + Ktr.T)

    mean_diag_before = float(np.mean(np.diag(Ktr)))

    if normalize_k and np.isfinite(mean_diag_before) and mean_diag_before > 0:
        Ktr = Ktr / mean_diag_before
        Kte = Kte / mean_diag_before

    Ktr = 0.5 * (Ktr + Ktr.T)
    mean_diag_after = float(np.mean(np.diag(Ktr)))

    return Ktr, Kte, mean_diag_before, mean_diag_after


def reml_profile_nll_intercept(log_lambda, eigvals, Uy, U1, df_reml):
    lam = float(np.exp(log_lambda))
    denom = eigvals + lam

    if np.any(denom <= 0):
        return np.inf

    logdet = float(np.sum(np.log(denom)))

    Vinv_1_1 = float(np.sum((U1 * U1) / denom))
    Vinv_1_y = float(np.sum((U1 * Uy) / denom))

    if not np.isfinite(Vinv_1_1) or Vinv_1_1 <= 0:
        return np.inf

    mu = Vinv_1_y / Vinv_1_1

    residual_eig = Uy - U1 * mu
    quad = float(np.sum((residual_eig * residual_eig) / denom))

    if not np.isfinite(quad) or quad <= 0:
        return np.inf

    nll = logdet + math.log(Vinv_1_1) + df_reml * math.log(quad / df_reml)

    if not np.isfinite(nll):
        return np.inf

    return float(nll)


def fit_reml_gblup_intercept_only(K, y, min_lambda=1e-6, max_lambda=1e6, grid_size=81):
    try:
        from scipy.optimize import minimize_scalar
    except Exception as e:
        raise SystemExit(
            "[ERROR] scipy is required for REML-GBLUP. "
            "Please install scipy in conda env."
        ) from e

    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    if n < 3:
        raise RuntimeError(f"too few samples for REML-GBLUP: n={n}")

    K = np.asarray(K, dtype=np.float64)
    K = 0.5 * (K + K.T)

    eigvals, U = np.linalg.eigh(K)

    min_eig = float(np.min(eigvals))
    lower_lambda = max(float(min_lambda), -min_eig + 1e-8)
    upper_lambda = float(max_lambda)

    if lower_lambda >= upper_lambda:
        upper_lambda = lower_lambda * 1e6

    one = np.ones(n, dtype=np.float64)

    Uy = U.T @ y
    U1 = U.T @ one

    df_reml = n - 1

    log_lower = math.log(lower_lambda)
    log_upper = math.log(upper_lambda)

    grid_size = int(grid_size)
    if grid_size < 21:
        grid_size = 21

    grid = np.linspace(log_lower, log_upper, grid_size)

    vals = np.asarray(
        [reml_profile_nll_intercept(x, eigvals, Uy, U1, df_reml) for x in grid],
        dtype=float,
    )

    finite = np.isfinite(vals)
    if not np.any(finite):
        raise RuntimeError("REML failed: all profile likelihood values are non-finite")

    best_i = int(np.nanargmin(vals))

    left_i = max(0, best_i - 2)
    right_i = min(len(grid) - 1, best_i + 2)

    opt_left = float(grid[left_i])
    opt_right = float(grid[right_i])

    if opt_left == opt_right:
        best_log_lambda = float(grid[best_i])
    else:
        res = minimize_scalar(
            lambda z: reml_profile_nll_intercept(z, eigvals, Uy, U1, df_reml),
            bounds=(opt_left, opt_right),
            method="bounded",
            options={"xatol": 1e-6},
        )

        if res.success and np.isfinite(res.fun):
            best_log_lambda = float(res.x)
        else:
            best_log_lambda = float(grid[best_i])

    lam = float(np.exp(best_log_lambda))

    denom = eigvals + lam

    Vinv_1_1 = float(np.sum((U1 * U1) / denom))
    Vinv_1_y = float(np.sum((U1 * Uy) / denom))
    mu = float(Vinv_1_y / Vinv_1_1)

    residual_eig = Uy - U1 * mu
    quad = float(np.sum((residual_eig * residual_eig) / denom))

    sigma_g2 = float(quad / df_reml)
    sigma_e2 = float(lam * sigma_g2)

    mean_diag = float(np.mean(np.diag(K)))

    denom_h2 = sigma_g2 * mean_diag + sigma_e2
    h2 = float((sigma_g2 * mean_diag) / denom_h2) if denom_h2 > 0 else np.nan

    nll = reml_profile_nll_intercept(best_log_lambda, eigvals, Uy, U1, df_reml)

    return {
        "lambda_reml": lam,
        "sigma_g2": sigma_g2,
        "sigma_e2": sigma_e2,
        "h2": h2,
        "mu": mu,
        "kinship_mean_diag": mean_diag,
        "reml_nll": float(nll),
        "min_eig": min_eig,
        "lower_lambda_used": lower_lambda,
        "upper_lambda_used": upper_lambda,
    }


def predict_gblup(Ktr, Kte, ytr, reml):
    ytr = np.asarray(ytr, dtype=np.float64)

    lam = float(reml["lambda_reml"])
    mu = float(reml["mu"])

    n = Ktr.shape[0]
    A = Ktr + lam * np.eye(n, dtype=np.float64)
    rhs = ytr - mu

    try:
        alpha = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, rhs, rcond=None)[0]

    pred = mu + Kte @ alpha

    return pred.astype(float)


def main():
    ap = argparse.ArgumentParser(
        description="REML-GBLUP using feature-derived kinship matrix and existing OOF folds."
    )

    ap.add_argument("--feature-npz", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--feature-name", required=True)

    ap.add_argument("--seed", type=int, default=20260427)

    ap.add_argument("--reml-min-lambda", type=float, default=1e-6)
    ap.add_argument("--reml-max-lambda", type=float, default=1e6)
    ap.add_argument("--reml-grid-size", type=int, default=81)

    ap.add_argument("--normalize-k", action="store_true", default=True)
    ap.add_argument("--no-normalize-k", dest="normalize_k", action="store_false")

    ap.add_argument("--out-oof", required=True)
    ap.add_argument("--out-metrics", required=True)

    args = ap.parse_args()

    X, y, folds = align_xy(args.feature_npz, args.folds)

    unique_folds = sorted(int(x) for x in folds["fold"].unique())

    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_meta = []

    print(
        f"[INFO] REML-GBLUP feature={args.feature_name} "
        f"trait={args.trait} X={X.shape} n_folds={len(unique_folds)} "
        f"folds={unique_folds}"
    )

    for fold in unique_folds:
        te = np.where(folds["fold"].values.astype(int) == fold)[0]
        tr = np.setdiff1d(np.arange(len(y)), te)

        Xtr = X[tr, :]
        Xte = X[te, :]
        ytr = y[tr]

        Ktr, Kte, mean_diag_before, mean_diag_after = build_kinship_from_features(
            Xtr=Xtr,
            Xte=Xte,
            normalize_k=args.normalize_k,
        )

        reml = fit_reml_gblup_intercept_only(
            K=Ktr,
            y=ytr,
            min_lambda=args.reml_min_lambda,
            max_lambda=args.reml_max_lambda,
            grid_size=args.reml_grid_size,
        )

        pred = predict_gblup(Ktr, Kte, ytr, reml)
        oof_pred[te] = pred

        md = metrics_dict(y[te], pred)

        meta = dict(reml)
        meta["fold"] = int(fold)
        meta["n_train"] = int(len(tr))
        meta["n_test"] = int(len(te))
        meta["fold_pearson_r"] = md["pearson_r"]
        meta["fold_r2"] = md["r2"]
        meta["fold_rmse"] = md["rmse"]
        meta["fold_mae"] = md["mae"]
        meta["kinship_mean_diag_before_norm"] = mean_diag_before
        meta["kinship_mean_diag_after_norm"] = mean_diag_after
        fold_meta.append(meta)

        print(
            f"[INFO] fold={fold} "
            f"n_train={len(tr)} n_test={len(te)} "
            f"pearson={md['pearson_r']:.4f} "
            f"r2={md['r2']:.4f} "
            f"rmse={md['rmse']:.4f} "
            f"lambda={reml['lambda_reml']:.6g} "
            f"sigma_g2={reml['sigma_g2']:.6g} "
            f"sigma_e2={reml['sigma_e2']:.6g} "
            f"h2={reml['h2']:.4f}"
        )

    if np.any(~np.isfinite(oof_pred)):
        raise RuntimeError("some OOF predictions are missing")

    oof = folds[["sample", "trait", "value", "fold"]].copy()
    oof = oof.rename(columns={"value": "y_true"})
    oof["y_pred"] = oof_pred
    oof["feature"] = args.feature_name
    oof["method"] = "reml_gblup"
    oof["feature_ncol"] = int(X.shape[1])

    md_all = metrics_dict(y, oof_pred)

    fold_meta_df = pd.DataFrame(fold_meta)

    def mean_col(col):
        if col not in fold_meta_df.columns:
            return np.nan
        vals = pd.to_numeric(fold_meta_df[col], errors="coerce")
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if len(vals) else np.nan

    reml_by_fold = ";".join(
        f"{int(r['fold'])}:"
        f"lambda={r.get('lambda_reml', np.nan)},"
        f"sg2={r.get('sigma_g2', np.nan)},"
        f"se2={r.get('sigma_e2', np.nan)},"
        f"h2={r.get('h2', np.nan)}"
        for _, r in fold_meta_df.iterrows()
    )

    metrics = pd.DataFrame(
        [{
            "trait": args.trait,
            "feature": args.feature_name,
            "method": "reml_gblup",
            "feature_ncol": int(X.shape[1]),

            **md_all,

            "n_folds": int(len(unique_folds)),
            "fold_ids": ",".join(map(str, unique_folds)),

            "mean_lambda_reml": mean_col("lambda_reml"),
            "mean_sigma_g2": mean_col("sigma_g2"),
            "mean_sigma_e2": mean_col("sigma_e2"),
            "mean_h2": mean_col("h2"),
            "mean_kinship_diag": mean_col("kinship_mean_diag"),
            "mean_kinship_diag_before_norm": mean_col("kinship_mean_diag_before_norm"),
            "mean_kinship_diag_after_norm": mean_col("kinship_mean_diag_after_norm"),

            "reml_by_fold": reml_by_fold,

            "feature_npz": args.feature_npz,
            "folds": args.folds,
        }]
    )

    os.makedirs(os.path.dirname(args.out_oof), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)

    oof.to_csv(args.out_oof, sep="\t", index=False)
    metrics.to_csv(args.out_metrics, sep="\t", index=False)

    touch_outputs_newer_than_inputs(
        outputs=[args.out_oof, args.out_metrics],
        inputs=[args.feature_npz, args.folds],
        plus_seconds=10.0,
    )

    print(metrics.to_string(index=False))
    print(f"[INFO] wrote {args.out_oof}")
    print(f"[INFO] wrote {args.out_metrics}")


if __name__ == "__main__":
    main()
