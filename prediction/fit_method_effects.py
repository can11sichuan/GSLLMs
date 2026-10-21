#!/usr/bin/env python3
import argparse
import math
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

try:
    import yaml
except Exception as exc:
    raise SystemExit("PyYAML is required. Run in a conda env with yaml installed.") from exc

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

try:
    from scipy.optimize import minimize_scalar
except Exception:
    minimize_scalar = None

warnings.filterwarnings("ignore", category=UserWarning)

EMBEDDING_MODELS = ("evo2", "dnabert2", "plantcad", "agront")
SUPPORTED_METHODS = ("reml_gblup", "gblup", "rrblup", "rf", "lightgbm")


def read_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_feature_label(label):
    f = str(label).lower()
    if f == "raw" or f.startswith("raw"):
        return "raw", 0
    for model in EMBEDDING_MODELS:
        if f.startswith(model):
            m = re.search(r"(?:^|_)w(\d+)(?:_|$)", f)
            return model, int(m.group(1)) if m else 0
    return "unknown", 0


def read_bim(path):
    bim = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["chr", "snp_id", "cm", "bp", "a1", "a2"],
        dtype={"chr": str, "snp_id": str},
    )
    bim["bp"] = pd.to_numeric(bim["bp"], errors="coerce")
    return bim[["snp_id", "chr", "bp", "a1", "a2"]]


def infer_chr_bp_from_snp_id(snp_ids):
    rows = []
    pat = re.compile(r"chr([^._]+)\.s[_-]?(\d+)", re.IGNORECASE)
    for snp in snp_ids:
        m = pat.search(str(snp))
        if m:
            rows.append((str(snp), m.group(1), int(m.group(2))))
        else:
            rows.append((str(snp), "", np.nan))
    return pd.DataFrame(rows, columns=["snp_id", "chr_inferred", "bp_inferred"])


def np_string_array(z, keys):
    for key in keys:
        if key in z.files:
            return np.asarray(z[key]).astype(str)
    return None


def load_feature_npz(path):
    z = np.load(path, allow_pickle=True)
    if "X" not in z.files or "samples" not in z.files:
        raise RuntimeError(f"feature NPZ must contain X and samples: {path}")
    X = np.asarray(z["X"], dtype=np.float32)
    samples = np.asarray(z["samples"]).astype(str)
    snp_ids = np_string_array(z, ["snp_id", "snp_ids", "variant_id", "feature_names"])
    if snp_ids is None:
        raise RuntimeError(f"could not find snp_id or feature_names in {path}; keys={z.files}")
    return X, samples, snp_ids


def find_folds_path(cfg, run_name, feature_label, feature_npz, trait):
    run_cfg = cfg["gp_runs"][run_name]
    base = str(run_cfg.get("folds", "") or "")
    candidates = []
    if base:
        candidates.append(os.path.join(base, f"{trait}.folds.tsv"))
        stem = os.path.basename(feature_npz)
        if stem.endswith(".npz"):
            stem = stem[:-4]
        candidates.append(os.path.join(base, stem, f"{trait}.folds.tsv"))
        candidates.append(os.path.join(base, feature_label, f"{trait}.folds.tsv"))
        candidates.append(os.path.join(base, "raw_selected", f"{trait}.folds.tsv"))
    root = str(run_cfg.get("root", "") or "")
    if root:
        candidates.append(os.path.join(root, "metadata", "folds", f"{trait}.folds.tsv"))
        candidates.append(os.path.join(root, "folds", feature_label, f"{trait}.folds.tsv"))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise RuntimeError(f"could not find folds for {run_name} {feature_label} {trait}; tried {candidates}")


def discover_feature_jobs(cfg, window, requested):
    jobs = {}
    first_run = next(iter(cfg["gp_runs"]))
    raw_path = cfg["gp_runs"][first_run].get("raw_feature_npz")
    if raw_path:
        jobs["raw"] = {"run": first_run, "path": raw_path}
    for run_name, run_cfg in cfg.get("gp_runs", {}).items():
        for label, path in (run_cfg.get("feature_npz") or {}).items():
            _, win = parse_feature_label(label)
            if int(win) == int(window):
                jobs[label] = {"run": run_name, "path": path}
    if requested:
        missing = [x for x in requested if x not in jobs]
        if missing:
            raise RuntimeError(f"requested feature labels not found for window {window}: {missing}")
        jobs = {k: jobs[k] for k in requested}
    return jobs


def align_xy(X_all, samples_all, folds_path):
    folds = pd.read_csv(folds_path, sep="\t")
    required = {"sample", "value", "fold"}
    missing = required - set(folds.columns)
    if missing:
        raise RuntimeError(f"fold file missing columns {missing}: {folds_path}")
    folds["sample"] = folds["sample"].astype(str)
    sample_to_idx = {s: i for i, s in enumerate(samples_all)}
    missing_samples = [s for s in folds["sample"] if s not in sample_to_idx]
    if missing_samples:
        raise RuntimeError(f"{len(missing_samples)} fold samples absent from feature matrix. First: {missing_samples[:5]}")
    idx = np.asarray([sample_to_idx[s] for s in folds["sample"]], dtype=np.int64)
    y = pd.to_numeric(folds["value"], errors="coerce").values.astype(np.float64)
    keep = np.isfinite(y)
    X = X_all[idx[keep], :]
    y = y[keep]
    folds = folds.loc[keep].reset_index(drop=True)
    return X, y, folds


def impute_and_standardize_train(X):
    X = np.asarray(X, dtype=np.float32)
    finite = np.isfinite(X)
    n = finite.sum(axis=0).astype(np.float32)
    sums = np.where(finite, X, 0.0).sum(axis=0)
    mean = np.divide(sums, n, out=np.zeros_like(sums, dtype=np.float32), where=n > 0)
    X_imp = np.where(finite, X, mean)
    centered = X_imp - mean
    var = np.divide((centered * centered).sum(axis=0), np.maximum(n - 1.0, 1.0), out=np.ones_like(mean), where=n > 1)
    scale = np.sqrt(var).astype(np.float32)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return (centered / scale).astype(np.float32)


def impute_nonfinite_train(X):
    X = np.asarray(X, dtype=np.float32)
    finite = np.isfinite(X)
    n = finite.sum(axis=0).astype(np.float32)
    sums = np.where(finite, X, 0.0).sum(axis=0)
    mean = np.divide(sums, n, out=np.zeros_like(sums, dtype=np.float32), where=n > 0)
    return np.where(finite, X, mean).astype(np.float32)


def parse_alpha_grid(value):
    vals = []
    for x in str(value).replace(";", ",").split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals or [1.0]


def rrblup_coef(Xs, y, alpha):
    y = np.asarray(y, dtype=np.float64)
    yc = y - float(np.mean(y))
    K = (Xs @ Xs.T).astype(np.float64)
    diag = np.diag_indices_from(K)
    K[diag] += float(alpha)
    dual = np.linalg.solve(K, yc)
    coef = Xs.T @ dual
    return np.asarray(coef, dtype=np.float64), float(alpha)


def gblup_coef_fixed_lambda(Xs, y, lam, normalize_k=True):
    y = np.asarray(y, dtype=np.float64)
    yc = y - float(np.mean(y))
    p = max(1, Xs.shape[1])
    scale = float(p)
    K = (Xs @ Xs.T).astype(np.float64) / scale
    K = 0.5 * (K + K.T)
    diag_mean = float(np.mean(np.diag(K)))
    if normalize_k and np.isfinite(diag_mean) and diag_mean > 0:
        K /= diag_mean
        scale *= diag_mean
    A = K + float(lam) * np.eye(K.shape[0], dtype=np.float64)
    dual = np.linalg.solve(A, yc)
    coef = Xs.T @ dual / scale
    return np.asarray(coef, dtype=np.float64), float(lam), diag_mean


def reml_profile_nll(log_lambda, eigvals, Uy, U1, df_reml):
    lam = float(np.exp(log_lambda))
    denom = eigvals + lam
    if np.any(denom <= 0):
        return np.inf
    logdet = float(np.sum(np.log(denom)))
    vinv_11 = float(np.sum((U1 * U1) / denom))
    vinv_1y = float(np.sum((U1 * Uy) / denom))
    if not np.isfinite(vinv_11) or vinv_11 <= 0:
        return np.inf
    mu = vinv_1y / vinv_11
    resid = Uy - U1 * mu
    quad = float(np.sum((resid * resid) / denom))
    if not np.isfinite(quad) or quad <= 0:
        return np.inf
    return logdet + math.log(vinv_11) + df_reml * math.log(quad / df_reml)


def fit_reml_lambda(K, y, min_lambda, max_lambda, grid_size):
    if minimize_scalar is None:
        raise RuntimeError("scipy is required for REML lambda estimation")
    y = np.asarray(y, dtype=np.float64)
    K = 0.5 * (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T)
    eigvals, U = np.linalg.eigh(K)
    min_eig = float(np.min(eigvals))
    lower = max(float(min_lambda), -min_eig + 1e-8)
    upper = max(float(max_lambda), lower * 10.0)
    Uy = U.T @ y
    U1 = U.T @ np.ones(len(y), dtype=np.float64)
    df_reml = len(y) - 1
    log_lower = math.log(lower)
    log_upper = math.log(upper)

    def obj(x):
        return reml_profile_nll(x, eigvals, Uy, U1, df_reml)

    grid = np.linspace(log_lower, log_upper, int(max(21, grid_size)))
    vals = np.asarray([obj(x) for x in grid], dtype=float)
    best_i = int(np.nanargmin(vals))
    left = float(grid[max(0, best_i - 2)])
    right = float(grid[min(len(grid) - 1, best_i + 2)])
    if left == right:
        log_lam = float(grid[best_i])
        converged = False
    else:
        opt = minimize_scalar(obj, bounds=(left, right), method="bounded", options={"xatol": 1e-5})
        log_lam = float(opt.x) if opt.success and np.isfinite(opt.fun) else float(grid[best_i])
        converged = bool(opt.success)
    return float(np.exp(log_lam)), converged


def reml_gblup_coef(Xs, y, min_lambda, max_lambda, grid_size, normalize_k=True):
    p = max(1, Xs.shape[1])
    K = (Xs @ Xs.T).astype(np.float64) / float(p)
    K = 0.5 * (K + K.T)
    diag_mean = float(np.mean(np.diag(K)))
    scale = float(p)
    if normalize_k and np.isfinite(diag_mean) and diag_mean > 0:
        K /= diag_mean
        scale *= diag_mean
    lam, converged = fit_reml_lambda(K, y, min_lambda, max_lambda, grid_size)
    y = np.asarray(y, dtype=np.float64)
    yc = y - float(np.mean(y))
    A = K + lam * np.eye(K.shape[0], dtype=np.float64)
    dual = np.linalg.solve(A, yc)
    coef = Xs.T @ dual / scale
    return np.asarray(coef, dtype=np.float64), lam, diag_mean, converged


def choose_gblup_lambda(Xs, y, alpha_grid, inner_cv, seed, normalize_k=True):
    if len(alpha_grid) <= 1:
        return float(alpha_grid[0])
    p = max(1, Xs.shape[1])
    K = (Xs @ Xs.T).astype(np.float64) / float(p)
    K = 0.5 * (K + K.T)
    diag_mean = float(np.mean(np.diag(K)))
    if normalize_k and np.isfinite(diag_mean) and diag_mean > 0:
        K /= diag_mean
    k = min(int(inner_cv), max(2, len(y) // 5))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    best_lam = float(alpha_grid[0])
    best_rmse = np.inf
    y = np.asarray(y, dtype=np.float64)
    for lam in alpha_grid:
        pred = np.full(len(y), np.nan, dtype=np.float64)
        for tr, va in cv.split(K):
            y_mean = float(np.mean(y[tr]))
            yc = y[tr] - y_mean
            A = K[np.ix_(tr, tr)] + float(lam) * np.eye(len(tr), dtype=np.float64)
            dual = np.linalg.solve(A, yc)
            pred[va] = y_mean + K[np.ix_(va, tr)] @ dual
        rmse = float(np.sqrt(np.nanmean((pred - y) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lam = float(lam)
    return best_lam


def feature_importance_model(method, X, y, args, fold_seed):
    X = impute_nonfinite_train(X)
    if method == "rf":
        model = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_features=args.rf_max_features,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=args.threads,
            random_state=fold_seed,
        )
    elif method == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError("lightgbm is not installed in the active environment") from exc
        model = LGBMRegressor(
            objective="regression",
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            num_leaves=args.lgbm_num_leaves,
            subsample=args.lgbm_subsample,
            colsample_bytree=args.lgbm_colsample_bytree,
            n_jobs=args.threads,
            random_state=fold_seed,
            verbosity=-1,
        )
    else:
        raise RuntimeError(method)
    model.fit(X, y)
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        raise RuntimeError(f"model {method} does not expose feature_importances_")
    return np.asarray(imp, dtype=np.float64)


def coef_to_snp_effect(values, snp_ids, signed=True):
    n_snps = len(snp_ids)
    p = len(values)
    if p == n_snps:
        signed_effect = values.copy()
        effect = np.abs(values) if signed else np.maximum(values, 0.0)
        n_components = 1
        component_mode = "raw_abs" if signed else "raw_importance"
    elif p % n_snps == 0:
        n_components = int(p // n_snps)
        mat = values.reshape(n_snps, n_components)
        signed_effect = mat[:, 0]
        effect = np.sqrt(np.sum(mat * mat, axis=1)) if signed else np.sum(np.maximum(mat, 0.0), axis=1)
        component_mode = "component_l2" if signed else "component_importance_sum"
    else:
        raise RuntimeError(f"cannot map feature values to SNPs: p={p}, n_snps={n_snps}")
    return signed_effect, effect, n_components, component_mode


def fit_effect_table(X, y, folds, snp_ids, trait, feature_label, method, args):
    effects = []
    signed_vals = []
    hypers = []
    unique_folds = sorted(int(x) for x in folds["fold"].unique())
    alpha_grid = parse_alpha_grid(args.alpha_grid)
    for fold in unique_folds:
        train_idx = np.where(folds["fold"].values.astype(int) != fold)[0]
        fold_seed = int(args.seed) + int(fold)
        Xtr = X[train_idx, :]
        ytr = y[train_idx]
        if method == "rrblup":
            Xs = impute_and_standardize_train(Xtr)
            alpha = float(alpha_grid[0])
            coef, hyper = rrblup_coef(Xs, ytr, alpha)
            signed_i, effect_i, n_components, component_mode = coef_to_snp_effect(coef, snp_ids, signed=True)
            hypers.append(f"{fold}:alpha={hyper:.6g}")
        elif method == "gblup":
            Xs = impute_and_standardize_train(Xtr)
            lam = choose_gblup_lambda(Xs, ytr, alpha_grid, args.inner_cv_folds, fold_seed, args.normalize_k)
            coef, hyper, diag_mean = gblup_coef_fixed_lambda(Xs, ytr, lam, args.normalize_k)
            signed_i, effect_i, n_components, component_mode = coef_to_snp_effect(coef, snp_ids, signed=True)
            hypers.append(f"{fold}:lambda={hyper:.6g},diag={diag_mean:.6g}")
        elif method == "reml_gblup":
            Xs = impute_and_standardize_train(Xtr)
            coef, lam, diag_mean, converged = reml_gblup_coef(
                Xs,
                ytr,
                args.reml_min_lambda,
                args.reml_max_lambda,
                args.reml_grid_size,
                args.normalize_k,
            )
            signed_i, effect_i, n_components, component_mode = coef_to_snp_effect(coef, snp_ids, signed=True)
            hypers.append(f"{fold}:lambda={lam:.6g},diag={diag_mean:.6g},conv={int(converged)}")
        elif method in {"rf", "lightgbm"}:
            imp = feature_importance_model(method, Xtr, ytr, args, fold_seed)
            signed_i, effect_i, n_components, component_mode = coef_to_snp_effect(imp, snp_ids, signed=False)
            hypers.append(f"{fold}:importance_sum={float(np.sum(imp)):.6g}")
        else:
            raise RuntimeError(f"unsupported method: {method}")
        effects.append(effect_i)
        signed_vals.append(signed_i)
        print(f"[INFO] fitted method={method} feature={feature_label} trait={trait} fold={fold} train_n={len(train_idx)}", file=sys.stderr)
    effect_mat = np.vstack(effects)
    signed_mat = np.vstack(signed_vals)
    out = pd.DataFrame(
        {
            "trait": trait,
            "method": method,
            "feature_label": feature_label,
            "snp_id": np.asarray(snp_ids).astype(str),
            "effect": effect_mat.mean(axis=0),
            "effect_std": effect_mat.std(axis=0, ddof=1) if effect_mat.shape[0] > 1 else 0.0,
            "signed_effect_pc1_or_raw": signed_mat.mean(axis=0),
            "n_fits": effect_mat.shape[0],
            "n_components_per_snp": n_components,
            "component_mode": component_mode,
            "fit_mode": "cv_mean",
            "hyper_by_fold": ";".join(hypers),
        }
    )
    sd = float(out["effect"].std(ddof=1))
    out["effect_z"] = (out["effect"] - float(out["effect"].mean())) / sd if sd > 0 else 0.0
    out["effect_rank"] = out["effect"].rank(ascending=False, method="first").astype(int)
    out["effect_percentile"] = 1.0 - (out["effect_rank"] - 1.0) / max(1.0, len(out) - 1.0)
    return out


def add_variant_map(effects, bim):
    inf = infer_chr_bp_from_snp_id(effects["snp_id"].values)
    out = effects.merge(bim, on="snp_id", how="left").merge(inf, on="snp_id", how="left")
    out["chr"] = out["chr"].fillna(out["chr_inferred"])
    out["bp"] = out["bp"].fillna(out["bp_inferred"])
    out = out.drop(columns=["chr_inferred", "bp_inferred"])
    return out


def default_methods(cfg):
    methods = list(cfg.get("plots", {}).get("methods", [])) or list(SUPPORTED_METHODS)
    return [m for m in methods if m in SUPPORTED_METHODS]


def main():
    ap = argparse.ArgumentParser(description="Fit model-derived SNP effects/importances for layered Manhattan plots.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--method", action="append", default=None, choices=SUPPORTED_METHODS)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--feature-label", action="append", default=None)
    ap.add_argument("--alpha-grid", default=None)
    ap.add_argument("--inner-cv-folds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rf-n-estimators", type=int, default=500)
    ap.add_argument("--rf-max-features", default="sqrt")
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--lgbm-n-estimators", type=int, default=1200)
    ap.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    ap.add_argument("--lgbm-num-leaves", type=int, default=31)
    ap.add_argument("--lgbm-subsample", type=float, default=0.8)
    ap.add_argument("--lgbm-colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reml-min-lambda", type=float, default=1e-6)
    ap.add_argument("--reml-max-lambda", type=float, default=1e6)
    ap.add_argument("--reml-grid-size", type=int, default=81)
    ap.add_argument("--no-normalize-k", dest="normalize_k", action="store_false")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--skip-manifest", action="store_true")
    args = ap.parse_args()

    cfg = read_config(args.config)
    args.seed = int(args.seed if args.seed is not None else cfg.get("random_seed", 20260427))
    args.alpha_grid = str(args.alpha_grid if args.alpha_grid is not None else cfg.get("method_effects", {}).get("alpha_grid", "1.0"))
    args.inner_cv_folds = int(args.inner_cv_folds if args.inner_cv_folds is not None else cfg.get("method_effects", {}).get("inner_cv_folds", 3))
    window = int(args.window or cfg.get("method_effects", {}).get("default_window", cfg.get("primary_window", 500)))
    methods = args.method or default_methods(cfg)
    outdir_base = args.outdir or os.path.join(str(cfg.get("outdir", "analysis_outputs/Maize1404")), "method_effects", args.trait)
    bim = read_bim(cfg["plink_bim"])
    jobs = discover_feature_jobs(cfg, window, args.feature_label)

    for method in methods:
        outdir = ensure_dir(os.path.join(outdir_base, method))
        written = []
        for label, job in jobs.items():
            feature_npz = job["path"]
            run_name = job["run"]
            if not os.path.exists(feature_npz):
                print(f"[WARN] missing feature NPZ for {label}: {feature_npz}", file=sys.stderr)
                continue
            folds_path = find_folds_path(cfg, run_name, label, feature_npz, args.trait)
            print(f"[INFO] loading method={method} feature={label}: {feature_npz}", file=sys.stderr)
            X_all, samples_all, snp_ids = load_feature_npz(feature_npz)
            X, y, folds = align_xy(X_all, samples_all, folds_path)
            print(f"[INFO] aligned method={method} feature={label}: X={X.shape} n={len(y)} folds={sorted(folds['fold'].unique())}", file=sys.stderr)
            effects = fit_effect_table(X, y, folds, snp_ids, args.trait, label, method, args)
            effects["source_run"] = run_name
            effects["feature_npz"] = feature_npz
            effects["folds"] = folds_path
            model, win = parse_feature_label(label)
            effects["embedding_model"] = model
            effects["window"] = win
            effects = add_variant_map(effects, bim)
            out_path = os.path.join(outdir, f"{args.trait}.{method}.{label}.effects.tsv.gz")
            effects.to_csv(out_path, sep="\t", index=False)
            written.append(out_path)
            print(f"[INFO] wrote {out_path}", file=sys.stderr)
            del X_all, X, y, folds, effects
        if not args.skip_manifest:
            manifest = os.path.join(outdir, f"{args.trait}.{method}.effects.manifest.txt")
            with open(manifest, "w", encoding="utf-8") as fh:
                for path in written:
                    fh.write(path + "\n")
            print(f"[INFO] wrote manifest {manifest}")


if __name__ == "__main__":
    main()
