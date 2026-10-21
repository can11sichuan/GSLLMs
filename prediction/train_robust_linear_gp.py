#!/usr/bin/env python3
"""Numerically stable rrBLUP and REML-GBLUP for sparse genotype features.

Features are centered using training-fold means and normalized once at the
whole-kernel level. Per-column unit-variance scaling is intentionally avoided
because it can make rare, fold-specific alleles produce extreme test values.
"""

import argparse
import importlib.util
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


REML_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_reml_gblup.py")


def load_reml_module():
    spec = importlib.util.spec_from_file_location("soybean_reml_base", REML_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_grid(text):
    return [float(value) for value in str(text).replace(";", ",").split(",") if value.strip()]


def load_and_align(feature_npz, folds_path):
    z = np.load(feature_npz, allow_pickle=True)
    X_all = np.asarray(z["X"], dtype=np.float32)
    samples = np.asarray(z["samples"]).astype(str)
    folds = pd.read_csv(folds_path, sep="\t")
    folds["sample"] = folds["sample"].astype(str)
    index = {sample: i for i, sample in enumerate(samples)}
    absent = [sample for sample in folds["sample"] if sample not in index]
    if absent:
        raise RuntimeError(f"{len(absent)} fold samples absent from {feature_npz}: {absent[:5]}")
    rows = np.asarray([index[sample] for sample in folds["sample"]], dtype=int)
    y = pd.to_numeric(folds["value"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y)
    return X_all[rows[keep], :], y[keep], folds.loc[keep].reset_index(drop=True)


def center_train_test(Xtrain, Xtest):
    Xtrain = np.asarray(Xtrain, dtype=np.float32)
    Xtest = np.asarray(Xtest, dtype=np.float32)
    finite = np.isfinite(Xtrain)
    count = finite.sum(axis=0)
    sums = np.where(finite, Xtrain, 0.0).sum(axis=0, dtype=np.float64)
    means = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0).astype(np.float32)
    train = np.where(finite, Xtrain, means).astype(np.float32) - means
    test = np.where(np.isfinite(Xtest), Xtest, means).astype(np.float32) - means
    return train, test, means


def build_normalized_kernel(Xtrain, Xtest):
    train_centered, test_centered, means = center_train_test(Xtrain, Xtest)
    Ktrain = (train_centered @ train_centered.T).astype(np.float64)
    Ktest = (test_centered @ train_centered.T).astype(np.float64)
    Ktrain = 0.5 * (Ktrain + Ktrain.T)
    mean_diag_before = float(np.mean(np.diag(Ktrain)))
    if not np.isfinite(mean_diag_before) or mean_diag_before <= 0:
        raise RuntimeError(f"nonpositive feature-kernel diagonal: {mean_diag_before}")
    Ktrain /= mean_diag_before
    Ktest /= mean_diag_before
    return Ktrain, Ktest, train_centered, test_centered, means, mean_diag_before


def kernel_ridge_predict(Ktrain, Ktest, ytrain, lam):
    intercept = float(np.mean(ytrain))
    system = Ktrain + float(lam) * np.eye(len(ytrain), dtype=np.float64)
    try:
        dual = np.linalg.solve(system, np.asarray(ytrain, dtype=np.float64) - intercept)
    except np.linalg.LinAlgError:
        dual = np.linalg.lstsq(system, np.asarray(ytrain, dtype=np.float64) - intercept, rcond=None)[0]
    return intercept + Ktest @ dual, intercept, dual


def center_kernel_split(kernel, train, validation):
    """Recenter a precomputed outer-fold kernel on an inner training split."""
    train_kernel = np.asarray(kernel[np.ix_(train, train)], dtype=np.float64)
    validation_kernel = np.asarray(kernel[np.ix_(validation, train)], dtype=np.float64)
    train_col_mean = train_kernel.mean(axis=0, keepdims=True)
    train_row_mean = train_kernel.mean(axis=1, keepdims=True)
    train_grand_mean = float(train_kernel.mean())
    validation_row_mean = validation_kernel.mean(axis=1, keepdims=True)
    centered_train = (
        train_kernel - train_col_mean - train_row_mean + train_grand_mean
    )
    centered_validation = (
        validation_kernel
        - validation_row_mean
        - train_col_mean
        + train_grand_mean
    )
    centered_train = 0.5 * (centered_train + centered_train.T)
    mean_diag = float(np.mean(np.diag(centered_train)))
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        raise RuntimeError(f"nonpositive inner-CV kernel diagonal: {mean_diag}")
    return centered_train / mean_diag, centered_validation / mean_diag


def spectral_kernel_ridge_predictions(Ktrain, Ktest, ytrain, lambdas):
    """Evaluate a lambda grid after one symmetric eigendecomposition."""
    intercept = float(np.mean(ytrain))
    centered_y = np.asarray(ytrain, dtype=np.float64) - intercept
    eigenvalues, eigenvectors = np.linalg.eigh(Ktrain)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_y = eigenvectors.T @ centered_y
    predictions = {}
    for lam in lambdas:
        dual = eigenvectors @ (projected_y / (eigenvalues + float(lam)))
        predictions[float(lam)] = intercept + Ktest @ dual
    return predictions


def choose_lambda_inner_cv(Kouter, ytrain, lambdas, inner_folds, seed):
    if len(lambdas) == 1:
        return float(lambdas[0]), pd.DataFrame(
            [{"lambda": float(lambdas[0]), "rmse": np.nan}]
        )
    splitter = KFold(
        n_splits=min(int(inner_folds), max(2, len(ytrain) // 10)),
        shuffle=True,
        random_state=int(seed),
    )
    squared_error = {float(lam): [] for lam in lambdas}
    for inner_train, validation in splitter.split(Kouter):
        Ktrain, Kvalidation = center_kernel_split(Kouter, inner_train, validation)
        predictions = spectral_kernel_ridge_predictions(
            Ktrain, Kvalidation, ytrain[inner_train], lambdas
        )
        for lam, prediction in predictions.items():
            squared_error[float(lam)].extend(
                (np.asarray(prediction) - ytrain[validation]) ** 2
            )
    scores = pd.DataFrame(
        [
            {"lambda": lam, "rmse": float(np.sqrt(np.mean(errors)))}
            for lam, errors in squared_error.items()
        ]
    ).sort_values(["rmse", "lambda"])
    return float(scores.iloc[0]["lambda"]), scores


def fit_predict_fold(method, Xtrain, ytrain, Xtest, args, fold_seed, reml_module=None):
    Ktrain, Ktest, train_centered, test_centered, means, raw_diag = build_normalized_kernel(
        Xtrain, Xtest
    )
    if method == "rrblup":
        lambdas = parse_grid(args.lambda_grid)
        lam, scores = choose_lambda_inner_cv(
            Ktrain, np.asarray(ytrain), lambdas, args.inner_cv_folds, fold_seed
        )
        prediction, intercept, dual = kernel_ridge_predict(Ktrain, Ktest, ytrain, lam)
        hyper = f"lambda={lam:.8g},inner_best_rmse={float(scores.iloc[0]['rmse']):.8g}"
        meta = {"lambda": lam, "intercept": intercept, "raw_kernel_diag": raw_diag}
    elif method == "reml_gblup":
        reml_module = reml_module or load_reml_module()
        reml = reml_module.fit_reml_gblup_intercept_only(
            Ktrain,
            ytrain,
            min_lambda=args.reml_min_lambda,
            max_lambda=args.reml_max_lambda,
            grid_size=args.reml_grid_size,
        )
        prediction = reml_module.predict_gblup(Ktrain, Ktest, ytrain, reml)
        lam = float(reml["lambda_reml"])
        intercept = float(reml["mu"])
        system = Ktrain + lam * np.eye(len(ytrain), dtype=np.float64)
        try:
            dual = np.linalg.solve(system, np.asarray(ytrain, dtype=np.float64) - intercept)
        except np.linalg.LinAlgError:
            dual = np.linalg.lstsq(system, np.asarray(ytrain, dtype=np.float64) - intercept, rcond=None)[0]
        hyper = f"lambda={lam:.8g},h2={float(reml['h2']):.8g}"
        meta = {**reml, "intercept": intercept, "raw_kernel_diag": raw_diag}
    else:
        raise RuntimeError(f"unsupported robust linear method: {method}")
    primal_coef = np.asarray(train_centered.T @ dual, dtype=np.float64) / float(raw_diag)
    return np.asarray(prediction, dtype=float), test_centered, primal_coef, hyper, meta


def metrics_dict(y, prediction):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    pearson = (
        float(np.corrcoef(y, prediction)[0, 1])
        if len(y) >= 3 and np.std(y) > 0 and np.std(prediction) > 0
        else np.nan
    )
    return {
        "n": int(len(y)),
        "pearson_r": pearson,
        "r2": float(r2_score(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-npz", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--method", choices=["rrblup", "reml_gblup"], required=True)
    parser.add_argument("--feature-name", required=True)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument(
        "--lambda-grid", default="1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000"
    )
    parser.add_argument("--inner-cv-folds", type=int, default=3)
    parser.add_argument("--reml-min-lambda", type=float, default=1e-6)
    parser.add_argument("--reml-max-lambda", type=float, default=1e6)
    parser.add_argument("--reml-grid-size", type=int, default=81)
    parser.add_argument("--out-oof", required=True)
    parser.add_argument("--out-metrics", required=True)
    args = parser.parse_args()

    X, y, folds = load_and_align(args.feature_npz, args.folds)
    fold_values = folds["fold"].astype(int).to_numpy()
    prediction = np.full(len(y), np.nan, dtype=float)
    hyper_by_fold = []
    reml_module = load_reml_module() if args.method == "reml_gblup" else None
    for fold in sorted(np.unique(fold_values)):
        test = np.where(fold_values == int(fold))[0]
        train = np.where(fold_values != int(fold))[0]
        pred, _, _, hyper, _ = fit_predict_fold(
            args.method,
            X[train, :],
            y[train],
            X[test, :],
            args,
            args.seed + int(fold),
            reml_module,
        )
        prediction[test] = pred
        hyper_by_fold.append(f"{fold}:{hyper}")
        fold_metric = metrics_dict(y[test], pred)
        print(
            f"[INFO] method={args.method} feature={args.feature_name} trait={args.trait} "
            f"fold={fold} pearson={fold_metric['pearson_r']:.5f} rmse={fold_metric['rmse']:.5f}",
            flush=True,
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("nonfinite OOF predictions")

    oof = folds[["sample", "trait", "value", "fold"]].copy().rename(
        columns={"value": "y_true"}
    )
    oof["y_pred"] = prediction
    oof["feature"] = args.feature_name
    oof["method"] = args.method
    oof["feature_ncol"] = X.shape[1]
    metrics = pd.DataFrame(
        [
            {
                "trait": args.trait,
                "feature": args.feature_name,
                "method": args.method,
                "feature_ncol": int(X.shape[1]),
                **metrics_dict(y, prediction),
                "linear_scaling": "train_centered_global_kernel_mean_diag_1",
                "lambda_grid": args.lambda_grid if args.method == "rrblup" else "REML",
                "chosen_hyper_by_fold": ";".join(hyper_by_fold),
                "feature_npz": args.feature_npz,
                "folds": args.folds,
            }
        ]
    )
    for path in [args.out_oof, args.out_metrics]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    oof.to_csv(args.out_oof, sep="\t", index=False)
    metrics.to_csv(args.out_metrics, sep="\t", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
