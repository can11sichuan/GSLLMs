#!/usr/bin/env python3
"""Train fixed-fold tabular predictors for a single MAF0.02 task."""

import argparse
import math
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, prediction):
    pearson = np.nan
    if len(y) >= 3 and np.std(y) > 0 and np.std(prediction) > 0:
        pearson = float(np.corrcoef(y, prediction)[0, 1])
    return {
        "n": int(len(y)),
        "pearson_r": pearson,
        "r2": float(r2_score(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
    }


def load_aligned(feature_npz, folds_path):
    data = np.load(feature_npz, allow_pickle=True)
    X_all = np.asarray(data["X"], dtype=np.float32)
    samples = np.asarray(data["samples"]).astype(str)
    if X_all.shape[0] != len(samples) or len(set(samples)) != len(samples):
        raise RuntimeError(f"invalid feature sample axis: {feature_npz}")
    folds = pd.read_csv(folds_path, sep="\t")
    required = {"sample", "trait", "value", "fold"}
    missing = required - set(folds.columns)
    if missing:
        raise RuntimeError(f"fold file missing columns {sorted(missing)}: {folds_path}")
    index = {sample: i for i, sample in enumerate(samples)}
    fold_samples = folds["sample"].astype(str).tolist()
    absent = [sample for sample in fold_samples if sample not in index]
    if absent:
        raise RuntimeError(f"{len(absent)} fold samples absent from features; first={absent[:5]}")
    rows = np.asarray([index[sample] for sample in fold_samples], dtype=int)
    y = pd.to_numeric(folds["value"], errors="coerce").to_numpy(float)
    keep = np.isfinite(y)
    return X_all[rows[keep]], y[keep], folds.loc[keep].reset_index(drop=True)


def impute_train_mean(Xtrain, Xtest):
    finite = np.isfinite(Xtrain)
    count = finite.sum(axis=0)
    means = np.divide(
        np.where(finite, Xtrain, 0.0).sum(axis=0, dtype=np.float64),
        count,
        out=np.zeros(Xtrain.shape[1], dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)
    return np.where(finite, Xtrain, means), np.where(np.isfinite(Xtest), Xtest, means)


def parse_max_features(value):
    value = str(value).strip()
    if value in {"sqrt", "log2"}:
        return value
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"invalid RF max_features: {value}") from error


def fit_predict_rf(Xtrain, ytrain, Xtest, n_estimators, max_features, min_samples_leaf, threads, seed):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_features=parse_max_features(max_features),
        min_samples_leaf=int(min_samples_leaf),
        n_jobs=int(threads),
        random_state=int(seed),
    )
    model.fit(Xtrain, ytrain)
    return model.predict(Xtest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-npz", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--feature-name", required=True)
    parser.add_argument("--method", choices=["rf", "lightgbm"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--rf-n-estimators", type=int, required=True)
    parser.add_argument("--rf-max-features", required=True)
    parser.add_argument("--rf-min-samples-leaf", type=int, required=True)
    parser.add_argument("--lgbm-n-estimators", type=int, required=True)
    parser.add_argument("--lgbm-learning-rate", type=float, required=True)
    parser.add_argument("--lgbm-num-leaves", type=int, required=True)
    parser.add_argument("--lgbm-subsample", type=float, required=True)
    parser.add_argument("--lgbm-colsample-bytree", type=float, required=True)
    parser.add_argument("--out-oof", required=True)
    parser.add_argument("--out-metrics", required=True)
    args = parser.parse_args()

    X, y, folds = load_aligned(args.feature_npz, args.folds)
    prediction = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(folds["fold"].astype(int).unique()):
        test = folds["fold"].astype(int).to_numpy() == fold
        train = ~test
        Xtrain, Xtest = impute_train_mean(X[train], X[test])
        if args.method == "rf":
            prediction[test] = fit_predict_rf(
                Xtrain, y[train], Xtest, args.rf_n_estimators,
                args.rf_max_features, args.rf_min_samples_leaf, args.threads, args.seed + fold,
            )
            print(f"[INFO] feature={args.feature_name} trait={args.trait} method={args.method} fold={fold}", flush=True)
            continue
        else:
            try:
                from lightgbm import LGBMRegressor
            except ImportError as error:
                raise RuntimeError("lightgbm is required by the GS environment") from error
            model = LGBMRegressor(
                objective="regression",
                n_estimators=args.lgbm_n_estimators,
                learning_rate=args.lgbm_learning_rate,
                num_leaves=args.lgbm_num_leaves,
                subsample=args.lgbm_subsample,
                colsample_bytree=args.lgbm_colsample_bytree,
                n_jobs=args.threads,
                random_state=args.seed + fold,
                verbosity=-1,
            )
        model.fit(Xtrain, y[train])
        prediction[test] = model.predict(Xtest)
        print(f"[INFO] feature={args.feature_name} trait={args.trait} method={args.method} fold={fold}", flush=True)
    if not np.isfinite(prediction).all():
        raise RuntimeError("missing OOF predictions")

    oof = folds[["sample", "trait", "value", "fold"]].rename(columns={"value": "y_true"}).copy()
    oof["y_pred"] = prediction
    oof["feature"] = args.feature_name
    oof["method"] = args.method
    oof["feature_ncol"] = int(X.shape[1])
    row = {"trait": args.trait, "feature": args.feature_name, "method": args.method, "feature_ncol": int(X.shape[1]), **metrics(y, prediction)}
    for path in (args.out_oof, args.out_metrics):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    oof.to_csv(args.out_oof, sep="\t", index=False)
    pd.DataFrame([row]).to_csv(args.out_metrics, sep="\t", index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
