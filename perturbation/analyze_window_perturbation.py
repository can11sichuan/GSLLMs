#!/usr/bin/env python3
import argparse
import glob
import hashlib
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except Exception as exc:
    raise SystemExit('PyYAML is required.') from exc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings(
    'ignore',
    message='X does not have valid feature names, but LGBMRegressor was fitted with feature names',
)

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

SUPPORTED_METHODS = {'reml_gblup', 'gblup', 'rrblup', 'rf', 'lightgbm'}
MODEL_COLORS = {
    'embedding_top': '#C1121F',
    'raw_top': '#3B6EA8',
    'embedding_bottom': '#777777',
    'random_chrom_matched': '#BBBBBB',
}
STRATEGY_LABELS = {
    'embedding_top': 'Top embedding windows',
    'raw_top': 'Top raw-effect windows',
    'embedding_bottom': 'Bottom embedding windows',
    'random_chrom_matched': 'Chromosome-matched random',
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_config(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def metric_values(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if len(y) < 3 or np.std(y) <= 0 or np.std(pred) <= 0:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y, pred)[0, 1])
    return {
        'n': int(len(y)),
        'pearson_r': pearson,
        'r2': float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
        'rmse': float(math.sqrt(mean_squared_error(y, pred))) if len(y) else np.nan,
        'mae': float(mean_absolute_error(y, pred)) if len(y) else np.nan,
    }


def np_string_array(z, keys):
    for key in keys:
        if key in z.files:
            return np.asarray(z[key]).astype(str)
    return None


def load_feature_npz(path):
    z = np.load(path, allow_pickle=True)
    if 'X' not in z.files or 'samples' not in z.files:
        raise RuntimeError(f'feature NPZ must contain X and samples: {path}')
    X = np.asarray(z['X'], dtype=np.float32)
    samples = np.asarray(z['samples']).astype(str)
    snp_ids = np_string_array(z, ['snp_id', 'snp_ids', 'variant_id', 'feature_names'])
    if snp_ids is None:
        raise RuntimeError(f'could not find SNP ids in {path}; keys={z.files}')
    return X, samples, snp_ids


def align_xy(X_all, samples_all, folds_path):
    folds = pd.read_csv(folds_path, sep='\t')
    required = {'sample', 'value', 'fold'}
    missing = required - set(folds.columns)
    if missing:
        raise RuntimeError(f'fold file missing columns {missing}: {folds_path}')
    folds['sample'] = folds['sample'].astype(str)
    sample_to_idx = {s: i for i, s in enumerate(samples_all)}
    absent = [s for s in folds['sample'] if s not in sample_to_idx]
    if absent:
        raise RuntimeError(f'{len(absent)} fold samples absent from feature matrix. First: {absent[:5]}')
    idx = np.asarray([sample_to_idx[s] for s in folds['sample']], dtype=np.int64)
    y = pd.to_numeric(folds['value'], errors='coerce').to_numpy(dtype=float)
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


def standardize_train_test(Xtr, Xte):
    """Impute and standardize using training-fold statistics only."""
    Xtr_imp, Xte_imp, means = impute_train_test(Xtr, Xte)
    centered = Xtr_imp - means
    scales = np.sqrt(np.mean(centered * centered, axis=0)).astype(np.float32)
    scales[~np.isfinite(scales) | (scales <= 0)] = 1.0
    return (centered / scales).astype(np.float32), ((Xte_imp - means) / scales).astype(np.float32)


def reml_profile_nll(log_lambda, eigvals, uy, u1, df_reml):
    lam = float(np.exp(log_lambda))
    denom = eigvals + lam
    if np.any(denom <= 0):
        return np.inf
    vinv_11 = float(np.sum((u1 * u1) / denom))
    vinv_1y = float(np.sum((u1 * uy) / denom))
    if not np.isfinite(vinv_11) or vinv_11 <= 0:
        return np.inf
    mu = vinv_1y / vinv_11
    residual = uy - u1 * mu
    quad = float(np.sum((residual * residual) / denom))
    if not np.isfinite(quad) or quad <= 0:
        return np.inf
    value = float(np.sum(np.log(denom))) + math.log(vinv_11) + df_reml * math.log(quad / df_reml)
    return value if np.isfinite(value) else np.inf


def fit_reml_dual(K, y, min_lambda, max_lambda, grid_size):
    try:
        from scipy.optimize import minimize_scalar
    except Exception as exc:
        raise RuntimeError('scipy is required for REML-GBLUP perturbation') from exc

    K = 0.5 * (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T)
    y = np.asarray(y, dtype=np.float64)
    eigvals, U = np.linalg.eigh(K)
    lower = max(float(min_lambda), -float(np.min(eigvals)) + 1e-8)
    upper = max(float(max_lambda), lower * 10.0)
    uy = U.T @ y
    u1 = U.T @ np.ones(len(y), dtype=np.float64)
    df_reml = len(y) - 1

    def objective(value):
        return reml_profile_nll(value, eigvals, uy, u1, df_reml)

    grid = np.linspace(math.log(lower), math.log(upper), int(max(21, grid_size)))
    values = np.asarray([objective(x) for x in grid], dtype=float)
    if not np.any(np.isfinite(values)):
        raise RuntimeError('REML failed: all profile likelihood values are non-finite')
    best = int(np.nanargmin(values))
    left = float(grid[max(0, best - 2)])
    right = float(grid[min(len(grid) - 1, best + 2)])
    if left == right:
        log_lambda = float(grid[best])
    else:
        result = minimize_scalar(objective, bounds=(left, right), method='bounded', options={'xatol': 1e-6})
        log_lambda = float(result.x) if result.success and np.isfinite(result.fun) else float(grid[best])
    lam = float(np.exp(log_lambda))
    denom = eigvals + lam
    vinv_11 = float(np.sum((u1 * u1) / denom))
    vinv_1y = float(np.sum((u1 * uy) / denom))
    mu = float(vinv_1y / vinv_11)
    A = K + lam * np.eye(len(y), dtype=np.float64)
    try:
        dual = np.linalg.solve(A, y - mu)
    except np.linalg.LinAlgError:
        dual = np.linalg.lstsq(A, y - mu, rcond=None)[0]
    return mu, np.asarray(dual, dtype=np.float64), lam


def fit_linear_predictor(method, Xtr, ytr, Xte, args, fold_seed):
    """Return baseline predictions and standardized-space coefficients.

    Setting a masked feature to its training mean is exactly setting its
    standardized value to zero. This makes repeated rrBLUP/GBLUP masks cheap
    and is algebraically identical to recomputing predictions from a masked
    test matrix.
    """
    Xtr_s, Xte_s = standardize_train_test(Xtr, Xte)
    if method == 'rrblup':
        model = Ridge(alpha=float(args.rrblup_alpha))
        model.fit(Xtr_s, ytr)
        coef = np.asarray(model.coef_, dtype=np.float64)
        baseline = np.asarray(model.predict(Xte_s), dtype=float)
        return baseline, Xte_s, coef, f'alpha={float(args.rrblup_alpha):.8g}'

    p = max(1, Xtr_s.shape[1])
    K = (Xtr_s @ Xtr_s.T).astype(np.float64) / float(p)
    K = 0.5 * (K + K.T)
    diag_mean = float(np.mean(np.diag(K)))
    scale = float(p)
    if method == 'reml_gblup' and args.reml_normalize_k and np.isfinite(diag_mean) and diag_mean > 0:
        K /= diag_mean
        scale *= diag_mean

    if method == 'reml_gblup':
        intercept, dual, hyper = fit_reml_dual(
            K,
            ytr,
            args.reml_min_lambda,
            args.reml_max_lambda,
            args.reml_grid_size,
        )
        hyper_text = f'lambda={hyper:.8g},diag={diag_mean:.8g}'
    elif method == 'gblup':
        intercept = float(np.mean(ytr))
        hyper = float(args.gblup_lambda)
        A = K + hyper * np.eye(len(ytr), dtype=np.float64)
        try:
            dual = np.linalg.solve(A, np.asarray(ytr, dtype=np.float64) - intercept)
        except np.linalg.LinAlgError:
            dual = np.linalg.lstsq(A, np.asarray(ytr, dtype=np.float64) - intercept, rcond=None)[0]
        hyper_text = f'lambda={hyper:.8g}'
    else:
        raise RuntimeError(method)

    coef = np.asarray(Xtr_s.T @ dual, dtype=np.float64) / scale
    baseline = intercept + np.asarray(Xte_s @ coef, dtype=float)
    return baseline, Xte_s, coef, hyper_text


def make_model(method, args, seed):
    if method == 'rf':
        return RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_features=args.rf_max_features,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=args.threads,
            random_state=seed,
        )
    if method == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError('lightgbm is not installed in the active environment') from exc
        return LGBMRegressor(
            objective='regression',
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            num_leaves=args.lgbm_num_leaves,
            subsample=args.lgbm_subsample,
            colsample_bytree=args.lgbm_colsample_bytree,
            n_jobs=args.threads,
            random_state=seed,
            verbosity=-1,
        )
    raise RuntimeError(f'unsupported method for perturbation: {method}')


def config_feature_label(model, window):
    return f'{model}_w{int(window)}'


def effect_paths(outdir, trait, method, model, window):
    label = config_feature_label(model, window)
    base = Path(outdir) / 'method_effects_all' / trait / method
    return base / f'{trait}.{method}.{label}.effects.tsv.gz', base / f'{trait}.{method}.raw.effects.tsv.gz', label


def load_effect_in_feature_order(path, snp_ids):
    usecols = ['snp_id', 'effect', 'effect_rank', 'chr', 'bp']
    eff = pd.read_csv(path, sep='\t', usecols=lambda c: c in usecols)
    eff['snp_id'] = eff['snp_id'].astype(str)
    eff['effect'] = pd.to_numeric(eff['effect'], errors='coerce')
    eff = eff.drop_duplicates('snp_id').set_index('snp_id')
    aligned = eff.reindex(pd.Index(np.asarray(snp_ids).astype(str), name='snp_id')).reset_index()
    missing = aligned['effect'].isna().sum()
    if missing:
        raise RuntimeError(f'{missing} SNP effects missing when aligning {path}')
    return aligned


def mask_columns(window_idx, n_components):
    window_idx = np.asarray(window_idx, dtype=np.int64)
    if window_idx.size == 0:
        return np.array([], dtype=np.int64)
    return (window_idx[:, None] * int(n_components) + np.arange(int(n_components), dtype=np.int64)).ravel()


def top_k(order, k):
    k = int(max(0, min(k, len(order))))
    return np.asarray(order[:k], dtype=np.int64)


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
            replace = fallback.size < need
            out.extend(rng.choice(fallback, size=need, replace=replace).tolist())
    return np.asarray(out, dtype=np.int64)


def choose_jobs(outdir, traits, top_traits, method_override=None):
    response_path = Path(outdir) / 'tables' / 'trait_embedding_response.tsv'
    best_path = Path(outdir) / 'tables' / 'best_embedding_by_trait_method.tsv'
    response = pd.read_csv(response_path, sep='\t')
    best = pd.read_csv(best_path, sep='\t')
    response = response.sort_values('best_delta_pearson_r_vs_raw', ascending=False)
    if traits:
        selected_traits = traits
    else:
        selected_traits = response['trait'].head(int(top_traits)).astype(str).tolist()
    jobs = []
    for trait in selected_traits:
        resp = response[response['trait'].astype(str).eq(str(trait))]
        if resp.empty:
            raise RuntimeError(f'trait absent from response table: {trait}')
        resp = resp.iloc[0]
        part_all = best[best['trait'].astype(str).eq(str(trait))].copy()
        if method_override:
            part = part_all[part_all['method'].astype(str).eq(str(method_override))].sort_values('delta_pearson_r_vs_raw', ascending=False).head(1)
        else:
            part = part_all[part_all['method'].astype(str).eq(str(resp['best_method']))]
            part = part[part['feature'].astype(str).eq(str(resp['best_feature']))]
        if part.empty:
            part = part_all.sort_values('delta_pearson_r_vs_raw', ascending=False).head(1)
        row = part.iloc[0].to_dict()
        row['trait_level_best_delta_pearson_r_vs_raw'] = float(resp['best_delta_pearson_r_vs_raw'])
        row['trait_response_class'] = str(resp.get('response_class', ''))
        jobs.append(row)
    return pd.DataFrame(jobs)


def build_evaluations(embed_effect, raw_effect, chroms, fractions, random_reps, seed):
    n_windows = len(embed_effect)
    embed_order = np.argsort(-np.asarray(embed_effect, dtype=float))
    raw_order = np.argsort(-np.asarray(raw_effect, dtype=float))
    bottom_order = np.argsort(np.asarray(embed_effect, dtype=float))
    evals = [{'strategy': 'baseline', 'fraction': 0.0, 'replicate': 0, 'window_idx': np.array([], dtype=np.int64)}]
    rng = np.random.default_rng(seed)
    for frac in fractions:
        frac = float(frac)
        if frac <= 0:
            continue
        k = int(max(1, math.ceil(frac * n_windows)))
        emb_sel = top_k(embed_order, k)
        evals.append({'strategy': 'embedding_top', 'fraction': frac, 'replicate': 0, 'window_idx': emb_sel})
        evals.append({'strategy': 'raw_top', 'fraction': frac, 'replicate': 0, 'window_idx': top_k(raw_order, k)})
        evals.append({'strategy': 'embedding_bottom', 'fraction': frac, 'replicate': 0, 'window_idx': top_k(bottom_order, k)})
        for rep in range(1, int(random_reps) + 1):
            evals.append({
                'strategy': 'random_chrom_matched',
                'fraction': frac,
                'replicate': rep,
                'window_idx': random_chrom_matched(chroms, emb_sel, rng),
            })
    return evals


def run_one_job(job, args):
    trait = str(job['trait'])
    method = str(job['method'])
    if method not in SUPPORTED_METHODS:
        raise RuntimeError(f'perturbation script currently supports {sorted(SUPPORTED_METHODS)}, got {method}')
    model = str(job['embedding_model'])
    window = int(job['window'])
    feature_npz = str(job['feature_npz'])
    folds_path = str(job['folds'])
    default_effect_path, default_raw_effect_path, effect_label = effect_paths(args.outdir, trait, method, model, window)
    effect_path = Path(str(job.get('effect_path', default_effect_path)))
    raw_effect_path = Path(str(job.get('raw_effect_path', default_raw_effect_path)))
    if not effect_path.exists():
        raise RuntimeError(f'missing embedding effect file: {effect_path}')
    if not raw_effect_path.exists():
        raise RuntimeError(f'missing raw effect file: {raw_effect_path}')

    X_all, samples_all, snp_ids = load_feature_npz(feature_npz)
    X, y, folds = align_xy(X_all, samples_all, folds_path)
    n_windows = len(snp_ids)
    if X.shape[1] % n_windows != 0:
        raise RuntimeError(f'feature columns are not windows x components: X={X.shape}, windows={n_windows}')
    n_components = X.shape[1] // n_windows

    embed_eff = load_effect_in_feature_order(effect_path, snp_ids)
    raw_eff = load_effect_in_feature_order(raw_effect_path, snp_ids)
    chroms = embed_eff['chr'].fillna('unknown').astype(str).to_numpy()
    embed_values = embed_eff['effect'].to_numpy(dtype=float)
    raw_values = raw_eff['effect'].to_numpy(dtype=float)
    job_key = f'{trait}:{method}:{model}:{window}'
    job_offset = int.from_bytes(hashlib.sha256(job_key.encode('utf-8')).digest()[:4], 'big')
    perturbation_seed = int(args.seed) + job_offset
    evals = build_evaluations(embed_values, raw_values, chroms, args.fractions, args.random_reps, perturbation_seed)
    pred_by_eval = {i: np.full(len(y), np.nan, dtype=float) for i in range(len(evals))}
    fold_hypers = []

    unique_folds = sorted(int(x) for x in folds['fold'].unique())
    for fold in unique_folds:
        tr = np.where(folds['fold'].astype(int).to_numpy() != fold)[0]
        te = np.where(folds['fold'].astype(int).to_numpy() == fold)[0]
        fold_seed = int(args.seed) + int(fold)
        if method in {'rrblup', 'gblup', 'reml_gblup'}:
            baseline, Xte_s, coef, hyper_text = fit_linear_predictor(
                method,
                X[tr, :],
                y[tr],
                X[te, :],
                args,
                fold_seed,
            )
            fold_hypers.append(f'{fold}:{hyper_text}')
            for i, ev in enumerate(evals):
                cols = mask_columns(ev['window_idx'], n_components)
                pred = baseline - Xte_s[:, cols] @ coef[cols] if cols.size else baseline
                pred_by_eval[i][te] = np.asarray(pred, dtype=float)
        else:
            Xtr, Xte, means = impute_train_test(X[tr, :], X[te, :])
            fitted = make_model(method, args, fold_seed)
            fitted.fit(Xtr, y[tr])
            fold_hypers.append(f'{fold}:default')
            for i, ev in enumerate(evals):
                cols = mask_columns(ev['window_idx'], n_components)
                if cols.size:
                    Xmask = Xte.copy()
                    Xmask[:, cols] = means[cols]
                    pred = fitted.predict(Xmask)
                else:
                    pred = fitted.predict(Xte)
                pred_by_eval[i][te] = pred
        print(f'[INFO] perturb trait={trait} method={method} feature={effect_label} fold={fold} done', flush=True)

    rows = []
    baseline_metrics = metric_values(y, pred_by_eval[0])
    baseline_pearson = baseline_metrics['pearson_r']
    for i, ev in enumerate(evals):
        met = metric_values(y, pred_by_eval[i])
        rows.append({
            'task_id': job.get('task_id', np.nan),
            'trait': trait,
            'method': method,
            'feature': str(job['feature']),
            'effect_feature_label': effect_label,
            'embedding_model': model,
            'window': window,
            'feature_npz': feature_npz,
            'folds': folds_path,
            'strategy': ev['strategy'],
            'fraction': float(ev['fraction']),
            'n_masked_windows': int(len(ev['window_idx'])),
            'n_components_per_window': int(n_components),
            'n_masked_columns': int(len(ev['window_idx']) * n_components),
            'replicate': int(ev['replicate']),
            'perturbation_seed': perturbation_seed,
            'fitted_hyper_by_fold': ';'.join(fold_hypers),
            'baseline_pearson_r': baseline_pearson,
            'pearson_drop_vs_baseline': float(baseline_pearson - met['pearson_r']) if np.isfinite(baseline_pearson) and np.isfinite(met['pearson_r']) else np.nan,
            'trait_level_best_delta_pearson_r_vs_raw': float(job.get('trait_level_best_delta_pearson_r_vs_raw', np.nan)),
            'trait_response_class': str(job.get('trait_response_class', '')),
            'original_embedding_pearson_r': pd.to_numeric(job.get('original_embedding_pearson_r', np.nan), errors='coerce'),
            'original_raw_pearson_r': pd.to_numeric(job.get('original_raw_pearson_r', np.nan), errors='coerce'),
            'original_delta_pearson_r_vs_raw': pd.to_numeric(job.get('original_delta_pearson_r_vs_raw', np.nan), errors='coerce'),
            **met,
        })
    metrics = pd.DataFrame(rows)

    embed_order = np.argsort(-embed_values)
    raw_order = np.argsort(-raw_values)
    top_rows = []
    top_n = int(min(args.top_window_n, n_windows))
    raw_top1 = set(top_k(raw_order, max(1, math.ceil(0.01 * n_windows))).tolist())
    for rank, idx in enumerate(embed_order[:top_n], start=1):
        top_rows.append({
            'task_id': job.get('task_id', np.nan),
            'trait': trait,
            'method': method,
            'feature': str(job['feature']),
            'effect_feature_label': effect_label,
            'embedding_model': model,
            'window': window,
            'embedding_rank': rank,
            'window_index': int(idx),
            'snp_id': str(snp_ids[idx]),
            'chr': str(embed_eff.loc[idx, 'chr']),
            'bp': embed_eff.loc[idx, 'bp'],
            'embedding_effect': float(embed_values[idx]),
            'raw_effect': float(raw_values[idx]),
            'in_raw_top1pct': int(idx in raw_top1),
        })
    top_windows = pd.DataFrame(top_rows)

    overlap_rows = []
    for frac in args.overlap_fractions:
        k = int(max(1, math.ceil(float(frac) * n_windows)))
        emb_set = set(top_k(embed_order, k).tolist())
        raw_set = set(top_k(raw_order, k).tolist())
        obs = len(emb_set & raw_set)
        expected = k * k / float(n_windows)
        if HAS_SCIPY:
            pval = float(stats.hypergeom.sf(obs - 1, n_windows, k, k))
        else:
            pval = np.nan
        overlap_rows.append({
            'task_id': job.get('task_id', np.nan),
            'trait': trait,
            'method': method,
            'feature': str(job['feature']),
            'effect_feature_label': effect_label,
            'embedding_model': model,
            'window': window,
            'top_fraction': float(frac),
            'top_k': int(k),
            'overlap_n': int(obs),
            'expected_overlap_n': float(expected),
            'fold_enrichment': float(obs / expected) if expected > 0 else np.nan,
            'hypergeom_p': pval,
        })
    overlap = pd.DataFrame(overlap_rows)
    return metrics, top_windows, overlap


def plot_mask_drop(metrics, outdir):
    plot_dir = ensure_dir(Path(outdir) / 'plots' / 'perturbation')
    rows = sorted(metrics['trait'].dropna().unique())
    n = len(rows)
    ncols = min(3, max(1, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, trait in zip(axes.ravel(), rows):
        sub = metrics[(metrics['trait'] == trait) & (metrics['strategy'] != 'baseline')].copy()
        for strategy in ['random_chrom_matched', 'embedding_bottom', 'raw_top', 'embedding_top']:
            ss = sub[sub['strategy'] == strategy]
            if ss.empty:
                continue
            grouped = ss.groupby('fraction')['pearson_drop_vs_baseline']
            x = np.array(sorted(grouped.groups.keys()), dtype=float) * 100.0
            if strategy == 'random_chrom_matched':
                mean = grouped.mean().reindex(sorted(grouped.groups.keys())).to_numpy(dtype=float)
                lo = grouped.quantile(0.025).reindex(sorted(grouped.groups.keys())).to_numpy(dtype=float)
                hi = grouped.quantile(0.975).reindex(sorted(grouped.groups.keys())).to_numpy(dtype=float)
                ax.fill_between(x, lo, hi, color=MODEL_COLORS[strategy], alpha=0.35, linewidth=0)
                ax.plot(x, mean, color='#555555', linewidth=1.1, label=STRATEGY_LABELS[strategy])
            else:
                mean = grouped.mean().reindex(sorted(grouped.groups.keys())).to_numpy(dtype=float)
                ax.plot(x, mean, marker='o', linewidth=1.5, color=MODEL_COLORS[strategy], label=STRATEGY_LABELS[strategy])
        base = metrics[(metrics['trait'] == trait) & (metrics['strategy'] == 'baseline')]['pearson_r'].iloc[0]
        method = metrics[metrics['trait'] == trait]['method'].iloc[0]
        feature = metrics[metrics['trait'] == trait]['effect_feature_label'].iloc[0]
        ax.axhline(0, color='black', linewidth=0.7)
        ax.set_title(f'{trait}: {method}, {feature}\nbaseline r={base:.3f}', fontsize=9)
        ax.set_xlabel('Masked windows (%)')
        ax.set_ylabel('Pearson r drop')
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(Path(plot_dir) / 'embedding_window_mask_drop_curves.png', dpi=300)
    fig.savefig(Path(plot_dir) / 'embedding_window_mask_drop_curves.pdf')
    plt.close(fig)


def plot_top1_gain(metrics, outdir):
    plot_dir = ensure_dir(Path(outdir) / 'plots' / 'perturbation')
    sub = metrics[(metrics['strategy'] == 'embedding_top') & (np.isclose(metrics['fraction'], 0.01))].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    gain_col = 'selected_delta_pearson_r_vs_raw' if 'selected_delta_pearson_r_vs_raw' in sub.columns else 'trait_level_best_delta_pearson_r_vs_raw'
    ax.scatter(sub[gain_col], sub['pearson_drop_vs_baseline'], s=54, color='#C1121F', alpha=0.88, edgecolor='white', linewidth=0.6)
    x = sub[gain_col].to_numpy(dtype=float)
    y = sub['pearson_drop_vs_baseline'].to_numpy(dtype=float)
    if len(sub) >= 3 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xs, intercept + slope * xs, color='black', linewidth=1.1)
        if HAS_SCIPY:
            r, p = stats.pearsonr(x, y)
            ax.text(0.02, 0.98, f'r={r:.3f}, p={p:.3g}, n={len(sub)}', transform=ax.transAxes, va='top', fontsize=8)
    for _, row in sub.iterrows():
        ax.annotate(str(row['trait']), (row[gain_col], row['pearson_drop_vs_baseline']), xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xlabel('Selected embedding gain vs raw SNP')
    ax.set_ylabel('Pearson r drop after masking top 1% embedding windows')
    ax.set_title('Prediction gain vs perturbation sensitivity')
    fig.tight_layout()
    fig.savefig(Path(plot_dir) / 'top1pct_mask_drop_vs_embedding_gain.png', dpi=300)
    fig.savefig(Path(plot_dir) / 'top1pct_mask_drop_vs_embedding_gain.pdf')
    plt.close(fig)


def plot_overlap(overlap, outdir):
    plot_dir = ensure_dir(Path(outdir) / 'plots' / 'perturbation')
    sub = overlap[np.isclose(overlap['top_fraction'], 0.01)].copy()
    if sub.empty:
        sub = overlap.copy()
    sub = sub.sort_values('fold_enrichment', ascending=False)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(sub)), 4.2))
    colors = ['#C1121F' if p < 0.05 else '#3B6EA8' for p in sub['hypergeom_p'].fillna(1.0)]
    ax.bar(sub['trait'], sub['fold_enrichment'], color=colors, alpha=0.88)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Top embedding vs raw-effect overlap enrichment')
    ax.set_xlabel('Trait')
    ax.set_title('Top 1% window concordance with raw-effect signal')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(Path(plot_dir) / 'top1pct_embedding_raw_overlap_enrichment.png', dpi=300)
    fig.savefig(Path(plot_dir) / 'top1pct_embedding_raw_overlap_enrichment.pdf')
    plt.close(fig)


def attach_selected_embedding_metrics(metrics, outdir):
    best_path = Path(outdir) / 'tables' / 'best_embedding_by_trait_method.tsv'
    if not best_path.exists():
        return metrics
    best = pd.read_csv(best_path, sep='\t')
    keep = [
        'trait',
        'method',
        'feature',
        'pearson_r',
        'raw_pearson_r',
        'delta_pearson_r_vs_raw',
        'delta_r2_vs_raw',
        'relative_rmse_reduction_vs_raw',
    ]
    keep = [c for c in keep if c in best.columns]
    annot = best[keep].drop_duplicates(['trait', 'method', 'feature']).rename(columns={
        'pearson_r': 'selected_embedding_pearson_r',
        'raw_pearson_r': 'selected_raw_pearson_r',
        'delta_pearson_r_vs_raw': 'selected_delta_pearson_r_vs_raw',
        'delta_r2_vs_raw': 'selected_delta_r2_vs_raw',
        'relative_rmse_reduction_vs_raw': 'selected_relative_rmse_reduction_vs_raw',
    })
    drop_cols = [c for c in annot.columns if c not in {'trait', 'method', 'feature'} and c in metrics.columns]
    metrics = metrics.drop(columns=drop_cols, errors='ignore')
    return metrics.merge(annot, on=['trait', 'method', 'feature'], how='left')


def write_report(metrics, overlap, outdir):
    report = Path(outdir) / 'embedding_window_mask_perturbation.md'
    top1 = metrics[(metrics['strategy'] == 'embedding_top') & (np.isclose(metrics['fraction'], 0.01))]
    rand1 = metrics[(metrics['strategy'] == 'random_chrom_matched') & (np.isclose(metrics['fraction'], 0.01))]
    lines = ['# Embedding Window Mask Perturbation', '']
    lines.append(f'- Jobs analyzed: {metrics[["trait", "method", "effect_feature_label"]].drop_duplicates().shape[0]}.')
    lines.append(f'- Metrics rows: {len(metrics)}.')
    if not top1.empty:
        lines.append(f'- Mean Pearson r drop after masking top 1% embedding windows: {top1["pearson_drop_vs_baseline"].mean():.5f}.')
    if not rand1.empty:
        lines.append(f'- Mean Pearson r drop after chromosome-matched random 1% masks: {rand1["pearson_drop_vs_baseline"].mean():.5f}.')
    if not top1.empty and not rand1.empty:
        per_trait_rand = rand1.groupby('trait')['pearson_drop_vs_baseline'].mean()
        comp = top1.set_index('trait')['pearson_drop_vs_baseline'].to_frame('top1_drop').join(per_trait_rand.rename('random1_mean_drop'))
        comp['excess_drop'] = comp['top1_drop'] - comp['random1_mean_drop']
        lines.append(f'- Mean excess top-1% drop over random: {comp["excess_drop"].mean():.5f}.')
    if not overlap.empty:
        o1 = overlap[np.isclose(overlap['top_fraction'], 0.01)]
        if not o1.empty:
            lines.append(f'- Mean top-1% embedding/raw overlap enrichment: {o1["fold_enrichment"].mean():.3f}.')
            lines.append(f'- Traits with top-1% overlap hypergeometric p < 0.05: {int((o1["hypergeom_p"] < 0.05).sum())}/{len(o1)}.')
    lines.append('')
    if not top1.empty:
        lines.append('## Top 1% mask sensitivity')
        gain_cols = ['selected_delta_pearson_r_vs_raw'] if 'selected_delta_pearson_r_vs_raw' in top1.columns else ['trait_level_best_delta_pearson_r_vs_raw']
        show = top1[['trait', 'method', 'effect_feature_label', 'baseline_pearson_r', 'pearson_r', 'pearson_drop_vs_baseline'] + gain_cols].sort_values('pearson_drop_vs_baseline', ascending=False)
        lines.append(show.to_markdown(index=False))
        lines.append('')
    report.write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Window-level occlusion perturbation for SNP-window embeddings.')
    ap.add_argument('--config', default='analysis/config/maize1404_downstream.yaml')
    ap.add_argument('--outdir', default='analysis_outputs/Maize1404')
    ap.add_argument('--trait', action='append', default=[])
    ap.add_argument('--top-traits', type=int, default=6)
    ap.add_argument('--method', default='', choices=[''] + sorted(SUPPORTED_METHODS))
    ap.add_argument('--jobs-file', default='', help='Exact task table; bypasses best-feature selection.')
    ap.add_argument('--task-id', type=int, default=None, help='Run one task_id from --jobs-file.')
    ap.add_argument('--fractions', type=float, nargs='+', default=[0.002, 0.005, 0.01, 0.02, 0.05, 0.10])
    ap.add_argument('--overlap-fractions', type=float, nargs='+', default=[0.01, 0.05, 0.10])
    ap.add_argument('--random-reps', type=int, default=20)
    ap.add_argument('--top-window-n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20260427)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--rf-n-estimators', type=int, default=500)
    ap.add_argument('--rf-max-features', default='sqrt')
    ap.add_argument('--rf-min-samples-leaf', type=int, default=1)
    ap.add_argument('--lgbm-n-estimators', type=int, default=1200)
    ap.add_argument('--lgbm-learning-rate', type=float, default=0.03)
    ap.add_argument('--lgbm-num-leaves', type=int, default=31)
    ap.add_argument('--lgbm-subsample', type=float, default=0.8)
    ap.add_argument('--lgbm-colsample-bytree', type=float, default=0.8)
    ap.add_argument('--rrblup-alpha', type=float, default=1.0)
    ap.add_argument('--gblup-lambda', type=float, default=1.0)
    ap.add_argument('--reml-min-lambda', type=float, default=1e-6)
    ap.add_argument('--reml-max-lambda', type=float, default=1e6)
    ap.add_argument('--reml-grid-size', type=int, default=81)
    ap.add_argument('--reml-normalize-k', action='store_true', default=True)
    ap.add_argument('--no-reml-normalize-k', dest='reml_normalize_k', action='store_false')
    ap.add_argument('--result-name', default='embedding_window_mask')
    ap.add_argument('--collect-metrics-glob', default='')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    ensure_dir(Path(args.outdir) / 'tables')
    ensure_dir(Path(args.outdir) / 'plots' / 'perturbation')

    result_prefix = Path(args.outdir) / 'tables' / args.result_name
    metrics_path = result_prefix.with_name(result_prefix.name + '_metrics.tsv')
    top_path = result_prefix.with_name(result_prefix.name + '_top_windows.tsv')
    overlap_path = result_prefix.with_name(result_prefix.name + '_overlap.tsv')
    job_path = result_prefix.with_name(result_prefix.name + '_jobs.tsv')
    ensure_dir(result_prefix.parent)

    if args.collect_metrics_glob:
        metric_files = sorted(glob.glob(args.collect_metrics_glob))
        if not metric_files:
            raise RuntimeError(f'no metric files matched: {args.collect_metrics_glob}')
        top_files = [x.replace('_metrics.tsv', '_top_windows.tsv') for x in metric_files]
        overlap_files = [x.replace('_metrics.tsv', '_overlap.tsv') for x in metric_files]
        metrics = pd.concat([pd.read_csv(x, sep='\t') for x in metric_files], ignore_index=True)
        metrics = attach_selected_embedding_metrics(metrics, args.outdir)
        top_windows = pd.concat([pd.read_csv(x, sep='\t') for x in top_files if os.path.exists(x)], ignore_index=True)
        overlap = pd.concat([pd.read_csv(x, sep='\t') for x in overlap_files if os.path.exists(x)], ignore_index=True)
        metrics.to_csv(metrics_path, sep='\t', index=False)
        top_windows.to_csv(top_path, sep='\t', index=False)
        overlap.to_csv(overlap_path, sep='\t', index=False)
        print(f'[INFO] collected {len(metric_files)} metric files into {metrics_path}')
        if not args.no_plots:
            plot_mask_drop(metrics, args.outdir)
            plot_top1_gain(metrics, args.outdir)
            plot_overlap(overlap, args.outdir)
            write_report(metrics, overlap, args.outdir)
            print(f'[INFO] wrote perturbation plots and report under {args.outdir}')
        return

    if args.jobs_file:
        jobs = pd.read_csv(args.jobs_file, sep='\t')
        if args.task_id is not None:
            if 'task_id' not in jobs.columns:
                raise RuntimeError(f'--jobs-file has no task_id column: {args.jobs_file}')
            jobs = jobs[pd.to_numeric(jobs['task_id'], errors='coerce').eq(int(args.task_id))].copy()
            if jobs.empty:
                raise RuntimeError(f'task_id {args.task_id} absent from {args.jobs_file}')
        if args.trait:
            jobs = jobs[jobs['trait'].astype(str).isin(args.trait)].copy()
        if args.method:
            jobs = jobs[jobs['method'].astype(str).eq(args.method)].copy()
        if jobs.empty:
            raise RuntimeError('no perturbation jobs remain after applying task filters')
    else:
        jobs = choose_jobs(args.outdir, args.trait, args.top_traits, args.method or None)
    jobs.to_csv(job_path, sep='\t', index=False)
    print(f'[INFO] wrote jobs {job_path}')

    metric_parts = []
    top_parts = []
    overlap_parts = []
    for _, job in jobs.iterrows():
        print(f'[INFO] running perturbation trait={job["trait"]} method={job["method"]} feature={job["feature"]}', flush=True)
        metrics, top_windows, overlap = run_one_job(job, args)
        metric_parts.append(metrics)
        top_parts.append(top_windows)
        overlap_parts.append(overlap)

    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics = attach_selected_embedding_metrics(metrics, args.outdir)
    top_windows = pd.concat(top_parts, ignore_index=True)
    overlap = pd.concat(overlap_parts, ignore_index=True)
    metrics.to_csv(metrics_path, sep='\t', index=False)
    top_windows.to_csv(top_path, sep='\t', index=False)
    overlap.to_csv(overlap_path, sep='\t', index=False)
    print(f'[INFO] wrote {metrics_path}')
    print(f'[INFO] wrote {top_path}')
    print(f'[INFO] wrote {overlap_path}')

    if not args.no_plots:
        plot_mask_drop(metrics, args.outdir)
        plot_top1_gain(metrics, args.outdir)
        plot_overlap(overlap, args.outdir)
        write_report(metrics, overlap, args.outdir)
        print(f'[INFO] wrote perturbation plots and report under {args.outdir}')


if __name__ == '__main__':
    main()
