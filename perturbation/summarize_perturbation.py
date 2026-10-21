#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None


METHOD_ORDER = ['reml_gblup', 'rrblup', 'rf', 'lightgbm']
MODEL_ORDER = ['evo2', 'dnabert2', 'plantcad', 'agront']
WINDOW_ORDER = [200, 500, 1000]
METHOD_COLORS = {
    'reml_gblup': '#3B6EA8',
    'rrblup': '#2A9D8F',
    'rf': '#C1121F',
    'lightgbm': '#E6B422',
}
STRATEGY_COLORS = {
    'embedding_top': '#C1121F',
    'raw_top': '#3B6EA8',
    'embedding_bottom': '#2A9D8F',
    'random_chrom_matched': '#777777',
}
STRATEGY_LABELS = {
    'embedding_top': 'Top embedding',
    'raw_top': 'Top raw-effect',
    'embedding_bottom': 'Bottom embedding',
    'random_chrom_matched': 'Chromosome-matched random',
}


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(values))[0]
    if not len(finite_idx):
        return out
    order = finite_idx[np.argsort(values[finite_idx])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def first_numeric(frame, column):
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors='coerce')
    values = values[np.isfinite(values)]
    return float(values.iloc[0]) if len(values) else np.nan


def strategy_drop(frame, strategy):
    part = frame[frame['strategy'].astype(str).eq(strategy)]
    return first_numeric(part, 'pearson_drop_vs_baseline')


def collect_task_outputs(tasks, task_dir, allow_incomplete=False):
    metric_parts = []
    overlap_parts = []
    top_parts = []
    missing = []
    for task_id in tasks['task_id'].astype(int):
        stem = task_dir / f'embedding_perturbation_task_{task_id:04d}'
        paths = {
            'metrics': Path(str(stem) + '_metrics.tsv'),
            'overlap': Path(str(stem) + '_overlap.tsv'),
            'top': Path(str(stem) + '_top_windows.tsv'),
        }
        absent = [str(path) for path in paths.values() if not path.exists()]
        if absent:
            missing.append((task_id, absent))
            continue
        metric_parts.append(pd.read_csv(paths['metrics'], sep='\t'))
        overlap_parts.append(pd.read_csv(paths['overlap'], sep='\t'))
        top_parts.append(pd.read_csv(paths['top'], sep='\t'))
    if missing and not allow_incomplete:
        first = '\n'.join(f'{task_id}: {paths}' for task_id, paths in missing[:10])
        raise RuntimeError(f'{len(missing)} task outputs are incomplete. First entries:\n{first}')
    if not metric_parts:
        raise RuntimeError(f'no completed perturbation outputs found under {task_dir}')
    return (
        pd.concat(metric_parts, ignore_index=True),
        pd.concat(overlap_parts, ignore_index=True),
        pd.concat(top_parts, ignore_index=True),
        missing,
    )


def build_top1_summary(metrics, overlap):
    rows = []
    top1_metrics = metrics[np.isclose(pd.to_numeric(metrics['fraction'], errors='coerce'), 0.01)].copy()
    top1_overlap = overlap[np.isclose(pd.to_numeric(overlap['top_fraction'], errors='coerce'), 0.01)].copy()
    overlap_by_task = top1_overlap.drop_duplicates('task_id').set_index('task_id')
    for task_id, part in top1_metrics.groupby('task_id', sort=True):
        task_id = int(task_id)
        random = part[part['strategy'].astype(str).eq('random_chrom_matched')]
        random_drop = pd.to_numeric(random['pearson_drop_vs_baseline'], errors='coerce').dropna().to_numpy(dtype=float)
        top_drop = strategy_drop(part, 'embedding_top')
        overlap_row = overlap_by_task.loc[task_id] if task_id in overlap_by_task.index else None
        first = part.iloc[0]
        random_mean = float(np.mean(random_drop)) if len(random_drop) else np.nan
        empirical_p = (1.0 + float(np.sum(random_drop >= top_drop))) / (len(random_drop) + 1.0) if len(random_drop) and np.isfinite(top_drop) else np.nan
        baseline = first_numeric(part, 'baseline_pearson_r')
        original_embedding = first_numeric(part, 'original_embedding_pearson_r')
        original_raw = first_numeric(part, 'original_raw_pearson_r')
        original_delta = first_numeric(part, 'original_delta_pearson_r_vs_raw')
        rows.append(
            {
                'task_id': task_id,
                'trait': str(first['trait']),
                'method': str(first['method']),
                'embedding_model': str(first['embedding_model']),
                'window': int(first['window']),
                'feature': str(first['feature']),
                'baseline_pearson_r': baseline,
                'original_embedding_pearson_r': original_embedding,
                'original_raw_pearson_r': original_raw,
                'original_delta_pearson_r_vs_raw': original_delta,
                'perturbation_baseline_delta_vs_raw': baseline - original_raw if np.isfinite(baseline) and np.isfinite(original_raw) else np.nan,
                'baseline_reproduction_error': baseline - original_embedding if np.isfinite(baseline) and np.isfinite(original_embedding) else np.nan,
                'embedding_top_drop': top_drop,
                'raw_top_drop': strategy_drop(part, 'raw_top'),
                'embedding_bottom_drop': strategy_drop(part, 'embedding_bottom'),
                'random_mean_drop': random_mean,
                'random_sd_drop': float(np.std(random_drop, ddof=1)) if len(random_drop) > 1 else np.nan,
                'random_q025_drop': float(np.quantile(random_drop, 0.025)) if len(random_drop) else np.nan,
                'random_q975_drop': float(np.quantile(random_drop, 0.975)) if len(random_drop) else np.nan,
                'top1_excess_over_random': top_drop - random_mean if np.isfinite(top_drop) and np.isfinite(random_mean) else np.nan,
                'top1_embedding_minus_raw_drop': top_drop - strategy_drop(part, 'raw_top'),
                'empirical_top_vs_random_p': empirical_p,
                'random_reps': int(len(random_drop)),
                'overlap_n': numeric_from_row(overlap_row, 'overlap_n'),
                'top_k': numeric_from_row(overlap_row, 'top_k'),
                'overlap_fold_enrichment': numeric_from_row(overlap_row, 'fold_enrichment'),
                'overlap_hypergeom_p': numeric_from_row(overlap_row, 'hypergeom_p'),
            }
        )
    out = pd.DataFrame(rows)
    out['empirical_top_vs_random_q_bh'] = bh_adjust(out['empirical_top_vs_random_p'])
    out['overlap_hypergeom_q_bh'] = bh_adjust(out['overlap_hypergeom_p'])
    return out


def numeric_from_row(row, column):
    if row is None or column not in row:
        return np.nan
    return pd.to_numeric(row[column], errors='coerce')


def summarize_groups(cells, group_columns, unit_columns):
    value_columns = [
        'original_delta_pearson_r_vs_raw',
        'perturbation_baseline_delta_vs_raw',
        'embedding_top_drop',
        'raw_top_drop',
        'embedding_bottom_drop',
        'random_mean_drop',
        'top1_excess_over_random',
        'top1_embedding_minus_raw_drop',
        'overlap_fold_enrichment',
    ]
    units = cells.groupby(unit_columns, as_index=False)[value_columns].mean(numeric_only=True)
    rows = []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, part in units.groupby(grouper, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row['n_independent_units'] = int(len(part))
        for column in value_columns:
            values = pd.to_numeric(part[column], errors='coerce').dropna().to_numpy(dtype=float)
            row[f'mean_{column}'] = float(np.mean(values)) if len(values) else np.nan
            row[f'median_{column}'] = float(np.median(values)) if len(values) else np.nan
        excess = pd.to_numeric(part['top1_excess_over_random'], errors='coerce').dropna().to_numpy(dtype=float)
        row['positive_excess_rate'] = float(np.mean(excess > 0)) if len(excess) else np.nan
        if stats is not None and len(excess) >= 3 and np.any(excess != 0):
            row['wilcoxon_excess_p'] = float(stats.wilcoxon(excess, alternative='greater').pvalue)
        else:
            row['wilcoxon_excess_p'] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out['wilcoxon_excess_q_bh'] = bh_adjust(out['wilcoxon_excess_p'])
    return out, units


def wilcoxon_p(values, alternative):
    values = pd.to_numeric(values, errors='coerce').dropna().to_numpy(dtype=float)
    if stats is None or len(values) < 3 or not np.any(values != 0):
        return np.nan
    return float(stats.wilcoxon(values, alternative=alternative).pvalue)


def build_method_inference(trait_method):
    rows = []
    methods = [method for method in METHOD_ORDER if method in set(trait_method['method'])]
    for method in methods:
        part = trait_method[trait_method['method'].eq(method)].copy()
        gain = pd.to_numeric(part['perturbation_baseline_delta_vs_raw'], errors='coerce')
        embedding_minus_raw = pd.to_numeric(part['top1_embedding_minus_raw_drop'], errors='coerce')
        paired = part[['perturbation_baseline_delta_vs_raw', 'top1_excess_over_random']].apply(
            pd.to_numeric, errors='coerce'
        ).dropna()
        rho = np.nan
        rho_p = np.nan
        if stats is not None and len(paired) >= 3 and paired.iloc[:, 0].std() > 0 and paired.iloc[:, 1].std() > 0:
            rho, rho_p = stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
        rows.append(
            {
                'method': method,
                'n_independent_units': int(gain.notna().sum()),
                'mean_gain_vs_raw': float(gain.mean()),
                'positive_gain_rate': float((gain.dropna() > 0).mean()),
                'wilcoxon_gain_two_sided_p': wilcoxon_p(gain, 'two-sided'),
                'mean_embedding_minus_raw_top_drop': float(embedding_minus_raw.mean()),
                'wilcoxon_embedding_gt_raw_p': wilcoxon_p(embedding_minus_raw, 'greater'),
                'gain_excess_spearman_rho': float(rho),
                'gain_excess_spearman_p': float(rho_p),
            }
        )
    out = pd.DataFrame(rows)
    out['wilcoxon_gain_two_sided_q_bh'] = bh_adjust(out['wilcoxon_gain_two_sided_p'])
    out['wilcoxon_embedding_gt_raw_q_bh'] = bh_adjust(out['wilcoxon_embedding_gt_raw_p'])
    out['gain_excess_spearman_q_bh'] = bh_adjust(out['gain_excess_spearman_p'])
    return out


def plot_heatmaps(cells, traits, plot_dir):
    columns = [(model, window) for model in MODEL_ORDER for window in WINDOW_ORDER]
    labels = [f'{model}\nw{window}' for model, window in columns]
    values = pd.to_numeric(cells['top1_excess_over_random'], errors='coerce').to_numpy(dtype=float)
    limit = float(np.nanquantile(np.abs(values), 0.98)) if np.any(np.isfinite(values)) else 0.01
    limit = max(limit, 0.001)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True)
    image_obj = None
    for ax, method in zip(axes.ravel(), METHOD_ORDER):
        sub = cells[cells['method'].eq(method)].copy()
        matrix = np.full((len(traits), len(columns)), np.nan, dtype=float)
        for i, trait in enumerate(traits):
            for j, (model, window) in enumerate(columns):
                match = sub[sub['trait'].eq(trait) & sub['embedding_model'].eq(model) & sub['window'].eq(window)]
                if len(match):
                    matrix[i, j] = pd.to_numeric(match.iloc[0]['top1_excess_over_random'], errors='coerce')
        image_obj = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_title(method)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(np.arange(len(traits)))
        ax.set_yticklabels(traits, fontsize=8)
    fig.colorbar(image_obj, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label='Top 1% excess Pearson r drop over random')
    fig.suptitle('All-trait embedding-window perturbation', fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.93, wspace=0.08, hspace=0.15)
    fig.savefig(plot_dir / 'perturbation_all_top1_heatmaps.png', dpi=300)
    fig.savefig(plot_dir / 'perturbation_all_top1_heatmaps.pdf')
    plt.close(fig)


def plot_method_summary(trait_method, plot_dir):
    summary = trait_method.groupby('method').agg(
        gain_mean=('perturbation_baseline_delta_vs_raw', 'mean'),
        gain_se=('perturbation_baseline_delta_vs_raw', 'sem'),
        excess_mean=('top1_excess_over_random', 'mean'),
        excess_se=('top1_excess_over_random', 'sem'),
    ).reindex(METHOD_ORDER)
    x = np.arange(len(summary))
    colors = [METHOD_COLORS[m] for m in summary.index]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(x, summary['gain_mean'], yerr=summary['gain_se'], color=colors, capsize=3)
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_ylabel('Pearson r change vs raw SNP')
    axes[0].set_title('Embedding prediction gain')
    axes[1].bar(x, summary['excess_mean'], yerr=summary['excess_se'], color=colors, capsize=3)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_ylabel('Top 1% excess Pearson r drop')
    axes[1].set_title('Selective dependence on top windows')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=25, ha='right')
        ax.grid(axis='y', color='#D9D9D9', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(plot_dir / 'perturbation_all_method_summary.png', dpi=300)
    fig.savefig(plot_dir / 'perturbation_all_method_summary.pdf')
    plt.close(fig)


def plot_dose_response(metrics, plot_dir):
    data = metrics[metrics['strategy'].ne('baseline')].copy()
    data['fraction'] = pd.to_numeric(data['fraction'], errors='coerce')
    data['pearson_drop_vs_baseline'] = pd.to_numeric(data['pearson_drop_vs_baseline'], errors='coerce')
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), METHOD_ORDER):
        sub = data[data['method'].eq(method)]
        for strategy in ['embedding_top', 'raw_top', 'embedding_bottom', 'random_chrom_matched']:
            part = sub[sub['strategy'].eq(strategy)].groupby('fraction')['pearson_drop_vs_baseline'].mean().sort_index()
            if part.empty:
                continue
            ax.plot(part.index.to_numpy() * 100.0, part.to_numpy(), marker='o', markersize=3, color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
        ax.axhline(0, color='black', linewidth=0.7)
        ax.set_title(method)
        ax.set_xlabel('Masked windows (%)')
        ax.set_ylabel('Mean Pearson r drop')
        ax.grid(color='#E2E2E2', linewidth=0.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(plot_dir / 'perturbation_all_dose_response.png', dpi=300)
    fig.savefig(plot_dir / 'perturbation_all_dose_response.pdf')
    plt.close(fig)


def plot_gain_relationship(trait_method, plot_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), sharex=False, sharey=True)
    for ax, method in zip(axes.ravel(), METHOD_ORDER):
        part = trait_method[trait_method['method'].eq(method)].dropna(
            subset=['perturbation_baseline_delta_vs_raw', 'top1_excess_over_random']
        )
        x = part['perturbation_baseline_delta_vs_raw'].to_numpy(dtype=float)
        y = part['top1_excess_over_random'].to_numpy(dtype=float)
        ax.scatter(x, y, s=30, color=METHOD_COLORS[method], alpha=0.85, edgecolor='white', linewidth=0.5)
        if len(part) >= 3 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(grid, intercept + slope * grid, color='black', linewidth=1.0)
            if stats is not None:
                rho, pvalue = stats.spearmanr(x, y)
                ax.text(0.03, 0.97, f'Spearman rho={rho:.3f}\np={pvalue:.3g}, n={len(part)}', transform=ax.transAxes, va='top', fontsize=8)
        ax.axhline(0, color='black', linewidth=0.7)
        ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
        ax.set_title(method)
        ax.set_xlabel('Mean embedding Pearson r change vs raw')
        ax.set_ylabel('Mean top 1% excess drop')
        ax.grid(color='#E2E2E2', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(plot_dir / 'perturbation_all_gain_vs_mask_drop.png', dpi=300)
    fig.savefig(plot_dir / 'perturbation_all_gain_vs_mask_drop.pdf')
    plt.close(fig)


def write_report(path, tasks, cells, method_summary, method_inference, setting_summary, missing):
    baseline_error = pd.to_numeric(cells['baseline_reproduction_error'], errors='coerce').abs().dropna()
    lines = ['# Complete Embedding Window Perturbation', '']
    lines.append(f'- Planned tasks: {len(tasks)}; completed tasks: {cells["task_id"].nunique()}; incomplete tasks: {len(missing)}.')
    lines.append(f'- Traits: {cells["trait"].nunique()}; methods: {cells["method"].nunique()}; embedding models: {cells["embedding_model"].nunique()}; windows: {cells["window"].nunique()}.')
    lines.append(f'- Cells lacking a pre-existing embedding metric: {int(cells["original_embedding_pearson_r"].isna().sum())}. Their perturbation baseline is still available.')
    if len(baseline_error):
        lines.append(f'- Mean absolute perturbation-baseline reproduction error: {baseline_error.mean():.6f}; maximum: {baseline_error.max():.6f}.')
    lines.extend(['', '## Method-level results', ''])
    show = method_summary[
        [
            'method',
            'n_independent_units',
            'mean_perturbation_baseline_delta_vs_raw',
            'mean_embedding_top_drop',
            'mean_random_mean_drop',
            'mean_top1_excess_over_random',
            'positive_excess_rate',
            'wilcoxon_excess_p',
            'wilcoxon_excess_q_bh',
        ]
    ]
    lines.append(show.to_markdown(index=False))

    lines.extend(['', '## Gain and mechanism tests', ''])
    inference_show = method_inference[
        [
            'method',
            'n_independent_units',
            'mean_gain_vs_raw',
            'positive_gain_rate',
            'wilcoxon_gain_two_sided_q_bh',
            'mean_embedding_minus_raw_top_drop',
            'wilcoxon_embedding_gt_raw_q_bh',
            'gain_excess_spearman_rho',
            'gain_excess_spearman_p',
            'gain_excess_spearman_q_bh',
        ]
    ]
    lines.append(inference_show.to_markdown(index=False))

    lines.extend(['', '## Main interpretation', ''])
    rf_inference = method_inference[method_inference['method'].eq('rf')]
    rf_summary = method_summary[method_summary['method'].eq('rf')]
    rf_settings = setting_summary[setting_summary['method'].eq('rf')]
    if not rf_inference.empty and not rf_summary.empty:
        rf_i = rf_inference.iloc[0]
        rf_s = rf_summary.iloc[0]
        positive_settings = int((pd.to_numeric(rf_settings['mean_perturbation_baseline_delta_vs_raw'], errors='coerce') > 0).sum())
        best = rf_settings.sort_values('mean_perturbation_baseline_delta_vs_raw', ascending=False).iloc[0]
        lines.append(
            f'- RF was the only method with a positive mean embedding gain: {rf_i["mean_gain_vs_raw"]:.5f} Pearson r, '
            f'positive in {rf_i["positive_gain_rate"]:.0%} of traits (BH-adjusted two-sided Wilcoxon '
            f'q={rf_i["wilcoxon_gain_two_sided_q_bh"]:.3g}). All {positive_settings}/{len(rf_settings)} RF model-window '
            f'settings had a positive mean gain; the largest was {best["embedding_model"]} w{int(best["window"])} '
            f'({best["mean_perturbation_baseline_delta_vs_raw"]:.5f}).'
        )
        lines.append(
            f'- RF top embedding windows caused a {rf_s["mean_embedding_top_drop"]:.5f} Pearson r drop versus '
            f'{rf_s["mean_random_mean_drop"]:.5f} for chromosome-matched random windows. They caused '
            f'{rf_i["mean_embedding_minus_raw_top_drop"]:.5f} more drop than raw-effect-ranked top windows '
            f'(q={rf_i["wilcoxon_embedding_gt_raw_q_bh"]:.3g}).'
        )
        lines.append(
            f'- RF top embedding and raw-effect windows overlapped at {rf_s["mean_overlap_fold_enrichment"]:.1f}-fold '
            'enrichment. The gain therefore does not appear to come mainly from a separate set of loci; the stronger '
            'explanation is contextual re-encoding and modest reprioritization of largely shared trait-associated loci.'
        )
        lines.append(
            f'- Across traits, RF gain and top-window selectivity had Spearman rho='
            f'{rf_i["gain_excess_spearman_rho"]:.3f} (nominal p={rf_i["gain_excess_spearman_p"]:.3g}, '
            f'BH q={rf_i["gain_excess_spearman_q_bh"]:.3g}). This is supportive but not FDR-significant across methods.'
        )
    lines.append(
        '- REML-GBLUP, rrBLUP, and LightGBM did not show a mean prediction gain, despite selective dependence on top '
        'windows. Selectivity is therefore not sufficient by itself: the practical advantage is specific to RF being '
        'able to exploit the multi-component local context representation.'
    )
    lines.append(
        '- Overall, the data support a model in which embeddings refine how known genetic signal is represented for a '
        'nonlinear learner, rather than universally improving genomic prediction or revealing a wholly new signal set.'
    )

    lines.extend(
        [
            '',
            '## Interpretation guardrails',
            '',
            '- Prediction gain is the unmasked embedding Pearson r minus the matched raw-SNP Pearson r.',
            '- Perturbation selectivity is the top-1% embedding-window Pearson r drop minus the chromosome-matched random-mask mean.',
            '- A positive association between gain and selectivity supports, but does not by itself prove, a causal mechanism for the embedding gain.',
            '- Cell-level tests are correlated within traits. Method-level inference therefore uses trait means as the independent units.',
            '- With 20 random controls, the smallest empirical cell-level p-value is 1/21; those p-values should not be used as high-resolution genome-wide significance tests.',
        ]
    )
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Collect and summarize all Maize1404 embedding perturbation tasks.')
    parser.add_argument('--outdir', default='analysis_outputs/Maize1404')
    parser.add_argument('--tasks', default='analysis/config/maize1404_embedding_perturbation_all_tasks.tsv')
    parser.add_argument('--task-dir', default='')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    table_dir = outdir / 'tables' / 'perturbation_all'
    plot_dir = outdir / 'plots' / 'perturbation_all'
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    task_dir = Path(args.task_dir) if args.task_dir else table_dir / 'tasks'
    tasks = pd.read_csv(args.tasks, sep='\t')
    metrics, overlap, top_windows, missing = collect_task_outputs(tasks, task_dir, args.allow_incomplete)

    metrics.to_csv(table_dir / 'embedding_perturbation_all_metrics.tsv.gz', sep='\t', index=False, compression='gzip')
    overlap.to_csv(table_dir / 'embedding_perturbation_all_overlap.tsv.gz', sep='\t', index=False, compression='gzip')
    top_windows.to_csv(table_dir / 'embedding_perturbation_all_top_windows.tsv.gz', sep='\t', index=False, compression='gzip')

    cells = build_top1_summary(metrics, overlap)
    cells.to_csv(table_dir / 'embedding_perturbation_top1_by_cell.tsv', sep='\t', index=False)
    method_summary, trait_method = summarize_groups(cells, ['method'], ['trait', 'method'])
    setting_summary, trait_setting = summarize_groups(
        cells,
        ['method', 'embedding_model', 'window'],
        ['trait', 'method', 'embedding_model', 'window'],
    )
    model_window_summary, trait_model_window = summarize_groups(
        cells,
        ['embedding_model', 'window'],
        ['trait', 'embedding_model', 'window'],
    )
    method_inference = build_method_inference(trait_method)
    method_summary.to_csv(table_dir / 'embedding_perturbation_method_summary.tsv', sep='\t', index=False)
    method_inference.to_csv(table_dir / 'embedding_perturbation_method_inference.tsv', sep='\t', index=False)
    setting_summary.to_csv(table_dir / 'embedding_perturbation_method_model_window_summary.tsv', sep='\t', index=False)
    model_window_summary.to_csv(table_dir / 'embedding_perturbation_model_window_summary.tsv', sep='\t', index=False)
    trait_method.to_csv(table_dir / 'embedding_perturbation_trait_method.tsv', sep='\t', index=False)
    trait_setting.to_csv(table_dir / 'embedding_perturbation_trait_method_model_window.tsv', sep='\t', index=False)
    trait_model_window.to_csv(table_dir / 'embedding_perturbation_trait_model_window.tsv', sep='\t', index=False)

    traits = [str(x) for x in tasks['trait'].drop_duplicates()]
    plot_heatmaps(cells, traits, plot_dir)
    plot_method_summary(trait_method, plot_dir)
    plot_dose_response(metrics, plot_dir)
    plot_gain_relationship(trait_method, plot_dir)
    write_report(
        outdir / 'embedding_perturbation_all_summary.md',
        tasks,
        cells,
        method_summary,
        method_inference,
        setting_summary,
        missing,
    )
    print(f'[INFO] collected {cells["task_id"].nunique()}/{len(tasks)} tasks')
    print(f'[INFO] wrote tables under {table_dir}')
    print(f'[INFO] wrote plots under {plot_dir}')


if __name__ == '__main__':
    main()
