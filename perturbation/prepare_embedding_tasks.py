#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except Exception as exc:
    raise SystemExit('PyYAML is required.') from exc


DEFAULT_METHODS = ['reml_gblup', 'rrblup', 'rf', 'lightgbm']
MODEL_ORDER = ['evo2', 'dnabert2', 'plantcad', 'agront']


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def numeric_or_nan(row, column):
    if row is None or column not in row:
        return np.nan
    return pd.to_numeric(row[column], errors='coerce')


def main():
    parser = argparse.ArgumentParser(description='Build the complete Maize1404 embedding perturbation task table.')
    parser.add_argument('--config', default='analysis/config/maize1404_downstream.yaml')
    parser.add_argument('--outdir', default='analysis_outputs/Maize1404')
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    parser.add_argument('--output', default='analysis/config/maize1404_embedding_perturbation_all_tasks.tsv')
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    traits = [str(x) for x in cfg['traits']]
    windows = [int(x) for x in cfg.get('windows', [200, 500, 1000])]
    methods = [str(x) for x in args.methods]
    gp_path = Path(args.outdir) / 'tables' / 'gp_delta_vs_raw.tsv'
    gp = pd.read_csv(gp_path, sep='\t')
    gp['window'] = pd.to_numeric(gp['window'], errors='coerce')

    rows = []
    missing_effects = []
    missing_original_metrics = []
    task_id = 0
    for trait in traits:
        for method in methods:
            for model in MODEL_ORDER:
                run_cfg = cfg['gp_runs'][model]
                for window in windows:
                    feature_label = f'{model}_w{window}'
                    effect_path = Path(args.outdir) / 'method_effects_all' / trait / method / f'{trait}.{method}.{feature_label}.effects.tsv.gz'
                    raw_effect_path = Path(args.outdir) / 'method_effects_all' / trait / method / f'{trait}.{method}.raw.effects.tsv.gz'
                    if not effect_path.exists() or not raw_effect_path.exists():
                        missing_effects.append((trait, method, feature_label, str(effect_path), str(raw_effect_path)))
                        continue

                    metadata = pd.read_csv(
                        effect_path,
                        sep='\t',
                        usecols=lambda value: value in {'feature_npz', 'folds', 'feature_label'},
                        nrows=1,
                    ).iloc[0]
                    feature_npz = str(metadata['feature_npz'])
                    folds = str(metadata['folds'])
                    if not os.path.exists(feature_npz):
                        raise RuntimeError(f'missing feature matrix referenced by {effect_path}: {feature_npz}')
                    if not os.path.exists(folds):
                        raise RuntimeError(f'missing fold table referenced by {effect_path}: {folds}')

                    emb_match = gp[
                        gp['trait'].astype(str).eq(trait)
                        & gp['method'].astype(str).eq(method)
                        & gp['embedding_model'].astype(str).eq(model)
                        & gp['window'].eq(window)
                        & gp['feature_class'].astype(str).eq('embedding')
                    ]
                    emb_row = emb_match.iloc[0] if len(emb_match) else None
                    raw_match = gp[
                        gp['trait'].astype(str).eq(trait)
                        & gp['method'].astype(str).eq(method)
                        & gp['source_run'].astype(str).eq(model)
                        & gp['feature_class'].astype(str).eq('raw')
                    ]
                    raw_row = raw_match.iloc[0] if len(raw_match) else None
                    if emb_row is None:
                        missing_original_metrics.append((trait, method, model, window))

                    original_raw = numeric_or_nan(emb_row, 'raw_pearson_r')
                    if not np.isfinite(original_raw):
                        original_raw = numeric_or_nan(raw_row, 'pearson_r')
                    rows.append(
                        {
                            'task_id': task_id,
                            'trait': trait,
                            'method': method,
                            'embedding_model': model,
                            'window': window,
                            'feature': str(emb_row['feature']) if emb_row is not None else feature_label,
                            'feature_npz': feature_npz,
                            'folds': folds,
                            'effect_path': str(effect_path),
                            'raw_effect_path': str(raw_effect_path),
                            'original_embedding_pearson_r': numeric_or_nan(emb_row, 'pearson_r'),
                            'original_raw_pearson_r': original_raw,
                            'original_delta_pearson_r_vs_raw': numeric_or_nan(emb_row, 'delta_pearson_r_vs_raw'),
                            'source_run': model,
                        }
                    )
                    task_id += 1

    if missing_effects:
        first = '\n'.join(map(str, missing_effects[:10]))
        raise RuntimeError(f'{len(missing_effects)} perturbation tasks lack effect inputs. First entries:\n{first}')

    tasks = pd.DataFrame(rows)
    expected = len(traits) * len(methods) * len(MODEL_ORDER) * len(windows)
    if len(tasks) != expected:
        raise RuntimeError(f'task coverage mismatch: expected {expected}, built {len(tasks)}')
    key = ['trait', 'method', 'embedding_model', 'window']
    if tasks.duplicated(key).any():
        raise RuntimeError(f'duplicate perturbation task keys:\n{tasks[tasks.duplicated(key, keep=False)].to_string(index=False)}')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(output, sep='\t', index=False)
    print(f'[INFO] wrote {len(tasks)} tasks to {output}')
    print(f'[INFO] coverage: traits={len(traits)} methods={len(methods)} models={len(MODEL_ORDER)} windows={len(windows)}')
    print(f'[INFO] tasks without a pre-existing embedding metric: {len(missing_original_metrics)}')
    for item in missing_original_metrics:
        print(f'[WARN] original metric absent: trait={item[0]} method={item[1]} model={item[2]} window={item[3]}')


if __name__ == '__main__':
    main()
