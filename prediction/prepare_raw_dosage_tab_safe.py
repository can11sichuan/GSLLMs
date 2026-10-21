#!/usr/bin/env python3
"""Build a dosage matrix without collapsing empty fields in PLINK2 .raw TSVs."""

import argparse
import json
import math
import os
import re

import numpy as np
import pandas as pd


FIXED_RAW_COLS = {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"}
VALID_ALLELES = {"A", "C", "G", "T"}


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def as_strings(values):
    return np.asarray(values).astype(str)


def load_variant_metadata(path):
    z = np.load(path, allow_pickle=True)
    required = {"snp_id", "a1", "a2"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(f"{path} is missing keys: {sorted(missing)}")
    return as_strings(z["snp_id"]), as_strings(z["a1"]), as_strings(z["a2"])


def read_plink_raw_tsv(path):
    # PLINK2 wrote missing dosages as empty tab-delimited fields. A whitespace
    # regex collapses those fields and silently assigns later values to the
    # wrong SNP columns. An explicit tab delimiter preserves their positions.
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="c",
    )
    if frame.empty:
        raise RuntimeError(f"empty PLINK raw table: {path}")
    if frame.columns.duplicated().any():
        dup = frame.columns[frame.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate columns in {path}: {dup[:10]}")
    sample_col = "IID" if "IID" in frame else "FID" if "FID" in frame else frame.columns[0]
    samples = frame[sample_col].astype(str).to_numpy()
    if len(set(samples)) != len(samples):
        raise RuntimeError(f"sample IDs are not unique in {path}")
    return frame, samples, sample_col


def raw_column_index(columns):
    genotype_columns = [c for c in columns if c not in FIXED_RAW_COLS]
    by_base = {}
    for column in genotype_columns:
        by_base.setdefault(column, []).append(column)
        if "_" in column:
            base, _ = column.rsplit("_", 1)
            by_base.setdefault(base, []).append(column)
    return set(genotype_columns), by_base


def choose_column(snp, a1, a2, exact, by_base):
    if snp in exact:
        return snp, "unknown_assume_A1"
    for allele, status in ((a1, "A1"), (a2, "A2")):
        candidate = f"{snp}_{allele}"
        if candidate in exact:
            return candidate, status
    candidates = by_base.get(snp, [])
    if not candidates:
        return None, "missing"
    column = candidates[0]
    allele = column.rsplit("_", 1)[-1].upper() if "_" in column else ""
    if allele == a1:
        return column, "A1"
    if allele == a2:
        return column, "A2"
    if allele in VALID_ALLELES:
        return column, f"other_counted_allele_{allele}"
    return column, "unknown_assume_A1"


def safe_corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return np.nan
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def compare_legacy(path, X, samples, snps):
    z = np.load(path, allow_pickle=True)
    legacy_samples = as_strings(z["samples"])
    legacy_snps = as_strings(z["snp_id"])
    sample_map = {value: i for i, value in enumerate(legacy_samples)}
    snp_map = {value: i for i, value in enumerate(legacy_snps)}
    missing_samples = [value for value in samples if value not in sample_map]
    missing_snps = [value for value in snps if value not in snp_map]
    if missing_samples or missing_snps:
        return {
            "legacy_npz": path,
            "comparable": 0,
            "missing_samples": len(missing_samples),
            "missing_snps": len(missing_snps),
        }
    rows = np.asarray([sample_map[value] for value in samples], dtype=int)
    cols = np.asarray([snp_map[value] for value in snps], dtype=int)
    legacy = np.asarray(z["X"])[np.ix_(rows, cols)].astype(np.float32)
    snp_corr = np.asarray([safe_corr(X[:, j], legacy[:, j]) for j in range(X.shape[1])])
    finite_corr = snp_corr[np.isfinite(snp_corr)]
    return {
        "legacy_npz": path,
        "comparable": 1,
        "missing_samples": 0,
        "missing_snps": 0,
        "matrix_pearson_r": safe_corr(X, legacy),
        "exact_cell_fraction": float(np.mean(X == legacy)),
        "mean_absolute_difference": float(np.mean(np.abs(X - legacy))),
        "snp_corr_median": float(np.median(finite_corr)) if finite_corr.size else np.nan,
        "snp_corr_q10": float(np.quantile(finite_corr, 0.10)) if finite_corr.size else np.nan,
        "snp_corr_q90": float(np.quantile(finite_corr, 0.90)) if finite_corr.size else np.nan,
    }


def parse_chromosome_position(snp):
    match = re.match(r"(?i)^chr([^_:.]+)[_:.](?:s[_-]?)?(\d+)$", str(snp))
    if not match:
        return "0", 0
    return match.group(1), int(match.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plink-raw", required=True)
    parser.add_argument("--genotype-npz", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-bim", required=True)
    parser.add_argument("--out-qc", required=True)
    parser.add_argument("--out-summary-json", required=True)
    parser.add_argument("--legacy-npz", action="append", default=[])
    args = parser.parse_args()

    snps, a1s, a2s = load_variant_metadata(args.genotype_npz)
    raw, samples, sample_col = read_plink_raw_tsv(args.plink_raw)
    exact, by_base = raw_column_index(raw.columns)

    X = np.empty((len(samples), len(snps)), dtype=np.float32)
    G = np.empty((len(samples), len(snps)), dtype=np.int8)
    report = []
    missing_columns = []
    invalid_values = []

    for j, (snp, a1, a2) in enumerate(zip(snps, a1s, a2s)):
        column, status = choose_column(str(snp), str(a1), str(a2), exact, by_base)
        if column is None:
            missing_columns.append(str(snp))
            continue
        raw_values = raw[column].replace("", np.nan)
        values = pd.to_numeric(raw_values, errors="coerce").to_numpy(dtype=float)
        observed = values[np.isfinite(values)]
        invalid = observed[~np.isin(observed, [0.0, 1.0, 2.0])]
        if invalid.size:
            invalid_values.append((str(snp), sorted(set(invalid.tolist()))[:5]))
        if status == "A2":
            values = 2.0 - values
        mean_value = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
        imputed = np.where(np.isfinite(values), values, mean_value)
        imputed = np.clip(imputed, 0.0, 2.0)
        X[:, j] = imputed.astype(np.float32)
        G[:, j] = np.clip(np.rint(imputed), 0, 2).astype(np.int8)
        report.append(
            {
                "snp_id": snp,
                "a1": a1,
                "a2": a2,
                "raw_column": column,
                "dosage_status": status,
                "n_missing": int(np.sum(~np.isfinite(values))),
                "missing_rate": float(np.mean(~np.isfinite(values))),
                "mean_impute_value": mean_value,
            }
        )

    if missing_columns:
        raise RuntimeError(f"{len(missing_columns)} SNPs lack raw columns; first: {missing_columns[:10]}")
    if invalid_values:
        raise RuntimeError(f"non-dosage values detected; first: {invalid_values[:5]}")

    for path in [args.out_npz, args.out_report, args.out_bim, args.out_qc, args.out_summary_json]:
        ensure_parent(path)

    np.savez_compressed(
        args.out_npz,
        X=X,
        G_state=G,
        samples=np.asarray(samples, dtype=object),
        snp_id=np.asarray(snps, dtype=object),
        a1=np.asarray(a1s, dtype=object),
        a2=np.asarray(a2s, dtype=object),
        feature_type=np.asarray("raw_dosage_A1_count_tab_safe", dtype=object),
    )
    report_frame = pd.DataFrame(report)
    report_frame.to_csv(args.out_report, sep="\t", index=False)

    with open(args.out_bim, "w", encoding="utf-8") as handle:
        for snp, a1, a2 in zip(snps, a1s, a2s):
            chrom, bp = parse_chromosome_position(snp)
            handle.write(f"{chrom}\t{snp}\t0\t{bp}\t{a1}\t{a2}\n")

    comparisons = [compare_legacy(path, X, samples, snps) for path in args.legacy_npz]
    pd.DataFrame(comparisons).to_csv(args.out_qc, sep="\t", index=False)
    summary = {
        "plink_raw": args.plink_raw,
        "genotype_npz": args.genotype_npz,
        "sample_column": sample_col,
        "n_samples": int(len(samples)),
        "n_snps": int(len(snps)),
        "raw_table_columns": int(raw.shape[1]),
        "true_missing_dosages": int(report_frame["n_missing"].sum()),
        "true_missing_rate": float(report_frame["n_missing"].sum() / X.size),
        "snps_with_missing": int((report_frame["n_missing"] > 0).sum()),
        "legacy_comparisons": comparisons,
    }
    with open(args.out_summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)

    print(f"[INFO] wrote corrected raw matrix {args.out_npz}: X={X.shape}")
    print(
        f"[INFO] preserved {summary['true_missing_dosages']} missing dosages "
        f"({summary['true_missing_rate']:.4%}) across {summary['snps_with_missing']} SNPs"
    )
    for row in comparisons:
        if row.get("comparable"):
            print(
                f"[QC] legacy={row['legacy_npz']} matrix_r={row['matrix_pearson_r']:.6f} "
                f"exact={row['exact_cell_fraction']:.4%} snp_r_median={row['snp_corr_median']:.6f}"
            )


if __name__ == "__main__":
    main()
