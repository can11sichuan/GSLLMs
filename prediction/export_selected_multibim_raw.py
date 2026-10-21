#!/usr/bin/env python3
import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections import OrderedDict

import pandas as pd

FIXED_RAW_COLS = {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"}


def natural_key(path):
    base = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", base)]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def read_snplist(path):
    snps = []
    seen = set()
    dup = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip().split()
            if not s:
                continue
            snp = s[0]
            if snp in seen:
                dup.append(snp)
            else:
                seen.add(snp)
                snps.append(snp)
    if not snps:
        raise SystemExit(f"[ERROR] empty selected SNP list: {path}")
    if dup:
        raise SystemExit(f"[ERROR] selected snplist contains duplicated SNP IDs. First 10: {dup[:10]}")
    return snps, seen


def expand_prefixes(prefix_glob):
    # Accept either /path/Chr*.prefix or /path/Chr*.prefix.bim
    g = prefix_glob
    if g.endswith(".bim"):
        bims = glob.glob(g)
    else:
        bims = glob.glob(g + ".bim")
    bims = sorted(set(bims), key=natural_key)
    prefixes = [x[:-4] for x in bims if x.endswith(".bim")]
    return prefixes


def read_bim_snps(prefix, selected_set):
    bim = prefix + ".bim"
    out = []
    with open(bim, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            snp = parts[1]
            if snp in selected_set:
                out.append(snp)
    return out


def run_cmd(cmd, log_fh):
    log_fh.write("[CMD] " + " ".join(shlex.quote(str(x)) for x in cmd) + "\n")
    log_fh.flush()
    p = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}: {' '.join(cmd)}")


def read_raw(path):
    # PLINK uses empty tab fields for missing dosages. A whitespace regex would
    # shift all following SNP columns left for that sample.
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="c",
    )
    if df.empty:
        raise RuntimeError(f"empty raw file: {path}")
    return df


def get_id_cols(df):
    if "FID" in df.columns and "IID" in df.columns:
        return ["FID", "IID"]
    if "IID" in df.columns:
        return ["IID"]
    if "FID" in df.columns:
        return ["FID"]
    return [df.columns[0]]


def genotype_cols(df):
    return [c for c in df.columns if c not in FIXED_RAW_COLS]


def merge_raw_tables(raw_paths):
    base = None
    base_id_cols = None
    merged_geno_cols = []

    for raw_path in raw_paths:
        df = read_raw(raw_path)
        id_cols = get_id_cols(df)
        gcols = genotype_cols(df)
        if not gcols:
            continue
        if base is None:
            fixed_cols = [c for c in df.columns if c in FIXED_RAW_COLS]
            # Preserve standard PLINK fixed columns when present.
            base = df[fixed_cols].copy() if fixed_cols else df[id_cols].copy()
            base_id_cols = id_cols
            for c in id_cols:
                if c not in base.columns:
                    base[c] = df[c]
        else:
            # Check sample order. If not identical, merge by ID columns.
            if len(df) != len(base) or not all(df[id_cols[0]].astype(str).values == base[base_id_cols[0]].astype(str).values):
                join_cols = [c for c in base_id_cols if c in df.columns]
                if not join_cols:
                    raise RuntimeError(f"cannot align samples for {raw_path}; base IDs={base_id_cols}, current IDs={id_cols}")
                tmp = base[join_cols].merge(df[join_cols + gcols], on=join_cols, how="left", sort=False)
                if tmp.shape[0] != base.shape[0]:
                    raise RuntimeError(f"sample merge changed row count for {raw_path}")
                df = tmp
        repeated = sorted(set(merged_geno_cols) & set(gcols))
        if repeated:
            raise RuntimeError(f"duplicated genotype columns across chromosome raw files. First 10: {repeated[:10]}")
        base = pd.concat([base.reset_index(drop=True), df[gcols].reset_index(drop=True)], axis=1)
        merged_geno_cols.extend(gcols)
    if base is None or not merged_geno_cols:
        raise RuntimeError("no genotype columns were exported from any chromosome prefix")
    return base, merged_geno_cols


def main():
    ap = argparse.ArgumentParser(
        description="Export selected SNP dosage from split-by-chromosome PLINK bed/bim/fam files and merge into one PLINK-like .raw file."
    )
    ap.add_argument("--plink-prefix-glob", required=True, help="Prefix glob, e.g. /path/Chr*.filterWild.biallele.snp ; do not add .bed/.bim/.fam")
    ap.add_argument("--snplist", required=True)
    ap.add_argument("--plink-bin", default="plink2")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--extra-args", default="--allow-extra-chr", help="Extra args passed to PLINK. Empty string allowed.")
    ap.add_argument("--out-raw", required=True)
    ap.add_argument("--out-report", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_raw)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_report)), exist_ok=True)

    selected_order, selected_set = read_snplist(args.snplist)
    prefixes = expand_prefixes(args.plink_prefix_glob)
    if not prefixes:
        raise SystemExit(f"[ERROR] no BIM files matched plink_prefix_glob: {args.plink_prefix_glob}")

    missing_triples = []
    for pfx in prefixes:
        for ext in [".bed", ".bim", ".fam"]:
            if not os.path.exists(pfx + ext):
                missing_triples.append(pfx + ext)
    if missing_triples:
        raise SystemExit("[ERROR] missing PLINK files:\n" + "\n".join(missing_triples[:20]))

    report_rows = []
    raw_paths = []
    exported_snps = set()
    extra_args = shlex.split(args.extra_args) if args.extra_args.strip() else []

    with tempfile.TemporaryDirectory(prefix="multibim_raw_") as tmpdir:
        log_path = args.out_report + ".plink.log"
        with open(log_path, "w", encoding="utf-8") as log_fh:
            log_fh.write(f"[INFO] selected SNPs: {len(selected_order)}\n")
            log_fh.write(f"[INFO] matched prefixes: {len(prefixes)}\n")
            for pfx in prefixes:
                snps = read_bim_snps(pfx, selected_set)
                label = os.path.basename(pfx)
                report_rows.append({"prefix": pfx, "n_selected_in_bim": len(snps)})
                if not snps:
                    eprint(f"[SKIP] {label}: 0 selected SNPs")
                    continue
                extract_path = os.path.join(tmpdir, label + ".extract.snplist")
                with open(extract_path, "w", encoding="utf-8") as f:
                    for s in snps:
                        f.write(s + "\n")
                out_prefix = os.path.join(tmpdir, label)
                cmd = [args.plink_bin, "--bfile", pfx, "--extract", extract_path, "--export", "A", "--threads", str(args.threads)]
                cmd.extend(extra_args)
                cmd.extend(["--out", out_prefix])
                run_cmd(cmd, log_fh)
                raw_path = out_prefix + ".raw"
                if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
                    raise RuntimeError(f"PLINK did not create raw file: {raw_path}")
                raw_paths.append(raw_path)
                exported_snps.update(snps)
                eprint(f"[OK] {label}: exported {len(snps)} selected SNPs")

            missing_selected = [s for s in selected_order if s not in exported_snps]
            if missing_selected:
                raise SystemExit(f"[ERROR] {len(missing_selected)} selected SNPs were not found in chromosome BIM files. First 20: {missing_selected[:20]}")

            merged, merged_cols = merge_raw_tables(raw_paths)
            # Reorder genotype columns roughly by selected snplist order when possible.
            fixed_cols = [c for c in merged.columns if c in FIXED_RAW_COLS]
            col_by_base = OrderedDict()
            for c in merged_cols:
                base = c.rsplit("_", 1)[0] if "_" in c else c
                # Keep first if duplicate base names; prepare_raw_dosage_matrix handles allele suffix.
                col_by_base.setdefault(base, c)
            ordered_gcols = [col_by_base[s] for s in selected_order if s in col_by_base]
            rest = [c for c in merged_cols if c not in set(ordered_gcols)]
            final_cols = fixed_cols + ordered_gcols + rest
            merged = merged[final_cols]
            merged.to_csv(args.out_raw, sep="\t", index=False)
            log_fh.write(f"[INFO] wrote merged raw: {args.out_raw}, shape={merged.shape}\n")
            log_fh.write(f"[INFO] genotype columns merged: {len(merged_cols)}\n")

    pd.DataFrame(report_rows).to_csv(args.out_report, sep="\t", index=False)
    print(f"[INFO] wrote {args.out_raw}")
    print(f"[INFO] wrote {args.out_report}")
    print(f"[INFO] PLINK logs: {args.out_report}.plink.log")


if __name__ == "__main__":
    main()
