#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Assemble sample-level SNP embeddings from VCF genotypes.

Match feature-window midpoints to normalized VCF coordinates. Sort SNP keys
lexicographically and dimensions numerically. Use reference embeddings for
missing or unrecognized genotypes to preserve a fixed feature count.
"""

import argparse
import csv
import pandas as pd
import pysam
from collections import defaultdict


# ================================
# Chromosome identifier normalization
# ================================
def normalize_chrom(chrom):
    """Normalize chromosome labels such as chr01, Chr1, and 01 to 1."""
    c = str(chrom).lower()
    if c.startswith('chr'):
        c = c[3:]
    # Remove leading zeros, for example converting 01 to 1.
    c = c.lstrip('0')
    if c == '':
        c = '0'
    return c


# ================================
# SNP-to-feature-column mapping
# ================================
def extract_snp_dim_map(columns):
    """Map normalized chromosome:position keys to embedding dimension columns.

    Parse coordinates after an optional pipe-delimited SNP identifier.
    Convert window intervals to integer midpoints to match VCF positions.
    """
    snp_map = defaultdict(list)

    for col in columns:
        if '_d' not in col:
            continue

        # Remove the embedding dimension suffix from the column name.
        base = col.split('_d')[0]

        # Extract genomic coordinates after the optional SNP identifier.
        if '|' in base:
            base = base.split('|', 1)[1]

        if ':' in base:
            chrom_raw, pos_raw = base.split(':', 1)
            chrom = normalize_chrom(chrom_raw)

            # Use the interval midpoint to match VCF positions.
            if '-' in pos_raw:
                start_str, end_str = pos_raw.split('-')
                try:
                    start_pos = int(start_str)
                    end_pos = int(end_str)
                    # Calculate the integer midpoint of the window.
                    pos = str((start_pos + end_pos) // 2)
                except ValueError:
                    pos = pos_raw
            else:
                pos = pos_raw

            snp_id = f"{chrom}:{pos}"
            snp_map[snp_id].append(col)

    return snp_map


# ================================
# Feature matrix loading
# ================================
def load_feature_dicts(path):
    print("[+] 正在加载特征矩阵 (Feature Matrix)...")

    df = pd.read_csv(path, na_values='-')
    df = df.fillna(0)
    df.set_index('sample', inplace=True)

    snp_dim_map = extract_snp_dim_map(df.columns)

    if len(snp_dim_map) == 0:
        raise ValueError("[!] 在特征 CSV 中未找到有效的 SNP 列 (需要包含 _d0, _d1 等标识)")

    # Sort dimensions numerically so that d2 precedes d10.
    for snp in snp_dim_map:
        snp_dim_map[snp] = sorted(snp_dim_map[snp], key=lambda x: int(x.split('_d')[-1]))

    first_snp = next(iter(snp_dim_map))
    dim = len(snp_dim_map[first_snp])

    print(f"[+] 解析完成: 提取到 {len(snp_dim_map)} 个有效特征窗口, 每个窗口维度 (dim) = {dim}")

    try:
        ref_row = df.loc['rice_4k_hom_ref_windows']
        alt_row = df.loc['rice_4k_hom_alt_windows']
        het_row = df.loc['rice_4k_het_iupac_windows']
    except KeyError as e:
        raise KeyError(f"[!] 特征文件中缺少必要的行: {e}. 请检查 sample 列是否包含 ref/alt/het 的指定名称。")

    ref_feats, alt_feats, het_feats = {}, {}, {}

    for snp, cols in snp_dim_map.items():
        ref_feats[snp] = ref_row[cols].astype(float).values.tolist()
        alt_feats[snp] = alt_row[cols].astype(float).values.tolist()
        het_feats[snp] = het_row[cols].astype(float).values.tolist()

    return snp_dim_map, ref_feats, alt_feats, het_feats, dim


# ================================
# Sample feature matrix assembly
# ================================
def main(vcf_path, feature_csv, out_csv):
    snp_dim_map, ref_feats, alt_feats, het_feats, dim = load_feature_dicts(feature_csv)
    ordered_snps = sorted(snp_dim_map.keys())

    print(f"[+] 正在读取 VCF: {vcf_path}")
    vcf = pysam.VariantFile(vcf_path)
    samples = list(vcf.header.samples)
    print(f"[+] 检测到样本数量: {len(samples)}")

    vcf_data = {}
    matched_count = 0
    for rec in vcf.fetch():
        chrom = normalize_chrom(rec.chrom)
        snp_id = f"{chrom}:{rec.pos}"

        if snp_id in snp_dim_map:
            vcf_data[snp_id] = {s: rec.samples[s].get('GT') for s in samples}
            matched_count += 1

    print(f"[+] VCF 位点匹配成功: {matched_count} / {len(snp_dim_map)}")
    if matched_count == 0:
        print("[!] 警告: VCF与特征矩阵位点匹配数为0，请检查染色体命名或位置坐标！")

    print("[+] 正在组装特征矩阵 (缺失/未知基因型将使用参考基因型填充)...")
    final_output = []

    for sample in samples:
        sample_row = [sample]

        for snp in ordered_snps:
            # Use the reference vector for missing or unrecognized genotypes.
            vec = ref_feats[snp]

            if snp in vcf_data:
                gt = vcf_data[snp][sample]

                if gt == (0, 0):
                    vec = ref_feats[snp]
                elif gt == (1, 1):
                    vec = alt_feats[snp]
                elif gt == (0, 1) or gt == (1, 0):
                    vec = het_feats[snp]

            sample_row.extend(vec)

        final_output.append(sample_row)

    print(f"[+] 正在写出到: {out_csv}")

    # Reuse feature column names in the same order as the assembled vectors.
    header = ['sample']
    for snp in ordered_snps:
        header.extend(snp_dim_map[snp])

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(final_output)

    print("[+] 任务完成！")


# ================================
# CLI
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCF to SNP Embedding Generator")
    parser.add_argument("--vcf", required=True, help="输入 VCF 文件路径")
    parser.add_argument("--feats", required=True, help="输入特征 CSV 文件路径")
    parser.add_argument("--out", required=True, help="输出 CSV 文件路径")

    args = parser.parse_args()
    main(args.vcf, args.feats, args.out)
