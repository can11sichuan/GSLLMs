#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate reference, alternate, and IUPAC-heterozygous FASTA windows from a VCF."""


import argparse
from pyfaidx import Fasta
import pysam

# IUPAC codes for heterozygous genotypes
iupac_map = {
    frozenset(['A', 'G']): 'R',
    frozenset(['A', 'C']): 'M',
    frozenset(['A', 'T']): 'W',
    frozenset(['C', 'G']): 'S',
    frozenset(['C', 'T']): 'Y',
    frozenset(['G', 'T']): 'K',
    frozenset(['A']): 'A',
    frozenset(['C']): 'C',
    frozenset(['G']): 'G',
    frozenset(['T']): 'T'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Generate three FASTA files for each SNP window: "
            "homo_ref, homo_alt, and het_iupac, ensuring each SNP is centered in its window."
        )
    )
    parser.add_argument('vcf', help="Input VCF file (.vcf or .vcf.gz with index)")
    parser.add_argument('ref', help="Reference genome FASTA file")
    parser.add_argument('out_prefix', help="Output prefix; generates <out_prefix>_hom_ref_windows.fa, etc.")
    parser.add_argument('win', type=int, help="Window half-size in bp, e.g. 200 to get ±200bp around each SNP")
    args = parser.parse_args()

    # Load reference genome
    ref = Fasta(args.ref)

    # Prepare VCF reader
    vcf = pysam.VariantFile(args.vcf)

    # Open three output FASTA files
    fout_ref = open(f"{args.out_prefix}_hom_ref_windows.fa", 'w')
    fout_alt = open(f"{args.out_prefix}_hom_alt_windows.fa", 'w')
    fout_het = open(f"{args.out_prefix}_het_iupac_windows.fa", 'w')

    # Process each SNP
    for rec in vcf.fetch():
        chrom = rec.chrom
        pos1 = rec.pos  # 1-based position
        ref_base = rec.ref
        alts = rec.alts
        if not alts:
            continue
        alt_base = alts[0]

        # Compute window boundaries
        seq_len = len(ref[chrom])
        start = max(1, pos1 - args.win)
        end = min(seq_len, pos1 + args.win)

        # Extract reference window
        seq_window_ref = ref[chrom][start-1:end].seq

        # Prepare mutable lists for alt and het
        seq_ref_list = list(seq_window_ref)
        seq_alt_list = seq_ref_list.copy()
        seq_het_list = seq_ref_list.copy()

        # SNP position in window (0-based)
        idx = pos1 - start
        seq_alt_list[idx] = alt_base
        seq_het_list[idx] = iupac_map.get(frozenset([ref_base, alt_base]), 'N')

        # Determine SNP ID and header
        snp_id = rec.id if rec.id and rec.id != '.' else f"{chrom}_{pos1}"
        header = f">{snp_id}|{chrom}:{start}-{end}\n"

        # Write sequences
        fout_ref.write(header)
        fout_ref.write(seq_window_ref + "\n")

        fout_alt.write(header)
        fout_alt.write(''.join(seq_alt_list) + "\n")

        fout_het.write(header)
        fout_het.write(''.join(seq_het_list) + "\n")

    # Close files
    fout_ref.close()
    fout_alt.close()
    fout_het.close()

    print(f"[+] Generated:")
    print(f"    {args.out_prefix}_hom_ref_windows.fa")
    print(f"    {args.out_prefix}_hom_alt_windows.fa")
    print(f"    {args.out_prefix}_het_iupac_windows.fa")
