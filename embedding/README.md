# Embedding stage

This stage should convert reference-genome sequence windows centered on the
selected SNPs into model embeddings, then convert the per-allele or per-SNP
embeddings into sample-level feature matrices.

## Required conceptual steps

1. Read a reference genome FASTA and SNP coordinates.
2. Extract fixed-width sequence windows around each SNP.
3. Create the alternate-allele and reference-allele sequence for each SNP.
4. Run the selected foundation model on unique sequences.
5. Aggregate hidden states into one vector per allele/window.
6. Combine allele-specific vectors with genotype dosage:
   - dosage 0: reference allele vector;
   - dosage 2: alternate allele vector;
   - dosage 1: arithmetic mean of the two vectors.
7. Save a feature matrix together with sample IDs, SNP IDs, window size,
   model name, and preprocessing metadata.

The model-specific entry points are included for Evo2, DNABERT-2, PlantCAD,
and AgroNT. They require a local model checkpoint supplied through
`--model-path`; checkpoints are not included. Do not add private FASTA files,
genotype files, sample IDs, or cluster paths.

Example interface:

```bash
python scripts/embedding/embed_snp_windows_evo2.py \
  --sites inputs/selected_sites.tsv \
  --fasta inputs/reference.fa \
  --flank 500 \
  --model-path /path/to/local/checkpoint \
  --outdir outputs/embedding/evo2_w500
```

The downstream scripts in `scripts/prediction/` consume the resulting NPZ
feature contract; the perturbation scripts consume the corresponding effect
tables and feature matrices.
