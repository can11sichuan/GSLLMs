#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from sklearn.decomposition import PCA
import joblib


def sarr(x):
    return np.asarray(x).astype(str)


def align_embedding_to_raw(raw_snps, emb_npz):
    emb_snps = sarr(emb_npz["snp_id"])
    idx = {s: i for i, s in enumerate(emb_snps)}
    missing = [s for s in raw_snps if s not in idx]
    if missing:
        raise RuntimeError(f"{len(missing)} raw SNPs not present in embedding NPZ. First 10: {missing[:10]}")
    order = np.asarray([idx[s] for s in raw_snps], dtype=np.int64)
    return order


def aggregate_matrix(Z, mode):
    if mode == "flatten":
        return Z.reshape(-1)
    if mode == "mean":
        return Z.mean(axis=0)
    if mode == "mean_std":
        return np.concatenate([Z.mean(axis=0), Z.std(axis=0)], axis=0)
    raise ValueError(f"unknown aggregation: {mode}")


def main():
    ap = argparse.ArgumentParser(description="Fit PCA on one reference individual's SNP embedding matrix and transform all individuals.")
    ap.add_argument("--raw-matrix", required=True, help="NPZ from prepare_raw_dosage_matrix.py")
    ap.add_argument("--genotype-npz", required=True, help="snp_genotype_embedding.npz from Evo2 embedding workflow")
    ap.add_argument("--n-components", type=int, default=8)
    ap.add_argument("--reference-sample", default="")
    ap.add_argument("--aggregation", choices=["flatten", "mean", "mean_std"], default="flatten")
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--out-features", required=True)
    ap.add_argument("--out-pca", required=True)
    ap.add_argument("--out-meta", required=True)
    args = ap.parse_args()

    raw = np.load(args.raw_matrix, allow_pickle=True)
    samples = sarr(raw["samples"])
    raw_snps = sarr(raw["snp_id"])
    G = raw["G_state"].astype(np.int16)
    if G.shape != (len(samples), len(raw_snps)):
        raise RuntimeError(f"G_state shape mismatch: {G.shape}, samples={len(samples)}, snps={len(raw_snps)}")

    emb = np.load(args.genotype_npz, allow_pickle=True)
    if "genotype_embedding" not in emb.files:
        raise RuntimeError(f"{args.genotype_npz} missing genotype_embedding")
    order = align_embedding_to_raw(raw_snps, emb)
    E = emb["genotype_embedding"][order, :, :].astype(np.float32)  # nSNP x 3 x dim
    if E.shape[0] != len(raw_snps) or E.shape[1] != 3:
        raise RuntimeError(f"unexpected genotype_embedding shape after alignment: {E.shape}")

    if args.reference_sample:
        matches = np.where(samples == args.reference_sample)[0]
        if len(matches) == 0:
            raise RuntimeError(f"reference_sample not found in samples: {args.reference_sample}")
        ref_idx = int(matches[0])
    else:
        ref_idx = 0
    reference_sample = str(samples[ref_idx])

    snp_index = np.arange(len(raw_snps), dtype=np.int64)
    ref_state = np.clip(G[ref_idx, :], 0, 2).astype(np.int64)
    ref_matrix = E[snp_index, ref_state, :]  # nSNP x dim

    if ref_matrix.shape[0] < args.n_components:
        raise RuntimeError(f"nSNP={ref_matrix.shape[0]} < n_components={args.n_components}")

    pca = PCA(n_components=args.n_components, svd_solver="randomized", random_state=args.seed)
    pca.fit(ref_matrix)

    first_Z = pca.transform(ref_matrix).astype(np.float32)
    first_feat = aggregate_matrix(first_Z, args.aggregation).astype(np.float32)
    X_feat = np.empty((len(samples), first_feat.shape[0]), dtype=np.float32)
    X_feat[ref_idx, :] = first_feat

    for i in range(len(samples)):
        if i == ref_idx:
            continue
        states = np.clip(G[i, :], 0, 2).astype(np.int64)
        M = E[snp_index, states, :]
        Z = pca.transform(M).astype(np.float32)
        X_feat[i, :] = aggregate_matrix(Z, args.aggregation).astype(np.float32)
        if (i + 1) % 100 == 0 or i == len(samples) - 1:
            print(f"[INFO] transformed {i + 1}/{len(samples)} samples")

    os.makedirs(os.path.dirname(args.out_features), exist_ok=True)
    np.savez_compressed(
        args.out_features,
        X=X_feat,
        samples=np.asarray(samples, dtype=object),
        snp_id=np.asarray(raw_snps, dtype=object),
        feature_type=np.asarray(f"evo2_pca{args.n_components}_{args.aggregation}", dtype=object),
        reference_sample=np.asarray(reference_sample, dtype=object),
        pca_components=np.asarray(args.n_components, dtype=np.int32),
        aggregation=np.asarray(args.aggregation, dtype=object),
        genotype_npz=np.asarray(args.genotype_npz, dtype=object),
    )
    joblib.dump(pca, args.out_pca)
    meta = {
        "raw_matrix": args.raw_matrix,
        "genotype_npz": args.genotype_npz,
        "n_samples": int(len(samples)),
        "n_snps": int(len(raw_snps)),
        "embedding_dim": int(E.shape[2]),
        "pca_components": int(args.n_components),
        "aggregation": args.aggregation,
        "feature_shape": list(X_feat.shape),
        "reference_sample": reference_sample,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    with open(args.out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] wrote features: {args.out_features}, shape={X_feat.shape}")
    print(f"[INFO] wrote PCA model: {args.out_pca}")
    print(f"[INFO] reference sample: {reference_sample}")


if __name__ == "__main__":
    main()
