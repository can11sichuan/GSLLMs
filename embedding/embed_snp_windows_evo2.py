#!/usr/bin/env python3
import argparse
import csv
import gc
import gzip
import hashlib
import json
import logging
import os
import re
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import pysam
import torch

DNA_RE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Embed selected PLINK BIM SNP windows with Evo2. For each SNP and flank size, "
            "two allele-centered sequences are embedded; genotype embeddings are derived as "
            "0=A2/A2, 1=A1/A2 mean, 2=A1/A1."
        )
    )
    p.add_argument("--sites", required=True, help="selected *.sites.tsv from select_random_bim_snps.py")
    p.add_argument("--fasta", required=True)
    p.add_argument("--flank", type=int, required=True, help="bp on each side of SNP; sequence length=2*flank+1")
    p.add_argument("--outdir", required=True)
    p.add_argument("--model-name", default="evo2_7b")
    p.add_argument("--model-path", required=True)
    p.add_argument("--layer-name", default="blocks.28.mlp.l3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--chunk-len", type=int, default=2048)
    p.add_argument("--embedding-method", choices=["gaussian_pool", "direct_token"], default="gaussian_pool")
    p.add_argument("--pool-sigma", type=float, default=None, help="default: token_length/6")
    p.add_argument("--heterozygote-method", choices=["mean"], default="mean")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--cache-root", default=None)
    p.add_argument("--save-vector-cache", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def md5_text(x: str) -> str:
    return hashlib.md5(x.encode("utf-8")).hexdigest()


def normalize_seq(seq: str) -> str:
    seq = "".join(seq.split()).upper()
    if not seq:
        raise ValueError("empty sequence")
    if not DNA_RE.match(seq):
        raise ValueError(f"sequence has non A/C/G/T/N characters, first 80 bp: {seq[:80]}")
    return seq


class Evo2Embedder:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        layer_name: str,
        device: str,
        chunk_len: int,
        pool_sigma: Optional[float],
        dtype: str,
        cache_root: Optional[str],
        flank: int,
        embedding_method: str,
        save_vector_cache: bool = False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.layer_name = layer_name
        self.device = device
        self.chunk_len = int(chunk_len)
        self.pool_sigma = pool_sigma
        self.out_dtype = np.float16 if dtype == "float16" else np.float32
        self.cache_root = cache_root
        self.flank = flank
        self.embedding_method = embedding_method
        self.save_vector_cache = save_vector_cache
        self.cache_hits = 0
        self.cache_miss = 0
        self._prepare_env()
        self.model, self.tokenizer = self._load_model()

        if self.cache_root:
            ensure_dir(self.vector_cache_dir())

    def _prepare_env(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def _load_model(self):
        try:
            import transformer_engine.pytorch as te
            real_fp8_autocast = te.fp8_autocast

            @contextmanager
            def fp8_autocast_force_disabled(*args, **kwargs):
                kwargs["enabled"] = False
                with real_fp8_autocast(*args, **kwargs):
                    yield

            te.fp8_autocast = fp8_autocast_force_disabled
            logging.info("Transformer Engine FP8 autocast disabled")
        except Exception as e:
            logging.info("Transformer Engine patch not applied: %s", e)

        import evo2
        from evo2 import Evo2
        import evo2.models as evo2_models

        package_dir = os.path.dirname(os.path.abspath(evo2.__file__))
        mp = os.path.abspath(self.model_path)
        if os.path.isdir(mp):
            model_dir = mp
            local_model_path = os.path.join(model_dir, f"{self.model_name}.pt")
        else:
            local_model_path = mp
            model_dir = os.path.dirname(mp)
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"model file not found: {local_model_path}")

        @contextmanager
        def pushd(path: str):
            old = os.getcwd()
            os.chdir(path)
            try:
                yield
            finally:
                os.chdir(old)

        evo2_obj = None
        try:
            with pushd(package_dir):
                evo2_obj = Evo2(self.model_name, local_path=local_model_path)
            logging.info("Loaded Evo2 with local_path: %s", local_model_path)
        except Exception as e:
            logging.warning("local_path loading failed, trying snapshot patch: %s", e)

        if evo2_obj is None:
            old_snapshot_download = getattr(evo2_models, "snapshot_download", None)
            try:
                def _local_snapshot_download(*args, **kwargs):
                    return model_dir

                evo2_models.snapshot_download = _local_snapshot_download
                with pushd(package_dir):
                    evo2_obj = Evo2(self.model_name)
                logging.info("Loaded Evo2 with local snapshot patch: %s", model_dir)
            finally:
                if old_snapshot_download is not None:
                    evo2_models.snapshot_download = old_snapshot_download

        evo2_obj.model = evo2_obj.model.to(self.device).eval()
        return evo2_obj, evo2_obj.tokenizer

    def encode_seq(self, seq: str):
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(seq)
        return self.tokenizer.tokenize(seq)

    def vector_cache_dir(self):
        safe_layer = self.layer_name.replace("/", "_").replace(".", "_")
        return os.path.join(
            self.cache_root,
            self.model_name,
            safe_layer,
            self.embedding_method,
            f"flank{self.flank}",
        )

    def vector_cache_key(self, seq: str):
        payload = {
            "seq": seq,
            "model_name": self.model_name,
            "layer_name": self.layer_name,
            "chunk_len": self.chunk_len,
            "pool_sigma": self.pool_sigma,
            "flank": self.flank,
            "embedding_method": self.embedding_method,
        }
        return md5_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def vector_cache_path(self, key: str):
        return os.path.join(self.vector_cache_dir(), f"{key}.npy")

    @torch.no_grad()
    def embed_full_sequence(self, seq: str):
        seq = normalize_seq(seq)
        ids = self.encode_seq(seq)
        if len(ids) == 0:
            raise ValueError("token length is 0")
        parts = []
        for s in range(0, len(ids), self.chunk_len):
            chunk_ids = ids[s:s + self.chunk_len]
            input_ids = torch.tensor(chunk_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            _, emb_dict = self.model(
                input_ids,
                return_embeddings=True,
                layer_names=[self.layer_name],
            )
            emb = emb_dict[self.layer_name].squeeze(0)
            part = emb.detach().float().cpu().numpy()
            parts.append(part)
            del input_ids, emb_dict, emb, part
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.concatenate(parts, axis=0)

    def gaussian_pool_one(self, full_emb: np.ndarray, center_1based: int):
        full_emb = np.asarray(full_emb, dtype=np.float32)
        L = int(full_emb.shape[0])
        center = max(0, min(L - 1, int(center_1based) - 1))
        sigma = self.pool_sigma if self.pool_sigma is not None else (L / 6.0)
        if sigma <= 0:
            raise ValueError(f"invalid pool sigma: {sigma}")
        idxs = np.arange(L, dtype=np.float32)
        weights = np.exp(-((idxs - float(center)) ** 2) / (2.0 * float(sigma) * float(sigma))).astype(np.float32)
        vec = (weights[:, None] * full_emb).sum(axis=0) / float(weights.sum())
        return vec.astype(self.out_dtype, copy=False)

    def direct_token_one(self, full_emb: np.ndarray, center_1based: int):
        full_emb = np.asarray(full_emb, dtype=np.float32)
        L = int(full_emb.shape[0])
        center = max(0, min(L - 1, int(center_1based) - 1))
        return full_emb[center].astype(self.out_dtype, copy=False)

    def embed_center_vector(self, seq: str, center_1based: int):
        seq = normalize_seq(seq)
        key = self.vector_cache_key(seq)
        if self.cache_root:
            path = self.vector_cache_path(key)
            if os.path.exists(path):
                self.cache_hits += 1
                return np.load(path), key
        full_emb = self.embed_full_sequence(seq)
        if self.embedding_method == "gaussian_pool":
            vec = self.gaussian_pool_one(full_emb, center_1based)
        elif self.embedding_method == "direct_token":
            vec = self.direct_token_one(full_emb, center_1based)
        else:
            raise ValueError(f"unknown embedding method: {self.embedding_method}")
        self.cache_miss += 1
        if self.cache_root and self.save_vector_cache:
            ensure_dir(self.vector_cache_dir())
            np.save(self.vector_cache_path(key), vec)
        del full_emb
        return vec, key


def read_sites(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"selected_order", "bim_chrom", "fasta_chrom", "snp_id", "pos", "a1", "a2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"sites file missing columns: {sorted(missing)}")
        rows = [dict(r) for r in reader]
    if not rows:
        raise RuntimeError(f"sites file is empty: {path}")
    return rows


def ref_match_status(ref_base, a1, a2):
    ref_base = ref_base.upper()
    if ref_base == a1 and ref_base == a2:
        return "both"
    if ref_base == a1:
        return "A1"
    if ref_base == a2:
        return "A2"
    return "neither"


def fetch_window(fasta, chrom, pos, flank, center_base):
    # pysam fetch is 0-based, half-open. pos is 1-based.
    left = fasta.fetch(chrom, pos - flank - 1, pos - 1).upper()
    right = fasta.fetch(chrom, pos, pos + flank).upper()
    if len(left) != flank or len(right) != flank:
        raise RuntimeError(
            f"bad window length for {chrom}:{pos}, flank={flank}, left={len(left)}, right={len(right)}"
        )
    return normalize_seq(left + center_base.upper() + right)


def object_array(values):
    return np.array(values, dtype=object)


def main():
    args = parse_args()
    ensure_dir(args.outdir)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sites = read_sites(args.sites)
    fasta = pysam.FastaFile(args.fasta)
    center_1based = args.flank + 1

    embedder = Evo2Embedder(
        model_name=args.model_name,
        model_path=args.model_path,
        layer_name=args.layer_name,
        device=args.device,
        chunk_len=args.chunk_len,
        pool_sigma=args.pool_sigma,
        dtype=args.dtype,
        cache_root=args.cache_root,
        flank=args.flank,
        embedding_method=args.embedding_method,
        save_vector_cache=args.save_vector_cache,
    )

    allele_vectors = []
    trace_rows = []
    ref_mismatch_n = 0

    logging.info("Embedding %d SNPs with flank=%d, sequence length=%d", len(sites), args.flank, 2 * args.flank + 1)

    for i, row in enumerate(sites, start=1):
        snp_id = row["snp_id"]
        fchr = row["fasta_chrom"]
        pos = int(row["pos"])
        a1 = row["a1"].upper()
        a2 = row["a2"].upper()

        ref_base = fasta.fetch(fchr, pos - 1, pos).upper()
        match = ref_match_status(ref_base, a1, a2)
        if match == "neither":
            ref_mismatch_n += 1
            logging.warning("FASTA center base matches neither BIM allele: %s %s:%d FASTA=%s A1=%s A2=%s", snp_id, fchr, pos, ref_base, a1, a2)

        seq_a1 = fetch_window(fasta, fchr, pos, args.flank, a1)
        seq_a2 = fetch_window(fasta, fchr, pos, args.flank, a2)

        vec_a1, hash_a1 = embedder.embed_center_vector(seq_a1, center_1based=center_1based)
        vec_a2, hash_a2 = embedder.embed_center_vector(seq_a2, center_1based=center_1based)
        allele_vectors.append(np.stack([vec_a1, vec_a2], axis=0))

        trace_rows.append({
            "selected_order": row["selected_order"],
            "snp_id": snp_id,
            "bim_chrom": row["bim_chrom"],
            "fasta_chrom": fchr,
            "pos": str(pos),
            "a1": a1,
            "a2": a2,
            "ref_fasta": ref_base,
            "ref_match": match,
            "flank": str(args.flank),
            "window_length": str(2 * args.flank + 1),
            "center_pos_1based": str(center_1based),
            "seq_hash_A1": hash_a1,
            "seq_hash_A2": hash_a2,
        })

        if i % args.log_every == 0 or i == len(sites):
            logging.info("completed %d/%d SNPs; vector_cache_hit=%d miss=%d", i, len(sites), embedder.cache_hits, embedder.cache_miss)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    allele_arr = np.stack(allele_vectors, axis=0)
    # allele_arr[:, 0, :] is A1; allele_arr[:, 1, :] is A2.
    # PLINK dosage is conventionally A1 allele count: 0=A2/A2, 1=A1/A2, 2=A1/A1.
    if args.heterozygote_method == "mean":
        geno0 = allele_arr[:, 1, :]
        geno1 = ((allele_arr[:, 0, :].astype(np.float32) + allele_arr[:, 1, :].astype(np.float32)) / 2.0).astype(allele_arr.dtype)
        geno2 = allele_arr[:, 0, :]
        genotype_arr = np.stack([geno0, geno1, geno2], axis=1)
    else:
        raise ValueError(f"unsupported heterozygote method: {args.heterozygote_method}")

    snp_ids = [r["snp_id"] for r in sites]
    bim_chroms = [r["bim_chrom"] for r in sites]
    fasta_chroms = [r["fasta_chrom"] for r in sites]
    positions = np.array([int(r["pos"]) for r in sites], dtype=np.int64)
    a1s = [r["a1"].upper() for r in sites]
    a2s = [r["a2"].upper() for r in sites]
    ref_bases = [r["ref_fasta"] for r in trace_rows]
    ref_matches = [r["ref_match"] for r in trace_rows]

    allele_npz = os.path.join(args.outdir, "snp_allele_embedding.npz")
    genotype_npz = os.path.join(args.outdir, "snp_genotype_embedding.npz")
    trace_tsv = os.path.join(args.outdir, "snp_window_trace.tsv")
    meta_json = os.path.join(args.outdir, "run_meta.json")

    common_meta = dict(
        snp_id=object_array(snp_ids),
        bim_chrom=object_array(bim_chroms),
        fasta_chrom=object_array(fasta_chroms),
        pos=positions,
        a1=object_array(a1s),
        a2=object_array(a2s),
        ref_fasta=object_array(ref_bases),
        ref_match=object_array(ref_matches),
        flank=np.array(args.flank, dtype=np.int32),
        window_length=np.array(2 * args.flank + 1, dtype=np.int32),
        center_pos_1based=np.array(center_1based, dtype=np.int32),
        model_name=np.array(args.model_name, dtype=object),
        model_path=np.array(args.model_path, dtype=object),
        layer_name=np.array(args.layer_name, dtype=object),
        chunk_len=np.array(args.chunk_len, dtype=np.int32),
        embedding_method=np.array(args.embedding_method, dtype=object),
        pool_sigma=np.array(-1 if args.pool_sigma is None else args.pool_sigma, dtype=np.float32),
        dtype=np.array(str(allele_arr.dtype), dtype=object),
    )

    np.savez_compressed(
        allele_npz,
        allele_embedding=allele_arr,
        allele_order=object_array(["A1_BIM_col5", "A2_BIM_col6"]),
        **common_meta,
    )
    np.savez_compressed(
        genotype_npz,
        genotype_embedding=genotype_arr,
        genotype_order=object_array(["0:A2/A2", "1:A1/A2_mean", "2:A1/A1"]),
        plink_dosage_is_A1_count=np.array(True),
        heterozygote_method=np.array(args.heterozygote_method, dtype=object),
        **common_meta,
    )

    with open(trace_tsv, "w", newline="") as f:
        fieldnames = [
            "selected_order", "snp_id", "bim_chrom", "fasta_chrom", "pos", "a1", "a2",
            "ref_fasta", "ref_match", "flank", "window_length", "center_pos_1based",
            "seq_hash_A1", "seq_hash_A2",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(trace_rows)

    with open(meta_json, "w") as f:
        json.dump({
            "sites": args.sites,
            "fasta": args.fasta,
            "flank": args.flank,
            "window_length": 2 * args.flank + 1,
            "n_snps": len(sites),
            "allele_embedding_shape": list(allele_arr.shape),
            "genotype_embedding_shape": list(genotype_arr.shape),
            "allele_order": ["A1_BIM_col5", "A2_BIM_col6"],
            "genotype_order": ["0:A2/A2", "1:A1/A2_mean", "2:A1/A1"],
            "plink_dosage_is_A1_count": True,
            "ref_mismatch_neither_count": ref_mismatch_n,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "layer_name": args.layer_name,
            "device": args.device,
            "chunk_len": args.chunk_len,
            "embedding_method": args.embedding_method,
            "pool_sigma": args.pool_sigma,
            "heterozygote_method": args.heterozygote_method,
            "dtype": str(allele_arr.dtype),
            "cache_root": args.cache_root,
            "save_vector_cache": args.save_vector_cache,
            "cache_hits": embedder.cache_hits,
            "cache_miss": embedder.cache_miss,
            "outputs": {
                "allele_npz": allele_npz,
                "genotype_npz": genotype_npz,
                "trace_tsv": trace_tsv,
            },
        }, f, indent=2, ensure_ascii=False)

    logging.info("wrote allele embedding: %s", allele_npz)
    logging.info("wrote genotype embedding: %s", genotype_npz)
    logging.info("wrote trace table: %s", trace_tsv)
    logging.info("wrote metadata: %s", meta_json)


if __name__ == "__main__":
    main()
