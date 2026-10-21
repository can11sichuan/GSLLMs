#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import re
from collections import Counter
from typing import Optional

import numpy as np
import pysam
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

DNA_RE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)
VOCAB_DNA_RE = re.compile(r"^[ACGT]+$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Embed selected PLINK BIM SNP windows with AgroNT / agro-nucleotide-transformer. "
            "For each SNP and flank size, two allele-centered sequences are embedded; "
            "genotype embeddings are derived as 0=A2/A2, 1=A1/A2 mean, 2=A1/A1. "
            "This script is parallel to embed_snp_windows_plantcad.py but uses PyTorch HF AgroNT."
        )
    )
    p.add_argument("--sites", required=True, help="selected *.sites.tsv from select_random_bim_snps.py")
    p.add_argument("--fasta", required=True)
    p.add_argument("--flank", type=int, required=True, help="bp on each side of SNP; sequence length=2*flank+1")
    p.add_argument("--outdir", required=True)
    p.add_argument("--model-name", default="agro-nucleotide-transformer-1b")
    p.add_argument("--model-path", required=True, help="local Hugging Face model directory")
    p.add_argument("--layer-name", default="last_hidden_state", help="last_hidden_state or hidden_states.<index>, e.g. hidden_states.-1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--embedding-method", choices=["gaussian_pool", "direct_token", "mean_pool"], default="gaussian_pool")
    p.add_argument("--pool-sigma", type=float, default=None, help="default: token_length/6, in token units")
    p.add_argument("--heterozygote-method", choices=["mean"], default="mean")
    p.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    p.add_argument("--cache-root", default=None)
    p.add_argument("--save-vector-cache", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-sites", type=int, default=0, help="debug only: use first N SNPs; 0 means all")
    p.add_argument("--kmer-size", type=int, default=0, help="0=auto infer from tokenizer vocab; AgroNT is usually 6")
    p.add_argument("--max-tokens", type=int, default=0, help="0=read from model config/tokenizer; fail if window exceeds this")
    p.add_argument("--add-special-tokens", action="store_true", help="default false; keep false for SNP-center token mapping")
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


def infer_kmer_size(tokenizer, fallback=6):
    try:
        vocab = tokenizer.get_vocab()
        lens = []
        for tok in vocab:
            t = str(tok).upper()
            if VOCAB_DNA_RE.match(t):
                lens.append(len(t))
        if lens:
            c = Counter(lens)
            k, n = c.most_common(1)[0]
            # Avoid choosing length 1 if the vocab contains single base helper tokens;
            # AgroNT vocab should be dominated by 6-mers.
            long_lens = [x for x in lens if x >= 3]
            if long_lens:
                k = Counter(long_lens).most_common(1)[0][0]
            return int(k)
    except Exception:
        pass
    return int(fallback)


def get_model_max_tokens(tokenizer, config, fallback=1024):
    candidates = []
    for obj, attrs in [
        (tokenizer, ["model_max_length"]),
        (config, ["max_position_embeddings", "n_positions", "max_positions", "seq_length"]),
    ]:
        for a in attrs:
            v = getattr(obj, a, None)
            if isinstance(v, int) and 0 < v < 10_000_000:
                candidates.append(v)
    # Some HF tokenizers use a huge sentinel as "unknown"; ignore it via bound above.
    return int(min(candidates)) if candidates else int(fallback)


class AgroNTEmbedder:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        layer_name: str,
        device: str,
        pool_sigma: Optional[float],
        dtype: str,
        cache_root: Optional[str],
        flank: int,
        embedding_method: str,
        save_vector_cache: bool = False,
        kmer_size: int = 0,
        max_tokens: int = 0,
        add_special_tokens: bool = False,
    ):
        self.model_name = model_name
        self.model_path = os.path.abspath(model_path)
        self.layer_name = layer_name
        self.device = device
        self.pool_sigma = pool_sigma
        self.out_dtype = np.float16 if dtype == "float16" else np.float32
        if dtype == "float16":
            self.torch_dtype = torch.float16
        elif dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
        self.cache_root = cache_root
        self.flank = flank
        self.embedding_method = embedding_method
        self.save_vector_cache = save_vector_cache
        self.add_special_tokens = bool(add_special_tokens)
        self.cache_hits = 0
        self.cache_miss = 0
        self._prepare_env()
        self.config, self.model, self.tokenizer = self._load_model()
        self.kmer_size = int(kmer_size) if int(kmer_size) > 0 else infer_kmer_size(self.tokenizer, fallback=6)
        self.max_tokens = int(max_tokens) if int(max_tokens) > 0 else get_model_max_tokens(self.tokenizer, self.config, fallback=1024)
        if self.cache_root:
            ensure_dir(self.vector_cache_dir())

    def _prepare_env(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _load_model(self):
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForMaskedLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        model.eval()
        return config, model, tokenizer

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
            "pool_sigma": self.pool_sigma,
            "flank": self.flank,
            "embedding_method": self.embedding_method,
            "kmer_size": self.kmer_size,
            "add_special_tokens": self.add_special_tokens,
        }
        return md5_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def vector_cache_path(self, key: str):
        return os.path.join(self.vector_cache_dir(), f"{key}.npy")

    def _tokenize_sequence(self, seq: str):
        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=self.add_special_tokens,
            truncation=False,
        )
        if "input_ids" not in enc:
            raise RuntimeError("tokenizer did not return input_ids")
        input_ids = enc["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError(f"unexpected input_ids shape: {tuple(input_ids.shape)}")
        n_tokens = int(input_ids.shape[1])
        if self.max_tokens > 0 and n_tokens > self.max_tokens:
            raise RuntimeError(
                f"AgroNT token length exceeds limit: n_tokens={n_tokens}, max_tokens={self.max_tokens}, "
                f"seq_len_bp={len(seq)}, flank={self.flank}. Try smaller flank or explicit chunking."
            )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def embed_full_sequence(self, seq: str):
        seq = normalize_seq(seq)
        inputs = self._tokenize_sequence(seq)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        if self.layer_name == "last_hidden_state":
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                emb = outputs.hidden_states[-1]
            else:
                raise RuntimeError("model output does not provide hidden_states")
        elif self.layer_name.startswith("hidden_states."):
            idx = int(self.layer_name.split(".", 1)[1])
            if outputs.hidden_states is None:
                raise RuntimeError("model output does not provide hidden_states")
            emb = outputs.hidden_states[idx]
        else:
            raise ValueError(
                f"unsupported layer_name for AgroNT: {self.layer_name}. "
                "Use last_hidden_state or hidden_states.<index>"
            )

        full_emb = emb.squeeze(0).detach().float().cpu().numpy()
        token_len = int(full_emb.shape[0])

        del inputs, outputs, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return full_emb, token_len

    def bp_center_to_token_index(self, center_1based: int, token_len: int):
        if self.add_special_tokens:
            # Special-token layouts differ between tokenizers. For SNP work, prefer --add-special-tokens false.
            # As a conservative fallback, put the SNP center around the middle token.
            approx = int(round((float(center_1based) - 1.0) / max(float(2 * self.flank + 1), 1.0) * max(token_len - 1, 0)))
        else:
            approx = int((int(center_1based) - 1) // max(int(self.kmer_size), 1))
        return max(0, min(token_len - 1, approx))

    def gaussian_pool_one(self, full_emb: np.ndarray, center_token_0based: int):
        full_emb = np.asarray(full_emb, dtype=np.float32)
        L = int(full_emb.shape[0])
        center = max(0, min(L - 1, int(center_token_0based)))
        sigma = self.pool_sigma if self.pool_sigma is not None else (L / 6.0)
        if sigma <= 0:
            raise ValueError(f"invalid pool sigma: {sigma}")
        idxs = np.arange(L, dtype=np.float32)
        weights = np.exp(-((idxs - float(center)) ** 2) / (2.0 * float(sigma) * float(sigma))).astype(np.float32)
        vec = (weights[:, None] * full_emb).sum(axis=0) / float(weights.sum())
        return vec.astype(self.out_dtype, copy=False)

    def direct_token_one(self, full_emb: np.ndarray, center_token_0based: int):
        full_emb = np.asarray(full_emb, dtype=np.float32)
        L = int(full_emb.shape[0])
        center = max(0, min(L - 1, int(center_token_0based)))
        return full_emb[center].astype(self.out_dtype, copy=False)

    def mean_pool(self, full_emb: np.ndarray):
        full_emb = np.asarray(full_emb, dtype=np.float32)
        return full_emb.mean(axis=0).astype(self.out_dtype, copy=False)

    def embed_center_vector(self, seq: str, center_1based: int):
        seq = normalize_seq(seq)
        key = self.vector_cache_key(seq)
        if self.cache_root:
            path = self.vector_cache_path(key)
            if os.path.exists(path):
                self.cache_hits += 1
                # old cache may not store token metadata; recompute token info cheaply
                n_tokens = int(self.tokenizer(seq, add_special_tokens=self.add_special_tokens, return_tensors="pt")["input_ids"].shape[1])
                center_token = self.bp_center_to_token_index(center_1based, n_tokens)
                return np.load(path), key, n_tokens, center_token

        full_emb, token_len = self.embed_full_sequence(seq)
        center_token = self.bp_center_to_token_index(center_1based, token_len)

        if self.embedding_method == "gaussian_pool":
            vec = self.gaussian_pool_one(full_emb, center_token)
        elif self.embedding_method == "direct_token":
            vec = self.direct_token_one(full_emb, center_token)
        elif self.embedding_method == "mean_pool":
            vec = self.mean_pool(full_emb)
        else:
            raise ValueError(f"unknown embedding method: {self.embedding_method}")

        self.cache_miss += 1
        if self.cache_root and self.save_vector_cache:
            ensure_dir(self.vector_cache_dir())
            np.save(self.vector_cache_path(key), vec)
        del full_emb
        return vec, key, token_len, center_token


def read_sites(path, max_sites=0):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"selected_order", "bim_chrom", "fasta_chrom", "snp_id", "pos", "a1", "a2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"sites file missing columns: {sorted(missing)}")
        rows = [dict(r) for r in reader]
    if max_sites and max_sites > 0:
        rows = rows[:max_sites]
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

    sites = read_sites(args.sites, max_sites=args.max_sites)
    fasta = pysam.FastaFile(args.fasta)
    center_1based = args.flank + 1

    embedder = AgroNTEmbedder(
        model_name=args.model_name,
        model_path=args.model_path,
        layer_name=args.layer_name,
        device=args.device,
        pool_sigma=args.pool_sigma,
        dtype=args.dtype,
        cache_root=args.cache_root,
        flank=args.flank,
        embedding_method=args.embedding_method,
        save_vector_cache=args.save_vector_cache,
        kmer_size=args.kmer_size,
        max_tokens=args.max_tokens,
        add_special_tokens=args.add_special_tokens,
    )

    logging.info("AgroNT model path: %s", embedder.model_path)
    logging.info("AgroNT inferred kmer_size=%d max_tokens=%d add_special_tokens=%s", embedder.kmer_size, embedder.max_tokens, embedder.add_special_tokens)
    logging.info("Embedding %d SNPs with flank=%d, sequence length=%d", len(sites), args.flank, 2 * args.flank + 1)

    allele_vectors = []
    trace_rows = []
    ref_mismatch_n = 0
    token_lengths = []
    center_tokens = []

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
            logging.warning(
                "FASTA center base matches neither BIM allele: %s %s:%d FASTA=%s A1=%s A2=%s",
                snp_id, fchr, pos, ref_base, a1, a2,
            )

        seq_a1 = fetch_window(fasta, fchr, pos, args.flank, a1)
        seq_a2 = fetch_window(fasta, fchr, pos, args.flank, a2)

        vec_a1, hash_a1, tok_a1, center_tok_a1 = embedder.embed_center_vector(seq_a1, center_1based=center_1based)
        vec_a2, hash_a2, tok_a2, center_tok_a2 = embedder.embed_center_vector(seq_a2, center_1based=center_1based)
        allele_vectors.append(np.stack([vec_a1, vec_a2], axis=0))

        token_lengths.append(max(tok_a1, tok_a2))
        center_tokens.append(max(center_tok_a1, center_tok_a2))

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
            "token_length_A1": str(tok_a1),
            "token_length_A2": str(tok_a2),
            "center_token_0based_A1": str(center_tok_a1),
            "center_token_0based_A2": str(center_tok_a2),
            "kmer_size": str(embedder.kmer_size),
            "seq_hash_A1": hash_a1,
            "seq_hash_A2": hash_a2,
        })

        if i % args.log_every == 0 or i == len(sites):
            logging.info(
                "completed %d/%d SNPs; vector_cache_hit=%d miss=%d; token_len=%s",
                i, len(sites), embedder.cache_hits, embedder.cache_miss, max(tok_a1, tok_a2),
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    allele_arr = np.stack(allele_vectors, axis=0)
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
        token_length=np.array(token_lengths, dtype=np.int32),
        center_token_0based=np.array(center_tokens, dtype=np.int32),
        kmer_size=np.array(embedder.kmer_size, dtype=np.int32),
        model_max_tokens=np.array(embedder.max_tokens, dtype=np.int32),
        model_name=np.array(args.model_name, dtype=object),
        model_path=np.array(args.model_path, dtype=object),
        layer_name=np.array(args.layer_name, dtype=object),
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
            "token_length_A1", "token_length_A2", "center_token_0based_A1", "center_token_0based_A2",
            "kmer_size", "seq_hash_A1", "seq_hash_A2",
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
            "embedding_method": args.embedding_method,
            "pool_sigma": args.pool_sigma,
            "heterozygote_method": args.heterozygote_method,
            "dtype": str(allele_arr.dtype),
            "tokenizer_kmer_size": embedder.kmer_size,
            "model_max_tokens": embedder.max_tokens,
            "add_special_tokens": embedder.add_special_tokens,
            "token_length_min": int(np.min(token_lengths)) if token_lengths else None,
            "token_length_max": int(np.max(token_lengths)) if token_lengths else None,
            "token_length_mean": float(np.mean(token_lengths)) if token_lengths else None,
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
