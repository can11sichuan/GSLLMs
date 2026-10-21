#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pysam
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

SCRIPT_VERSION = "2026-05-13-dnabert2-tuple-centerchunk-v2"

DNA_RE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Embed selected PLINK BIM SNP windows with DNABERT-2. "
            "For each SNP and flank size, two allele-centered sequences are embedded; "
            "genotype embeddings are derived as 0=A2/A2, 1=A1/A2 mean, 2=A1/A1. "
            "If token count exceeds model limit, tokens are chunked around the SNP-centered token "
            "and merged back before pooling."
        )
    )
    p.add_argument("--sites", required=True, help="selected *.sites.tsv from select_random_bim_snps.py")
    p.add_argument("--fasta", required=True)
    p.add_argument("--flank", type=int, required=True, help="bp on each side of SNP; sequence length=2*flank+1")
    p.add_argument("--outdir", required=True)

    p.add_argument("--model-name", default="DNABERT-2-117M")
    p.add_argument("--model-path", required=True, help="local Hugging Face model snapshot directory")
    p.add_argument("--layer-name", default="last_hidden_state", help="last_hidden_state or hidden_states.<index>")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")

    p.add_argument(
        "--chunk-len",
        type=int,
        default=2048,
        help="maximum DNA bp length allowed before rejecting raw window; normally should be >= window_length",
    )
    p.add_argument(
        "--max-model-tokens",
        type=int,
        default=512,
        help="maximum token count per forward pass; if tokenized sequence exceeds this, chunk around SNP token",
    )

    p.add_argument("--embedding-method", choices=["gaussian_pool", "direct_token", "mean_pool"], default="gaussian_pool")
    p.add_argument("--pool-sigma", type=float, default=None, help="default: sequence_length/6, in bp units")
    p.add_argument("--heterozygote-method", choices=["mean"], default="mean")
    p.add_argument("--sequence-case", choices=["upper", "lower", "as_is"], default="upper", help="case conversion before tokenization; old DNABERT2 workflow used upper")

    p.add_argument("--cache-root", default=None)
    p.add_argument("--save-vector-cache", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def md5_text(x: str) -> str:
    return hashlib.md5(x.encode("utf-8")).hexdigest()


def normalize_seq(seq: str, sequence_case: str = "upper") -> str:
    seq = "".join(seq.split())
    if sequence_case == "upper":
        seq = seq.upper()
    elif sequence_case == "lower":
        seq = seq.lower()
    elif sequence_case == "as_is":
        pass
    else:
        raise ValueError(f"invalid sequence_case: {sequence_case}")
    if not seq:
        raise ValueError("empty sequence")
    if not DNA_RE.match(seq):
        raise ValueError(f"sequence has non A/C/G/T/N characters, first 80 bp: {seq[:80]}")
    return seq


def object_array(values):
    return np.array(values, dtype=object)


def read_sites(path: str):
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


def ref_match_status(ref_base: str, a1: str, a2: str) -> str:
    ref_base = ref_base.upper()
    if ref_base == a1 and ref_base == a2:
        return "both"
    if ref_base == a1:
        return "A1"
    if ref_base == a2:
        return "A2"
    return "neither"


def fetch_window(fasta: pysam.FastaFile, chrom: str, pos: int, flank: int, center_base: str) -> str:
    left = fasta.fetch(chrom, pos - flank - 1, pos - 1).upper()
    right = fasta.fetch(chrom, pos, pos + flank).upper()
    if len(left) != flank or len(right) != flank:
        raise RuntimeError(
            f"bad window length for {chrom}:{pos}, flank={flank}, left={len(left)}, right={len(right)}"
        )
    return normalize_seq(left + center_base.upper() + right)


class DNABERT2Embedder:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        layer_name: str,
        device: str,
        chunk_len: int,
        max_model_tokens: int,
        pool_sigma: Optional[float],
        dtype: str,
        cache_root: Optional[str],
        flank: int,
        embedding_method: str,
        sequence_case: str = "upper",
        save_vector_cache: bool = False,
    ):
        self.model_name = model_name
        self.model_path = os.path.abspath(model_path)
        self.layer_name = layer_name
        self.device = device
        self.chunk_len = int(chunk_len)
        self.max_model_tokens = int(max_model_tokens)
        self.pool_sigma = pool_sigma
        self.out_dtype = np.float16 if dtype == "float16" else np.float32
        self.torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        self.cache_root = cache_root
        self.flank = flank
        self.embedding_method = embedding_method
        self.sequence_case = sequence_case
        self.save_vector_cache = save_vector_cache

        self.cache_hits = 0
        self.cache_miss = 0
        self.offset_mode = "unknown"

        self.n_chunked_sequences = 0
        self.max_tokens_observed = 0
        self.max_chunks_observed = 0

        self._prepare_env()
        self.model, self.tokenizer = self._load_model()

        if self.cache_root:
            ensure_dir(self.vector_cache_dir())

    def _prepare_env(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            config=config,
            local_files_only=True,
        ).to(self.device)

        if self.torch_dtype == torch.float16:
            model = model.half()
        else:
            model = model.float()

        model.eval()
        # The bundled MosaicBERT remote code selects its Triton attention
        # kernel whenever attention dropout is zero.  That kernel currently
        # fails to compile with the Triton version in the DNABert environment;
        # eval-mode dropout is inert, so a nonzero branch flag preserves the
        # deterministic inference values while using the PyTorch fallback.
        forced_pytorch_attention = 0
        for module in model.modules():
            if hasattr(module, "p_dropout") and hasattr(module, "dropout"):
                module.p_dropout = 1.0
                forced_pytorch_attention += 1
        if forced_pytorch_attention:
            # The fallback combines ALiBi biases created in float32 with Q/K/V.
            # Keep its compute in float32 to avoid mixed-dtype matmul failures;
            # pooled vectors are still cast to the configured output dtype.
            model = model.float()
            print(
                f"[INFO] forced PyTorch attention for {forced_pytorch_attention} "
                "DNABERT2 attention modules with float32 compute "
                "(Triton compatibility fallback)"
            )
        return model, tokenizer

    def vector_cache_dir(self):
        safe_layer = self.layer_name.replace("/", "_").replace(".", "_")
        return os.path.join(
            self.cache_root,
            self.model_name,
            safe_layer,
            self.embedding_method,
            f"flank{self.flank}",
            f"maxtok{self.max_model_tokens}",
        )

    def vector_cache_key(self, seq: str):
        payload = {
            "seq": seq,
            "model_name": self.model_name,
            "layer_name": self.layer_name,
            "chunk_len": self.chunk_len,
            "max_model_tokens": self.max_model_tokens,
            "pool_sigma": self.pool_sigma,
            "flank": self.flank,
            "embedding_method": self.embedding_method,
            "sequence_case": self.sequence_case,
        }
        return md5_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def vector_cache_path(self, key: str):
        return os.path.join(self.vector_cache_dir(), f"{key}.npy")

    def _fallback_uniform_offsets(self, seq_len: int, n_tokens: int) -> List[Tuple[int, int]]:
        edges = np.linspace(0.0, float(seq_len), num=n_tokens + 1)
        offsets = []
        for i in range(n_tokens):
            s = int(np.floor(edges[i]))
            e = int(np.ceil(edges[i + 1]))
            if e <= s:
                e = min(seq_len, s + 1)
            offsets.append((s, e))
        return offsets

    def _tokenize_with_offsets(self, seq: str):
        seq = normalize_seq(seq, self.sequence_case)

        try:
            enc = self.tokenizer(
                seq,
                return_tensors="pt",
                add_special_tokens=False,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", None)
            offsets = enc.get("offset_mapping", None)
            if offsets is not None:
                offsets = offsets[0].tolist()
                self.offset_mode = "return_offsets_mapping"
                return input_ids, attention_mask, offsets
        except TypeError:
            pass
        except Exception:
            pass

        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)

        if getattr(enc, "encodings", None):
            offsets = enc.encodings[0].offsets
            if offsets is not None:
                self.offset_mode = "encodings.offsets"
                return input_ids, attention_mask, offsets

        offsets = self._fallback_uniform_offsets(len(seq), int(input_ids.shape[1]))
        self.offset_mode = "uniform_fallback"
        return input_ids, attention_mask, offsets

    def _extract_last_hidden_and_hidden_states(self, outputs):
        last_hidden_state = None
        hidden_states = None

        if isinstance(outputs, (tuple, list)):
            if len(outputs) > 0 and torch.is_tensor(outputs[0]):
                last_hidden_state = outputs[0]
            for obj in outputs[1:]:
                if isinstance(obj, (tuple, list)) and len(obj) > 0 and all(torch.is_tensor(x) for x in obj):
                    hidden_states = obj
                    break
        else:
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            hidden_states = getattr(outputs, "hidden_states", None)
            if isinstance(outputs, dict):
                if last_hidden_state is None:
                    last_hidden_state = outputs.get("last_hidden_state", None)
                if hidden_states is None:
                    hidden_states = outputs.get("hidden_states", None)

        return last_hidden_state, hidden_states

    @torch.no_grad()
    def _forward_on_token_chunk(self, input_ids_chunk: torch.Tensor, attention_mask_chunk: Optional[torch.Tensor]):
        input_ids_chunk = input_ids_chunk.to(self.device)
        if attention_mask_chunk is not None:
            attention_mask_chunk = attention_mask_chunk.to(self.device)

        outputs = self.model(
            input_ids=input_ids_chunk,
            attention_mask=attention_mask_chunk,
            output_hidden_states=True,
            return_dict=False,
        )

        last_hidden_state, hidden_states = self._extract_last_hidden_and_hidden_states(outputs)

        if self.layer_name == "last_hidden_state":
            if last_hidden_state is None:
                raise RuntimeError("model output does not provide last_hidden_state / outputs[0]")
            emb = last_hidden_state
        elif self.layer_name.startswith("hidden_states."):
            idx = int(self.layer_name.split(".", 1)[1])
            if hidden_states is None:
                raise RuntimeError("model output does not provide hidden_states")
            emb = hidden_states[idx]
        else:
            raise ValueError(
                f"unsupported layer_name for DNABERT2: {self.layer_name}. "
                "Use last_hidden_state or hidden_states.<index>"
            )

        arr = emb.squeeze(0).detach().float().cpu().numpy()

        del input_ids_chunk, attention_mask_chunk, outputs, emb, last_hidden_state, hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return arr

    def _token_covering_center(self, offsets, center0: int) -> int:
        candidates = []
        for i, (s, e) in enumerate(offsets):
            s = int(s)
            e = int(e)
            if s <= center0 < e:
                candidates.append(i)
        if candidates:
            return candidates[0]

        mids = self._offset_midpoints_bp(offsets)
        return int(np.argmin(np.abs(mids - float(center0))))

    def _offset_midpoints_bp(self, offsets):
        mids = []
        for s, e in offsets:
            s = int(s)
            e = int(e)
            if e <= s:
                mids.append(float(s) + 0.5)
            else:
                mids.append((float(s) + float(e - 1)) / 2.0)
        return np.asarray(mids, dtype=np.float32)

    def _make_token_chunks_centered(self, n_tokens: int, center_idx: int) -> List[Tuple[int, int]]:
        if n_tokens <= self.max_model_tokens:
            return [(0, n_tokens)]

        half = self.max_model_tokens // 2
        center_start = max(0, center_idx - half)
        center_end = min(n_tokens, center_start + self.max_model_tokens)
        center_start = max(0, center_end - self.max_model_tokens)

        chunks = [(center_start, center_end)]

        left_end = center_start
        while left_end > 0:
            start = max(0, left_end - self.max_model_tokens)
            chunks.insert(0, (start, left_end))
            left_end = start

        right_start = center_end
        while right_start < n_tokens:
            end = min(n_tokens, right_start + self.max_model_tokens)
            chunks.append((right_start, end))
            right_start = end

        return chunks

    @torch.no_grad()
    def embed_full_sequence(self, seq: str, center_1based: int):
        seq = normalize_seq(seq, self.sequence_case)
        if len(seq) > self.chunk_len:
            raise ValueError(
                f"sequence length {len(seq)} exceeds chunk_len {self.chunk_len}. "
                f"Please set chunk_len >= window_length ({len(seq)})."
            )

        input_ids, attention_mask, offsets = self._tokenize_with_offsets(seq)
        n_tokens = int(input_ids.shape[1])
        self.max_tokens_observed = max(self.max_tokens_observed, n_tokens)

        center0 = int(center_1based) - 1
        center_token_idx = self._token_covering_center(offsets, center0)
        chunks = self._make_token_chunks_centered(n_tokens, center_token_idx)

        if len(chunks) > 1:
            self.n_chunked_sequences += 1
        self.max_chunks_observed = max(self.max_chunks_observed, len(chunks))

        pieces = []
        piece_offsets = []

        for start, end in chunks:
            ids_chunk = input_ids[:, start:end]
            mask_chunk = attention_mask[:, start:end] if attention_mask is not None else None
            arr_chunk = self._forward_on_token_chunk(ids_chunk, mask_chunk)
            if arr_chunk.shape[0] != (end - start):
                raise RuntimeError(
                    f"chunk embedding length {arr_chunk.shape[0]} does not match token slice length {end - start}"
                )
            pieces.append(arr_chunk)
            piece_offsets.extend(offsets[start:end])

        full_arr = np.concatenate(pieces, axis=0)
        if full_arr.shape[0] != len(piece_offsets):
            raise RuntimeError(
                f"merged token length {full_arr.shape[0]} does not match merged offsets {len(piece_offsets)}"
            )

        return full_arr, piece_offsets, n_tokens, len(chunks), center_token_idx

    def gaussian_pool_one(self, full_emb: np.ndarray, offsets, center_1based: int):
        full_emb = np.asarray(full_emb, dtype=np.float32)
        center0 = int(center_1based) - 1
        sigma = self.pool_sigma if self.pool_sigma is not None else (self.flank * 2 + 1) / 6.0
        if sigma <= 0:
            raise ValueError(f"invalid pool sigma: {sigma}")

        mids = self._offset_midpoints_bp(offsets)
        weights = np.exp(-((mids - float(center0)) ** 2) / (2.0 * float(sigma) * float(sigma))).astype(np.float32)
        if not np.any(weights > 0):
            raise RuntimeError("all gaussian weights are zero")

        vec = (weights[:, None] * full_emb).sum(axis=0) / float(weights.sum())
        return vec.astype(self.out_dtype, copy=False)

    def direct_token_one(self, full_emb: np.ndarray, offsets, center_1based: int):
        full_emb = np.asarray(full_emb, dtype=np.float32)
        center0 = int(center_1based) - 1
        idx = self._token_covering_center(offsets, center0)
        return full_emb[idx].astype(self.out_dtype, copy=False)

    def mean_pool(self, full_emb: np.ndarray):
        full_emb = np.asarray(full_emb, dtype=np.float32)
        return full_emb.mean(axis=0).astype(self.out_dtype, copy=False)

    def embed_center_vector(self, seq: str, center_1based: int):
        seq = normalize_seq(seq, self.sequence_case)
        key = self.vector_cache_key(seq)

        if self.cache_root:
            path = self.vector_cache_path(key)
            if os.path.exists(path):
                self.cache_hits += 1
                vec = np.load(path)
                meta = {
                    "token_count": -1,
                    "n_chunks": -1,
                    "center_token_idx": -1,
                    "cache_hit": True,
                }
                return vec, key, meta

        full_emb, offsets, token_count, n_chunks, center_token_idx = self.embed_full_sequence(seq, center_1based)

        if self.embedding_method == "gaussian_pool":
            vec = self.gaussian_pool_one(full_emb, offsets, center_1based)
        elif self.embedding_method == "direct_token":
            vec = self.direct_token_one(full_emb, offsets, center_1based)
        elif self.embedding_method == "mean_pool":
            vec = self.mean_pool(full_emb)
        else:
            raise ValueError(f"unknown embedding method: {self.embedding_method}")

        self.cache_miss += 1
        if self.cache_root and self.save_vector_cache:
            ensure_dir(self.vector_cache_dir())
            np.save(self.vector_cache_path(key), vec)

        del full_emb
        meta = {
            "token_count": int(token_count),
            "n_chunks": int(n_chunks),
            "center_token_idx": int(center_token_idx),
            "cache_hit": False,
        }
        return vec, key, meta


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("DNABERT2 embed script version: %s", SCRIPT_VERSION)

    sites = read_sites(args.sites)
    fasta = pysam.FastaFile(args.fasta)
    center_1based = args.flank + 1

    embedder = DNABERT2Embedder(
        model_name=args.model_name,
        model_path=args.model_path,
        layer_name=args.layer_name,
        device=args.device,
        chunk_len=args.chunk_len,
        max_model_tokens=args.max_model_tokens,
        pool_sigma=args.pool_sigma,
        dtype=args.dtype,
        cache_root=args.cache_root,
        flank=args.flank,
        embedding_method=args.embedding_method,
        sequence_case=args.sequence_case,
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
            logging.warning(
                "FASTA center base matches neither BIM allele: %s %s:%d FASTA=%s A1=%s A2=%s",
                snp_id, fchr, pos, ref_base, a1, a2,
            )

        seq_a1 = fetch_window(fasta, fchr, pos, args.flank, a1)
        seq_a2 = fetch_window(fasta, fchr, pos, args.flank, a2)

        vec_a1, hash_a1, meta_a1 = embedder.embed_center_vector(seq_a1, center_1based=center_1based)
        vec_a2, hash_a2, meta_a2 = embedder.embed_center_vector(seq_a2, center_1based=center_1based)

        allele_vectors.append(np.stack([vec_a1, vec_a2], axis=0))

        trace_rows.append({
            "selected_order": row["selected_order"],
            "snp_id": snp_id,
            "snp_uid": row.get("snp_uid", snp_id),
            "source_label": row.get("source_label", ""),
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
            "token_count_A1": str(meta_a1["token_count"]),
            "token_count_A2": str(meta_a2["token_count"]),
            "n_chunks_A1": str(meta_a1["n_chunks"]),
            "n_chunks_A2": str(meta_a2["n_chunks"]),
            "center_token_idx_A1_0based": str(meta_a1["center_token_idx"]),
            "center_token_idx_A2_0based": str(meta_a2["center_token_idx"]),
        })

        if i == 1:
            logging.info("token offset mode: %s", embedder.offset_mode)

        if i % args.log_every == 0 or i == len(sites):
            logging.info(
                "completed %d/%d SNPs; cache_hit=%d miss=%d; max_tokens_observed=%d; max_chunks_observed=%d",
                i,
                len(sites),
                embedder.cache_hits,
                embedder.cache_miss,
                embedder.max_tokens_observed,
                embedder.max_chunks_observed,
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

    selected_orders = np.array([int(r.get("selected_order", i + 1)) for i, r in enumerate(sites)], dtype=np.int32)
    snp_ids = [r["snp_id"] for r in sites]
    snp_uids = [r.get("snp_uid", r["snp_id"]) for r in sites]
    source_labels = [r.get("source_label", "") for r in sites]
    source_prefixes = [r.get("source_prefix", "") for r in sites]
    source_bims = [r.get("source_bim", "") for r in sites]
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
        selected_order=selected_orders,
        snp_id=object_array(snp_ids),
        snp_uid=object_array(snp_uids),
        source_label=object_array(source_labels),
        source_prefix=object_array(source_prefixes),
        source_bim=object_array(source_bims),
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
        max_model_tokens=np.array(args.max_model_tokens, dtype=np.int32),
        embedding_method=np.array(args.embedding_method, dtype=object),
        pool_sigma=np.array(-1 if args.pool_sigma is None else args.pool_sigma, dtype=np.float32),
        dtype=np.array(str(allele_arr.dtype), dtype=object),
        sequence_case=np.array(args.sequence_case, dtype=object),
        script_version=np.array(SCRIPT_VERSION, dtype=object),
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
            "selected_order", "snp_id", "snp_uid", "source_label", "bim_chrom", "fasta_chrom", "pos", "a1", "a2",
            "ref_fasta", "ref_match", "flank", "window_length", "center_pos_1based",
            "seq_hash_A1", "seq_hash_A2",
            "token_count_A1", "token_count_A2",
            "n_chunks_A1", "n_chunks_A2",
            "center_token_idx_A1_0based", "center_token_idx_A2_0based",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(trace_rows)

    with open(meta_json, "w") as f:
        json.dump({
            "script_version": SCRIPT_VERSION,
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
            "max_model_tokens": args.max_model_tokens,
            "embedding_method": args.embedding_method,
            "pool_sigma": args.pool_sigma,
            "heterozygote_method": args.heterozygote_method,
            "dtype": str(allele_arr.dtype),
            "sequence_case": args.sequence_case,
            "cache_root": args.cache_root,
            "save_vector_cache": args.save_vector_cache,
            "cache_hits": embedder.cache_hits,
            "cache_miss": embedder.cache_miss,
            "offset_mode": embedder.offset_mode,
            "n_chunked_sequences": embedder.n_chunked_sequences,
            "max_tokens_observed": embedder.max_tokens_observed,
            "max_chunks_observed": embedder.max_chunks_observed,
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
