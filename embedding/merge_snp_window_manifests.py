#!/usr/bin/env python3
import argparse
import csv
import os
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_args():
    p = argparse.ArgumentParser(description="Merge per-window SNP trace files into one manifest.")
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--flanks", nargs="+", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    rows = []
    header = None
    for trace in args.traces:
        if not os.path.exists(trace):
            raise FileNotFoundError(trace)
        with open(trace) as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                raise RuntimeError(f"missing header: {trace}")
            if header is None:
                header = list(reader.fieldnames)
            elif list(reader.fieldnames) != header:
                raise RuntimeError(f"inconsistent header in {trace}")
            n = 0
            for r in reader:
                r["trace_file"] = trace
                rows.append(r)
                n += 1
            eprint(f"[INFO] read {trace}: {n} rows")

    if header is None:
        raise RuntimeError("no input traces")
    out_header = header + ["trace_file"]
    rows.sort(key=lambda r: (int(r.get("flank", 0)), int(r.get("selected_order", 0))))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_header, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    eprint(f"[INFO] wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
