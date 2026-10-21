#!/usr/bin/env python3
"""Create the official 23-trait Maize1404 method-effect task table."""
import argparse
from pathlib import Path

import pandas as pd
import yaml


OFFICIAL_METHODS = ["reml_gblup", "rrblup", "rf", "lightgbm"]
MODEL_ORDER = ["evo2", "dnabert2", "plantcad", "agront"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    rows = []
    task_id = 0
    for trait in map(str, cfg["traits"]):
        for method in OFFICIAL_METHODS:
            rows.append({
                "task_id": task_id,
                "trait": trait,
                "method": method,
                "window": 500,
                "feature": "raw",
            })
            task_id += 1
            for window in map(int, cfg.get("windows", [200, 500, 1000])):
                for model in MODEL_ORDER:
                    rows.append({
                        "task_id": task_id,
                        "trait": trait,
                        "method": method,
                        "window": window,
                        "feature": f"{model}_w{window}",
                    })
                    task_id += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    print(f"[INFO] wrote {len(rows)} official method-effect tasks: {output}")


if __name__ == "__main__":
    main()
