# -*- coding: utf-8 -*-

"""Convert a sample-by-feature CSV matrix to compressed HDF5 in row chunks."""


import os
import argparse
import numpy as np
import pandas as pd
import h5py


def csv_to_hdf5(input_csv, output_h5, chunksize=1000):

    """Write CSV features and sample identifiers to HDF5 in row chunks."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(input_csv)

    print("Reading header...")

    header = pd.read_csv(input_csv, nrows=0)
    columns = header.columns.tolist()

    sample_col = columns[0]
    feature_cols = columns[1:]

    feature_count = len(feature_cols)

    print("Feature count:", feature_count)

    total_rows = sum(1 for _ in open(input_csv)) - 1

    print("Total samples:", total_rows)

    chunk_rows = min(chunksize, total_rows)

    print("Chunk shape:", (chunk_rows, feature_count))

    with h5py.File(output_h5, "w") as hf:

        # Compressed genotype feature matrix.
        X_ds = hf.create_dataset(
            "X",
            shape=(total_rows, feature_count),
            dtype="float32",
            compression="lzf",
            chunks=(chunk_rows, feature_count)
        )

        # Sample identifiers.
        sample_ds = hf.create_dataset(
            "samples",
            shape=(total_rows,),
            dtype=h5py.string_dtype()
        )

        # Feature column names.
        col_ds = hf.create_dataset(
            "columns",
            data=np.array(feature_cols, dtype="S")
        )

        reader = pd.read_csv(
            input_csv,
            chunksize=chunksize
        )

        start = 0

        for chunk_id, chunk in enumerate(reader):

            samples = chunk.iloc[:, 0].astype(str).values
            X = chunk.iloc[:, 1:].astype("float32").values

            end = start + len(chunk)

            X_ds[start:end] = X
            sample_ds[start:end] = samples

            print(f"chunk {chunk_id+1} written {len(chunk)} rows")

            start = end

    print("Finished:", output_h5)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_csv")
    parser.add_argument("output_h5")
    parser.add_argument("--chunksize", type=int, default=1000)

    args = parser.parse_args()

    csv_to_hdf5(
        args.input_csv,
        args.output_h5,
        args.chunksize
    )
