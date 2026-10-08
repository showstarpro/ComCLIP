#!/usr/bin/env python3
"""
Reformat the raw CSV produced by `clip_benchmark.cli build` into a cleaner
pivot table: rows = datasets, columns = model labels, values = primary accuracy.

Usage:
    python reformat_csv.py <input_csv> [--output <output_csv>]

If --output is not specified, the result overwrites the input file.
"""

import argparse
import os
import sys
import math
import csv
from collections import OrderedDict


def make_model_label(model, pretrained):
    if pretrained == "openai":
        return f"{model} (OpenAI)"
    basename = os.path.basename(pretrained)
    name = os.path.splitext(basename)[0]
    return f"{model} ({name})"


def dataset_short(dataset):
    return dataset.replace("wds/", "").replace("vtab/", "")


def safe_float(val):
    try:
        v = float(val)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def primary_metric(row):
    for key in ["acc1", "lp_acc1"]:
        v = safe_float(row.get(key, ""))
        if v is not None:
            return v
    return None


def reformat(input_path, output_path):
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    table = OrderedDict()
    all_datasets = []

    for row in rows:
        model = row.get("model", "")
        pretrained = row.get("pretrained", "")
        dataset = row.get("dataset", "")
        if not model or not dataset:
            continue

        label = make_model_label(model, pretrained)
        ds = dataset_short(dataset)
        val = primary_metric(row)

        if label not in table:
            table[label] = OrderedDict()
        if ds not in all_datasets:
            all_datasets.append(ds)
        if val is not None:
            table[label][ds] = val

    model_names = list(table.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + model_names)
        for ds in all_datasets:
            row_out = [ds]
            for m in model_names:
                v = table[m].get(ds)
                row_out.append("%.4f" % v if v is not None else "")
            writer.writerow(row_out)

    print(f"reformatted csv written to: {output_path}")
    print(f"  {len(all_datasets)} datasets x {len(model_names)} models")


def main():
    parser = argparse.ArgumentParser(description="Reformat clip_benchmark build CSV")
    parser.add_argument("input_csv", help="Raw CSV from clip_benchmark build")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: overwrite input)")
    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"Error: {args.input_csv} not found", file=sys.stderr)
        sys.exit(1)

    output_path = args.output if args.output else args.input_csv
    reformat(args.input_csv, output_path)


if __name__ == "__main__":
    main()
