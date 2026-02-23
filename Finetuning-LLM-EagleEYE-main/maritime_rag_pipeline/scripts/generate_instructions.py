#!/usr/bin/env python
"""
generate_instructions.py
------------------------

This script synthesises instruction–response pairs from the maritime dataset.  These
pairs are suitable for supervised fine‑tuning of a language model.  Each example
consists of an ``instruction`` (the question), an optional ``input`` (kept
empty here for simplicity) and an ``output`` (the expected answer).

The generation logic draws random samples from the dataset and produces
questions about vessel arrivals, dwell times and vessel type distributions.
You can extend the templates or add your own functions to cover additional
tasks (e.g. congestion prediction, emissions estimation) once you enrich the
underlying data.

Usage:
    python generate_instructions.py --input data/final_data.csv --output training_data/instructions.jsonl \
        --num_samples 20000

Arguments:
    --input        Path to the cleaned CSV file (required).
    --output       Path to write the JSONL file containing instruction examples.
    --num_samples  Number of instruction–response pairs to generate (default: 10000).

"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import List, Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic instruction–response pairs")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    return parser.parse_args()


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure arrival and departure times are parsed as datetimes and add a date column."""
    df = df.copy()
    # Coerce to datetime; errors='coerce' converts invalid strings to NaT
    df["arrival_dt"] = pd.to_datetime(df.get("portArrival"), errors="coerce")
    df["departure_dt"] = pd.to_datetime(df.get("portDeparture"), errors="coerce")
    df["arrival_date"] = df["arrival_dt"].dt.date
    # Compute dwell hours where both times are available
    df["dwell_hours"] = (df["departure_dt"] - df["arrival_dt"]).dt.total_seconds() / 3600.0
    return df


def generate_count_by_port_date(df: pd.DataFrame) -> Dict[str, str]:
    """Generate a question counting total arrivals at a random port on a random date."""
    row = df.sample(1).iloc[0]
    port = row["portName"]
    date = row["arrival_date"]
    count = int((df["portName"] == port) & (df["arrival_date"] == date)).sum()
    return {
        "instruction": f"How many vessels arrived at {port} on {date.isoformat()}?",
        "input": "",
        "output": str(count),
    }


def generate_count_by_port_type_date(df: pd.DataFrame) -> Dict[str, str]:
    """Generate a question counting vessels of a specific type arriving at a port on a date."""
    row = df.sample(1).iloc[0]
    port = row["portName"]
    date = row["arrival_date"]
    vtype = row["ais_VesselType"]
    mask = (df["portName"] == port) & (df["arrival_date"] == date) & (df["ais_VesselType"] == vtype)
    count = int(mask.sum())
    return {
        "instruction": f"How many {vtype} vessels arrived at {port} on {date.isoformat()}?",
        "input": "",
        "output": str(count),
    }


def generate_median_dwell_by_port_date(df: pd.DataFrame) -> Dict[str, str]:
    """Generate a question asking for median dwell time at a port on a date."""
    row = df.dropna(subset=["dwell_hours"]).sample(1).iloc[0]
    port = row["portName"]
    date = row["arrival_date"]
    dwell_vals = df[(df["portName"] == port) & (df["arrival_date"] == date)]["dwell_hours"].dropna()
    if dwell_vals.empty:
        median = 0.0
    else:
        median = float(dwell_vals.median())
    # round to one decimal place for readability
    answer = f"{median:.1f}"
    return {
        "instruction": f"What was the median dwell time (in hours) for vessels at {port} on {date.isoformat()}?",
        "input": "",
        "output": answer,
    }


def generate_unique_types_by_port_date(df: pd.DataFrame) -> Dict[str, str]:
    """Generate a question asking which vessel types arrived at a port on a date."""
    row = df.sample(1).iloc[0]
    port = row["portName"]
    date = row["arrival_date"]
    types = df[(df["portName"] == port) & (df["arrival_date"] == date)]["ais_VesselType"].dropna().unique()
    if len(types) == 0:
        answer = "none"
    else:
        answer = ", ".join(sorted(str(t) for t in types))
    return {
        "instruction": f"List the vessel types that arrived at {port} on {date.isoformat()}.",
        "input": "",
        "output": answer,
    }


TEMPLATES = [
    generate_count_by_port_date,
    generate_count_by_port_type_date,
    generate_median_dwell_by_port_date,
    generate_unique_types_by_port_date,
]


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output
    num_samples = args.num_samples

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")
    df = parse_dates(df)
    # Filter out rows without essential fields
    df = df.dropna(subset=["portName", "arrival_date"])
    print(f"After parsing dates: {len(df):,} rows remain")
    if df.empty:
        raise ValueError("No valid rows available after parsing dates")

    examples: List[Dict[str, str]] = []
    for i in range(num_samples):
        template_fn = random.choice(TEMPLATES)
        try:
            ex = template_fn(df)
            examples.append(ex)
        except Exception as exc:
            # Skip any templates that fail due to missing data
            continue
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_samples} examples")

    # Write to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()