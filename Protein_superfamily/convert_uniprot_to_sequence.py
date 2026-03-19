"""
convert_uniprot_to_sequence.py

Reads all *_origin.csv files from:
    data/train_val_data_unsampled_unfeaturized/

For each file:
  - Fetches the amino acid sequence for every unique UniProt ID via the UniProt REST API
  - Adds a 'target_sequence' column
  - Renames 'canonical_smiles' -> 'compound_iso_smiles' (notebook-compatible)
  - Saves the result to:
    data/newTrainVal/<protein_class>_converted.csv

Usage:
    python convert_uniprot_to_sequence.py

"""

import os
import time
import argparse
import requests
import pandas as pd
from pathlib import Path


def fetch_sequence(uniprot_id: str, retries: int = 3, delay: float = 0.2) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                lines = response.text.strip().split("\n")
                sequence = "".join(lines[1:])
                return sequence
            elif response.status_code == 404:
                print(f"  [WARN] UniProt ID not found: {uniprot_id}")
                return None
            else:
                print(f"  [WARN] HTTP {response.status_code} for {uniprot_id}, attempt {attempt+1}/{retries}")
        except requests.exceptions.RequestException as e:
            print(f"  [WARN] Request error for {uniprot_id}: {e}, attempt {attempt+1}/{retries}")
        time.sleep(delay * (attempt + 1))  
    print(f"  [ERROR] Failed to fetch sequence for {uniprot_id} after {retries} attempts.")
    return None


def build_sequence_cache(uniprot_ids: list, retries: int, delay: float) -> dict:
    unique_ids = list(dict.fromkeys(uniprot_ids))  
    cache = {}
    total = len(unique_ids)
    print(f"  Fetching sequences for {total} unique UniProt IDs...")
    for i, uid in enumerate(unique_ids, 1):
        if i % 50 == 0 or i == total:
            print(f"  Progress: {i}/{total}")
        cache[uid] = fetch_sequence(uid, retries=retries, delay=delay)
        time.sleep(delay)
    fetched = sum(1 for v in cache.values() if v is not None)
    print(f"  Done. {fetched}/{total} sequences fetched successfully.")
    return cache

def process_file(csv_path: Path, output_dir: Path, retries: int, delay: float):
    print(f"\nProcessing: {csv_path.name}")
    df = pd.read_csv(csv_path)

    required = {"uniprot_id"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [SKIP] Missing required columns {missing} in {csv_path.name}")
        return

    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")

    sequence_cache = build_sequence_cache(df["uniprot_id"].tolist(), retries, delay)
    df["target_sequence"] = df["uniprot_id"].map(sequence_cache)

    n_missing = df["target_sequence"].isna().sum()
    if n_missing:
        missing_ids = df.loc[df["target_sequence"].isna(), "uniprot_id"].unique()
        print(f"  [WARN] {n_missing} rows have no sequence. UniProt IDs: {list(missing_ids[:10])}"
              + (" ..." if len(missing_ids) > 10 else ""))
        
    if "canonical_smiles" in df.columns:
        df = df.rename(columns={"canonical_smiles": "compound_iso_smiles"})
        print("  Renamed 'canonical_smiles' -> 'compound_iso_smiles'")

    out_name = csv_path.stem.replace("_origin", "") + "_converted.csv"
    out_path = output_dir / out_name
    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}  ({len(df)} rows, {df['target_sequence'].notna().sum()} with sequence)")


def main():
    parser = argparse.ArgumentParser(description="Convert UniProt IDs to AA sequences in origin CSVs.")
    parser.add_argument("--input_dir",  default="data/train_val_data_unsampled_unfeaturized",
                        help="Folder containing *_origin.csv files")
    parser.add_argument("--output_dir", default="data/newTrainVal",
                        help="Folder to save converted CSVs")
    parser.add_argument("--delay",   type=float, default=0.2,
                        help="Seconds between UniProt API calls (be polite to the server)")
    parser.add_argument("--retries", type=int,   default=3,
                        help="Number of retries on API failure")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input  : {input_dir.resolve()}")
    print(f"Output : {output_dir.resolve()}")

    csv_files = sorted(input_dir.glob("*_origin.csv"))
    if not csv_files:
        print(f"[ERROR] No *_origin.csv files found in {input_dir}")
        return

    print(f"\nFound {len(csv_files)} file(s): {[f.name for f in csv_files]}")

    for csv_path in csv_files:
        process_file(csv_path, output_dir, retries=args.retries, delay=args.delay)

    print("\n All files processed.")


if __name__ == "__main__":
    main()
