"""Clear specific columns in a CSV file while preserving headers.

Usage:
    uv run python Suitability/scripts/clear_csv_columns.py path/to/data.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

COLUMNS_TO_CLEAR = [
    "StartDate",
    "EndDate",
    "IPAddress",
    "RecordedDate",
    "ResponseId",
    "RecipientLastName",
    "RecipientFirstName",
    "RecipientEmail",
    "ExternalReference",
    "LocationLatitude",
    "LocationLongitude",
    "DistributionChannel",
    "UserLanguage",
    "Q2.1",
    "Q2.2",
    "Q2.3",
    "Q2.7",
    "Q6.4",
    "1_Q5.1",
    "2_Q5.1",
    "3_Q5.1",
    "4_Q5.1",
    "5_Q5.1",
    "6_Q5.1",
    "7_Q5.1",
    "8_Q5.1",
    "9_Q5.1",
    "10_Q5.1",
    "11_Q5.1",
    "12_Q5.1",
    "13_Q5.1",
    "14_Q5.1",
    "15_Q5.1",
    "16_Q5.1",
    "17_Q5.1",
    "18_Q5.1",
    "19_Q5.1",
    "20_Q5.1",
    "PROLIFIC_PID",
    "PID",
]


def main():
    parser = argparse.ArgumentParser(
        description="Clear data in specific CSV columns while preserving headers."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file to modify")
    args = parser.parse_args()

    if not args.csv_path.exists():
        parser.error(f"File not found: {args.csv_path}")

    if not COLUMNS_TO_CLEAR:
        print("Error: COLUMNS_TO_CLEAR is empty. Add column names to the list.")
        sys.exit(1)

    print(f"Loading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    missing = [c for c in COLUMNS_TO_CLEAR if c not in df.columns]
    if missing:
        print(f"Warning: columns not found in CSV: {missing}")

    found = [c for c in COLUMNS_TO_CLEAR if c in df.columns]
    if not found:
        print("No matching columns to clear.")
        sys.exit(1)

    print(f"Clearing {len(found)} columns:")
    for col in found:
        print(f"  - {col}")
        df[col] = ""

    df.to_csv(args.csv_path, index=False)
    print("\nDone!")


if __name__ == "__main__":
    main()
