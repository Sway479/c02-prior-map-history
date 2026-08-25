#!/usr/bin/env python3
"""Verify the fixed C02 result-ID set and provenance fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from export_authoritative_result_map import verify_result_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_map", type=Path)
    args = parser.parse_args()
    result = pd.read_csv(args.result_map)
    verify_result_map(result)
    if result[["source_table", "source_selector"]].isna().any().any():
        raise RuntimeError("result map contains missing provenance fields")
    print(f"AUTHORITATIVE_RESULT_MAP_VERIFY_PASS rows={len(result)}")


if __name__ == "__main__":
    main()
