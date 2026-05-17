#!/usr/bin/env python3
"""
Analyze and explain the virtual combination counts
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_FILE = BASE_DIR / "data" / "PhosIrDB.csv"
DEFAULT_VIRTUAL_FILE = BASE_DIR / "data" / "ir_assemble.csv"


def analyze_combinations(data_file=None, virtual_file=None):
    """Analyze the origin of ~272k combinations"""

    data_path = Path(data_file) if data_file else DEFAULT_DATA_FILE
    virtual_path = Path(virtual_file) if virtual_file else DEFAULT_VIRTUAL_FILE

    print("=" * 80)
    print("Combination Count Analysis")
    print("=" * 80)
    
    # 1. Load original dataset
    print(f"\n1) Load dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Rows: {len(df):,}")
    
    # 2. Analyze L1 ligands
    print("\n2) Analyze L1 ligands:")
    l1_unique = df['L1'].dropna().unique()
    l1_count = len(l1_unique)
    print(f"   Unique L1: {l1_count:,}")
    print(f"   Examples: {l1_unique[:3].tolist()}")
    
    # 3. Analyze L2 ligands
    print("\n3) Analyze L2 ligands:")
    l2_unique = df['L2'].dropna().unique()
    l2_count = len(l2_unique)
    print(f"   Unique L2: {l2_count:,}")
    print(f"   Examples: {l2_unique[:3].tolist()}")
    
    # 4. Analyze L3 ligands
    print("\n4) Analyze L3 ligands:")
    l3_unique = df['L3'].dropna().unique()
    l3_count = len(l3_unique)
    print(f"   Unique L3: {l3_count:,}")
    print(f"   Examples: {l3_unique[:3].tolist()}")
    
    # 5. Union of L1 and L2 (for L1=L2 combinations)
    print("\n5) L1 U L2 union analysis:")
    l1_l2_union = np.unique(np.concatenate([l1_unique, l2_unique]))
    l1_l2_count = len(l1_l2_union)
    print(f"   Unique L1 U L2: {l1_l2_count:,}")
    print(f"   Note: L1=L2 is required (same ligand), select from union")
    
    # 6. Compute total combinations
    print("\n6) Compute total combinations:")
    print(f"   Scheme: L1=L2 (same ligand) x L3 (all candidates)")
    print(f"   Choices for L1=L2: {l1_l2_count:,}")
    print(f"   Choices for L3: {l3_count:,}")
    print(f"   Total = {l1_l2_count:,} x {l3_count:,} = {l1_l2_count * l3_count:,}")
    
    # 7. Validate generated combinations file
    print("\n7) Validate generated combinations file:")
    if virtual_path.exists():
        actual_df = pd.read_csv(virtual_path)
        actual_count = len(actual_df)
        print(f"   Actual rows in {virtual_path.name}: {actual_count:,}")
        
        # Validate L1=L2
        same_count = (actual_df['L1'] == actual_df['L2']).sum()
        print(f"   Rows with L1=L2: {same_count:,}")
        print(f"   Check: {'OK all rows satisfy L1=L2' if same_count == actual_count else 'WARNING some rows violate L1=L2'}")
    
    # 8. Detailed statistics
    print("\n8) Ligand distribution stats:")
    print(f"   L1-only: {len(set(l1_unique) - set(l2_unique)):,}")
    print(f"   L2-only: {len(set(l2_unique) - set(l1_unique)):,}")
    print(f"   Intersection (L1 & L2): {len(set(l1_unique) & set(l2_unique)):,}")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  - L1 library: {l1_count:,}")
    print(f"  - L2 library: {l2_count:,}")
    print(f"  - L3 library: {l3_count:,}")
    print(f"  - L1 U L2 merged library: {l1_l2_count:,}")
    print(f"  - Theoretical combinations: {l1_l2_count:,} x {l3_count:,} = {l1_l2_count * l3_count:,}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Analyze virtual combination counts")
    parser.add_argument("--data", default=None,
                        help="Original dataset path (default: data/PhosIrDB.csv)")
    parser.add_argument("--virtual", "--assemble", dest="virtual", default=None,
                        help="Virtual database path (default: data/ir_assemble.csv)")

    args = parser.parse_args()
    analyze_combinations(data_file=args.data, virtual_file=args.virtual)


if __name__ == "__main__":
    main()
