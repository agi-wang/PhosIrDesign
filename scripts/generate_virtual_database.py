#!/usr/bin/env python3
"""
Generate virtual database - reassemble all L1/L2/L3 combinations and predict
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import joblib
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phosirdesign.core.feature_extractor import FeatureExtractor
from phosirdesign.models.base import load_model

def load_original_data(data_file):
    """Load original data and extract unique ligands"""
    df = pd.read_csv(data_file)
    
    # Extract all unique L1, L2, L3
    l1_unique = df['L1'].dropna().unique()
    l2_unique = df['L2'].dropna().unique()
    l3_unique = df['L3'].dropna().unique()
    
    print("Original data stats:")
    print(f"  L1: {len(l1_unique)} unique ligands")
    print(f"  L2: {len(l2_unique)} unique ligands")
    print(f"  L3: {len(l3_unique)} unique ligands")
    print(f"  Original combinations: {len(df)}")
    
    return l1_unique, l2_unique, l3_unique, df

def generate_all_combinations(l1_unique, l2_unique, l3_unique, max_combinations=None):
    """Generate all possible L1/L2/L3 combinations
    Note: L1 and L2 are constrained to be the same ligand (L1=L2)
    """
    
    # Merge unique values from L1 and L2 (shared pool)
    l12_unique = np.unique(np.concatenate([l1_unique, l2_unique]))
    
    print("\nCombination strategy:")
    print(f"  L1/L2 shared pool: {len(l12_unique)} ligands")
    print(f"  L3 pool: {len(l3_unique)} ligands")
    
    # Generate all combinations
    all_combinations = []
    
    # L1=L2 cases (symmetric ligands)
    total_possible = len(l12_unique) * len(l3_unique)
    print(f"  Theoretical combinations: {total_possible:,} (L1=L2)")
    
    if max_combinations and total_possible > max_combinations:
        print(f"WARNING: Limiting combinations to: {max_combinations:,}")
        # Random sampling
        import random
        random.seed(42)
        sampled_indices = random.sample(range(total_possible), min(max_combinations, total_possible))
        sampled_indices.sort()
        
        count = 0
        for idx, (l12, l3) in enumerate(product(l12_unique, l3_unique)):
            if idx in sampled_indices:
                all_combinations.append({
                    'L1': l12,
                    'L2': l12,
                    'L3': l3
                })
                count += 1
                if count >= max_combinations:
                    break
    else:
        # Generate all L1=L2 combinations
        for l12 in l12_unique:
            for l3 in l3_unique:
                all_combinations.append({
                    'L1': l12,
                    'L2': l12,
                    'L3': l3
                })
    
    # Create DataFrame
    assembled_df = pd.DataFrame(all_combinations)
    print(f"Generated combinations: {len(assembled_df):,}")
    
    return assembled_df

def remove_existing_combinations(assembled_df, original_df):
    """Remove existing combinations, keep only new ones"""
    
    # Create combo keys
    assembled_df['combo_key'] = assembled_df['L1'] + '|' + assembled_df['L2'] + '|' + assembled_df['L3']
    original_df['combo_key'] = original_df['L1'] + '|' + original_df['L2'] + '|' + original_df['L3']
    
    # Identify new combinations
    existing_keys = set(original_df['combo_key'].dropna())
    new_df = assembled_df[~assembled_df['combo_key'].isin(existing_keys)].copy()
    
    # Drop helper column
    new_df = new_df.drop('combo_key', axis=1)
    
    print(f"New combinations (excluding existing): {len(new_df):,}")
    
    return new_df

def extract_features_for_prediction(df, feature_type='combined'):
    print("\nExtracting molecular features...")
    extractor = FeatureExtractor(
        feature_type=feature_type,
        morgan_radius=2,
        morgan_bits=1024,
        use_cache=True,
        descriptor_count=85
    )
    features_list = []
    valid_indices = []
    for idx, row in df.iterrows():
        try:
            smiles_list = [row['L1'], row['L2'], row['L3']]
            smiles_list = [s for s in smiles_list if pd.notna(s) and s != '']
            if smiles_list:
                features = extractor.extract_combination(
                    smiles_list,
                    feature_type=feature_type,
                    combination_method='mean'
                )
                if features is not None:
                    features_list.append(features)
                    valid_indices.append(idx)
                    if len(features_list) % 100 == 0:
                        print(f"  Processed: {len(features_list)} combinations")
        except Exception:
            continue
    if features_list:
        X = np.vstack(features_list)
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"INFO: Successfully extracted features: {len(X)} combinations")
        return X, df_valid
    else:
        print("ERROR: No features extracted")
        return None, None

def load_trained_model(project_dir, model_name='xgboost', target='PLQY'):
    """Load trained model from standard AutoML output paths."""

    project_path = Path(project_dir)

    possible_dirs = [
        project_path / 'all_models' / 'automl_train' / model_name / 'models',
        project_path / model_name / 'models',
        project_path / 'models' / model_name,
    ]

    model_dir = None
    for d in possible_dirs:
        if d.exists():
            model_dir = d
            break

    if model_dir is None:
        # Auto-discover latest Project_Output_* models
        root = project_path.parent if project_path.name == 'Project_Output' else project_path
        candidates = []
        try:
            for d in root.glob('Project_Output_*'):
                mdir = d / 'all_models' / 'automl_train' / model_name / 'models'
                if mdir.exists():
                    candidates.append(mdir)
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_dir = candidates[0]
                print(f"INFO: Switched to latest model directory: {model_dir}")
        except Exception:
            pass

    if model_dir is None or not model_dir.exists():
        print(f"ERROR: Model directory not found: {project_path}/{model_name}/models")
        return None

    print(f"INFO: Model directory: {model_dir}")

    # Find model files for the target
    model_files = list(model_dir.glob(f"*{target}*.joblib"))

    if not model_files:
        # Try other possible naming patterns
        if target == 'wavelength':
            model_files = list(model_dir.glob("*wavelength*.joblib")) + \
                         list(model_dir.glob("*Max_wavelength*.joblib"))
        elif target == 'PLQY':
            model_files = list(model_dir.glob("*PLQY*.joblib")) + \
                         list(model_dir.glob("*plqy*.joblib"))
        elif target == 'tau':
            model_files = list(model_dir.glob("*tau*.joblib")) + \
                         list(model_dir.glob("*lifetime*.joblib"))

    if model_files:
        # Use latest model file
        model_file = sorted(model_files)[-1]
        print(f"INFO: Loaded model: {model_file.name}")
        model = load_model(model_file)
        return model
    else:
        print(f"ERROR: No model file found for target: {target}")
        return None

def predict_properties(X, df_valid, project_dir, model_name='xgboost'):
    """Predict molecular properties using trained models"""
    
    print("\nLoading models and predicting...")
    
    predictions = {}
    
    # Predict three targets
    targets = {
        'Max_wavelength(nm)': 'wavelength',
        'PLQY': 'PLQY',
        'tau(s*10^-6)': 'tau'
    }
    
    for target_col, target_key in targets.items():
        model = load_trained_model(project_dir, model_name, target_key)
        
        if model:
            print(f"  Predicting {target_col}...")
            try:
                pred = model.predict(X)
                predictions[target_col] = pred
                
                # Prediction stats
                print(f"    Range: [{pred.min():.3f}, {pred.max():.3f}]")
                print(f"    Mean: {pred.mean():.3f}")
                print(f"    Std: {pred.std():.3f}")
            except Exception as e:
                print(f"    ERROR: Prediction failed: {e}")
                predictions[target_col] = np.zeros(len(X))
        else:
            print(f"  WARNING: Skipping {target_col} (no model)")
            predictions[target_col] = np.zeros(len(X))
    
    # Add prediction results to DataFrame
    for col, pred in predictions.items():
        df_valid[f'Predicted_{col}'] = pred
    
    return df_valid

def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description='Generate virtual database')
    
    parser.add_argument('--data', '-d', default='data/PhosIrDB.csv',
                       help='Original data file path')
    parser.add_argument('--project', '-p', default='Project_Output',
                       help='Training project directory')
    parser.add_argument('--model', '-m', default='xgboost',
                       help='Model to use')
    parser.add_argument('--output', '-o', default='data/ir_assemble.csv',
                       help='Output file name')
    parser.add_argument('--max-combinations', type=int,
                       help='Max combinations limit')
    parser.add_argument('--include-existing', action='store_true',
                       help='Include existing combinations')
    parser.add_argument('--feature-type', default='combined',
                       choices=['morgan', 'descriptors', 'combined'],
                       help='Feature type')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generate virtual database")
    print("=" * 60)
    print(f"Original data: {args.data}")
    print(f"Project dir: {args.project}")
    print(f"Model: {args.model}")
    print(f"Feature type: {args.feature_type}")
    
    # 1) Load original data
    print("\n" + "-" * 40)
    print("Step 1: Load original data")
    l1_unique, l2_unique, l3_unique, original_df = load_original_data(args.data)
    
    # 2) Generate all combinations
    print("\n" + "-" * 40)
    print("Step 2: Generate all combinations")
    assembled_df = generate_all_combinations(
        l1_unique, l2_unique, l3_unique, 
        max_combinations=args.max_combinations
    )
    
    # 3) Optionally remove existing combinations
    if not args.include_existing:
        print("\n" + "-" * 40)
        print("Step 3: Remove existing combinations")
        assembled_df = remove_existing_combinations(assembled_df, original_df)
    
    # Save combinations file
    assembled_file = args.output.replace('.csv', '_combinations.csv')
    assembled_df.to_csv(assembled_file, index=False)
    print(f"\nINFO: Saved combinations file: {assembled_file}")
    
    # 4) If models are not trained, save combinations for later prediction
    project_path = Path(args.project)
    automl_dir = project_path / 'all_models' / 'automl_train'
    if not automl_dir.exists():
        assembled_df.to_csv(args.output, index=False)
        print(f"\nINFO: Saved combinations for later prediction: {args.output}")
        print("\n" + "=" * 60)
        print("Virtual database stats:")
        print("-" * 40)
        print(f"Total combinations: {len(assembled_df):,}")
        print("\n" + "=" * 60)
        print("INFO: Virtual database generation completed")
        print("=" * 60)
        return
    
    # 5) If models exist, extract features and predict
    print("\n" + "-" * 40)
    print("Step 4: Extract molecular features")
    X, df_valid = extract_features_for_prediction(assembled_df, args.feature_type)
    if X is None:
        print("ERROR: Feature extraction failed")
        return
    print("\n" + "-" * 40)
    print("Step 5: Predict molecular properties")
    df_predicted = predict_properties(X, df_valid, args.project, args.model)
    print("\n" + "-" * 40)
    print("Step 6: Save virtual database")
    output_file = args.output
    df_predicted.to_csv(output_file, index=False)
    print(f"INFO: Virtual database saved: {output_file}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("Virtual database stats:")
    print("-" * 40)
    print(f"Total combinations: {len(df_predicted):,}")
    
    # Find best combinations
    if 'Predicted_PLQY' in df_predicted.columns:
        # Highest-PLQY combination
        best_plqy_idx = df_predicted['Predicted_PLQY'].idxmax()
        best_plqy = df_predicted.loc[best_plqy_idx]
        print(f"\nTop PLQY combination:")
        print(f"  L1: {best_plqy['L1'][:30]}...")
        print(f"  L2: {best_plqy['L2'][:30]}...")
        print(f"  L3: {best_plqy['L3'][:30]}...")
        print(f"  Predicted PLQY: {best_plqy['Predicted_PLQY']:.3f}")
    
    if 'Predicted_Max_wavelength(nm)' in df_predicted.columns:
        # Longest-wavelength combination
        best_wl_idx = df_predicted['Predicted_Max_wavelength(nm)'].idxmax()
        best_wl = df_predicted.loc[best_wl_idx]
        print(f"\nLongest wavelength combination:")
        print(f"  L1: {best_wl['L1'][:30]}...")
        print(f"  L2: {best_wl['L2'][:30]}...")
        print(f"  L3: {best_wl['L3'][:30]}...")
        print(f"  Predicted wavelength: {best_wl['Predicted_Max_wavelength(nm)']:.1f} nm")
    
    # Save top candidate combinations
    top_candidates = df_predicted.nlargest(100, 'Predicted_PLQY')
    top_file = output_file.replace('.csv', '_top100.csv')
    top_candidates.to_csv(top_file, index=False)
    print(f"\nINFO: Top 100 candidates saved: {top_file}")
    
    print("\n" + "=" * 60)
    print("INFO: Virtual database generation completed")
    print("=" * 60)

if __name__ == "__main__":
    main()
