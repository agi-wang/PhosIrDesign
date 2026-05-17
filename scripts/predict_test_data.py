#!/usr/bin/env python3
"""
Predict post-screening synthesized experimental emitters with trained XGBoost models.

Input: ours.csv
Output: test_predict.csv
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import argparse
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from phosirdesign.core.feature_extractor import FeatureExtractor
from phosirdesign.models.base import load_model

def load_models(project_dir, model_name='xgboost', use_intersection=False):
    """Load trained models"""
    print("\n" + "="*80)
    print("Load pretrained models")
    print("-"*80)
    
    models = {}
    
    # Choose model directory based on intersection flag
    if use_intersection:
        # Intersection-trained models usually live under intersection subdir
        model_dir = Path(project_dir) / model_name / 'intersection' / f'{model_name}_intersection' / 'models'
        if not model_dir.exists():
            # Try alternative paths
            model_dir = Path(project_dir) / model_name / 'intersection' / 'models'
            if not model_dir.exists():
                model_dir = Path(project_dir) / f'{model_name}_intersection' / 'models'
        print(f"INFO: Using intersection-trained models")
    else:
        # Try multiple possible model paths
        model_dir = Path(project_dir) / 'all_models' / 'automl_train' / model_name / 'models'
        if not model_dir.exists():
            # Try alternate exported-results path
            model_dir = Path(project_dir) / model_name / 'models'
        print(f"INFO: Using full-data-trained models")
    
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        # Try to auto-discover latest Project_Output_* model dir
        root = Path(project_dir).parent if Path(project_dir).name == 'Project_Output' else Path(project_dir)
        candidates = []
        try:
            for d in root.glob('Project_Output_*'):
                mdir = d / 'all_models' / 'automl_train' / model_name / 'models'
                if mdir.exists():
                    candidates.append(mdir)
            if candidates:
                # Choose most recently modified dir
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_dir = candidates[0]
                print(f"INFO: Auto-switched to latest model directory: {model_dir}")
            else:
                print("WARNING: No model directory found under recent Project_Output_* dirs")
        except Exception:
            pass
        if not model_dir.exists():
            return models
    
    print(f"INFO: Model directory: {model_dir}")
    
    # Find model files (load wavelength and PLQY only)
    for model_file in model_dir.glob("*.joblib"):
        filename = model_file.stem
        if 'wavelength' in filename.lower():
            models['wavelength'] = load_model(model_file)
            print(f"INFO: Wavelength model: {model_file.name}")
        elif 'plqy' in filename.lower():
            models['PLQY'] = load_model(model_file)
            print(f"INFO: PLQY model: {model_file.name}")
    
    print(f"\nINFO: Loaded {len(models)} models")
    return models

def extract_features_for_test(df, feature_type='combined', combination_method='mean', descriptor_count=85):
    """Extract test data features"""
    print("\n" + "="*80)
    print("Feature extraction")
    print("-"*80)
    print(f"  - Samples: {len(df):,}")
    print(f"  - Feature type: {feature_type}")
    
    extractor = FeatureExtractor(
        feature_type=feature_type,
        morgan_radius=2,
        morgan_bits=1024,
        use_cache=True,
        descriptor_count=descriptor_count
    )
    
    features_list = []
    valid_indices = []
    failed_count = 0
    
    start_counter = time.perf_counter()
    
    # Extract features
    for idx, row in df.iterrows():
        try:
            # Extract combination features
            smiles_list = [row['L1'], row['L2'], row['L3']]
            features = extractor.extract_combination(
                smiles_list,
                feature_type=feature_type,
                combination_method=combination_method
            )
            
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
            else:
                failed_count += 1
                print(f"  WARNING: Row {idx}: feature extraction failed")
        except Exception as e:
            failed_count += 1
            print(f"  WARNING: Row {idx}: {str(e)}")
    
    elapsed = time.perf_counter() - start_counter
    
    if features_list:
        X = np.vstack(features_list)
        df_valid = df.iloc[valid_indices].copy()
        print(f"\nINFO: Feature extraction completed:")
        print(f"  - Success: {len(X):,} samples")
        print(f"  - Fail: {failed_count:,} samples")
        print(f"  - Time: {elapsed:.2f} s")
        return X, df_valid
    else:
        print(f"\nERROR: Feature extraction failed")
        return None, None

def predict_test(models, X, df_valid):
    """Predict test data"""
    print("\n" + "="*80)
    print("Batch prediction")
    print("-"*80)
    print(f"  - Samples: {len(X):,}")
    
    predictions = {}
    
    # Predict each target
    for target, model in models.items():
        print(f"\nPredict {target}...")
        start_counter = time.perf_counter()
        
        preds = model.predict(X)
        predictions[target] = preds
        
        elapsed = time.perf_counter() - start_counter
        print(f"  INFO: Completed:")
        print(f"    - Min: {preds.min():.6f}")
        print(f"    - Max: {preds.max():.6f}")
        print(f"    - Mean: {preds.mean():.6f}")
        print(f"    - Std: {preds.std():.6f}")
        print(f"    - Time: {elapsed:.3f} s")
    
    # Add predictions to DataFrame
    if 'wavelength' in predictions:
        df_valid['Predicted_wavelength'] = predictions['wavelength']
    if 'PLQY' in predictions:
        df_valid['Predicted_PLQY'] = predictions['PLQY']
    
    return df_valid

def compare_with_actual(df):
    """Compare with actual when available"""
    print("\n" + "="*80)
    print("Prediction result analysis")
    print("-"*80)
    
    # Check if actual values exist
    has_actual_wavelength = 'Max_wavelength(nm)' in df.columns and not df['Max_wavelength(nm)'].isna().all()
    has_actual_plqy = 'PLQY' in df.columns and not df['PLQY'].isna().all()
    
    if has_actual_wavelength and 'Predicted_wavelength' in df.columns:
        actual = df['Max_wavelength(nm)'].dropna()
        pred = df.loc[actual.index, 'Predicted_wavelength']
        
        if len(actual) > 0:
            mae = np.abs(actual - pred).mean()
            rmse = np.sqrt(((actual - pred) ** 2).mean())
            r2 = 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
            
            print(f"\nWavelength prediction performance:")
            print(f"  - Samples: {len(actual)}")
            print(f"  - MAE: {mae:.2f} nm")
            print(f"  - RMSE: {rmse:.2f} nm")
            print(f"  - R^2: {r2:.4f}")
    
    if has_actual_plqy and 'Predicted_PLQY' in df.columns:
        # Handle PLQY units (convert percentage to fraction)
        actual = df['PLQY'].dropna()
        if actual.max() > 1.5:  # may be percentage
            actual = actual / 100
        
        pred = df.loc[actual.index, 'Predicted_PLQY']
        
        if len(actual) > 0:
            mae = np.abs(actual - pred).mean()
            rmse = np.sqrt(((actual - pred) ** 2).mean())
            r2 = 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
            
            print(f"\nPLQY prediction performance:")
            print(f"  - Samples: {len(actual)}")
            print(f"  - MAE: {mae:.4f}")
            print(f"  - RMSE: {rmse:.4f}")
            print(f"  - R^2: {r2:.4f}")
    
    # PLQY distribution
    if 'Predicted_PLQY' in df.columns:
        plqy = df['Predicted_PLQY']
        print(f"\nPLQY distribution:")
        print(f"  - Min: {plqy.min():.4f}")
        print(f"  - 25%: {plqy.quantile(0.25):.4f}")
        print(f"  - Median: {plqy.median():.4f}")
        print(f"  - 75%: {plqy.quantile(0.75):.4f}")
        print(f"  - Max: {plqy.max():.4f}")
        print(f"  - Mean: {plqy.mean():.4f}")
        
        # Range distribution
        ranges = [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
        print(f"\n  PLQY range distribution:")
        for min_val, max_val in ranges:
            count = ((plqy >= min_val) & (plqy < max_val)).sum()
            pct = 100 * count / len(plqy)
            print(f"    [{min_val:.1f}, {max_val:.1f}): {count:4d} ({pct:5.1f}%)")
    
    # Top 5 high PLQY samples
    if 'Predicted_PLQY' in df.columns:
        print(f"\nTop 5 high PLQY predictions:")
        top5 = df.nlargest(5, 'Predicted_PLQY')
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"\n  #{i} (row: {idx}):")
            print(f"    PLQY predicted: {row['Predicted_PLQY']:.4f}")
            if 'Predicted_wavelength' in df.columns:
                print(f"    Wavelength predicted: {row['Predicted_wavelength']:.1f} nm")
            if 'PLQY' in row and pd.notna(row['PLQY']):
                actual_plqy = row['PLQY']
                if actual_plqy > 1.5:
                    actual_plqy = actual_plqy / 100
                print(f"    PLQY actual: {actual_plqy:.4f}")
            if 'Max_wavelength(nm)' in row and pd.notna(row['Max_wavelength(nm)']):
                print(f"    Wavelength actual: {row['Max_wavelength(nm)']:.1f} nm")

def _sample_label(row, fallback_index):
    for column in ("Abbreviation_in_the_article", "Unnamed: 0", "Name", "name", "ID", "id"):
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column])
    return str(fallback_index)

def emit_ours_predictions_event(progress_file, model_name, output_path, df_predicted, max_rows=20):
    """Send a compact ours.csv prediction preview to the workflow TUI."""
    if not progress_file:
        return

    rows = []
    for idx, row in df_predicted.head(max_rows).iterrows():
        item = {
            "sample": _sample_label(row, idx),
        }
        if "Predicted_wavelength" in row and pd.notna(row["Predicted_wavelength"]):
            item["Predicted_wavelength"] = float(row["Predicted_wavelength"])
        if "Predicted_PLQY" in row and pd.notna(row["Predicted_PLQY"]):
            item["Predicted_PLQY"] = float(row["Predicted_PLQY"])
        rows.append(item)

    event = {
        "type": "ours_predictions",
        "model": model_name,
        "output_path": str(output_path),
        "rows": rows,
    }
    progress_path = Path(progress_file)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")

def _find_latest_project_dir() -> str:
    """Find latest Project_Output_* directory under current working dir"""
    cwd = Path.cwd()
    projects = [d for d in cwd.glob('Project_Output_*') if d.is_dir()]
    if not projects:
        return ''
    projects.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(projects[0])

def main():
    parser = argparse.ArgumentParser(description='Predict test dataset')
    parser.add_argument('--project', '-p',
                       help='Model project directory (default: latest Project_Output_* autodetect)')
    parser.add_argument('--input', '-i', 
                       default=None,
                       help='Post-screening synthesized experimental data file (default: PROJECT/ours.csv)')
    parser.add_argument('--output', '-o',
                       help='Output file (default: PROJECT/test_predict.csv)')
    parser.add_argument('--model', '-m',
                       default='xgboost',
                       help='Model type')
    parser.add_argument('--intersection', action='store_true',
                       help='Use intersection-trained models (train only on samples with all targets)')
    parser.add_argument('--combination-method', default='mean',
                       choices=['mean', 'sum', 'concat'],
                       help='Feature merge method for multiple ligands')
    parser.add_argument('--descriptor-count', type=int, default=85,
                       help='Number of molecular descriptors')
    
    args = parser.parse_args()
    
    # Auto-detect project directory
    if not args.project:
        latest = _find_latest_project_dir()
        if latest:
            args.project = latest
            print(f"INFO: Auto-selected project dir: {args.project}")
        else:
            args.project = 'Project_Output'
            print("INFO: No Project_Output_* dir found, fallback to default: Project_Output")
    
    # Resolve default paths
    if not args.output:
        if args.intersection:
            args.output = f"{args.project}/test_predict_intersection.csv"
        else:
            args.output = f"{args.project}/test_predict.csv"
    if not args.input:
        args.input = f"{args.project}/ours.csv"
    
    print("="*80)
    print("Test data prediction")
    print("="*80)
    print(f"\nConfig:")
    print(f"  - Input file: {args.input}")
    print(f"  - Model dir candidate: {args.project}/{args.model}")
    print(f"  - Model type: {'Intersection-trained' if args.intersection else 'Full-data-trained'}")
    print(f"  - Output file: {args.output}")
    print(f"  - Combination method: {args.combination_method}")
    print(f"  - Descriptor count: {args.descriptor_count}")
    
    start_counter = time.perf_counter()
    
    # 1. Load test data
    print(f"\nLoading test data...")
    df = pd.read_csv(args.input)
    print(f"INFO: Loaded {len(df):,} samples")
    print(f"INFO: Columns: {', '.join(df.columns[:10])}")
    
    # 2. Load models
    models = load_models(args.project, args.model, use_intersection=args.intersection)
    if not models:
        print("ERROR: No models found")
        return
    
    # 3. Extract features
    X, df_valid = extract_features_for_test(
        df,
        combination_method=args.combination_method,
        descriptor_count=args.descriptor_count
    )
    if X is None:
        print("ERROR: Feature extraction failed")
        return
    
    # 4. Predict
    df_predicted = predict_test(models, X, df_valid)
    
    # 5. Analyze results
    compare_with_actual(df_predicted)
    
    # 6. Save results
    print(f"\nSaving predictions...")
    
    # Save full data with original and prediction columns
    output_df = df.copy()
    
    # Merge predictions back into original DataFrame
    if 'Predicted_wavelength' in df_predicted.columns:
        output_df.loc[df_predicted.index, 'Predicted_wavelength'] = df_predicted['Predicted_wavelength']
    if 'Predicted_PLQY' in df_predicted.columns:
        output_df.loc[df_predicted.index, 'Predicted_PLQY'] = df_predicted['Predicted_PLQY']
    
    # Save file
    output_df.to_csv(args.output, index=False)
    print(f"INFO: Saved to: {args.output}")
    print(f"INFO: Rows: {len(output_df):,}")
    print(f"INFO: Successful predictions: {len(df_predicted):,}")
    
    # Save only successful predictions
    valid_output = args.output.replace('.csv', '_valid_only.csv')
    df_predicted.to_csv(valid_output, index=False)
    print(f"INFO: Valid predictions: {valid_output}")
    emit_ours_predictions_event(
        os.environ.get("TEST_PREDICTION_PROGRESS_FILE") or os.environ.get("TRAIN_PROGRESS_FILE"),
        args.model,
        args.output,
        df_predicted,
    )
    
    # Notes for intersection-trained models
    if args.intersection:
        print(f"\n  NOTE: Using intersection-trained models")
        print(f"     These models are trained on samples with all targets present")
        print(f"     Often more consistent but may generalize slightly less")
    
    # Summary
    total_time = time.perf_counter() - start_counter
    processed_samples = len(df_predicted) if df_predicted is not None else len(df)
    speed = processed_samples / total_time if total_time > 0 else 0.0
    print("\n" + "="*80)
    print("INFO: Prediction completed!")
    print(f"  - Total time: {total_time:.2f} s")
    print(f"  - Processing speed: {speed:.2f} samples/s")
    print("="*80)

if __name__ == "__main__":
    main()
