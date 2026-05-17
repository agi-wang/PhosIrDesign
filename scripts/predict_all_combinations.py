#!/usr/bin/env python3
"""
Run full predictions on all combinations using trained models
Shows detailed progress; no output size limits
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
import time
import sys
import json
sys.path.insert(0, str(Path(__file__).parent.parent))

from phosirdesign.core.feature_extractor import FeatureExtractor
from phosirdesign.models.base import load_model

# Global performance record
performance_stats = {
    'steps': [],
    'total_duration_seconds': 0.0
}

def load_models(project_dir, model_name='xgboost', use_intersection=False):
    """Load trained models
    
    Args:
        project_dir: Project directory
        model_name: Model name
        use_intersection: Whether to use intersection-trained models
    """
    step_start = time.perf_counter()
    print("\n" + "="*80)
    print("Step 1: Load models")
    print("-"*80)
    
    models = {}
    
    # Try multiple possible model paths
    possible_paths = [
        # AutoML training path
        Path(project_dir) / 'all_models' / 'automl_train' / model_name / 'models',
        Path(project_dir) / '*' / 'automl_train' / model_name / 'models',
        # Standard paths
        Path(project_dir) / model_name / 'models',
        Path(project_dir) / 'models' / model_name,
    ]
    
    # Add intersection paths when requested
    if use_intersection:
        # Intersection-trained models usually live under an intersection subdirectory
        possible_paths.extend([
            Path(project_dir) / model_name / 'intersection' / f'{model_name}_intersection' / 'models',
            Path(project_dir) / model_name / 'intersection' / 'models',
        ])
    
    # Find existing model directory
    model_dir = None
    for path in possible_paths:
        if '*' in str(path):
            # Handle wildcard path
            matches = list(Path(project_dir).glob(str(path.relative_to(Path(project_dir)))))
            if matches:
                model_dir = matches[0]
                break
        elif path.exists():
            model_dir = path
            break
    
    if model_dir is None:
        print(f"ERROR: Model directory not found: {project_dir}/{model_name}/models")
        print("  Tried paths:")
        for path in possible_paths[:3]:
            print(f"    - {path}")
        return models
    print(f"INFO: Found model directory: {model_dir}")
    if 'automl_train' in str(model_dir):
        print("INFO: Using AutoML-trained models")
    elif 'intersection' in str(model_dir):
        print("INFO: Using intersection-trained models")
    else:
        print("INFO: Using regular-trained models")
    
    # Find model files (load wavelength and PLQY only)
    for model_file in model_dir.glob("*.joblib"):
        filename = model_file.stem
        if 'wavelength' in filename.lower():
            models['wavelength'] = load_model(model_file)
            print(f"INFO: Wavelength model: {model_file.name}")
        elif 'plqy' in filename.lower():
            models['PLQY'] = load_model(model_file)
            print(f"INFO: PLQY model: {model_file.name}")
        # Skip tau models
    
    print(f"\nINFO: Loaded {len(models)} models")
    
    step_time = time.perf_counter() - step_start
    performance_stats['steps'].append({
        'name': 'Model loading',
        'time_seconds': step_time,
        'details': f'Loaded {len(models)} models'
    })
    return models

def extract_features_batch(df, feature_type='combined', batch_size=1000):
    """Extract features in batches with detailed progress"""
    step_start = time.perf_counter()
    print("\n" + "="*80)
    print("Step 2: Feature extraction")
    print("-"*80)
    print("Config:")
    print(f"  - Feature type: {feature_type}")
    print(f"  - Batch size: {batch_size:,}")
    print(f"  - Total samples: {len(df):,}")
    print("\nStart extracting...")
    
    extractor = FeatureExtractor(
        feature_type=feature_type,
        morgan_radius=2,
        morgan_bits=1024,
        use_cache=True
    )
    
    n_samples = len(df)
    features_list = []
    valid_indices = []
    failed_count = 0
    
    start_counter = time.perf_counter()
    
    for i in range(0, n_samples, batch_size):
        batch_start_counter = time.perf_counter()
        batch_end = min(i + batch_size, n_samples)
        batch_df = df.iloc[i:batch_end]
        
        batch_valid = 0
        for idx, row in batch_df.iterrows():
            try:
                # Extract combination features
                smiles_list = [row['L1'], row['L2'], row['L3']]
                features = extractor.extract_combination(smiles_list)
                
                if features is not None:
                    features_list.append(features)
                    valid_indices.append(idx)
                    batch_valid += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                continue
        
        # Compute speed and remaining time
        batch_time = time.perf_counter() - batch_start_counter
        elapsed_time = time.perf_counter() - start_counter
        processed = batch_end
        rate = processed / elapsed_time if elapsed_time > 0 else 0
        remaining = (n_samples - processed) / rate if rate > 0 else 0
        
        # Show progress (every 1000 samples or every 10 batches)
        if i % (batch_size * 10) == 0 or batch_end == n_samples:
            print(f"\r  Progress: {processed:,}/{n_samples:,} ({100*processed/n_samples:.1f}%) | "
                  f"Success: {len(valid_indices):,} | Fail: {failed_count:,} | "
                  f"Speed: {rate:.0f} samples/s | "
                  f"Remaining: {remaining/60:.1f} min", end='', flush=True)
    
    print()  
    
    total_time = time.perf_counter() - start_counter
    if features_list:
        X = np.vstack(features_list)
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"\nINFO: Feature extraction completed:")
        print(f"  - Success: {len(X):,} samples")
        print(f"  - Fail: {failed_count:,} samples")
        print(f"  - Success rate: {100*len(X)/n_samples:.1f}%")
        print(f"  - Total time: {total_time/60:.1f} min")
        print(f"  - Avg speed: {n_samples/total_time:.0f} samples/s")
        
        step_time = time.perf_counter() - step_start
        performance_stats['steps'].append({
        'name': 'Feature extraction',
            'time_seconds': step_time,
            'samples_processed': n_samples,
            'samples_success': len(X),
            'samples_failed': failed_count,
            'speed_samples_per_sec': n_samples/total_time,
            'details': f'{len(X):,}/{n_samples:,} samples succeeded'
        })
        return X, df_valid
    else:
        print(f"\nERROR: Feature extraction failed: no valid features")
        return None, None

def predict_batch(models, X, df_valid, batch_size=10000):
    """Predict in batches with detailed progress"""
    step_start = time.perf_counter()
    print("\n" + "="*80)
    print("Step 3: Batch prediction")
    print("-"*80)
    print("Config:")
    print(f"  - Samples: {len(X):,}")
    print(f"  - Batch size: {batch_size:,}")
    print(f"  - Models: {len(models)}")
    
    predictions = {}
    target_times = {}
    
    # Predict each target
    for target_idx, (target, model) in enumerate(models.items(), 1):
        print(f"\nPredict target {target_idx}/{len(models)}: {target}")
        
        n_samples = len(X)
        preds = []
        
        start_counter = time.perf_counter()
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            batch_pred = model.predict(batch_X)
            preds.append(batch_pred)
            
            # Compute progress
            elapsed = time.perf_counter() - start_counter
            rate = batch_end / elapsed if elapsed > 0 else 0
            remaining = (n_samples - batch_end) / rate if rate > 0 else 0
            
            # Show progress
            print(f"\r  Progress: {batch_end:,}/{n_samples:,} ({100*batch_end/n_samples:.1f}%) | "
                  f"Speed: {rate:.0f} samples/s | "
                  f"Remaining: {remaining:.0f}s", end='', flush=True)
        
        predictions[target] = np.concatenate(preds)
        
        # Record timing
        target_time = time.perf_counter() - start_counter
        target_times[target] = target_time
        
        # Stats
        print(f"\nINFO: Completed: {target}")
        print(f"    - Min: {predictions[target].min():.6f}")
        print(f"    - Max: {predictions[target].max():.6f}")
        print(f"    - Mean: {predictions[target].mean():.6f}")
        print(f"    - Std: {predictions[target].std():.6f}")
        print(f"    - Time: {target_time:.3f}s")
        print(f"    - Speed: {n_samples/target_time:.0f} samples/s")
    
    # Add predictions to DataFrame
    print("\nINFO: Adding predictions to DataFrame...")
    if 'wavelength' in predictions:
        df_valid['Predicted_wavelength'] = predictions['wavelength']
    if 'PLQY' in predictions:
        df_valid['Predicted_PLQY'] = predictions['PLQY']
    
    step_time = time.perf_counter() - step_start
    prediction_speed = len(X) / step_time if step_time > 0 else 0
    
    performance_stats['steps'].append({
        'name': 'Batch prediction',
        'time_seconds': step_time,
        'samples': len(X),
        'models': len(models),
        'prediction_speed_samples_per_sec': prediction_speed,
        'target_times': target_times,
        'details': f'Predicted {len(models)} targets, speed: {prediction_speed:.0f} samples/s'
    })
    
    return df_valid

def analyze_results(df):
    """Analyze prediction results"""
    print("\n" + "="*80)
    print("Step 4: Result analysis")
    print("-"*80)
    
    if 'Predicted_PLQY' in df.columns:
        print("\nSTATS: PLQY distribution:")
        plqy = df['Predicted_PLQY']
        print(f"  - Min: {plqy.min():.4f}")
        print(f"  - 25%: {plqy.quantile(0.25):.4f}")
        print(f"  - Median: {plqy.median():.4f}")
        print(f"  - 75%: {plqy.quantile(0.75):.4f}")
        print(f"  - Max: {plqy.max():.4f}")
        print(f"  - Mean: {plqy.mean():.4f}")
        print(f"  - Std: {plqy.std():.4f}")
        
        # PLQY range distribution
        print("\n  PLQY range distribution:")
        ranges = [
            (0.9, 1.0, "0.9-1.0"),
            (0.8, 0.9, "0.8-0.9"),
            (0.7, 0.8, "0.7-0.8"),
            (0.6, 0.7, "0.6-0.7"),
            (0.5, 0.6, "0.5-0.6"),
            (0.0, 0.5, "0.0-0.5")
        ]
        for min_val, max_val, label in ranges:
            count = ((plqy >= min_val) & (plqy < max_val)).sum()
            pct = 100 * count / len(plqy)
            print(f"    {label}: {count:,} ({pct:.1f}%)")
        
        # Top 10 PLQY
        print("\nTop 10 PLQY combinations:")
        top10 = df.nlargest(10, 'Predicted_PLQY')
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            print(f"\n  #{i}:")
            print(f"    PLQY: {row['Predicted_PLQY']:.4f}")
            if 'Predicted_wavelength' in df.columns:
                print(f"    Wavelength: {row['Predicted_wavelength']:.1f} nm")
            if i <= 3:
                print(f"    L1/L2: {row['L1'][:60]}...")
                print(f"    L3: {row['L3'][:60]}...")
    
    if 'Predicted_wavelength' in df.columns:
        print("\nSTATS: Wavelength distribution:")
        wl = df['Predicted_wavelength']
        print(f"  - Min: {wl.min():.1f} nm")
        print(f"  - 25%: {wl.quantile(0.25):.1f} nm")
        print(f"  - Median: {wl.median():.1f} nm")
        print(f"  - 75%: {wl.quantile(0.75):.1f} nm")
        print(f"  - Max: {wl.max():.1f} nm")
        print(f"  - Mean: {wl.mean():.1f} nm")

def save_performance_stats(output_dir):
    """Save performance statistics tables"""
    print("\nSaving performance statistics...")
    
    # Create performance statistics table
    perf_data = []
    for step in performance_stats['steps']:
        row = {
            'Step': step['name'],
            'Time_seconds': f"{step['time_seconds']:.3f}",
            'Time_minutes': f"{step['time_seconds']/60:.3f}",
            'Details': step['details']
        }
        
        # Add extra info - ensure fields have values
        if 'samples_processed' in step:
            row['Samples_processed'] = f"{step['samples_processed']:,}"
            row['Samples_success'] = f"{step['samples_success']:,}"
            row['Samples_failed'] = f"{step['samples_failed']:,}"
            row['Speed_samples_per_sec'] = f"{step['speed_samples_per_sec']:.0f}"
        elif 'samples' in step:  
            row['Samples_processed'] = f"{step['samples']:,}"
            row['Samples_success'] = f"{step['samples']:,}"
            row['Samples_failed'] = '0'
            if 'prediction_speed_samples_per_sec' in step:
                row['Speed_samples_per_sec'] = f"{step['prediction_speed_samples_per_sec']:.0f}"
            else:
                row['Speed_samples_per_sec'] = '-'
        else:
            row['Samples_processed'] = '-'
            row['Samples_success'] = '-'
            row['Samples_failed'] = '-'
            row['Speed_samples_per_sec'] = '-'
        
        if 'target_times' in step:
            for target, t in step['target_times'].items():
                row[f'{target}_prediction_time_seconds'] = f"{t:.3f}"
        
        perf_data.append(row)
    
    # Add total row
    total_time = performance_stats.get('total_duration_seconds', 0.0)
    perf_data.append({
        'Step': 'Total',
        'Time_seconds': f"{total_time:.3f}",
        'Time_minutes': f"{total_time/60:.3f}",
        'Details': f"Completed {len(performance_stats['steps'])} steps",
        'Samples_processed': '-',
        'Samples_success': '-',
        'Samples_failed': '-',
        'Speed_samples_per_sec': '-'
    })
    
    # Save as CSV
    perf_df = pd.DataFrame(perf_data)
    perf_file = Path(output_dir) / 'performance_statistics.csv'
    perf_df.to_csv(perf_file, index=False, encoding='utf-8-sig')
    print(f"  [INFO] Performance statistics: {perf_file}")
    
    # Save as JSON
    json_file = Path(output_dir) / 'performance_statistics.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(performance_stats, f, ensure_ascii=True, indent=2)
    print(f"  [INFO] Detailed statistics: {json_file}")
    
    # Print performance table
    print("\nPerformance summary:")
    print("-" * 80)
    print(perf_df.to_string(index=False))
    print("-" * 80)
    
    # Print performance summary
    print("\n" + "="*80)
    print("SUMMARY: Performance Metrics")
    print("="*80)
    
    # Extract key performance data
    feature_speed = 0
    prediction_speed = 0
    total_samples = 0
    
    for step in performance_stats['steps']:
        if step['name'] == 'Feature extraction' and 'speed_samples_per_sec' in step:
            feature_speed = step['speed_samples_per_sec']
            total_samples = step.get('samples_processed', 0)
        elif step['name'] == 'Batch prediction' and 'prediction_speed_samples_per_sec' in step:
            prediction_speed = step['prediction_speed_samples_per_sec']
            if total_samples == 0:
                total_samples = step.get('samples', 0)
    
    total_time = performance_stats.get('total_duration_seconds', 0.0)
    end_to_end_speed = total_samples / total_time if total_time > 0 else 0
    
    print(f"  SAMPLES: {total_samples:,}")
    print(f"  TIME: Total {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"  FEATURE: Extraction speed {feature_speed:,.0f} samples/s")
    print(f"  PRED: Prediction speed {prediction_speed:,.0f} samples/s")
    print(f"  END2END: End-to-end speed {end_to_end_speed:,.0f} samples/s")
    
    if prediction_speed > 100000:
        print(f"  FAST: Ultra-fast prediction {prediction_speed/1000:.0f}k samples/s!")
    
    print("="*80)
    
    # Save detailed performance report
    performance_report = {
        'summary': {
            'total_samples': total_samples,
            'total_time_seconds': total_time,
            'feature_extraction_speed': feature_speed,
            'prediction_speed': prediction_speed,
            'end_to_end_speed': end_to_end_speed
        },
        'steps': performance_stats['steps']
    }
    
    report_file = Path(output_dir) / 'performance_detailed.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(performance_report, f, ensure_ascii=True, indent=2)
    print(f"\nINFO: Detailed performance report: {report_file}")
    
    return perf_df

def save_results(df, output_file):
    """Save results, including sorted versions"""
    step_start = time.perf_counter()
    print("\n" + "="*80)
    print("Step 5: Save results")
    print("-"*80)
    
    # Save full prediction results
    print(f"\nSaving full prediction results...")
    df.to_csv(output_file, index=False)
    print(f"  [INFO] File: {output_file}")
    print(f"  [INFO] Rows: {len(df):,}")
    
    # Save sorted by PLQY
    if 'Predicted_PLQY' in df.columns:
        sorted_file = output_file.replace('.csv', '_sorted_by_plqy.csv')
        df_sorted = df.sort_values('Predicted_PLQY', ascending=False)
        df_sorted.to_csv(sorted_file, index=False)
        print(f"INFO: PLQY sorted: {sorted_file}")
        
        # Save filtered results for thresholds
        thresholds = [0.9, 0.8, 0.7]
        for threshold in thresholds:
            filtered = df[df['Predicted_PLQY'] >= threshold]
            if len(filtered) > 0:
                threshold_file = output_file.replace('.csv', f'_plqy_{threshold:.1f}+.csv')
                filtered.to_csv(threshold_file, index=False)
                print(f"  [INFO] PLQY>={threshold}: {threshold_file} ({len(filtered):,})")
    
    step_time = time.perf_counter() - step_start
    performance_stats['steps'].append({
        'name': 'Save results',
        'time_seconds': step_time,
        'details': f'Saved {len(df):,} predictions'
    })

def main():
    parser = argparse.ArgumentParser(description='Predict properties for all combinations')
    parser.add_argument('--project', '-p',
                       help='Model project directory (default: Project_Output)')
    parser.add_argument('--input', '-i', 
                       help='Combination file (default: PROJECT/ir_assemble.csv)')
    parser.add_argument('--output', '-o',
                       help='Output file (default: PROJECT/ir_assemble_predicted_all.csv)')
    parser.add_argument('--intersection', action='store_true',
                       help='Use intersection-trained models (trained on samples with all targets)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for feature extraction')
    parser.add_argument('--predict-batch', type=int, default=10000,
                       help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # Set default project directory to fixed name
    if not args.project:
        args.project = 'Project_Output'
    
    # Set default input and output paths
    if not args.input:
        args.input = f"{args.project}/ir_assemble.csv"
    if not args.output:
        if args.intersection:
            args.output = f"{args.project}/ir_assemble_predicted_intersection.csv"
        else:
            args.output = f"{args.project}/ir_assemble_predicted_all.csv"
    
    print("="*80)
    print("Full prediction pipeline - 272,104 combinations")
    print("="*80)
    print("\nConfig:")
    print(f"  - Project: {args.project}")
    print(f"  - Model type: {'Intersection-trained' if args.intersection else 'Full-data-trained'}")
    print(f"  - Input file: {args.input}")
    print(f"  - Output file: {args.output}")
    
    # Record elapsed runtime
    total_start = time.perf_counter()
    
    # 1. Load combinations
    print(f"\nLoading combination file...")
    df = pd.read_csv(args.input)
    print(f"INFO: Loaded {len(df):,} combinations")
    
    # Validate L1=L2
    same_count = (df['L1'] == df['L2']).sum()
    print(f"INFO: L1=L2 validation: {same_count:,}/{len(df):,}")
    
    # 2. Load models
    models = load_models(args.project, use_intersection=args.intersection)
    if not models:
        print("ERROR: No models found")
        return
    
    # 3. Extract features
    X, df_valid = extract_features_batch(df, batch_size=args.batch_size)
    if X is None:
        print("ERROR: Feature extraction failed")
        return
    
    # 4. Predict
    df_predicted = predict_batch(models, X, df_valid, batch_size=args.predict_batch)
    
    # 5. Analyze results
    analyze_results(df_predicted)
    
    # 6. Save results
    save_results(df_predicted, args.output)
    
    performance_stats['total_duration_seconds'] = time.perf_counter() - total_start
    
    # 7. Save performance statistics
    output_dir = Path(args.output).parent
    save_performance_stats(output_dir)
    
    # Summary
    total_time = performance_stats['total_duration_seconds']
    processed_samples = len(df_predicted) if df_predicted is not None else len(df)
    speed = processed_samples / total_time if total_time > 0 else 0.0
    print("\n" + "="*80)
    print("Prediction completed!")
    print(f"  - Total time: {total_time:.3f} s ({total_time/60:.3f} min)")
    print(f"  - Processing speed: {speed:.2f} samples/s")
    print("="*80)

if __name__ == "__main__":
    main()
