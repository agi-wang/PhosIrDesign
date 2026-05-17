#!/usr/bin/env python3
"""
Standalone script to generate stratified performance analysis
Generates PLQY confusion matrix and related analysis plots from training results
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from phosirdesign.visualization.stratified_analysis import generate_stratified_analysis


def load_predictions_from_project(project_dir: Path) -> dict:
    """
    Load prediction results from a project directory
    
    Args:
        project_dir: project directory path
    
    Returns:
        dict of predictions
    """
    predictions = {}
    
    print(f"INFO: Loading predictions from project dir: {project_dir}")
    
    # Method 1: find CV prediction result files
    for model_dir in project_dir.glob('*/'):
        # Find cv_predictions files
        for result_file in model_dir.glob('results/cv_predictions_*.csv'):
            target = result_file.stem.replace('cv_predictions_', '')
            
            # Read prediction results
            df = pd.read_csv(result_file)
            if 'actual' in df.columns and 'predicted' in df.columns:
                predictions[target] = {
                    'actual': df['actual'].values,
                    'predicted': df['predicted'].values
                }
                print(f"  INFO: Loaded {target}: {len(df)} predictions")
    
    # Method 2: extract from training_results.json
    if not predictions:
        print("  INFO: No cv_predictions found, trying training_results.json...")
        
        for model_dir in project_dir.glob('*/'):
            results_file = model_dir / 'results' / 'training_results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract fold predictions for each model
                for model_name, model_data in results.items():
                    if isinstance(model_data, dict) and 'targets' in model_data:
                        for target, target_data in model_data['targets'].items():
                            if 'fold_results' in target_data:
                                all_actual = []
                                all_predicted = []
                                
                                for fold in target_data['fold_results']:
                                    if 'predictions' in fold:
                                        pred_data = fold['predictions']
                                        if isinstance(pred_data, dict):
                                            all_actual.extend(pred_data.get('actual', []))
                                            all_predicted.extend(pred_data.get('predicted', []))
                                
                                if all_actual:
                                    clean_target = target.replace('(', '').replace(')', '').replace('*', 'x')
                                    key = f'{clean_target}_{model_name}'
                                    predictions[key] = {
                                        'actual': np.array(all_actual),
                                        'predicted': np.array(all_predicted)
                                    }
                                    print(f"  INFO: Loaded {key}: {len(all_actual)} predictions")
    
    # Method 3: check AutoML training outputs
    if not predictions:
        print("  INFO: Trying AutoML training outputs...")
        
        # Find all_models/automl_train directory
        automl_dir = project_dir / 'all_models' / 'automl_train'
        if automl_dir.exists():
            for model_dir in automl_dir.glob('*/'):
                model_name = model_dir.name
                
                # Find CSV prediction files
                for pred_file in model_dir.glob('exports/csv/*_all_predictions.csv'):
                    try:
                        df = pd.read_csv(pred_file)
                        
                        # Parse target from filename
                        filename = pred_file.stem  # e.g.: xgboost_PLQY_all_predictions
                        parts = filename.split('_')
                        
                        # Find target name (after model name, before date)
                        target = None
                        for i, part in enumerate(parts):
                            if part == model_name and i < len(parts) - 1:
                                # Target name may contain multiple parts
                                target_parts = []
                                for j in range(i+1, len(parts)):
                                    if parts[j].isdigit() and len(parts[j]) == 8:  # date part
                                        break
                                    target_parts.append(parts[j])
                                if target_parts:
                                    target = '_'.join(target_parts)
                                    break
                        
                        if not target:
                            # Fallback: check column names
                            if 'Max_wavelength(nm)' in str(pred_file):
                                target = 'wavelength'
                            elif 'PLQY' in str(pred_file):
                                target = 'PLQY'
                            elif 'tau' in str(pred_file):
                                target = 'tau'
                        
                        if target and 'true' in df.columns and 'predicted' in df.columns:
                            key = f'{target}_{model_name}'
                            predictions[key] = {
                                'actual': df['true'].values,
                                'predicted': df['predicted'].values
                            }
                            print(f"  INFO: Loaded {key}: {len(df)} predictions")
                    except Exception as e:
                        print(f"  WARNING: Failed to read {pred_file.name}: {e}")
    
    # Method 4: check exported-results directory
    if not predictions:
        print("  INFO: Trying exports directory...")
        
        for export_file in project_dir.glob('*/exports/json/*_complete.json'):
            try:
                with open(export_file, 'r') as f:
                    data = json.load(f)
                
                if 'cv_results' in data:
                    cv_data = data['cv_results']
                    if 'all_predictions' in cv_data:
                        target = data.get('target_column', 'unknown')
                        model = data.get('model_type', 'unknown')
                        
                        all_pred = cv_data['all_predictions']
                        if all_pred and len(all_pred) > 0:
                            # Extract actual and predicted values
                            actuals = []
                            predicteds = []
                            
                            for pred in all_pred:
                                if isinstance(pred, dict) and 'actual' in pred and 'predicted' in pred:
                                    actuals.append(pred['actual'])
                                    predicteds.append(pred['predicted'])
                            
                            if actuals:
                                clean_target = target.replace('(', '').replace(')', '').replace('*', 'x')
                                key = f'{clean_target}_{model}'
                                predictions[key] = {
                                    'actual': np.array(actuals),
                                    'predicted': np.array(predicteds)
                                }
                                print(f"  INFO: Loaded {key}: {len(actuals)} predictions")
            except Exception as e:
                print(f"  WARNING: Failed to read {export_file.name}: {e}")
    
    return predictions


def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(
        description='Generate stratified performance analysis figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_stratified_analysis.py --project Project_Output_YYYYMMDD_HHMMSS --output Project_Output_YYYYMMDD_HHMMSS/figures
  python generate_stratified_analysis.py --project results/training_001
        """
    )
    
    parser.add_argument(
        '--project', '-p',
        type=str,
        required=True,
        help='Project directory path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory path (default: PROJECT/figures)'
    )
    
    parser.add_argument(
        '--targets', '-t',
        type=str,
        nargs='+',
        default=None,
        help='Targets to analyze (default: all)'
    )
    
    args = parser.parse_args()
    
    # Set paths
    project_dir = Path(args.project)
    if not project_dir.exists():
        print(f"ERROR: Project directory not found: {project_dir}")
        return 1
    
    output_dir = Path(args.output) if args.output else project_dir / 'figures'
    
    print("="*60)
    print("Stratified performance analysis")
    print("="*60)
    print(f"Project dir: {project_dir}")
    print(f"Output dir: {output_dir}")
    
    # Load prediction data
    print("\nLoading predictions...")
    predictions = load_predictions_from_project(project_dir)
    
    if not predictions:
        print("ERROR: No prediction data found")
        return 1
    
    print(f"\nFound {len(predictions)} prediction datasets")
    
    # Generate stratified analysis
    print("\nGenerating analysis plots...")
    results = generate_stratified_analysis(
        predictions=predictions,
        output_dir=output_dir,
        targets=args.targets
    )
    
    print("\n" + "="*60)
    print("INFO: Stratified analysis completed")
    print("="*60)
    print(f"Output dir: {output_dir / 'stratified_analysis'}")
    
    # List generated files
    analysis_dir = output_dir / 'stratified_analysis'
    if analysis_dir.exists():
        print("\nGenerated files:")
        for file in sorted(analysis_dir.rglob('*')):
            if file.is_file():
                print(f"  - {file.relative_to(output_dir)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
