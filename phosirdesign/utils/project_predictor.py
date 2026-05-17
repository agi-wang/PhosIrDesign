#!/usr/bin/env python3
"""
Project-level batch predictor
Manage and execute batch prediction tasks across a project
"""

import json
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Import required modules
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from phosirdesign.core.feature_extractor import FeatureExtractor
from phosirdesign.utils.batch_predictor_v2 import BatchPredictorV2
from phosirdesign.utils.file_feature_cache import FileFeatureCache
from phosirdesign.utils.timing import TimingTracker
from phosirdesign.models.base import ModelFactory, load_model


class ProjectPredictor:
    """Project-level batch predictor"""
    
    def __init__(self, project_dir: str, verbose: bool = True):
        """
        Initialize project predictor
        
        Args:
            project_dir: Project directory path
            verbose: Show detailed output
        """
        self.project_dir = Path(project_dir)
        self.verbose = verbose
        
        if not self.project_dir.exists():
            raise ValueError(f"Project directory not found: {project_dir}")
        
        # Load project info
        self.models = {}
        self.configs = {}
        self.metadata = {}
        
        # Initialize timing tracker
        self.timing = TimingTracker(f"project_predictor_{project_dir}")
        
        # Scan and load all models
        with self.timing.measure('load_models'):
            self._load_all_models()
        
        # Initialize batch predictor
        self.batch_predictor = BatchPredictorV2(
            batch_size=5000,
            show_progress=verbose
        )
        
        if self.verbose:
            print(f"Loaded project: {self.project_dir}")
            print(f"   Found {len(self.models)} models")
    
    def _load_all_models(self):
        """Load all models in the project"""
        # Find all model files
        model_files = list(self.project_dir.rglob("*.joblib"))
        
        for model_file in model_files:
            parsed = self._parse_model_name(model_file.stem)
            if not parsed:
                continue

            model_type, target = parsed
            key = f"{model_type}_{target}"

            try:
                model = load_model(model_file)

                target_mappings = {
                    'Max_wavelength_nm': 'Max_wavelength(nm)',
                    'tau_sx10-6': 'tau(s*10^-6)',
                    'PLQY': 'PLQY'
                }
                original_target = target_mappings.get(target, target)

                self.models[key] = {
                    'model': model,
                    'path': str(model_file),
                    'type': model_type,
                    'target': target,
                    'original_target': original_target,
                    'name': model_file.stem
                }

                config_file = model_file.parent.parent / 'config.yaml'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        self.configs[key] = yaml.safe_load(f)

                exports_dir = model_file.parent.parent / "exports"
                summary_files = []
                if exports_dir.exists():
                    for f in exports_dir.glob(f"{model_type}_*_summary.json"):
                        if original_target in f.name:
                            summary_files.append(f)
                            break
                if summary_files:
                    with open(summary_files[0], 'r') as f:
                        summary = json.load(f)
                        self.models[key]['performance'] = {
                            'r2': summary.get('mean_r2', 0),
                            'r2_std': summary.get('std_r2', 0),
                            'rmse': summary.get('mean_rmse', 0),
                            'rmse_std': summary.get('std_rmse', 0),
                            'mae': summary.get('mean_mae', 0),
                            'mae_std': summary.get('std_mae', 0)
                        }

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to load model {model_file}: {e}")

    @staticmethod
    def _parse_model_name(model_name: str) -> Optional[Tuple[str, str]]:
        """Parse persisted model filename into (model_type, target)."""
        supported = sorted(ModelFactory.get_supported_models(), key=len, reverse=True)

        for model_type in supported:
            prefix = f"{model_type}_"
            if not model_name.startswith(prefix):
                continue

            remainder = model_name[len(prefix):]
            if remainder.endswith("_final"):
                target = remainder[:-len("_final")]
            else:
                parts = remainder.rsplit("_", 1)
                target = parts[0] if len(parts) == 2 else remainder

            if target:
                return model_type, target

        return None
    
    def list_models(self) -> pd.DataFrame:
        """
        List all available models
        
        Returns:
            DataFrame of model information
        """
        if not self.models:
            return pd.DataFrame()
        
        data = []
        for key, info in self.models.items():
            perf = info.get('performance', {})
            
            # Format values with standard deviation
            def format_metric(mean_key, std_key, decimals=4):
                mean_val = perf.get(mean_key, 'N/A')
                std_val = perf.get(std_key, 0)
                if isinstance(mean_val, (int, float)):
                    if std_val > 0:
                        return f"{mean_val:.{decimals}f}+/-{std_val:.{decimals}f}"
                    else:
                        return f"{mean_val:.{decimals}f}"
                return 'N/A'
            
            row = {
                'Model': info['type'],
                'Target': info.get('original_target', info['target']),
                'R^2 (mean+/-std)': format_metric('r2', 'r2_std', 4),
                'RMSE (mean+/-std)': format_metric('rmse', 'rmse_std', 2),
                'MAE (mean+/-std)': format_metric('mae', 'mae_std', 2)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        if self.verbose:
            print("\nProject model list:")
            print(df.to_string(index=False))
        
        return df
    
    def predict_all_models(self, data_path: str, output_dir: str = None,
                          smiles_columns: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Predict with all models
        
        Args:
            data_path: Input data file path
            output_dir: Output directory
            smiles_columns: SMILES column names
        
        Returns:
            Dict containing prediction results for all models
        """
        # Read data
        with self.timing.measure('data_loading', {'file': data_path}):
            df = pd.read_csv(data_path)
        print(f"\nLoading data: {data_path}")
        print(f"   Samples: {len(df)}")
        
        # Record overall prediction throughput
        self.timing.calculate_throughput('data_loading', len(df))
        
        # Set output directory
        if output_dir is None:
            output_dir = self.project_dir / "batch_predictions"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default SMILES columns
        if smiles_columns is None:
            smiles_columns = ['L1', 'L2', 'L3']
        
        results = {}
        
        print(f"\nStarting batch prediction ({len(self.models)} models)...")
        
        for i, (key, model_info) in enumerate(self.models.items(), 1):
            print(f"\n[{i}/{len(self.models)}] Predict: {key}")
            
            try:
                with self.timing.measure(f'predict_{key}', {'model': key, 'samples': len(df)}):
                    # Get config
                    config = self.configs.get(key, {})
                    feature_config = config.get('feature', {})
                    
                    # Create feature extractor
                    feature_extractor = FeatureExtractor(
                        feature_type=feature_config.get('feature_type', 'combined'),
                        morgan_bits=feature_config.get('morgan_bits', 1024),
                        morgan_radius=feature_config.get('morgan_radius', 2),
                        use_cache=True
                    )
                    
                    # Perform prediction
                    pred_values, failed_indices = self.batch_predictor.predict_with_cache(
                        df=df,
                        model=model_info['model'],
                        feature_extractor=feature_extractor,
                        smiles_columns=smiles_columns,
                        feature_type=feature_config.get('feature_type', 'combined'),
                        combination_method=feature_config.get('combination_method', 'mean'),
                        input_file=str(data_path)
                    )
                    
                    # Create prediction result DataFrame
                    predictions = df.copy()
                    pred_col = f"Predicted_{model_info.get('original_target', model_info['target'])}"
                    # Use predicted values directly (failed_indices already set to NaN)
                    predictions[pred_col] = pred_values
                    
                    # Save results
                    output_file = output_dir / f"{key}_predictions.csv"
                    predictions.to_csv(output_file, index=False)
                    print(f"   Saved to: {output_file}")
                    
                    results[key] = predictions
                
                # Calculate throughput
                self.timing.calculate_throughput(f'predict_{key}', len(df))
                
            except Exception as e:
                print(f"   Prediction failed: {e}")
                continue
        
        # Generate summary file
        self._generate_summary(results, output_dir)
        
        print(f"\nBatch prediction completed")
        print(f"   Results directory: {output_dir}")
        
        # Print timing stats
        if self.verbose:
            print("\n" + "="*50)
            print("Prediction timing statistics")
            print("="*50)
            self.timing.print_summary()
            
            # Save timing report
            try:
                timing_file = output_dir / "timing_report.json"
                self.timing.save_report(timing_file, format='json')
                print(f"\nTiming report saved to: {timing_file}")
            except Exception as e:
                print(f"Warning: Failed to save timing report: {e}")
        
        return results
    
    def predict_best_models(self, data_path: str, output_path: str = None,
                           smiles_columns: List[str] = None) -> pd.DataFrame:
        """
        Predict using the best model per target
        
        Args:
            data_path: Input data file path
            output_path: Output file path
            smiles_columns: SMILES column names
        
        Returns:
            DataFrame containing predictions from best models
        """
        # Find best model per target
        best_models = self._find_best_models()
        
        if not best_models:
            print("No performance metrics found; cannot select best models")
            return pd.DataFrame()
        
        # Read data
        df = pd.read_csv(data_path)
        result_df = df.copy()
        
        print(f"\nLoading data: {data_path}")
        print(f"   Samples: {len(df)}")
        
        # Default SMILES columns
        if smiles_columns is None:
            smiles_columns = ['L1', 'L2', 'L3']
        
        print(f"\nPredicting with best models...")
        
        for target, model_key in best_models.items():
            model_info = self.models[model_key]
            print(f"\nTarget: {target}")
            print(f"  Best model: {model_info['type']} (R^2={model_info['performance']['r2']:.4f})")
            
            try:
                # Get config
                config = self.configs.get(model_key, {})
                feature_config = config.get('feature', {})
                
                # Create feature extractor
                feature_extractor = FeatureExtractor(
                    feature_type=feature_config.get('feature_type', 'combined'),
                    morgan_bits=feature_config.get('morgan_bits', 1024),
                    morgan_radius=feature_config.get('morgan_radius', 2),
                    use_cache=True
                )
                
                # Perform prediction
                pred_values, failed_indices = self.batch_predictor.predict_with_cache(
                    df=df,
                    model=model_info['model'],
                    feature_extractor=feature_extractor,
                    smiles_columns=smiles_columns,
                    feature_type=feature_config.get('feature_type', 'combined'),
                    combination_method=feature_config.get('combination_method', 'mean'),
                    input_file=str(data_path)
                )
                
                # Add to results (use predicted values; failed_indices already NaN)
                result_df[f"Best_{target}"] = pred_values
                
            except Exception as e:
                print(f"  Prediction failed: {e}")
                continue
        
        # Save results
        if output_path is None:
            output_path = "best_predictions.csv"
        
        result_df.to_csv(output_path, index=False)
        print(f"\nBest model prediction completed")
        print(f"   Saved to: {output_path}")
        
        return result_df
    
    def predict_ensemble(self, data_path: str, output_path: str = None,
                        smiles_columns: List[str] = None,
                        method: str = 'mean') -> pd.DataFrame:
        """
        Ensemble prediction (multiple models)
        
        Args:
            data_path: Input data file path
            output_path: Output file path
            smiles_columns: SMILES column names
            method: Ensemble method ('mean', 'median', 'weighted')
        
        Returns:
            DataFrame with ensemble prediction results
        """
        # Perform predictions for all models first
        all_predictions = self.predict_all_models(
            data_path=data_path,
            output_dir=None,
            smiles_columns=smiles_columns
        )
        
        if not all_predictions:
            return pd.DataFrame()
        
        # Read original data
        df = pd.read_csv(data_path)
        result_df = df.copy()
        
        print(f"\nRunning ensemble prediction (method: {method})...")
        
        # Group by targets
        targets = {}
        for key in all_predictions.keys():
            target = self.models[key]['target']
            if target not in targets:
                targets[target] = []
            targets[target].append(key)
        
        # Ensemble per target
        for target, model_keys in targets.items():
            print(f"\nTarget: {target}")
            print(f"  Participating models: {len(model_keys)}")
            
            # Collect predictions
            predictions = []
            weights = []
            
            for key in model_keys:
                pred_col = f"Predicted_{target}"
                if pred_col in all_predictions[key].columns:
                    predictions.append(all_predictions[key][pred_col].values)
                    
                    # If weighted, use R^2 as weights
                    if method == 'weighted' and 'performance' in self.models[key]:
                        r2 = self.models[key]['performance'].get('r2', 0)
                        weights.append(max(r2, 0))
                    else:
                        weights.append(1.0)
            
            if predictions:
                predictions = np.array(predictions)
                
                if method == 'mean':
                    ensemble_pred = np.mean(predictions, axis=0)
                elif method == 'median':
                    ensemble_pred = np.median(predictions, axis=0)
                elif method == 'weighted':
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    ensemble_pred = np.average(predictions, axis=0, weights=weights)
                else:
                    raise ValueError(f"Unsupported ensemble method: {method}")
                
                result_df[f"Ensemble_{target}"] = ensemble_pred
                print(f"  Ensemble complete")
        
        # Save results
        if output_path is None:
            output_path = "ensemble_predictions.csv"
        
        result_df.to_csv(output_path, index=False)
        print(f"\nEnsemble prediction completed")
        print(f"   Saved to: {output_path}")
        
        return result_df
    
    def _find_best_models(self) -> Dict[str, str]:
        """
        Find the best model per target
        
        Returns:
            Mapping from target to best model key
        """
        best_models = {}
        
        # Group by target (using original target names)
        targets = {}
        for key, info in self.models.items():
            target = info.get('original_target', info['target'])
            if target not in targets:
                targets[target] = []
            if 'performance' in info:
                targets[target].append((key, info['performance'].get('r2', -1)))
        
        # Select best model for each target
        for target, models in targets.items():
            if models:
                # Sort by R^2 and pick the highest
                models.sort(key=lambda x: x[1], reverse=True)
                best_models[target] = models[0][0]
        
        return best_models
    
    def _generate_summary(self, results: Dict[str, pd.DataFrame], output_dir: Path):
        """Generate prediction summary"""
        summary = {
            'project': str(self.project_dir),
            'models_used': len(results),
            'predictions': {}
        }
        
        for key, df in results.items():
            # Find prediction column
            pred_cols = [col for col in df.columns if col.startswith('Predicted_')]
            if pred_cols:
                pred_col = pred_cols[0]
                summary['predictions'][key] = {
                    'file': f"{key}_predictions.csv",
                    'samples': len(df),
                    'mean': float(df[pred_col].mean()),
                    'std': float(df[pred_col].std()),
                    'min': float(df[pred_col].min()),
                    'max': float(df[pred_col].max())
                }
        
        # Save summary
        summary_file = output_dir / 'prediction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary file: {summary_file}")
    
    def get_project_info(self) -> Dict:
        """
        Get project information
        
        Returns:
            Project info dict
        """
        info = {
            'project_path': str(self.project_dir),
            'models_count': len(self.models),
            'models': {},
            'targets': set(),
            'best_models': {}
        }
        
        # Collect model info
        for key, model_info in self.models.items():
            model_type = model_info['type']
            target = model_info['target']
            
            info['targets'].add(target)
            
            if model_type not in info['models']:
                info['models'][model_type] = []
            
            info['models'][model_type].append({
                'target': target,
                'performance': model_info.get('performance', {})
            })
        
        # Find best models
        best = self._find_best_models()
        for target, model_key in best.items():
            model_info = self.models[model_key]
            info['best_models'][target] = {
                'model': model_info['type'],
                'r2': model_info.get('performance', {}).get('r2', 'N/A')
            }
        
        info['targets'] = list(info['targets'])
        
        return info


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Project-level batch prediction')
    parser.add_argument('project', help='Project directory')
    parser.add_argument('--data', required=True, help='Prediction data file')
    parser.add_argument('--mode', default='all', 
                       choices=['all', 'best', 'ensemble'],
                       help='Prediction mode')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--smiles-columns', help='SMILES column names (comma-separated)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all models')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = ProjectPredictor(args.project)
    
    if args.list_models:
        predictor.list_models()
        return
    
    # Parse SMILES columns
    smiles_columns = None
    if args.smiles_columns:
        smiles_columns = args.smiles_columns.split(',')
    
    # Run prediction
    if args.mode == 'all':
        predictor.predict_all_models(
            data_path=args.data,
            output_dir=args.output,
            smiles_columns=smiles_columns
        )
    elif args.mode == 'best':
        predictor.predict_best_models(
            data_path=args.data,
            output_path=args.output,
            smiles_columns=smiles_columns
        )
    elif args.mode == 'ensemble':
        predictor.predict_ensemble(
            data_path=args.data,
            output_path=args.output,
            smiles_columns=smiles_columns
        )


if __name__ == '__main__':
    main()
