#!/usr/bin/env python3
"""
Training log recording module
Comprehensive recorder similar to TensorBoard that saves training data for reporting
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Optional, Union, Any
import hashlib
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict, field
import yaml


# ========================================
#           Data Class Definitions
# ========================================

@dataclass
class FoldResult:
    """Training results for a single fold"""
    fold_id: int
    train_indices: List[int]
    val_indices: List[int]
    train_predictions: Optional[np.ndarray] = None
    val_predictions: np.ndarray = None
    train_true: Optional[np.ndarray] = None
    val_true: np.ndarray = None
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    training_history: Optional[List[Dict]] = None
    timing: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict (handles NumPy arrays)"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


@dataclass
class ExperimentResult:
    """Complete experiment result"""
    experiment_id: str
    run_sequence: int
    model_type: str
    target: str
    feature_type: str
    n_samples: int
    n_features: int
    n_folds: int
    fold_results: List[FoldResult]
    final_model_metrics: Optional[Dict[str, float]] = None
    hyperparameters: Dict = field(default_factory=dict)
    data_info: Dict = field(default_factory=dict)
    system_info: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dict"""
        result = asdict(self)
        result['fold_results'] = [fold.to_dict() for fold in self.fold_results]
        return result


# ========================================
#           Training Logger
# ========================================

class TrainingLogger:
    """Training process recorder"""
    
    def __init__(self, 
                 project_name: str,
                 base_dir: str = "training_logs",
                 auto_save: bool = True,
                 save_plots: bool = True):
        """
        Initialize training logger
        
        Args:
            project_name: Project name
            base_dir: Base save directory
            auto_save: Auto-save flag
            save_plots: Save plots flag
        """
        self.project_name = project_name
        self.base_dir = Path(base_dir)
        self.auto_save = auto_save
        self.save_plots = save_plots
        
        # Create project directory structure
        self.project_dir = self.base_dir / project_name
        self.create_directory_structure()
        
        # Current experiment
        self.current_experiment = None
        self.experiment_history = []
        
        # Real-time records
        self.current_fold_data = {}
        self.global_metrics = {}
        
        print("INFO Training logger initialized")
        print(f"   Project: {project_name}")
        print(f"   Save path: {self.project_dir}")
    
    def create_directory_structure(self):
        """Create directory structure"""
        directories = [
            self.project_dir,
            self.project_dir / "experiments",
            self.project_dir / "models",
            self.project_dir / "predictions",
            self.project_dir / "plots",
            self.project_dir / "plots" / "fold_results",
            self.project_dir / "plots" / "comparison",
            self.project_dir / "plots" / "feature_importance",
            self.project_dir / "exports",
            self.project_dir / "exports" / "csv",
            self.project_dir / "exports" / "excel",
            self.project_dir / "exports" / "json",
            self.project_dir / "checkpoints"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    #           Experiment Management
    # ========================================
    
    def start_experiment(self, 
                        model_type: str,
                        target: str,
                        feature_type: str,
                        hyperparameters: Dict,
                        n_folds: int = 10,
                        **kwargs):
        """
        Start a new experiment
        
        Args:
            model_type: Model type
            target: Target variable
            feature_type: Feature type
            hyperparameters: Hyperparameters
            n_folds: Number of CV folds
            **kwargs: Additional info
        """
        experiment_id = f"{model_type}_{target}"
        run_sequence = len(self.experiment_history) + 1
        
        self.current_experiment = {
            'experiment_id': experiment_id,
            'run_sequence': run_sequence,
            'model_type': model_type,
            'target': target,
            'feature_type': feature_type,
            'hyperparameters': hyperparameters,
            'n_folds': n_folds,
            'fold_results': [],
            'start_counter': time.perf_counter(),
            'timing': {},
            **kwargs
        }
        
        # Save experiment config
        config_path = self.project_dir / "experiments" / f"{experiment_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'experiment_id': experiment_id,
                'run_sequence': run_sequence,
                'model_type': model_type,
                'target': target,
                'feature_type': feature_type,
                'hyperparameters': hyperparameters,
                'n_folds': n_folds,
                **kwargs
            }, f, indent=2, ensure_ascii=True)
        
        print(f"\nINFO Experiment started: {experiment_id}")
        
        return experiment_id
    
    def end_experiment(self, final_metrics: Optional[Dict] = None):
        """
        End current experiment
        
        Args:
            final_metrics: Final evaluation metrics
        """
        if self.current_experiment is None:
            return
        
        duration = time.perf_counter() - self.current_experiment['start_counter']
        self.current_experiment['duration_seconds'] = duration
        self.current_experiment['final_metrics'] = final_metrics
        
        # Save full experiment results
        self.save_experiment_results()
        
        # Generate report
        self.generate_experiment_report()
        
        # Add to history
        self.experiment_history.append(self.current_experiment)
        
        print(f"\nINFO Experiment ended: {self.current_experiment['experiment_id']}")
        print(f"   Duration: {duration:.2f}s")
        
        self.current_experiment = None

    def add_timing(self, key: str, seconds: float):
        """Add timing record for current experiment"""
        try:
            if self.current_experiment is not None:
                timing = self.current_experiment.get('timing', {})
                timing[key] = float(seconds)
                self.current_experiment['timing'] = timing
        except Exception:
            pass
    
    # ========================================
    #           Fold Logging
    # ========================================
    
    def log_fold_start(self, fold_id: int, train_indices: List[int], val_indices: List[int]):
        """
        Record fold start
        
        Args:
            fold_id: Fold index
            train_indices: Training indices
            val_indices: Validation indices
        """
        self.current_fold_data = {
            'fold_id': fold_id,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'start_counter': time.perf_counter()
        }
        
        print(f"\n  INFO Fold {fold_id} started")
        print(f"     Train samples: {len(train_indices)}")
        print(f"     Val samples: {len(val_indices)}")
    
    def log_fold_end(self, 
                     y_train: np.ndarray,
                     y_train_pred: np.ndarray,
                     y_val: np.ndarray,
                     y_val_pred: np.ndarray,
                     metrics: Dict[str, float],
                     feature_importance: Optional[Dict] = None,
                     **kwargs):
        """
        Record fold end
        
        Args:
            y_train: Training true values
            y_train_pred: Training predictions
            y_val: Validation true values
            y_val_pred: Validation predictions
            metrics: Evaluation metrics
            feature_importance: Feature importance
            **kwargs: Additional info
        """
        if not self.current_fold_data:
            return
        
        duration = time.perf_counter() - self.current_fold_data['start_counter']
        
        # Create fold result
        fold_result = FoldResult(
            fold_id=self.current_fold_data['fold_id'],
            train_indices=self.current_fold_data['train_indices'],
            val_indices=self.current_fold_data['val_indices'],
            train_predictions=y_train_pred,
            val_predictions=y_val_pred,
            train_true=y_train,
            val_true=y_val,
            metrics=metrics,
            feature_importance=feature_importance,
            timing={'duration_seconds': duration}
        )
        
        # Append to current experiment
        if self.current_experiment:
            self.current_experiment['fold_results'].append(fold_result)
        
        # Save fold data
        if self.auto_save:
            self.save_fold_data(fold_result)
        
        # Generate fold plots
        if self.save_plots:
            self.plot_fold_results(fold_result)
        
        print(f"     INFO Fold {fold_result.fold_id} completed (duration: {duration:.2f}s)")
        print(f"       Validation RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"       Validation R^2: {metrics.get('r2', 0):.4f}")
    
    # ========================================
    #           Data Saving
    # ========================================
    
    def save_fold_data(self, fold_result: FoldResult):
        """Save single fold data"""
        if not self.current_experiment:
            return
        
        exp_id = self.current_experiment['experiment_id']
        fold_id = fold_result.fold_id
        
        # Save prediction CSV
        pred_df = pd.DataFrame({
            'fold': fold_id,
            'split': ['train'] * len(fold_result.train_true) + ['val'] * len(fold_result.val_true),
            'true': np.concatenate([fold_result.train_true, fold_result.val_true]),
            'predicted': np.concatenate([fold_result.train_predictions, fold_result.val_predictions]),
            'error': np.concatenate([
                fold_result.train_true - fold_result.train_predictions,
                fold_result.val_true - fold_result.val_predictions
            ])
        })
        
        csv_path = self.project_dir / "predictions" / f"{exp_id}_fold{fold_id}.csv"
        pred_df.to_csv(csv_path, index=False)
        
        # Save raw NumPy arrays (for exact reproduction)
        np_path = self.project_dir / "predictions" / f"{exp_id}_fold{fold_id}.npz"
        np.savez(np_path,
                train_true=fold_result.train_true,
                train_pred=fold_result.train_predictions,
                val_true=fold_result.val_true,
                val_pred=fold_result.val_predictions,
                train_indices=fold_result.train_indices,
                val_indices=fold_result.val_indices)
    
    def save_experiment_results(self):
        """Save complete experiment results"""
        if not self.current_experiment:
            return
        
        exp_id = self.current_experiment['experiment_id']
        
        # Aggregate all fold results
        all_val_true = []
        all_val_pred = []
        all_metrics = {key: [] for key in ['rmse', 'mae', 'r2', 'mape']}
        
        for fold_result in self.current_experiment['fold_results']:
            all_val_true.extend(fold_result.val_true)
            all_val_pred.extend(fold_result.val_predictions)
            for key in all_metrics:
                if key in fold_result.metrics:
                    all_metrics[key].append(fold_result.metrics[key])
        
        # Compute summary metrics
        summary_metrics = {
            f"{key}_mean": np.mean(values) if values else 0
            for key, values in all_metrics.items()
        }
        summary_metrics.update({
            f"{key}_std": np.std(values) if values else 0
            for key, values in all_metrics.items()
        })
        
        # Save summary CSV
        summary_df = pd.DataFrame({
            'true': all_val_true,
            'predicted': all_val_pred,
            'error': np.array(all_val_true) - np.array(all_val_pred),
            'absolute_error': np.abs(np.array(all_val_true) - np.array(all_val_pred)),
            'percentage_error': np.abs((np.array(all_val_true) - np.array(all_val_pred)) / 
                                     np.array(all_val_true)) * 100
        })
        
        csv_path = self.project_dir / "exports" / "csv" / f"{exp_id}_all_predictions.csv"
        summary_df.to_csv(csv_path, index=False)
        
        cfg = self.current_experiment.get('config', {}) if self.current_experiment else {}
        exp_formats = []
        if isinstance(cfg, dict):
            exp_conf_export = cfg.get('export')
            if isinstance(exp_conf_export, dict):
                exp_formats = exp_conf_export.get('formats', [])
        if 'excel' in exp_formats:
            excel_path = self.project_dir / "exports" / "excel" / f"{exp_id}_results.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='All_Predictions', index=False)
                fold_metrics_df = pd.DataFrame([
                    {
                        'fold': fold.fold_id,
                        **fold.metrics
                    }
                    for fold in self.current_experiment['fold_results']
                ])
                fold_metrics_df.to_excel(writer, sheet_name='Fold_Metrics', index=False)
                summary_metrics_df = pd.DataFrame([summary_metrics])
                summary_metrics_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Save complete JSON
        json_path = self.project_dir / "exports" / "json" / f"{exp_id}_complete.json"
        with open(json_path, 'w') as f:
                json.dump({
                    **{k: v for k, v in self.current_experiment.items()
                       if k not in ['fold_results', 'start_counter']},
                'fold_results': [fold.to_dict() for fold in self.current_experiment['fold_results']],
                'summary_metrics': summary_metrics
            }, f, indent=2, ensure_ascii=True)
        
        # Save pickle (full Python object)
        pickle_path = self.project_dir / "experiments" / f"{exp_id}_complete.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.current_experiment, f)
        
        # Save JSON summary (for automl analyze command)
        json_summary_path = self.project_dir / "exports" / f"{exp_id}_summary.json"
        with open(json_summary_path, 'w') as f:
            json.dump({
                'experiment_id': exp_id,
                'model': self.current_experiment.get('model_type', 'unknown'),
                'target': self.current_experiment.get('target', 'unknown'),
                    'feature_type': self.current_experiment.get('feature_type', 'unknown'),
                    'n_folds': self.current_experiment.get('n_folds', 0),
                    'run_sequence': self.current_experiment.get('run_sequence', 0),
                'mean_rmse': summary_metrics.get('rmse_mean', 0),
                'std_rmse': summary_metrics.get('rmse_std', 0),
                'mean_mae': summary_metrics.get('mae_mean', 0),
                'std_mae': summary_metrics.get('mae_std', 0),
                'mean_r2': summary_metrics.get('r2_mean', 0),
                'std_r2': summary_metrics.get('r2_std', 0),
                'total_duration': self.current_experiment.get('duration_seconds', 0),
                'hyperparameters': self.current_experiment.get('hyperparameters', {})
            }, f, indent=2, ensure_ascii=True)
    
    # ========================================
    #           Visualization
    # ========================================
    
    def plot_fold_results(self, fold_result: FoldResult):
        """Plot single fold results"""
        if not self.current_experiment:
            return
        
        exp_id = self.current_experiment['experiment_id']
        fold_id = fold_result.fold_id
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Scatter: True vs Predicted
        ax = axes[0, 0]
        ax.scatter(fold_result.val_true, fold_result.val_predictions, alpha=0.5)
        ax.plot([fold_result.val_true.min(), fold_result.val_true.max()],
                [fold_result.val_true.min(), fold_result.val_true.max()],
                'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'Fold {fold_id}: True vs Predicted')
        ax.text(0.05, 0.95, f"R^2 = {fold_result.metrics.get('r2', 0):.4f}",
                transform=ax.transAxes, va='top')
        
        # 2. Residual plot
        ax = axes[0, 1]
        residuals = fold_result.val_true - fold_result.val_predictions
        ax.scatter(fold_result.val_predictions, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predictions')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Fold {fold_id}: Residual Plot')
        
        # 3. Error distribution
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, edgecolor='black')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Fold {fold_id}: Error Distribution')
        ax.axvline(x=0, color='r', linestyle='--')
        
        # 4. Q-Q plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Fold {fold_id}: Q-Q Plot')
        
        plt.suptitle(f'Experiment: {exp_id}', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.project_dir / "plots" / "fold_results" / f"{exp_id}_fold{fold_id}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_experiment_summary(self):
        """Plot experiment summary charts"""
        if not self.current_experiment or not self.current_experiment['fold_results']:
            return
        
        exp_id = self.current_experiment['experiment_id']
        
        # Collect all data
        all_val_true = []
        all_val_pred = []
        fold_metrics = []
        
        for fold_result in self.current_experiment['fold_results']:
            all_val_true.extend(fold_result.val_true)
            all_val_pred.extend(fold_result.val_predictions)
            fold_metrics.append({
                'fold': fold_result.fold_id,
                **fold_result.metrics
            })
        
        all_val_true = np.array(all_val_true)
        all_val_pred = np.array(all_val_pred)
        
        # Create summary plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Overall scatter
        ax = axes[0, 0]
        ax.scatter(all_val_true, all_val_pred, alpha=0.3)
        ax.plot([all_val_true.min(), all_val_true.max()],
                [all_val_true.min(), all_val_true.max()],
                'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title('All Folds: True vs Predicted')
        
        # 2. RMSE per fold
        ax = axes[0, 1]
        fold_df = pd.DataFrame(fold_metrics)
        ax.bar(fold_df['fold'], fold_df['rmse'])
        ax.set_xlabel('Fold')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE by Fold')
        ax.axhline(y=fold_df['rmse'].mean(), color='r', linestyle='--', 
                   label=f"Mean: {fold_df['rmse'].mean():.4f}")
        ax.legend()
        
        # 3. R^2 per fold
        ax = axes[0, 2]
        ax.bar(fold_df['fold'], fold_df['r2'])
        ax.set_xlabel('Fold')
        ax.set_ylabel('R^2')
        ax.set_title('R^2 by Fold')
        ax.axhline(y=fold_df['r2'].mean(), color='r', linestyle='--',
                   label=f"Mean: {fold_df['r2'].mean():.4f}")
        ax.legend()
        
        # 4. Error box plot
        ax = axes[1, 0]
        errors_by_fold = []
        for fold_result in self.current_experiment['fold_results']:
            errors = fold_result.val_true - fold_result.val_predictions
            errors_by_fold.append(errors)
        ax.boxplot(errors_by_fold, labels=range(1, len(errors_by_fold)+1))
        ax.set_xlabel('Fold')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error Distribution by Fold')
        ax.axhline(y=0, color='r', linestyle='--')
        
        # 5. Learning curve (if available)
        ax = axes[1, 1]
        if fold_df.shape[0] > 1:
            ax.plot(fold_df['fold'], fold_df['rmse'], 'o-', label='Validation RMSE')
            ax.set_xlabel('Fold')
            ax.set_ylabel('RMSE')
            ax.set_title('Cross-Validation Performance')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Not enough folds for learning curve',
                   ha='center', va='center', transform=ax.transAxes)
        
        # 6. Metrics summary
        ax = axes[1, 2]
        ax.axis('off')
        metrics_text = f"""
        Summary Metrics:
        
        RMSE: {fold_df['rmse'].mean():.4f} +/- {fold_df['rmse'].std():.4f}
        MAE:  {fold_df['mae'].mean():.4f} +/- {fold_df['mae'].std():.4f}
        R^2:  {fold_df['r2'].mean():.4f} +/- {fold_df['r2'].std():.4f}
        
        Model: {self.current_experiment['model_type']}
        Target: {self.current_experiment['target']}
        Feature: {self.current_experiment['feature_type']}
        Folds: {self.current_experiment['n_folds']}
        """
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
                fontsize=10, va='top', family='monospace')
        
        plt.suptitle(f'Experiment Summary: {exp_id}', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.project_dir / "plots" / f"{exp_id}_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ========================================
    #           Report Generation
    # ========================================
    
    def generate_experiment_report(self):
        """Generate experiment HTML report"""
        if not self.current_experiment:
            return
        
        exp_id = self.current_experiment['experiment_id']
        
        # Generate summary plots
        self.plot_experiment_summary()
        
        # Collect data
        fold_metrics = pd.DataFrame([
            {
                'fold': fold.fold_id,
                **fold.metrics
            }
            for fold in self.current_experiment['fold_results']
        ])
        
        # Generate table rows
        table_rows = []
        for _, row in fold_metrics.iterrows():
            table_rows.append(f"""
                <tr>
                    <td>{row['fold']}</td>
                    <td>{row['rmse']:.4f}</td>
                    <td>{row['mae']:.4f}</td>
                    <td>{row['r2']:.4f}</td>
                    <td>{row.get('mape', 0):.2f}</td>
                </tr>
            """)
        table_rows_html = ''.join(table_rows)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report: {exp_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-box {{ 
                    display: inline-block; 
                    padding: 10px; 
                    margin: 5px;
                    background: #f0f0f0; 
                    border-radius: 5px;
                }}
                .plot {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Training Report: {exp_id}</h1>
            
            <h2>Experiment Configuration</h2>
            <div class="metric-box">
                <strong>Model:</strong> {self.current_experiment['model_type']}<br>
                <strong>Target:</strong> {self.current_experiment['target']}<br>
                <strong>Feature Type:</strong> {self.current_experiment['feature_type']}<br>
                <strong>Cross-Validation:</strong> {self.current_experiment['n_folds']} folds<br>
                <strong>Duration:</strong> {self.current_experiment.get('duration_seconds', 0):.2f} seconds
            </div>
            
            <h2>Performance Summary</h2>
            <div class="metric-box">
                <strong>RMSE:</strong> {fold_metrics['rmse'].mean():.4f} +/- {fold_metrics['rmse'].std():.4f}<br>
                <strong>MAE:</strong> {fold_metrics['mae'].mean():.4f} +/- {fold_metrics['mae'].std():.4f}<br>
                <strong>R^2:</strong> {fold_metrics['r2'].mean():.4f} +/- {fold_metrics['r2'].std():.4f}
            </div>
            
            <h2>Fold-by-Fold Results</h2>
            <table>
                <tr>
                    <th>Fold</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R^2</th>
                    <th>MAPE (%)</th>
                </tr>
                {table_rows_html}
            </table>
            
            <h2>Visualizations</h2>
            <div class="plot">
                <img src="../plots/{exp_id}_summary.png" alt="Summary Plot">
            </div>
            
            <h2>Hyperparameters</h2>
            <pre>{json.dumps(self.current_experiment.get('hyperparameters', {}), indent=2)}</pre>
            
            <hr>
                <p><small>Run sequence: {self.current_experiment.get('run_sequence', 0)}</small></p>
        </body>
        </html>
        """
        
        report_path = self.project_dir / "exports" / f"{exp_id}_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"   Report generated: {report_path}")
    
    # ========================================
    #           Comparison Utilities
    # ========================================
    
    def compare_experiments(self, experiment_ids: List[str] = None):
        """
        Compare multiple experiments
        
        Args:
            experiment_ids: List of experiment IDs to compare; None for all
        """
        # Load all experiments
        experiments = []
        exp_dir = self.project_dir / "experiments"
        
        for pkl_file in exp_dir.glob("*_complete.pkl"):
            with open(pkl_file, 'rb') as f:
                exp = pickle.load(f)
                if experiment_ids is None or exp['experiment_id'] in experiment_ids:
                    experiments.append(exp)
        
        if not experiments:
            print("No experiment data found")
            return
        
        # Create comparison table
        comparison_data = []
        for exp in experiments:
            fold_results = exp['fold_results']
            metrics = pd.DataFrame([fold.metrics for fold in fold_results])
            
            comparison_data.append({
                'experiment_id': exp['experiment_id'],
                'model': exp['model_type'],
                'target': exp['target'],
                'feature': exp['feature_type'],
                'rmse_mean': metrics['rmse'].mean(),
                'rmse_std': metrics['rmse'].std(),
                'r2_mean': metrics['r2'].mean(),
                'r2_std': metrics['r2'].std(),
                'duration': exp.get('duration_seconds', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison results
        comparison_path = self.project_dir / "exports" / "experiment_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # Generate comparison plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE comparison
        ax = axes[0]
        x = range(len(comparison_df))
        ax.bar(x, comparison_df['rmse_mean'], yerr=comparison_df['rmse_std'], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45)
        ax.set_ylabel('RMSE')
        ax.set_title('Model Comparison: RMSE')
        
        # R^2 comparison
        ax = axes[1]
        ax.bar(x, comparison_df['r2_mean'], yerr=comparison_df['r2_std'], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45)
        ax.set_ylabel('R^2')
        ax.set_title('Model Comparison: R^2')
        
        plt.tight_layout()
        comparison_plot_path = self.project_dir / "plots" / "comparison" / "model_comparison.png"
        plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Comparison results saved: {comparison_path}")
        
        return comparison_df
    
    # ========================================
    #           Utilities
    # ========================================
    
    def get_best_model(self, metric: str = 'rmse', ascending: bool = True):
        """
        Get best model
        
        Args:
            metric: Evaluation metric
            ascending: Sort ascending (True means smaller is better)
        
        Returns:
            Best experiment info
        """
        experiments = []
        exp_dir = self.project_dir / "experiments"
        
        for pkl_file in exp_dir.glob("*_complete.pkl"):
            with open(pkl_file, 'rb') as f:
                exp = pickle.load(f)
                fold_results = exp['fold_results']
                metrics = pd.DataFrame([fold.metrics for fold in fold_results])
                exp['mean_' + metric] = metrics[metric].mean()
                experiments.append(exp)
        
        if not experiments:
            return None
        
        # Sort
        experiments.sort(key=lambda x: x['mean_' + metric], reverse=not ascending)
        best_exp = experiments[0]
        
        print(f"Best model ({metric}):")
        print(f"   Experiment ID: {best_exp['experiment_id']}")
        print(f"   Model: {best_exp['model_type']}")
        print(f"   {metric}: {best_exp['mean_' + metric]:.4f}")
        
        return best_exp
    
    def export_for_publication(self, experiment_id: str, output_dir: str = None):
        """
        Export data for publication-oriented reporting
        
        Args:
            experiment_id: Experiment ID
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = self.project_dir / "exports" / "publication_ready"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load experiment
        pkl_path = self.project_dir / "experiments" / f"{experiment_id}_complete.pkl"
        with open(pkl_path, 'rb') as f:
            exp = pickle.load(f)
        
        # Export prediction data
        all_true = []
        all_pred = []
        for fold in exp['fold_results']:
            all_true.extend(fold.val_true)
            all_pred.extend(fold.val_predictions)
        
        pred_df = pd.DataFrame({
            'true_value': all_true,
            'predicted_value': all_pred
        })
        pred_df.to_csv(output_dir / f"{experiment_id}_predictions.csv", index=False)
        
        # Export metrics
        metrics_df = pd.DataFrame([fold.metrics for fold in exp['fold_results']])
        metrics_df.to_csv(output_dir / f"{experiment_id}_fold_metrics.csv", index=False)
        
        # Copy plots
        plot_src = self.project_dir / "plots" / f"{experiment_id}_summary.png"
        if plot_src.exists():
            shutil.copy(plot_src, output_dir / f"{experiment_id}_summary.png")
        
        print(f"   Publication-ready data exported: {output_dir}")
        
        return output_dir


# ========================================
#           Convenience Functions
# ========================================

def create_logger(project_name: str, **kwargs) -> TrainingLogger:
    """Convenience function to create TrainingLogger"""
    return TrainingLogger(project_name, **kwargs)


def load_experiment(experiment_path: Union[str, Path]) -> Dict:
    """Load experiment results"""
    with open(experiment_path, 'rb') as f:
        return pickle.load(f)


def plot_publication_figure(true_values: np.ndarray,
                            predicted_values: np.ndarray,
                            title: str = None,
                            save_path: str = None):
    """
    Generate publication-quality plot
    
    Args:
        true_values: True values
        predicted_values: Predicted values
        title: Plot title
        save_path: Save path
    """
    # Set compact publication style
    plt.style.use('seaborn-v0_8-paper')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Scatter
    ax.scatter(true_values, predicted_values, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    
    # Diagonal
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Compute R^2
    from sklearn.metrics import r2_score
    r2 = r2_score(true_values, predicted_values)
    
    # Labels
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add R^2 text
    ax.text(0.05, 0.95, f'R^2 = {r2:.4f}', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig
