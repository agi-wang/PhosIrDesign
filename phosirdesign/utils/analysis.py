#!/usr/bin/env python3
"""
Analysis module for AutoML system
Provides functionality to analyze and compare experiment results
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ResultsAnalyzer:
    """Analyze and compare AutoML experiment results"""
    
    def __init__(self, run_dir: Path):
        """
        Initialize analyzer with a run directory
        
        Args:
            run_dir: Path to the run directory containing experiments
        """
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise ValueError(f"Run directory does not exist: {run_dir}")
    
    def extract_metrics_from_html(self, html_file: Path) -> Dict:
        """Extract metrics from HTML report"""
        with open(html_file, 'r') as f:
            content = f.read()
        
        # Extract model info
        model_match = re.search(r'<strong>Model:</strong>\s*(\w+)', content)
        target_match = re.search(r'<strong>Target:</strong>\s*([^<]+)', content)
        
        # Extract performance metrics
        rmse_match = re.search(r'<strong>RMSE:</strong>\s*([\d.]+)\s*(?:\+/-|\u00B1)\s*([\d.]+)', content)
        mae_match = re.search(r'<strong>MAE:</strong>\s*([\d.]+)\s*(?:\+/-|\u00B1)\s*([\d.]+)', content)
        r2_match = re.search(r'<strong>R(?:\^2|\u00B2):</strong>\s*([\d.]+)\s*(?:\+/-|\u00B1)\s*([\d.]+)', content)
        
        # Extract duration
        duration_match = re.search(r'<strong>Duration:</strong>\s*([\d.]+)\s*seconds', content)
        
        # Extract fold count
        fold_match = re.search(r'<strong>Cross-Validation:</strong>\s*(\d+)\s*folds', content)
        
        result = {
            'model': model_match.group(1) if model_match else 'unknown',
            'target': target_match.group(1) if target_match else 'unknown',
            'mean_rmse': float(rmse_match.group(1)) if rmse_match else 0,
            'std_rmse': float(rmse_match.group(2)) if rmse_match else 0,
            'mean_mae': float(mae_match.group(1)) if mae_match else 0,
            'std_mae': float(mae_match.group(2)) if mae_match else 0,
            'mean_r2': float(r2_match.group(1)) if r2_match else 0,
            'std_r2': float(r2_match.group(2)) if r2_match else 0,
            'duration': float(duration_match.group(1)) if duration_match else 0,
            'n_folds': int(fold_match.group(1)) if fold_match else 0
        }
        
        return result
    
    def collect_automl_results(self) -> pd.DataFrame:
        """Collect AutoML results (hyperparameter search removed)."""
        # Optuna search outputs are no longer produced; return empty DataFrame.
        return pd.DataFrame()
    
    def collect_all_results(self) -> pd.DataFrame:
        """Collect all results from HTML reports and JSON summaries in the run directory"""
        # Try AutoML results first
        df = self.collect_automl_results()
        if not df.empty:
            return df
        
        # Fallback to original method
        results = []
        
        # First try to find JSON summaries (preferred)
        json_files = list(self.run_dir.rglob("*_summary.json"))
        
        if json_files:
            # Use JSON summaries if available
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Convert to consistent format
                    result = {
                        'model': data.get('model', 'unknown'),
                        'target': data.get('target', 'unknown'),
                        'mean_rmse': data.get('mean_rmse', 0),
                        'std_rmse': data.get('std_rmse', 0),
                        'mean_mae': data.get('mean_mae', 0),
                        'std_mae': data.get('std_mae', 0),
                        'mean_r2': data.get('mean_r2', 0),
                        'std_r2': data.get('std_r2', 0),
                        'mean_mape': data.get('mean_mape', 0),
                        'std_mape': data.get('std_mape', 0),
                        'duration': data.get('total_duration', 0),
                        'n_folds': data.get('n_folds', 0),
                        'report_path': str(json_file.relative_to(self.run_dir))
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Error processing {json_file}: {e}")
        
        # Fall back to HTML reports if no JSON summaries
        if not results:
            html_files = list(self.run_dir.rglob("*_report.html"))
            
            for html_file in html_files:
                # Skip analysis reports
                if 'analysis_report' in html_file.name:
                    continue
                    
                try:
                    metrics = self.extract_metrics_from_html(html_file)
                    metrics['report_path'] = str(html_file.relative_to(self.run_dir))
                    results.append(metrics)
                except Exception as e:
                    print(f"Warning: Error processing {html_file}: {e}")
        
        return pd.DataFrame(results)
    
    def get_best_models(self, df: pd.DataFrame) -> Dict:
        """Find the best model for each target based on R^2 score"""
        best_models = {}
        
        for target in df['target'].unique():
            target_df = df[df['target'] == target]
            best_idx = target_df['mean_r2'].idxmax()
            best_row = target_df.loc[best_idx]
            
            best_models[target] = {
                'model': best_row['model'],
                'r2': best_row['mean_r2'],
                'r2_std': best_row['std_r2'],
                'rmse': best_row['mean_rmse'],
                'rmse_std': best_row['std_rmse'],
                'mae': best_row['mean_mae'],
                'mae_std': best_row['std_mae'],
                'duration': best_row.get('duration', 0)
            }
        
        return best_models
    
    def generate_report(self, output_format: str = 'text') -> str:
        """
        Generate analysis report
        
        Args:
            output_format: Format of the report ('text', 'json', 'html')
        
        Returns:
            Report string in the specified format
        """
        df = self.collect_all_results()
        
        if df.empty:
            return "No results found in the specified directory."
        
        if output_format == 'json':
            return self._generate_json_report(df)
        elif output_format == 'html':
            return self._generate_html_report(df)
        else:
            return self._generate_text_report(df)
    
    def _generate_text_report(self, df: pd.DataFrame) -> str:
        """Generate text format report"""
        lines = []
        
        # Header
        lines.append("=" * 100)
        lines.append(f"AUTOML ANALYSIS REPORT - {self.run_dir.name}")
        lines.append("=" * 100)
        lines.append(f"Run directory: {self.run_dir.name}")
        lines.append("")
        
        # Summary statistics
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 100)
        lines.append(f"Total experiments: {len(df)}")
        if 'duration' in df.columns:
            lines.append(f"Total training time: {df['duration'].sum():.2f} seconds")
        lines.append(f"Models tested: {', '.join(sorted(df['model'].unique()))}")
        lines.append(f"Targets tested: {', '.join(sorted(df['target'].unique()))}")
        if 'n_trials' in df.columns and df['n_trials'].sum() > 0:
            lines.append(f"Total optimization trials: {df['n_trials'].sum()}")
        lines.append("")
        
        # Performance by target
        targets = sorted(df['target'].unique())
        for target in targets:
            target_df = df[df['target'] == target].copy()
            target_df = target_df.sort_values('mean_r2', ascending=False)
            
            lines.append(f"TARGET: {target}")
            lines.append("-" * 100)
            
            # Check if MAPE data is available
            has_mape = 'mean_mape' in df.columns and df['mean_mape'].sum() > 0
            
            if has_mape:
                lines.append(f"{'Model':<15} {'R^2 (+/-std)':<18} {'RMSE (+/-std)':<18} {'MAE (+/-std)':<18} {'MAPE (+/-std)':<18} {'Folds':<6}")
            else:
                lines.append(f"{'Model':<15} {'R^2 (+/-std)':<18} {'RMSE (+/-std)':<18} {'MAE (+/-std)':<18} {'Folds':<6}")
            lines.append("-" * 100)
            
            for idx, row in target_df.iterrows():
                # Mark best model
                model_name = row['model']
                if idx == target_df.index[0]:
                    model_name = f"{model_name} [BEST]"
                
                r2_str = f"{row['mean_r2']:.4f}+/-{row['std_r2']:.4f}" if row['std_r2'] > 0 else f"{row['mean_r2']:.4f}"
                rmse_str = f"{row['mean_rmse']:.3f}+/-{row['std_rmse']:.3f}" if row['std_rmse'] > 0 else f"{row['mean_rmse']:.3f}"
                mae_str = f"{row['mean_mae']:.3f}+/-{row['std_mae']:.3f}" if row['std_mae'] > 0 else f"{row['mean_mae']:.3f}"
                
                if has_mape and 'mean_mape' in row and row['mean_mape'] > 0:
                    mape_str = f"{row['mean_mape']:.2f}%+/-{row.get('std_mape', 0):.2f}%" if row.get('std_mape', 0) > 0 else f"{row['mean_mape']:.2f}%"
                    lines.append(f"{model_name:<15} {r2_str:<18} {rmse_str:<18} {mae_str:<18} {mape_str:<18} {row.get('n_folds', 10):<6}")
                else:
                    lines.append(f"{model_name:<15} {r2_str:<18} {rmse_str:<18} {mae_str:<18} {row.get('n_folds', 10):<6}")
            
            # Best model summary
            best = target_df.iloc[0]
            lines.append(f"\nBest model: {best['model']} (R^2={best['mean_r2']:.4f}+/-{best['std_r2']:.4f}, RMSE={best['mean_rmse']:.3f}+/-{best['std_rmse']:.3f})")
            
            # Add CV stability analysis
            if best['std_r2'] > 0:
                cv_coefficient = (best['std_r2'] / best['mean_r2']) * 100 if best['mean_r2'] > 0 else 0
                lines.append(f"   CV Stability: {cv_coefficient:.1f}% variation (lower is better)")
            
            lines.append("")
        
        # Model rankings
        lines.append("MODEL RANKINGS (Average R^2 across all targets)")
        lines.append("-" * 100)
        
        model_avg = df.groupby('model')['mean_r2'].mean().sort_values(ascending=False)
        for rank, (model, avg_r2) in enumerate(model_avg.items(), 1):
            lines.append(f"{rank}. {model:<15} {avg_r2:.4f}")
        
        lines.append("")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def _generate_json_report(self, df: pd.DataFrame) -> str:
        """Generate JSON format report"""
        best_models = self.get_best_models(df)
        
        report = {
            'run_directory': str(self.run_dir),
            'summary': {
                'total_experiments': len(df),
                'total_duration': df['duration'].sum(),
                'models': sorted(df['model'].unique().tolist()),
                'targets': sorted(df['target'].unique().tolist())
            },
            'best_models': best_models,
            'all_results': df.to_dict('records'),
            'model_rankings': df.groupby('model')['mean_r2'].mean().sort_values(ascending=False).to_dict()
        }
        
        return json.dumps(report, indent=2)
    
    def _generate_html_report(self, df: pd.DataFrame) -> str:
        """Generate HTML format report with enhanced CV statistics"""
        best_models = self.get_best_models(df)
        
        # Check if we have MAPE data
        has_mape = 'mean_mape' in df.columns and df['mean_mape'].sum() > 0
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AutoML Analysis Report - {self.run_dir.name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
        th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #e8f4f8; transition: background 0.3s; }}
        tr.best-row {{ background: #d4f1d4 !important; font-weight: bold; }}
        .metric-card {{ display: inline-block; padding: 15px; margin: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
        .best-model {{ background: #27ae60; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .run-meta {{ color: #7f8c8d; font-size: 0.9em; }}
        .star {{ color: gold; font-size: 1.2em; }}
        .metric-value {{ font-family: 'Courier New', monospace; }}
        .std-value {{ color: #7f8c8d; font-size: 0.9em; }}
        .cv-stability {{ background: #f39c12; color: white; padding: 5px 10px; border-radius: 3px; margin-left: 10px; font-size: 0.9em; }}
        .tooltip {{ position: relative; display: inline-block; cursor: help; }}
        .tooltip .tooltiptext {{ visibility: hidden; width: 200px; background-color: #555; color: #fff; text-align: center; border-radius: 6px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s; }}
        .tooltip:hover .tooltiptext {{ visibility: visible; opacity: 1; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AutoML Analysis Report</h1>
        <p class="run-meta">Directory: {self.run_dir}</p>
        
        <h2>Summary Statistics</h2>
        <div>
            <div class="metric-card">
                <strong>Total Experiments:</strong> {len(df)}
            </div>
            <div class="metric-card">
                <strong>Models Tested:</strong> {len(df['model'].unique())}
            </div>
            <div class="metric-card">
                <strong>Targets:</strong> {len(df['target'].unique())}
            </div>
            <div class="metric-card">
                <strong>CV Folds:</strong> {df['n_folds'].max() if 'n_folds' in df.columns else 'N/A'}
            </div>
        </div>
"""
        
        # Add results by target
        for target in sorted(df['target'].unique()):
            target_df = df[df['target'] == target].copy()
            target_df = target_df.sort_values('mean_r2', ascending=False)
            
            html += f"""
        <h2>Target: {target}</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>R^2 Score (+/-std)</th>
                <th>RMSE (+/-std)</th>
                <th>MAE (+/-std)</th>"""
            
            if has_mape:
                html += "<th>MAPE (+/-std)</th>"
            
            html += """<th>CV Folds</th>
                <th>Trials</th>
            </tr>
"""
            for rank, (idx, row) in enumerate(target_df.iterrows(), 1):
                row_class = 'class="best-row"' if rank == 1 else ''
                star = '[BEST]' if rank == 1 else ''
                
                r2_str = f"<span class='metric-value'>{row['mean_r2']:.4f}</span> <span class='std-value'>+/-{row['std_r2']:.4f}</span>" if row['std_r2'] > 0 else f"{row['mean_r2']:.4f}"
                rmse_str = f"<span class='metric-value'>{row['mean_rmse']:.3f}</span> <span class='std-value'>+/-{row['std_rmse']:.3f}</span>" if row['std_rmse'] > 0 else f"{row['mean_rmse']:.3f}"
                mae_str = f"<span class='metric-value'>{row['mean_mae']:.3f}</span> <span class='std-value'>+/-{row['std_mae']:.3f}</span>" if row['std_mae'] > 0 else f"{row['mean_mae']:.3f}"
                
                html += f"""
            <tr {row_class}>
                <td>{rank}</td>
                <td><strong>{row['model']}</strong> {star}</td>
                <td>{r2_str}</td>
                <td>{rmse_str}</td>
                <td>{mae_str}</td>"""
                
                if has_mape and 'mean_mape' in row:
                    mape_str = f"<span class='metric-value'>{row['mean_mape']:.2f}%</span> <span class='std-value'>+/-{row.get('std_mape', 0):.2f}%</span>" if row.get('std_mape', 0) > 0 else f"{row.get('mean_mape', 0):.2f}%"
                    html += f"<td>{mape_str}</td>"
                
                html += f"""
                <td>{row.get('n_folds', 'N/A')}</td>
                <td>{row.get('n_trials', 'N/A')}</td>
            </tr>
"""
            html += "</table>"
            
            best = best_models[target]
            cv_coefficient = (best['r2_std'] / best['r2']) * 100 if best['r2'] > 0 and best.get('r2_std', 0) > 0 else 0
            
            html += f"""
        <div class="best-model">
            Best Model: <strong>{best['model']}</strong><br>
            - R^2: {best['r2']:.4f} +/- {best.get('r2_std', 0):.4f}<br>
            - RMSE: {best['rmse']:.3f} +/- {best.get('rmse_std', 0):.3f}<br>
            - MAE: {best['mae']:.3f} +/- {best.get('mae_std', 0):.3f}
            {f'<span class="cv-stability">CV Stability: {cv_coefficient:.1f}%</span>' if cv_coefficient > 0 else ''}
        </div>
"""
        
        # Add model rankings
        model_avg = df.groupby('model')['mean_r2'].mean().sort_values(ascending=False)
        html += """
        <h2>Overall Model Rankings</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Average R^2 Score</th>
            </tr>
"""
        for rank, (model, avg_r2) in enumerate(model_avg.items(), 1):
            html += f"""
            <tr>
                <td>{rank}</td>
                <td><strong>{model}</strong></td>
                <td>{avg_r2:.4f}</td>
            </tr>
"""
        html += """
        </table>
    </div>
</body>
</html>
"""
        return html
    
    def save_report(self, output_path: Optional[Path] = None, output_format: str = 'text'):
        """
        Save analysis report to file
        
        Args:
            output_path: Path to save the report (default: run_dir/analysis_report.{ext})
            output_format: Format of the report ('text', 'json', 'html', 'csv')
        """
        if output_format == 'csv':
            # Direct CSV export with all CV statistics
            df = self.collect_all_results()
            if df.empty:
                print("No results found to export.")
                return
            
            if output_path is None:
                output_path = self.run_dir / "analysis_results.csv"
            
            # Ensure all columns are included
            columns_order = ['model', 'target', 'mean_r2', 'std_r2', 'mean_rmse', 'std_rmse', 
                           'mean_mae', 'std_mae', 'mean_mape', 'std_mape', 'n_folds', 'n_trials']
            
            # Add missing columns with default values
            for col in columns_order:
                if col not in df.columns:
                    df[col] = 0 if 'mean' in col or 'std' in col else 'N/A'
            
            # Sort by target and R2 score
            df = df.sort_values(['target', 'mean_r2'], ascending=[True, False])
            
            # Add best model indicator
            df['is_best'] = False
            for target in df['target'].unique():
                target_mask = df['target'] == target
                best_idx = df[target_mask]['mean_r2'].idxmax()
                df.loc[best_idx, 'is_best'] = True
            
            # Save to CSV
            df[columns_order + ['is_best']].to_csv(output_path, index=False)
            print(f"CSV results with CV statistics saved to: {output_path}")
            
            # Print summary
            print(f"\nSummary:")
            for target in df['target'].unique():
                best_model = df[(df['target'] == target) & df['is_best']].iloc[0]
                print(f"  {target}: {best_model['model']} (R^2={best_model['mean_r2']:.4f}+/-{best_model['std_r2']:.4f})")
            
            return
        
        # Generate report for other formats
        report = self.generate_report(output_format)
        
        if output_path is None:
            ext = {'text': 'txt', 'json': 'json', 'html': 'html'}.get(output_format, 'txt')
            output_path = self.run_dir / f"analysis_report.{ext}"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {output_path}")
        
        # Also save CSV for further analysis
        df = self.collect_all_results()
        if not df.empty:
            csv_path = self.run_dir / "analysis_results.csv"
            
            # Same CSV export logic as above
            columns_order = ['model', 'target', 'mean_r2', 'std_r2', 'mean_rmse', 'std_rmse', 
                           'mean_mae', 'std_mae', 'mean_mape', 'std_mape', 'n_folds', 'n_trials']
            
            for col in columns_order:
                if col not in df.columns:
                    df[col] = 0 if 'mean' in col or 'std' in col else 'N/A'
            
            df = df.sort_values(['target', 'mean_r2'], ascending=[True, False])
            
            df['is_best'] = False
            for target in df['target'].unique():
                target_mask = df['target'] == target
                best_idx = df[target_mask]['mean_r2'].idxmax()
                df.loc[best_idx, 'is_best'] = True
            
            df[columns_order + ['is_best']].to_csv(csv_path, index=False)
            print(f"CSV data with CV statistics saved to: {csv_path}")


def analyze_command(args):
    """
    Command handler for automl analyze
    
    Args:
        args: Parsed command line arguments
    """
    analyzer = ResultsAnalyzer(args.run_dir)
    
    # Generate and save report
    analyzer.save_report(output_format=args.format)
    
    # Also print to console if requested
    if args.print:
        print(analyzer.generate_report('text'))
