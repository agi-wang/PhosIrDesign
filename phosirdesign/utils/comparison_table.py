#!/usr/bin/env python3
"""
Model performance comparison table generator
Generates compact model-comparison tables
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ComparisonTableGenerator:
    """Model comparison table generator"""
    
    def __init__(self, results_dir: str, highlight_best: bool = True):
        """
        Initialize table generator
        
        Args:
            results_dir: Directory containing all model results
            highlight_best: Whether to highlight the best model
        """
        self.results_dir = Path(results_dir)
        self.highlight_best = highlight_best
        self.results = []
        self.models_data = {}
        
    def collect_all_results(self) -> pd.DataFrame:
        """
        Collect training results for all models
        
        Returns:
            DataFrame containing all results
        """
        results = []
        
        # Traverse subdirectories to find result files
        for json_file in self.results_dir.rglob('*_summary.json'):
            # Skip prediction summary files; process training results only
            if json_file.name == 'prediction_summary.json':
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract model info
                model_type = data.get('model', data.get('model_type', 'unknown'))
                target = data.get('target', 'unknown')
                
                # Skip invalid data
                if model_type == 'unknown' or target == 'unknown':
                    continue
                
                # Format model name
                model_name = self._format_model_name(model_type)
                
                # Format target name
                target_display = self._format_target_name(target)
                
                # Extract CV results - support two formats
                if 'cv_results' in data:
                    cv_results = data['cv_results']
                    r2_mean = cv_results.get('r2_mean', 0)
                    r2_std = cv_results.get('r2_std', 0)
                    rmse_mean = cv_results.get('rmse_mean', 0)
                    rmse_std = cv_results.get('rmse_std', 0)
                    mae_mean = cv_results.get('mae_mean', 0)
                    mae_std = cv_results.get('mae_std', 0)
                else:
                    # Extract directly from root level
                    r2_mean = data.get('mean_r2', 0)
                    r2_std = data.get('std_r2', 0)
                    rmse_mean = data.get('mean_rmse', 0)
                    rmse_std = data.get('std_rmse', 0)
                    mae_mean = data.get('mean_mae', 0)
                    mae_std = data.get('std_mae', 0)
                
                # Collect results
                result = {
                    'Object': target_display,
                    'Algorithm': model_name,
                    'R2_mean': r2_mean,
                    'R2_std': r2_std,
                    'RMSE_mean': rmse_mean,
                    'RMSE_std': rmse_std,
                    'MAE_mean': mae_mean,
                    'MAE_std': mae_std,
                    'n_samples': data.get('n_samples', 0),
                    'n_folds': data.get('n_folds', 10)
                }
                
                results.append(result)
                
                # Store detailed data
                key = f"{model_type}_{target}"
                self.models_data[key] = data
                
            except Exception as e:
                print(f"WARNING Could not read file {json_file}: {e}")
                continue
        
        if not results:
            print("ERROR No model results found")
            return pd.DataFrame()
        
        self.results = results
        return pd.DataFrame(results)
    
    def _format_model_name(self, model_type: str) -> str:
        """Format model name"""
        name_mapping = {
            'xgboost': 'XGBoost Regressor',
            'lightgbm': 'LightGBM Regressor',
            'catboost': 'CatBoost Regressor',
            'random_forest': 'Random Forest Regressor',
            'gradient_boosting': 'Gradient Boosting Regressor',
            'adaboost': 'AdaBoost Regressor',
            'extra_trees': 'Extra Trees Regressor',
            'svr': 'SVR',
            'knn': 'K-Nearest Neighbors Regressor',
            'decision_tree': 'Decision Tree Regressor',
            'ridge': 'Ridge Regressor',
            'lasso': 'Lasso Regressor',
            'elastic_net': 'Elastic Net Regressor',
            'mlp': 'MLP (Neural Network)',
            'mlp_torch_mps': 'MLP Torch/MPS'
        }
        return name_mapping.get(model_type.lower(), model_type)
    
    def _format_target_name(self, target: str) -> str:
        """Format target name"""
        name_mapping = {
            'Max_wavelength(nm)': 'Max Wavelength (nm)',
            'PLQY': 'PLQY',
            'tau(s*10^-6)': 'Tau (s*10^-6)'
        }
        return name_mapping.get(target, target)
    
    def generate_markdown_table(self, df: pd.DataFrame = None, 
                               decimal_places: Dict[str, int] = None) -> str:
        """
        Generate comparison table in Markdown format
        
        Args:
            df: Results DataFrame
            decimal_places: Decimal places configuration
        
        Returns:
            Markdown table string
        """
        if df is None:
            df = self.collect_all_results()
        
        if df.empty:
            return "No results found"
        
        if decimal_places is None:
            decimal_places = {'r2': 4, 'rmse': 4, 'mae': 4}
        
        # Group by target
        markdown = "# Model Performance Comparison\n\n"
        markdown += f"**Models Evaluated**: {df['Algorithm'].nunique()}\n\n"
        
        for target in df['Object'].unique():
            target_df = df[df['Object'] == target].copy()
            
            # Format values
            target_df['R^2'] = target_df.apply(
                lambda x: f"{x['R2_mean']:.{decimal_places['r2']}f}+/-{x['R2_std']:.{decimal_places['r2']}f}", 
                axis=1
            )
            target_df['RMSE'] = target_df.apply(
                lambda x: f"{x['RMSE_mean']:.{decimal_places['rmse']}f}+/-{x['RMSE_std']:.{decimal_places['rmse']}f}", 
                axis=1
            )
            target_df['MAE'] = target_df.apply(
                lambda x: f"{x['MAE_mean']:.{decimal_places['mae']}f}+/-{x['MAE_std']:.{decimal_places['mae']}f}", 
                axis=1
            )
            
            # Find best model
            best_idx = target_df['R2_mean'].idxmax()
            
            markdown += f"## Target: {target}\n\n"
            markdown += "| Algorithm | R^2 | RMSE | MAE |\n"
            markdown += "|-----------|-----|------|-----|\n"
            
            # Sort by R^2 descending
            target_df = target_df.sort_values('R2_mean', ascending=False)
            
            for idx, row in target_df.iterrows():
                if self.highlight_best and idx == best_idx:
                    # Highlight best model
                    markdown += f"| **{row['Algorithm']}** [BEST] | **{row['R^2']}** | **{row['RMSE']}** | **{row['MAE']}** |\n"
                else:
                    markdown += f"| {row['Algorithm']} | {row['R^2']} | {row['RMSE']} | {row['MAE']} |\n"
            
            markdown += "\n"
        
        # Add best model summary
        markdown += "## Best Models Summary\n\n"
        markdown += "| Target | Best Model | R^2 | RMSE |\n"
        markdown += "|--------|------------|-----|------|\n"
        
        for target in df['Object'].unique():
            target_df = df[df['Object'] == target]
            best_idx = target_df['R2_mean'].idxmax()
            best = target_df.loc[best_idx]
            markdown += f"| {target} | {best['Algorithm']} | "
            markdown += f"{best['R2_mean']:.{decimal_places['r2']}f}+/-{best['R2_std']:.{decimal_places['r2']}f} | "
            markdown += f"{best['RMSE_mean']:.{decimal_places['rmse']}f}+/-{best['RMSE_std']:.{decimal_places['rmse']}f} |\n"
        
        return markdown
    
    def generate_latex_table(self, df: pd.DataFrame = None,
                            decimal_places: Dict[str, int] = None) -> str:
        """
        Generate comparison table in LaTeX format
        
        Args:
            df: Results DataFrame
            decimal_places: Decimal places configuration
        
        Returns:
            LaTeX table string
        """
        if df is None:
            df = self.collect_all_results()
        
        if df.empty:
            return "% No results found"
        
        if decimal_places is None:
            decimal_places = {'r2': 4, 'rmse': 4, 'mae': 4}
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Prediction performance with different models}\n"
        latex += "\\begin{tabular}{llrrr}\n"
        latex += "\\toprule\n"
        latex += "Objects & Algorithms & R\\textsuperscript{2} & RMSE & MAE \\\\\n"
        latex += "\\midrule\n"
        
        for target in df['Object'].unique():
            target_df = df[df['Object'] == target].copy()
            
            # Sort by R^2 descending
            target_df = target_df.sort_values('R2_mean', ascending=False)
            
            # Find best model
            best_idx = target_df['R2_mean'].idxmax()
            
            for i, (idx, row) in enumerate(target_df.iterrows()):
                if i == 0:
                    # First row shows target name
                    latex += f"{target} & "
                else:
                    latex += " & "
                
                # Format values
                r2_str = f"{row['R2_mean']:.{decimal_places['r2']}f}$\\pm${row['R2_std']:.{decimal_places['r2']}f}"
                rmse_str = f"{row['RMSE_mean']:.{decimal_places['rmse']}f}$\\pm${row['RMSE_std']:.{decimal_places['rmse']}f}"
                mae_str = f"{row['MAE_mean']:.{decimal_places['mae']}f}$\\pm${row['MAE_std']:.{decimal_places['mae']}f}"
                
                if self.highlight_best and idx == best_idx:
                    # Highlight best model (bold)
                    latex += f"\\textbf{{{row['Algorithm']}}} & \\textbf{{{r2_str}}} & \\textbf{{{rmse_str}}} & \\textbf{{{mae_str}}} \\\n"
                else:
                    latex += f"{row['Algorithm']} & {r2_str} & {rmse_str} & {mae_str} \\\n"
            
            # Add separator
            if target != df['Object'].unique()[-1]:
                latex += "\\midrule\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def generate_html_table(self, df: pd.DataFrame = None,
                           decimal_places: Dict[str, int] = None) -> str:
        """
        Generate comparison table in HTML format
        
        Args:
            df: Results DataFrame
            decimal_places: Decimal places configuration
        
        Returns:
            HTML table string
        """
        if df is None:
            df = self.collect_all_results()
        
        if df.empty:
            return "<p>No results found</p>"
        
        if decimal_places is None:
            decimal_places = {'r2': 4, 'rmse': 4, 'mae': 4}
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                    font-family: Arial, sans-serif;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #4CAF50;
                    color: white;
                }
                tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .best-model {
                    font-weight: bold;
                    color: #d32f2f;
                }
                .target-header {
                    background-color: #e8f5e9;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
        <h2>Model Performance Comparison</h2>
        """
        
        html += "<table>\n"
        html += "<tr><th>Objects</th><th>Algorithms</th><th>R^2</th><th>RMSE</th><th>MAE</th></tr>\n"
        
        for target in df['Object'].unique():
            target_df = df[df['Object'] == target].copy()
            
            # Sort by R^2 descending
            target_df = target_df.sort_values('R2_mean', ascending=False)
            
            # Find best model
            best_idx = target_df['R2_mean'].idxmax()
            
            for i, (idx, row) in enumerate(target_df.iterrows()):
                if self.highlight_best and idx == best_idx:
                    html += '<tr class="best-model">'
                else:
                    html += '<tr>'
                
                if i == 0:
                    html += f'<td rowspan="{len(target_df)}" class="target-header">{target}</td>'
                
                # Format values
                r2_str = f"{row['R2_mean']:.{decimal_places['r2']}f}+/-{row['R2_std']:.{decimal_places['r2']}f}"
                rmse_str = f"{row['RMSE_mean']:.{decimal_places['rmse']}f}+/-{row['RMSE_std']:.{decimal_places['rmse']}f}"
                mae_str = f"{row['MAE_mean']:.{decimal_places['mae']}f}+/-{row['MAE_std']:.{decimal_places['mae']}f}"
                
                html += f"<td>{row['Algorithm']}</td>"
                html += f"<td>{r2_str}</td>"
                html += f"<td>{rmse_str}</td>"
                html += f"<td>{mae_str}</td>"
                html += "</tr>\n"
        
        html += "</table>\n"
        html += "</body></html>"
        
        return html
    
    def generate_csv_table(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate comparison table in CSV format
        
        Args:
            df: Results DataFrame
        
        Returns:
            Formatted DataFrame
        """
        if df is None:
            df = self.collect_all_results()
        
        if df.empty:
            return pd.DataFrame()
        
        # Create output DataFrame
        output_df = pd.DataFrame()
        
        for target in df['Object'].unique():
            target_df = df[df['Object'] == target].copy()
            
            # Sort by R^2 descending
            target_df = target_df.sort_values('R2_mean', ascending=False)
            
            # Add target column
            target_df['Target'] = target
            
            # Reorder columns
            target_df = target_df[['Target', 'Algorithm', 'R2_mean', 'R2_std', 
                                   'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std']]
            
            output_df = pd.concat([output_df, target_df], ignore_index=True)
        
        return output_df
    
    def export_all_formats(self, output_dir: str = None, 
                          formats: List[str] = None,
                          decimal_places: Dict[str, int] = None) -> Dict[str, str]:
        """
        Export tables in multiple formats
        
        Args:
            output_dir: Output directory
            formats: List of formats to export
            decimal_places: Decimal places configuration
        
        Returns:
            Dict of file paths per format
        """
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
        
        if formats is None:
            formats = ['markdown', 'html', 'latex', 'csv']
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect results
        df = self.collect_all_results()
        if df.empty:
            print("ERROR No results found; cannot generate tables")
            return {}
        
        exported_files = {}
        
        # Generate formats
        if 'markdown' in formats:
            md_content = self.generate_markdown_table(df, decimal_places)
            md_file = output_dir / "comparison_table.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            exported_files['markdown'] = str(md_file)
            print(f"INFO Markdown table saved: {md_file}")
        
        if 'html' in formats:
            html_content = self.generate_html_table(df, decimal_places)
            html_file = output_dir / "comparison_table.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            exported_files['html'] = str(html_file)
            print(f"INFO HTML table saved: {html_file}")
        
        if 'latex' in formats:
            latex_content = self.generate_latex_table(df, decimal_places)
            latex_file = output_dir / "comparison_table.tex"
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            exported_files['latex'] = str(latex_file)
            print(f"INFO LaTeX table saved: {latex_file}")
        
        if 'csv' in formats:
            csv_df = self.generate_csv_table(df)
            csv_file = output_dir / "comparison_table.csv"
            csv_df.to_csv(csv_file, index=False)
            exported_files['csv'] = str(csv_file)
            print(f"INFO CSV table saved: {csv_file}")
        
        return exported_files
    
    def get_best_models(self, metric: str = 'r2') -> Dict[str, Dict]:
        """
        Get the best model for each target
        
        Args:
            metric: Metric to select by (r2, rmse, mae)
        
        Returns:
            Dict containing the best model per target
        """
        df = self.collect_all_results()
        if df.empty:
            return {}
        
        best_models = {}
        
        for target in df['Object'].unique():
            target_df = df[df['Object'] == target]
            
            if metric == 'r2':
                best_idx = target_df['R2_mean'].idxmax()
            elif metric == 'rmse':
                best_idx = target_df['RMSE_mean'].idxmin()
            elif metric == 'mae':
                best_idx = target_df['MAE_mean'].idxmin()
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            best = target_df.loc[best_idx]
            
            best_models[target] = {
                'algorithm': best['Algorithm'],
                'r2': f"{best['R2_mean']:.4f}+/-{best['R2_std']:.4f}",
                'rmse': f"{best['RMSE_mean']:.4f}+/-{best['RMSE_std']:.4f}",
                'mae': f"{best['MAE_mean']:.4f}+/-{best['MAE_std']:.4f}",
                'n_samples': best['n_samples']
            }
        
        return best_models


def main():
    """Command-line interface for comparison-table generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate model comparison tables')
    parser.add_argument('results_dir', help='Directory containing model results')
    parser.add_argument('--formats', nargs='+', default=['markdown', 'html', 'latex', 'csv'],
                       help='Formats to generate')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--no-highlight', action='store_true', help='Do not highlight best model')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ComparisonTableGenerator(
        results_dir=args.results_dir,
        highlight_best=not args.no_highlight
    )
    
    # Export all formats
    exported = generator.export_all_formats(
        output_dir=args.output,
        formats=args.formats
    )
    
    # Show best models
    print("\n" + "="*60)
    print("Best Models Summary")
    print("="*60)
    best_models = generator.get_best_models()
    for target, info in best_models.items():
        print(f"\n{target}:")
        print(f"  Best Model: {info['algorithm']}")
        print(f"  R^2: {info['r2']}")
        print(f"  RMSE: {info['rmse']}")
        print(f"  MAE: {info['mae']}")


if __name__ == '__main__':
    main()
