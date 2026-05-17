#!/usr/bin/env python3
"""
ML Pipeline Helper Functions
Consolidated Python utilities for the ML pipeline
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_intersection(args):
    """Create intersection dataset (samples with both wavelength and PLQY)"""
    print(f"Creating intersection data from {args.input}")
    
    # Read data
    df = pd.read_csv(args.input)
    original_count = len(df)
    
    # Filter for samples with both wavelength and PLQY
    mask = df[['Max_wavelength(nm)', 'PLQY']].notna().all(axis=1)
    df_intersection = df[mask].copy()
    
    # Remove tau column if present
    if 'tau(s*10^-6)' in df_intersection.columns:
        df_intersection = df_intersection.drop('tau(s*10^-6)', axis=1)
    
    # Save
    df_intersection.to_csv(args.output, index=False)
    
    intersection_count = len(df_intersection)
    percentage = 100 * intersection_count / original_count
    
    print(f"Intersection data created:")
    print(f"   Original: {original_count} samples")
    print(f"   Intersection: {intersection_count} samples ({percentage:.1f}%)")
    print(f"   Saved to: {args.output}")
    
    return intersection_count


def find_best_model(args):
    """Find the best model based on R^2 scores"""
    output_dir = Path(args.output_dir)
    models = args.models.split()
    
    best_r2 = -1
    best_model_dir = None
    best_model_name = None
    
    # Check all model directories
    for model in models:
        # Check intersection models first (usually better)
        for suffix in ['_intersection', '']:
            model_dir = output_dir / f"{model}{suffix}"
            
            # Find summary files
            summary_files = list(model_dir.rglob('*_summary.json'))
            
            for summary_file in summary_files:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                # Get R^2 for wavelength (primary target)
                if 'Max_wavelength' in str(summary_file):
                    r2 = data.get('mean_r2', 0)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_dir = model_dir
                        best_model_name = f"{model}{suffix}"
    
    if best_model_dir:
        print(f"Best model: {best_model_name} (R^2={best_r2:.4f})")
        print(str(best_model_dir))
    else:
        print("No models found")
    
    return str(best_model_dir) if best_model_dir else ""


def merge_predictions(args):
    """Merge wavelength and PLQY predictions"""
    print(f"Merging predictions...")
    
    # Read original data
    df = pd.read_csv(args.data)
    
    # Read predictions
    wavelength_df = pd.read_csv(args.wavelength)
    plqy_df = pd.read_csv(args.plqy)
    
    # Add predictions
    df['Predicted_wavelength'] = wavelength_df['predictions']
    df['Predicted_PLQY'] = plqy_df['predictions']
    
    # Filter high PLQY if requested
    if args.filter_plqy and args.filter_plqy > 0:
        high_plqy = df[df['Predicted_PLQY'] >= args.filter_plqy].copy()
        
        if len(high_plqy) > 0:
            # Sort by PLQY
            high_plqy = high_plqy.sort_values('Predicted_PLQY', ascending=False)
            
            # Save high PLQY candidates
            high_plqy_file = args.output.replace('.csv', f'_plqy_{args.filter_plqy}+.csv')
            
            # Limit to top N if specified
            if args.top_n and args.top_n > 0:
                high_plqy = high_plqy.head(args.top_n)
            
            high_plqy.to_csv(high_plqy_file, index=False)
            print(f"Found {len(high_plqy)} candidates with PLQY >= {args.filter_plqy}")
            print(f"   Saved to: {high_plqy_file}")
            
            # Show top 5
            if len(high_plqy) > 0:
                print("\nTop candidates:")
                for i, row in high_plqy.head(5).iterrows():
                    print(f"  {i+1}. PLQY={row['Predicted_PLQY']:.4f}, Wavelength={row['Predicted_wavelength']:.1f} nm")
    
    # Save merged predictions
    df.to_csv(args.output, index=False)
    print(f"Merged predictions saved to: {args.output}")
    print(f"   Samples: {len(df)}")
    print(f"   Wavelength: {df['Predicted_wavelength'].mean():.1f} +/- {df['Predicted_wavelength'].std():.1f} nm")
    print(f"   PLQY: {df['Predicted_PLQY'].mean():.4f} +/- {df['Predicted_PLQY'].std():.4f}")


def generate_comparison(args):
    """Generate model comparison table"""
    output_dir = Path(args.output_dir)
    models = args.models.split()
    
    print("\n" + "="*80)
    print("MODEL COMPARISON (10-fold Cross-Validation)")
    print("="*80)
    
    results = []
    
    # Collect results from all models
    for model in models:
        for suffix in ['', '_intersection']:
            model_dir = output_dir / f"{model}{suffix}"
            
            # Find summary files
            summary_files = list(model_dir.rglob('*_summary.json'))
            
            for summary_file in summary_files:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                target = data.get('target', 'unknown')
                
                results.append({
                    'Model': model.upper(),
                    'Type': 'Intersection' if suffix else 'Regular',
                    'Target': target.replace('Max_wavelength(nm)', 'Wavelength')
                                   .replace('tau(s*10^-6)', 'Lifetime'),
                    'R2_mean': data.get('mean_r2', 0),
                    'R2_std': data.get('std_r2', 0),
                    'RMSE_mean': data.get('mean_rmse', 0),
                    'RMSE_std': data.get('std_rmse', 0),
                    'MAE_mean': data.get('mean_mae', 0),
                    'MAE_std': data.get('std_mae', 0),
                    'Samples': data.get('n_samples', 0)
                })
    
    if not results:
        print("No results found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Format for display
    df['R2'] = df.apply(lambda x: f"{x['R2_mean']:.4f} +/- {x['R2_std']:.4f}", axis=1)
    df['RMSE'] = df.apply(lambda x: f"{x['RMSE_mean']:.2f} +/- {x['RMSE_std']:.2f}", axis=1)
    df['MAE'] = df.apply(lambda x: f"{x['MAE_mean']:.2f} +/- {x['MAE_std']:.2f}", axis=1)
    
    # Display by target
    for target in df['Target'].unique():
        print(f"\n{target}:")
        print("-"*70)
        target_df = df[df['Target'] == target][['Model', 'Type', 'R2', 'RMSE', 'MAE']]
        target_df = target_df.sort_values(['Type', 'Model'])
        print(target_df.to_string(index=False))
    
    # Find best models
    print("\n" + "="*80)
    print("BEST MODELS")
    print("="*80)
    
    best_models = []
    for target in df['Target'].unique():
        target_df = df[df['Target'] == target]
        if len(target_df) > 0:
            best_idx = target_df['R2_mean'].idxmax()
            best = target_df.loc[best_idx]
            best_models.append({
                'Target': target,
                'Best Model': f"{best['Model']} ({best['Type']})",
                'R^2': f"{best['R2_mean']:.4f} +/- {best['R2_std']:.4f}",
                'RMSE': f"{best['RMSE_mean']:.2f} +/- {best['RMSE_std']:.2f}"
            })
    
    best_df = pd.DataFrame(best_models)
    best_df.to_csv(output_dir / 'best_models.csv', index=False)
    print(best_df.to_string(index=False))
    
    print(f"\nComparison saved to: {output_dir}/model_comparison.csv")
    print(f"Best models saved to: {output_dir}/best_models.csv")
    
    # Generate LaTeX if requested
    if args.generate_latex:
        generate_latex_table(df, output_dir)
    
    # Generate plots if requested
    if args.generate_plots:
        generate_comparison_plots(df, output_dir)


def generate_latex_table(df, output_dir):
    """Generate LaTeX table"""
    latex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{adjustbox}
\begin{document}

\begin{table}[htbp]
\centering
\caption{Model Performance Comparison (10-fold Cross-Validation)}
\adjustbox{width=\textwidth}{
\begin{tabular}{lllrrr}
\toprule
Model & Type & Target & R$^2$ & RMSE & MAE \\
\midrule
"""
    
    for _, row in df.iterrows():
        latex_content += f"{row['Model']} & {row['Type']} & {row['Target']} & "
        latex_content += f"{row['R2']} & {row['RMSE']} & {row['MAE']} \\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
}
\end{table}
\end{document}"""
    
    latex_file = output_dir / 'model_comparison.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to: {latex_file}")


def generate_comparison_plots(df, output_dir):
    """Generate comparison plots"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # R^2 comparison bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        plot_df = df.copy()
        plot_df['Model_Type'] = plot_df['Model'] + ' (' + plot_df['Type'] + ')'
        
        # Plot
        sns.barplot(data=plot_df, x='Model_Type', y='R2_mean', hue='Target', ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel('R^2 Score')
        ax.set_title('Model R^2 Comparison (10-fold CV)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = output_dir / 'model_comparison.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {plot_file}")
    except ImportError:
        print("WARNING: Matplotlib not available, skipping plots")


def generate_report(args):
    """Generate final JSON report"""
    output_dir = Path(args.output_dir)
    
    report = {
        'output_dir': str(output_dir),
        'configuration': {},
        'results': {},
        'files': []
    }
    
    # Read configuration if exists
    config_file = output_dir / 'config.txt'
    if config_file.exists():
        with open(config_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    report['configuration'][key] = value
    
    # Collect results
    if (output_dir / 'model_comparison.csv').exists():
        df = pd.read_csv(output_dir / 'model_comparison.csv')
        report['results']['models_trained'] = len(df)
        report['results']['best_r2'] = df['R2_mean'].max()
        report['results']['best_model'] = df.loc[df['R2_mean'].idxmax(), 'Model']
    
    if (output_dir / 'test_predictions.csv').exists():
        df = pd.read_csv(output_dir / 'test_predictions.csv')
        report['results']['test_samples'] = len(df)
    
    if (output_dir / 'virtual_predictions.csv').exists():
        df = pd.read_csv(output_dir / 'virtual_predictions.csv')
        report['results']['virtual_samples'] = len(df)
    
    # List all files
    for file in output_dir.rglob('*'):
        if file.is_file():
            report['files'].append(str(file.relative_to(output_dir)))
    
    # Save report
    report_file = output_dir / 'final_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFinal report saved to: {report_file}")
    print(f"   Files generated: {len(report['files'])}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ML Pipeline Helper Functions')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create intersection command
    intersection_parser = subparsers.add_parser('create_intersection')
    intersection_parser.add_argument('--input', required=True, help='Input CSV file')
    intersection_parser.add_argument('--output', required=True, help='Output CSV file')
    
    # Find best model command
    best_parser = subparsers.add_parser('find_best_model')
    best_parser.add_argument('--output-dir', required=True, help='Output directory')
    best_parser.add_argument('--models', required=True, help='Space-separated model list')
    
    # Merge predictions command
    merge_parser = subparsers.add_parser('merge_predictions')
    merge_parser.add_argument('--data', required=True, help='Original data CSV')
    merge_parser.add_argument('--wavelength', required=True, help='Wavelength predictions CSV')
    merge_parser.add_argument('--plqy', required=True, help='PLQY predictions CSV')
    merge_parser.add_argument('--output', required=True, help='Output CSV file')
    merge_parser.add_argument('--filter-plqy', type=float, help='Filter PLQY threshold')
    merge_parser.add_argument('--top-n', type=int, help='Keep only top N candidates')
    
    # Generate comparison command
    comparison_parser = subparsers.add_parser('generate_comparison')
    comparison_parser.add_argument('--output-dir', required=True, help='Output directory')
    comparison_parser.add_argument('--models', required=True, help='Space-separated model list')
    comparison_parser.add_argument('--generate-latex', type=bool, default=False, help='Generate LaTeX')
    comparison_parser.add_argument('--generate-plots', type=bool, default=False, help='Generate plots')
    comparison_parser.add_argument('--generate-html', type=bool, default=False, help='Generate HTML')
    
    # Generate report command
    report_parser = subparsers.add_parser('generate_report')
    report_parser.add_argument('--output-dir', required=True, help='Output directory')
    report_parser.add_argument('--config', help='Configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'create_intersection':
        create_intersection(args)
    elif args.command == 'find_best_model':
        find_best_model(args)
    elif args.command == 'merge_predictions':
        merge_predictions(args)
    elif args.command == 'generate_comparison':
        generate_comparison(args)
    elif args.command == 'generate_report':
        generate_report(args)


if __name__ == '__main__':
    main()
