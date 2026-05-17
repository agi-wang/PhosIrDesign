#!/usr/bin/env python3
"""
Generate multi-model comparison tables and visualizations
Includes 10-fold cross-validation means and standard deviations
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def collect_model_results(output_dir):
    """Collect training results for all models"""
    output_dir = Path(output_dir)
    results = []
    
    # 1) Try AutoML training results first
    automl_dir = output_dir / 'all_models' / 'automl_train'
    if automl_dir.exists():
        print(f"  Scanning AutoML training results: {automl_dir}")
        model_dirs = [d for d in automl_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            
            # Find summary JSON files
            summary_files = list(model_dir.glob('exports/*_summary.json'))
            
            for summary_file in summary_files:
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                        
                        # Extract target name
                        target = data.get('target', '')
                        
                        # Normalize target name
                        if 'wavelength' in target.lower():
                            display_target = 'Max_wavelength(nm)'
                        elif 'plqy' in target.lower():
                            display_target = 'PLQY'
                        elif 'tau' in target.lower():
                            display_target = 'tau(s*10^-6)'
                        else:
                            display_target = target
                        
                        # Format model display name
                        model_display_name = model_name.upper()
                        if model_name == 'mlp':
                            model_display_name = 'MLP (Neural Network)'
                        elif model_name == 'mlp_torch_mps':
                            model_display_name = 'MLP Torch/MPS'
                        elif model_name == 'svr':
                            model_display_name = 'SVR'
                        elif model_name == 'knn':
                            model_display_name = 'KNN'
                        elif model_name == 'xgboost':
                            model_display_name = 'XGBoost'
                        elif model_name == 'lightgbm':
                            model_display_name = 'LightGBM'
                        elif model_name == 'catboost':
                            model_display_name = 'CatBoost'
                        elif model_name == 'adaboost':
                            model_display_name = 'AdaBoost'
                        elif model_name == 'decision_tree':
                            model_display_name = 'Decision Tree'
                        elif model_name == 'elastic_net':
                            model_display_name = 'Elastic Net'
                        elif model_name == 'gradient_boosting':
                            model_display_name = 'Gradient Boosting'
                        elif model_name == 'lasso':
                            model_display_name = 'Lasso'
                        elif model_name == 'random_forest':
                            model_display_name = 'Random Forest'
                        elif model_name == 'ridge':
                            model_display_name = 'Ridge'

                        results.append({
                            'Model': model_display_name,
                            'Type': 'AutoML',
                            'Target': display_target,
                            'R2_mean': data.get('mean_r2', 0),
                            'R2_std': data.get('std_r2', 0),
                            'RMSE_mean': data.get('mean_rmse', 0),
                            'RMSE_std': data.get('std_rmse', 0),
                            'MAE_mean': data.get('mean_mae', 0),
                            'MAE_std': data.get('std_mae', 0),
                            'N_folds': data.get('n_folds', 10),
                            'Training_samples': data.get('n_samples', 0),
                            'Test_samples': 0  # AutoML uses cross-validation
                        })
                        print(f"INFO: Loaded summary for {model_name} - {display_target}")
                except Exception as e:
                    print(f"ERROR: Failed reading {summary_file}: {e}")
    
    # 2) If AutoML results not found, scan regular paths
    if not results:
        print(f"INFO: Scanning regular training results...")
        # Scan all model directories
        model_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            
            # Skip non-model directories
            if model_name in ['tables', 'figures', 'reports', 'all_models']:
                continue
            
            # Find regular model results
            train_dirs = list(model_dir.glob('train*/'))
            for train_dir in train_dirs:
                result_file = train_dir / 'results' / 'training_results.json'
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        
                        for target in ['Max_wavelength(nm)', 'PLQY', 'tau(s*10^-6)']:
                            if target in data:
                                target_data = data[target]
                                
                                # Extract 10-fold results
                                fold_results = target_data.get('fold_results', [])
                                if fold_results:
                                    r2_scores = [fold['metrics']['r2'] for fold in fold_results]
                                    rmse_scores = [fold['metrics']['rmse'] for fold in fold_results]
                                    mae_scores = [fold['metrics']['mae'] for fold in fold_results]
                                    
                                    results.append({
                                        'Model': model_name.upper(),
                                        'Type': 'Regular',
                                        'Target': target,
                                        'R2_mean': np.mean(r2_scores),
                                        'R2_std': np.std(r2_scores),
                                        'RMSE_mean': np.mean(rmse_scores),
                                        'RMSE_std': np.std(rmse_scores),
                                        'MAE_mean': np.mean(mae_scores),
                                        'MAE_std': np.std(mae_scores),
                                        'N_folds': len(fold_results),
                                        'Training_samples': target_data.get('training_samples', 0),
                                        'Test_samples': target_data.get('test_samples', 0)
                                    })
            
            # Find intersection model results
            intersection_dirs = list(model_dir.glob('intersection/*/'))
            for inter_dir in intersection_dirs:
                result_file = inter_dir / 'results' / 'training_results.json'
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        
                        for target in ['Max_wavelength(nm)', 'PLQY']:  # Intersection does not include tau
                            if target in data:
                                target_data = data[target]
                                
                                fold_results = target_data.get('fold_results', [])
                                if fold_results:
                                    r2_scores = [fold['metrics']['r2'] for fold in fold_results]
                                    rmse_scores = [fold['metrics']['rmse'] for fold in fold_results]
                                    mae_scores = [fold['metrics']['mae'] for fold in fold_results]
                                    
                                    results.append({
                                        'Model': model_name.upper(),
                                        'Type': 'Intersection',
                                        'Target': target,
                                        'R2_mean': np.mean(r2_scores),
                                        'R2_std': np.std(r2_scores),
                                        'RMSE_mean': np.mean(rmse_scores),
                                        'RMSE_std': np.std(rmse_scores),
                                        'MAE_mean': np.mean(mae_scores),
                                        'MAE_std': np.std(mae_scores),
                                        'N_folds': len(fold_results),
                                        'Training_samples': target_data.get('training_samples', 0),
                                        'Test_samples': target_data.get('test_samples', 0)
                                    })
    
    return pd.DataFrame(results)

def generate_comparison_table(df, output_dir):
    """Generate comparison tables"""
    output_dir = Path(output_dir)
    
    # Create formatted table
    df_formatted = df.copy()
    
    # Add formatted columns (ASCII)
    df_formatted['R^2 (mean +/- std)'] = df_formatted.apply(
        lambda x: f"{x['R2_mean']:.4f} +/- {x['R2_std']:.4f}", axis=1
    )
    df_formatted['RMSE (mean +/- std)'] = df_formatted.apply(
        lambda x: f"{x['RMSE_mean']:.2f} +/- {x['RMSE_std']:.2f}", axis=1
    )
    df_formatted['MAE (mean +/- std)'] = df_formatted.apply(
        lambda x: f"{x['MAE_mean']:.2f} +/- {x['MAE_std']:.2f}", axis=1
    )
    
    # Simplify target names
    df_formatted['Target_clean'] = df_formatted['Target'].apply(
        lambda x: x.replace('Max_wavelength(nm)', 'Wavelength')
                  .replace('tau(s*10^-6)', 'Lifetime')
    )
    
    # Save detailed table
    df_formatted.to_csv(output_dir / 'model_comparison_detailed.csv', index=False)
    
    # Create simplified display table
    display_cols = ['Model', 'Type', 'Target_clean', 'R^2 (mean +/- std)', 
                   'RMSE (mean +/- std)', 'MAE (mean +/- std)', 'N_folds', 'Training_samples']
    df_display = df_formatted[display_cols].copy()
    df_display.columns = ['Model', 'Type', 'Target', 'R^2', 'RMSE', 'MAE', 'Folds', 'Samples']
    
    # Group by type and target
    print("\n" + "="*100)
    print("Model Performance Comparison (10-fold Cross-Validation)")
    print("="*100)
    
    for target in df_display['Target'].unique():
        print(f"\nTarget: {target}")
        print("-"*80)
        
        target_df = df_display[df_display['Target'] == target]
        
        # Show Regular and Intersection models separately
        for model_type in ['Regular', 'Intersection']:
            type_df = target_df[target_df['Type'] == model_type]
            if len(type_df) > 0:
                print(f"\n{model_type} Models:")
                type_df = type_df.sort_values('Model')
                print(type_df[['Model', 'R^2', 'RMSE', 'MAE', 'Samples']].to_string(index=False))
    
    return df_formatted

def find_best_models(df):
    """Find the best model per target"""
    best_models = []
    
    for target in df['Target'].unique():
        target_df = df[df['Target'] == target]
        
        # Find model with highest R^2
        best_r2_idx = target_df['R2_mean'].idxmax()
        best_r2 = target_df.loc[best_r2_idx]
        
        # Find model with lowest RMSE
        best_rmse_idx = target_df['RMSE_mean'].idxmin()
        best_rmse = target_df.loc[best_rmse_idx]
        
        best_models.append({
            'Target': target.replace('Max_wavelength(nm)', 'Wavelength').replace('tau(s*10^-6)', 'Lifetime'),
            'Best_R2_Model': f"{best_r2['Model']} ({best_r2['Type']})",
            'Best_R2': f"{best_r2['R2_mean']:.4f} +/- {best_r2['R2_std']:.4f}",
            'Best_RMSE_Model': f"{best_rmse['Model']} ({best_rmse['Type']})",
            'Best_RMSE': f"{best_rmse['RMSE_mean']:.2f} +/- {best_rmse['RMSE_std']:.2f}"
        })
    
    best_df = pd.DataFrame(best_models)
    
    print("\n" + "="*100)
    print("Best Model Summary")
    print("="*100)
    print(best_df.to_string(index=False))
    
    return best_df

def plot_model_comparison(df, output_dir):
    """Generate model comparison visualizations"""
    output_dir = Path(output_dir)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. R^2 comparison bar charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    targets = df['Target'].unique()
    for idx, target in enumerate(targets[:3]):  # show up to 3 targets
        ax = axes[idx] if len(targets) > 1 else axes
        
        target_df = df[df['Target'] == target]
        
        
        # Prepare data
        models = target_df['Model'].values
        r2_means = target_df['R2_mean'].values
        r2_stds = target_df['R2_std'].values
        types = target_df['Type'].values
        
        # Set colors
        colors = ['blue' if t == 'Regular' else 'orange' for t in types]
        
        # Draw bar chart
        x = np.arange(len(models))
        ax.bar(x, r2_means, yerr=r2_stds, capsize=5, color=colors, alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel('R^2 Score')
        ax.set_title(f"{target.replace('Max_wavelength(nm)', 'Wavelength').replace('tau(s*10^-6)', 'Lifetime')}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n({t[:3]})" for m, t in zip(models, types)], rotation=45, ha='right')
        ax.set_ylim([0, 1])
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(r2_means, r2_stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Model R^2 Comparison (10-fold CV)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'model_r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap: model performance matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pivot table
    pivot_data = df.pivot_table(
        values='R2_mean',
        index=['Model', 'Type'],
        columns='Target',
        aggfunc='mean'
    )
    
    # Simplify column names
    pivot_data.columns = [col.replace('Max_wavelength(nm)', 'Wavelength')
                          .replace('tau(s*10^-6)', 'Lifetime') 
                          for col in pivot_data.columns]
    
    # Draw heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'R^2 Score'}, vmin=0, vmax=1)
    plt.title('Model Performance Heatmap (R^2 Scores)', fontsize=14, fontweight='bold')
    plt.ylabel('Model (Type)')
    plt.xlabel('Target')
    plt.tight_layout()
    plt.savefig(fig_dir / 'model_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"INFO: Saved figures to: {fig_dir}")

def generate_latex_table(df, output_dir):
    """Generate LaTeX table"""
    output_dir = Path(output_dir)
    
    latex_content = r'''\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{adjustbox}
\begin{document}

\begin{table}[htbp]
\centering
\caption{Model Performance Comparison with 10-fold Cross-Validation}
\label{tab:model_comparison}
\adjustbox{width=\textwidth}{
\begin{tabular}{llllrrr}
\toprule
\multirow{2}{*}{Model} & \multirow{2}{*}{Type} & \multirow{2}{*}{Target} & 
\multicolumn{1}{c}{R$^2$} & \multicolumn{1}{c}{RMSE} & \multicolumn{1}{c}{MAE} & \multirow{2}{*}{Samples} \\
& & & (mean $\pm$ std) & (mean $\pm$ std) & (mean $\pm$ std) & \\
\midrule
'''
    
    # Sort data
    df_sorted = df.sort_values(['Target', 'Type', 'Model'])
    
    for _, row in df_sorted.iterrows():
        model = row['Model']
        type_str = row['Type']
        target = row['Target'].replace('Max_wavelength(nm)', 'Wavelength').replace('tau(s*10^-6)', 'Lifetime')
        r2 = f"{row['R2_mean']:.4f} $\\pm$ {row['R2_std']:.4f}"
        rmse = f"{row['RMSE_mean']:.2f} $\\pm$ {row['RMSE_std']:.2f}"
        mae = f"{row['MAE_mean']:.2f} $\\pm$ {row['MAE_std']:.2f}"
        samples = row['Training_samples']
        
        latex_content += f"{model} & {type_str} & {target} & {r2} & {rmse} & {mae} & {samples} \\\\\n"
    
    latex_content += r'''\bottomrule
\end{tabular}
}
\end{table}

\end{document}'''
    
    # Save LaTeX file
    latex_file = output_dir / 'model_comparison_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_content)
    
    print(f"INFO: Saved LaTeX table: {latex_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate multi-model comparison tables and visualizations')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory path (contains training results)')
    parser.add_argument('--format', type=str, default='all',
                       choices=['csv', 'latex', 'plot', 'all'],
                       help='Output format')
    
    args = parser.parse_args()
    
    print("\n" + "="*100)
    print("Generate Model Comparison Report")
    print("="*100)
    print(f"Output directory: {args.output_dir}")
    
    # Collect results
    print("\nINFO: Collecting model training results...")
    df = collect_model_results(args.output_dir)
    
    if len(df) == 0:
        print("ERROR: No training results found")
        return
    
    print(f"INFO: Found {len(df)} model results")
    print(f"   Models: {df['Model'].unique().tolist()}")
    print(f"   Types: {df['Type'].unique().tolist()}")
    print(f"   Targets: {df['Target'].unique().tolist()}")
    
    # Generate tables
    df_formatted = generate_comparison_table(df, args.output_dir)
    
    # Find best models
    best_df = find_best_models(df)
    best_df.to_csv(Path(args.output_dir) / 'best_models_summary.csv', index=False)
    
    # Generate visualizations
    if args.format in ['plot', 'all']:
        print("\nINFO: Generating visualizations...")
        plot_model_comparison(df, args.output_dir)
    
    # Generate LaTeX table
    if args.format in ['latex', 'all']:
        print("\nINFO: Generating LaTeX table...")
        generate_latex_table(df, args.output_dir)
    
    print("\n" + "="*100)
    print("INFO: Report generation completed!")
    print("="*100)

if __name__ == '__main__':
    main()
