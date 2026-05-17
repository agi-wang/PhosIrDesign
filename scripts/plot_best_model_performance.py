#!/usr/bin/env python3
"""
Generate performance plots for best models
Includes prediction scatter plots and PLQY confusion matrices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error
import json

def find_best_models(comparison_file):
    """Find best models from comparison table"""
    df = pd.read_csv(comparison_file)
    
    best_models = {}
    for target in df['Target'].unique():
        target_df = df[df['Target'] == target]
        best_idx = target_df['R2_mean'].idxmax()
        best_model = target_df.loc[best_idx]
        
        # Normalize target name
        if 'wavelength' in target.lower():
            key = 'wavelength'
        elif 'plqy' in target.lower():
            key = 'plqy'
        else:
            continue
            
        best_models[key] = {
            'model': best_model['Model'].lower(),
            'r2': best_model['R2_mean'],
            'mae': best_model['MAE_mean'],
            'rmse': best_model['RMSE_mean']
        }
        
    return best_models

def load_predictions(project_dir, model_name, target):
    """Load cross-validation prediction results for a model"""
    csv_dir = Path(project_dir) / "all_models" / "automl_train" / model_name / "exports" / "csv"
    target_patterns = {
        "wavelength": [
            f"{model_name}_Max_wavelength(nm)_all_predictions.csv",
            f"{model_name}_Max_wavelength*all_predictions.csv",
        ],
        "plqy": [
            f"{model_name}_PLQY_all_predictions.csv",
            f"{model_name}_PLQY*all_predictions.csv",
        ],
    }
    patterns = target_patterns.get(target, [f"{model_name}_{target}*all_predictions.csv"])
    files = []
    for pattern in patterns:
        files.extend(csv_dir.glob(pattern))
        if files:
            break
    
    if not files:
        tried = ", ".join(str(csv_dir / pattern) for pattern in patterns)
        print(f"WARNING: Prediction file not found. Tried: {tried}")
        return None
        
    # Read prediction data
    df = pd.read_csv(files[0])
    return df

def plot_scatter(true_values, pred_values, target_name, metrics, output_dir):
    """Plot prediction scatter"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set colors
    if 'wavelength' in target_name.lower():
        color = '#2E86AB'
        unit = ' (nm)'
    else:
        color = '#F24236'
        unit = ''
    
    # Scatter points
    ax.scatter(true_values, pred_values, 
              alpha=0.6, s=20, color=color, edgecolors='none')
    
    # Add diagonal
    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', lw=2, alpha=0.8, label='Perfect prediction')
    
    # Add metric text
    text = f"MAE = {metrics['mae']:.2f}{unit}\n"
    text += f"R^2 = {metrics['r2']:.2f}"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
           fontsize=16, fontweight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Axis labels
    if 'wavelength' in target_name.lower():
        ax.set_xlabel('Experimental Wavelength (nm)', fontsize=14)
        ax.set_ylabel('Predicted Wavelength (nm)', fontsize=14)
        title = 'Wavelength Prediction Performance'
    else:
        ax.set_xlabel('Experimental PLQY', fontsize=14)
        ax.set_ylabel('Predicted PLQY', fontsize=14)
        title = 'PLQY Prediction Performance'
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Equal axis range
    ax.set_aspect('equal', adjustable='box')
    
    # Save figure
    output_file = output_dir / f'{target_name}_scatter.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"INFO: Saved scatter: {output_file}")
    return output_file

def plot_confusion_matrix_with_bins(true_values, pred_values, bins, labels, title_suffix, output_dir, filename_suffix):
    """Plot confusion matrix with specific bins"""
    
    # Bin continuous values
    true_cats = pd.cut(true_values, bins=bins, labels=labels, include_lowest=True)
    pred_cats = pd.cut(pred_values, bins=bins, labels=labels, include_lowest=True)
    
    # Remove NaNs
    valid_mask = ~(true_cats.isna() | pred_cats.isna())
    true_cats_clean = true_cats[valid_mask]
    pred_cats_clean = pred_cats[valid_mask]
    
    # Check data sufficiency
    if len(true_cats_clean) == 0:
        print(f"  [WARNING] No valid data to generate confusion matrix ({title_suffix})")
        return None, None
    
    # Compute confusion matrix
    cm = confusion_matrix(true_cats_clean, pred_cats_clean, labels=labels)
    
    # Normalize by row
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Handle NaNs for rows with all zeros
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Figure size
    n_bins = len(labels)
    if n_bins <= 5:
        figsize = (10, 8)
        annot_size = 10
    elif n_bins <= 10:
        figsize = (14, 12)
        annot_size = 8
    else:
        figsize = (16, 14)
        annot_size = 6
    
    # Heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color scheme
    if '10x10' in title_suffix:
        cmap = 'YlOrRd'
    elif '5x5' in title_suffix:
        cmap = 'Blues'
    else:
        cmap = 'viridis'
    
    # Create annotations showing counts and percentages
    annot_data = []
    for i in range(len(labels)):
        row = []
        for j in range(len(labels)):
            if cm[i, j] > 0:
                row.append(f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})')
            else:
                row.append('')
        annot_data.append(row)
    
    sns.heatmap(cm_normalized, annot=annot_data, fmt='', 
                cmap=cmap, vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion'},
                annot_kws={'size': annot_size},
                ax=ax)
    
    # Rotate tick labels for readability
    if n_bins > 5:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    ax.set_xlabel('Predicted PLQY Range', fontsize=12)
    ax.set_ylabel('Experimental PLQY Range', fontsize=12)
    ax.set_title(f'PLQY Confusion Matrix - {title_suffix}', fontsize=14, fontweight='bold')
    
    # Add diagonal accuracy
    diagonal_acc = np.trace(cm_normalized) / len(labels)
    ax.text(0.02, 0.98, f'Diagonal Accuracy: {diagonal_acc:.2%}', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save figure
    output_file = output_dir / f'plqy_confusion_matrix_{filename_suffix}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"INFO: Saved confusion matrix: {output_file}")
    
    # Save confusion matrix data
    cm_df = pd.DataFrame(cm, 
                        index=[f'True_{l}' for l in labels],
                        columns=[f'Pred_{l}' for l in labels])
    cm_file = output_dir / f'plqy_confusion_matrix_{filename_suffix}.csv'
    cm_df.to_csv(cm_file)
    
    # Save normalized version
    cm_norm_df = pd.DataFrame(cm_normalized, 
                             index=[f'True_{l}' for l in labels],
                             columns=[f'Pred_{l}' for l in labels])
    cm_norm_file = output_dir / f'plqy_confusion_matrix_{filename_suffix}_normalized.csv'
    cm_norm_df.to_csv(cm_norm_file)
    
    return output_file, cm_file

def plot_plqy_confusion_matrix(true_values, pred_values, output_dir):
    """Plot PLQY confusion matrices with multiple configurations"""
    
    results = {}
    
    # Create subdirectory for confusion matrices
    cm_dir = output_dir / 'confusion_matrices'
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n  Generating multiple confusion matrix configurations...")
    
    # 1. 10x10 matrix (0.1 intervals)
    print("    - 10x10 matrix (0.1 intervals)...")
    bins_10x10 = [i/10 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    labels_10x10 = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
    
    img_10x10, data_10x10 = plot_confusion_matrix_with_bins(
        true_values, pred_values, 
        bins_10x10, labels_10x10,
        '10x10 (0.1 intervals)',
        cm_dir, '10x10'
    )
    results['10x10'] = {'image': str(img_10x10), 'data': str(data_10x10)}
    
    # 2. 5x5 matrix (0.2 intervals)
    print("    - 5x5 matrix (0.2 intervals)...")
    bins_5x5 = [i/5 for i in range(6)]  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels_5x5 = [f'{i/5:.1f}-{(i+1)/5:.1f}' for i in range(5)]
    
    img_5x5, data_5x5 = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_5x5, labels_5x5,
        '5x5 (0.2 intervals)',
        cm_dir, '5x5'
    )
    results['5x5'] = {'image': str(img_5x5), 'data': str(data_5x5)}
    
    # 3. Custom ranges - Low/Medium/High (3x3)
    print("    - 3x3 matrix (Low/Medium/High)...")
    bins_3x3 = [0, 0.3, 0.7, 1.0]
    labels_3x3 = ['Low (0-0.3)', 'Medium (0.3-0.7)', 'High (0.7-1.0)']
    
    img_3x3, data_3x3 = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_3x3, labels_3x3,
        'Low-Medium-High',
        cm_dir, '3x3_custom'
    )
    results['3x3_custom'] = {'image': str(img_3x3), 'data': str(data_3x3)}
    
    # 4. Fine-grained focus on high PLQY
    print("    - High PLQY focus matrix...")
    bins_high = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    labels_high = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.7', 
                   '0.7-0.8', '0.8-0.85', '0.85-0.9', '0.9-0.95', '0.95-1.0']
    
    img_high, data_high = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_high, labels_high,
        'High PLQY Focus',
        cm_dir, 'high_plqy_focus'
    )
    results['high_plqy_focus'] = {'image': str(img_high), 'data': str(data_high)}
    
    # 5. Custom 4x4 matrix (0-0.1, 0.1-0.4, 0.4-0.7, 0.7-1.0)
    print("    - 4x4 matrix (0-0.1, 0.1-0.4, 0.4-0.7, 0.7-1.0)...")
    bins_4x4 = [0, 0.1, 0.4, 0.7, 1.0]
    labels_4x4 = ['0-0.1', '0.1-0.4', '0.4-0.7', '0.7-1.0']

    img_4x4, data_4x4 = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_4x4, labels_4x4,
        '4x4 (0-0.1, 0.1-0.4, 0.4-0.7, 0.7-1.0)',
        cm_dir, '4x4_custom'
    )
    results['4x4_custom'] = {'image': str(img_4x4), 'data': str(data_4x4)}
    
    # Keep original simple matrix (backward compatibility)
    bins_simple = [0, 0.1, 0.5, 1.0]
    labels_simple = ['0-0.1', '0.1-0.5', '0.5-1.0']
    img_simple, data_simple = plot_confusion_matrix_with_bins(
        true_values, pred_values,
        bins_simple, labels_simple,
        'Simple (Original)',
        output_dir, 'simple'
    )
    
    # Summary
    print("\n  Confusion matrices created:")
    print(f"    - 10x10 (fine): {cm_dir}/plqy_confusion_matrix_10x10.png")
    print(f"    - 5x5 (medium): {cm_dir}/plqy_confusion_matrix_5x5.png")
    print(f"    - 3x3 (simple): {cm_dir}/plqy_confusion_matrix_3x3_custom.png")
    print(f"    - High PLQY focus: {cm_dir}/plqy_confusion_matrix_high_plqy_focus.png")
    print(f"    - 4x4 (custom): {cm_dir}/plqy_confusion_matrix_4x4_custom.png")
    
    return results

def save_prediction_data(true_values, pred_values, target_name, output_dir):
    """Save prediction data to CSV"""
    df = pd.DataFrame({
        'true': true_values,
        'predicted': pred_values,
        'error': pred_values - true_values,
        'absolute_error': np.abs(pred_values - true_values),
        'percentage_error': 100 * np.abs(pred_values - true_values) / np.abs(true_values)
    })
    
    output_file = output_dir / f'best_model_{target_name}_predictions.csv'
    df.to_csv(output_file, index=False)
    print(f"INFO: Saved prediction data: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate best model performance plots')
    parser.add_argument('--project', '-p', required=True,
                       help='Project directory')
    parser.add_argument('--output', '-o', 
                       help='Output directory (default: project/figures/model_performance)')
    
    args = parser.parse_args()
    
    project_dir = Path(args.project)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_dir / 'figures' / 'model_performance'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Generate best model performance plots")
    print("="*80)
    print(f"Project directory: {project_dir}")
    print(f"Output directory: {output_dir}")
    
    # 1. Find best models
    comparison_file = project_dir / 'model_comparison_detailed.csv'
    if not comparison_file.exists():
        print(f"ERROR: Comparison file not found: {comparison_file}")
        return
        
    print("\nFinding best models...")
    best_models = find_best_models(comparison_file)
    
    for target, info in best_models.items():
        print(f"  {target.upper()}: {info['model'].upper()} (R^2={info['r2']:.4f}, MAE={info['mae']:.4f})")
    
    # 2. Process each target
    results = {}
    
    for target_key, model_info in best_models.items():
        print(f"\nProcessing {target_key.upper()} predictions...")
        
        # Determine target column name
        if target_key == 'wavelength':
            target_col = 'Max_wavelength(nm)'
        else:
            target_col = 'PLQY'
            
        # Load prediction data
        pred_df = load_predictions(project_dir, model_info['model'], target_col)
        if pred_df is None:
            continue
            
        true_values = pred_df['true'].values
        pred_values = pred_df['predicted'].values
        
        # Compute metrics
        metrics = {
            'r2': r2_score(true_values, pred_values),
            'mae': mean_absolute_error(true_values, pred_values),
            'rmse': np.sqrt(np.mean((true_values - pred_values)**2))
        }
        
        print(f"  Actual metrics: R^2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
        
        # Plot scatter
        plot_scatter(true_values, pred_values, target_key, metrics, output_dir)
        
        # Save data
        save_prediction_data(true_values, pred_values, target_key, output_dir)
        
        results[target_key] = {
            'model': model_info['model'],
            'metrics': metrics,
            'n_samples': len(true_values)
        }
        
        # If PLQY, also generate confusion matrices
        if target_key == 'plqy':
            plot_plqy_confusion_matrix(true_values, pred_values, output_dir)
    
    # 3. Save summary info
    summary = {
        'project': str(project_dir),
        'best_models': best_models,
        'actual_performance': results
    }
    
    summary_file = output_dir / 'performance_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(f"\nINFO: Saved performance summary: {summary_file}")
    
    print("\n" + "="*80)
    print("Best model performance plots generated!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    for f in output_dir.glob('*'):
        if f.is_file():
            print(f"  - {f.name}")

if __name__ == "__main__":
    main()
