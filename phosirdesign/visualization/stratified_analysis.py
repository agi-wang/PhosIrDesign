#!/usr/bin/env python3
"""
Stratified performance analysis module
Generates performance analysis figures across value ranges
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def plot_plqy_confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str = "PLQY Prediction Accuracy by Range",
    bins: List[float] = None,
    labels: List[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Dict:
    """
    Generate confusion matrix heatmap for PLQY prediction accuracy
    
    Args:
        actual: array of actual values
        predicted: array of predicted values
        output_path: output directory path
        title: figure title
        bins: bin boundaries, default [0, 0.1, 0.5, 1.0]
        labels: bin labels, default ['0-0.1', '0.1-0.5', '0.5-1.0']
        figsize: figure size
    
    Returns:
        dict containing confusion matrix and statistics
    """
    # Set default binning
    if bins is None:
        bins = [0, 0.1, 0.5, 1.0]
    if labels is None:
        labels = ['0-0.1', '0.1-0.5', '0.5-1.0']
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        print("WARNING: No valid data to generate confusion matrix")
        return {}
    
    # Bin actual and predicted values
    actual_binned = pd.cut(actual, bins=bins, labels=labels, include_lowest=True)
    predicted_binned = pd.cut(predicted, bins=bins, labels=labels, include_lowest=True)
    
    # Remove NaNs after binning
    mask2 = ~(actual_binned.isna() | predicted_binned.isna())
    actual_binned = actual_binned[mask2]
    predicted_binned = predicted_binned[mask2]
    # Filter original arrays accordingly
    actual_filtered = actual[mask2]
    predicted_filtered = predicted[mask2]
    
    # Create confusion matrix
    cm = confusion_matrix(actual_binned, predicted_binned, labels=labels)
    
    # Normalize to row-wise percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use blue color palette
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f',
                cmap=cmap,
                vmin=0, 
                vmax=1,
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Accuracy'},
                ax=ax,
                square=True,
                linewidths=1,
                linecolor='white')
    
    ax.set_xlabel('Predicted PLQY Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual PLQY Range', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'plqy_confusion_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute statistics per range
    range_stats = {}
    for i, label in enumerate(labels):
        mask = actual_binned == label
        if mask.sum() > 0:
            range_actual = actual_filtered[actual_binned == label]
            range_predicted = predicted_filtered[actual_binned == label]
            
            range_stats[label] = {
                'count': len(range_actual),
                'accuracy': cm_normalized[i, i],  # diagonal value
                'r2': r2_score(range_actual, range_predicted) if len(range_actual) > 1 else 0,
                'rmse': np.sqrt(mean_squared_error(range_actual, range_predicted)),
                'mae': mean_absolute_error(range_actual, range_predicted)
            }
    
    # Save confusion matrix data
    cm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    cm_df.to_csv(output_path / 'plqy_confusion_matrix.csv')
    
    # Save range statistics
    stats_df = pd.DataFrame(range_stats).T
    stats_df.to_csv(output_path / 'plqy_range_statistics.csv')
    
    print(f"INFO: PLQY confusion matrix saved: {output_file}")
    
    return {
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'range_statistics': range_stats,
        'total_samples': len(actual)
    }


def plot_performance_by_range(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    target_name: str = "PLQY",
    bins: List[float] = None,
    labels: List[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> Dict:
    """
    Generate performance comparison across value ranges
    
    Args:
        actual: array of actual values
        predicted: array of predicted values
        output_path: output directory path
        target_name: target variable name
        bins: bin boundaries
        labels: bin labels
        figsize: figure size
    
    Returns:
        dict containing performance statistics
    """
    # Set default bins by target
    if bins is None:
        if "PLQY" in target_name:
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            labels = ['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
        elif "wavelength" in target_name.lower():
            bins = [400, 450, 500, 550, 600, 650, 700]
            labels = ['400-450', '450-500', '500-550', '550-600', '600-650', '650-700']
        else:
            # Automatic binning
            bins = np.percentile(actual[~np.isnan(actual)], [0, 20, 40, 60, 80, 100])
            labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        print("WARNING: No valid data to generate performance analysis")
        return {}
    
    # Group data by bins
    actual_binned = pd.cut(actual, bins=bins, labels=labels, include_lowest=True)
    
    # Compute metrics per range
    performance_data = []
    for label in labels:
        mask = actual_binned == label
        if mask.sum() > 1:  # At least 2 samples to compute R^2
            range_actual = actual[mask]
            range_predicted = predicted[mask]
            
            performance_data.append({
                'Range': label,
                'Count': len(range_actual),
                'R^2': r2_score(range_actual, range_predicted),
                'RMSE': np.sqrt(mean_squared_error(range_actual, range_predicted)),
                'MAE': mean_absolute_error(range_actual, range_predicted)
            })
    
    if not performance_data:
        print("WARNING: Not enough data to generate performance analysis")
        return {}
    
    perf_df = pd.DataFrame(performance_data)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # R^2 bar chart
    ax = axes[0]
    bars = ax.bar(perf_df['Range'], perf_df['R^2'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Range', fontsize=11)
    ax.set_ylabel('R^2 Score', fontsize=11)
    ax.set_title('R^2 Score by Range', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(perf_df['Range'], rotation=45, ha='right')
    
    # Add numeric labels on bars
    for bar, value in zip(bars, perf_df['R^2']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # RMSE bar chart
    ax = axes[1]
    bars = ax.bar(perf_df['Range'], perf_df['RMSE'], color='coral', alpha=0.8)
    ax.set_xlabel('Range', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('RMSE by Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(perf_df['Range'], rotation=45, ha='right')
    
    # Sample count bar chart
    ax = axes[2]
    bars = ax.bar(perf_df['Range'], perf_df['Count'], color='seagreen', alpha=0.8)
    ax.set_xlabel('Range', fontsize=11)
    ax.set_ylabel('Sample Count', fontsize=11)
    ax.set_title('Sample Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(perf_df['Range'], rotation=45, ha='right')
    
    # Add numeric labels on bars
    for bar, value in zip(bars, perf_df['Count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'{target_name} Performance by Range', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save data
    csv_file = output_path / f'{target_name.lower().replace(" ", "_")}_performance_by_range.csv'
    perf_df.to_csv(csv_file, index=False)
    plt.close()
    
    print(f"INFO: Performance analysis data saved: {csv_file}")
    
    return {
        'performance_df': perf_df,
        'total_samples': len(actual),
        'num_ranges': len(perf_df)
    }


def generate_stratified_analysis(
    predictions: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    targets: List[str] = None
) -> Dict:
    """
    Generate full stratified performance analysis
    
    Args:
        predictions: dict in format {target: {'actual': array, 'predicted': array}}
        output_dir: output directory
        targets: list of targets to analyze (None for all)
    
    Returns:
        analysis results dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectory for stratified analysis
    analysis_dir = output_dir / 'stratified_analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # Determine targets to analyze
    if targets is None:
        targets = list(predictions.keys())
    
    for target in targets:
        if target not in predictions:
            print(f"WARNING: No prediction data found for target {target}")
            continue
        
        print(f"\nAnalyzing target: {target}")
        print("-" * 40)
        
        actual = predictions[target].get('actual')
        predicted = predictions[target].get('predicted')
        
        if actual is None or predicted is None:
            print(f"WARNING: Target {target} missing actual or predicted values")
            continue
        
        # Create target subdirectory
        target_dir = analysis_dir / target.replace('(', '').replace(')', '').replace('*', 'x')
        target_dir.mkdir(exist_ok=True)
        
        # Generate PLQY confusion matrix
        if "PLQY" in target:
            cm_result = plot_plqy_confusion_matrix(
                actual, predicted, target_dir,
                title=f"{target} Prediction Accuracy by Range"
            )
            results[f"{target}_confusion_matrix"] = cm_result
        
        # Generate performance analysis
        perf_result = plot_performance_by_range(
            actual, predicted, target_dir,
            target_name=target
        )
        results[f"{target}_performance"] = perf_result
    
    # Generate summary report
    generate_summary_report(results, analysis_dir)
    
    print(f"\nINFO: Stratified performance analysis completed")
    print(f"Output directory: {analysis_dir}")
    
    return results


def generate_summary_report(results: Dict, output_dir: Path):
    """
    Generate summary report for stratified analysis
    
    Args:
        results: analysis results dict
        output_dir: output directory
    """
    report_lines = []
    report_lines.append("# Stratified Performance Analysis Report\n")
    report_lines.append("\n## Analysis Summary\n")
    
    for key, value in results.items():
        if 'confusion_matrix' in key:
            target = key.replace('_confusion_matrix', '')
            report_lines.append(f"\n### {target} - Confusion Matrix Analysis\n")
            
            if 'range_statistics' in value:
                report_lines.append("| Range | Sample Count | Accuracy | R^2 | RMSE | MAE |\n")
                report_lines.append("|-------|--------------|----------|-----|------|-----|\n")
                
                for range_name, stats in value['range_statistics'].items():
                    report_lines.append(
                        f"| {range_name} | {stats['count']} | "
                        f"{stats['accuracy']:.3f} | {stats['r2']:.3f} | "
                        f"{stats['rmse']:.3f} | {stats['mae']:.3f} |\n"
                    )
        
        elif 'performance' in key:
            target = key.replace('_performance', '')
            report_lines.append(f"\n### {target} - Performance Analysis\n")
            
            if 'performance_df' in value and value['performance_df'] is not None:
                df = value['performance_df']
                report_lines.append(df.to_markdown(index=False))
                report_lines.append("\n")
    
    # Save report
    report_path = output_dir / 'stratified_analysis_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    print(f"INFO: Summary report saved: {report_path}")
