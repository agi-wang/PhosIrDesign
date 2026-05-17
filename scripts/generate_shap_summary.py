#!/usr/bin/env python3
"""
Generate SHAP summary plots for publication-ready reporting

This script creates publication-ready SHAP visualizations:
- Figure J(a): Beeswarm summary plot for wavelength model
- Figure J(b): Beeswarm summary plot for PLQY model  
- Feature importance bar charts
- Top feature dependence plots

Requirements:
- Trained models (.joblib files)
- Training data with SMILES columns
- SHAP library
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Publication-ready plotting configuration
# =============================================================================
try:
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Arial' in available_fonts:
        FONT_FAMILY = 'Arial'
    elif 'Helvetica' in available_fonts:
        FONT_FAMILY = 'Helvetica'
    else:
        FONT_FAMILY = 'DejaVu Sans'
except:
    FONT_FAMILY = 'sans-serif'

PUBLICATION_RCPARAMS = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': FONT_FAMILY,
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
}
plt.rcParams.update(PUBLICATION_RCPARAMS)

# Color-blind friendly palette
COLORBLIND_PALETTE = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
}


def find_best_models(project_dir):
    """Find best models for each target in the project directory."""
    project_path = Path(project_dir)
    models = {}
    
    # Search patterns for model files
    search_paths = [
        project_path / 'all_models' / 'automl_train',
        project_path / 'models',
        project_path,
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        # Find all joblib files
        for model_file in search_path.glob('**/*final*.joblib'):
            name = model_file.stem.lower()
            
            # Determine target type
            if 'wavelength' in name or 'max_wavelength' in name:
                if 'wavelength' not in models:
                    models['wavelength'] = model_file
            elif 'plqy' in name:
                if 'plqy' not in models:
                    models['plqy'] = model_file
    
    return models


def load_data_and_features(data_path, smiles_columns=['L1', 'L2', 'L3'], sample_size=200):
    """Load data and extract features for SHAP analysis."""
    from phosirdesign.core.feature_extractor import FeatureExtractor
    
    df = pd.read_csv(data_path)
    print(f"INFO: Loaded {len(df)} samples from {data_path}")
    
    # Sample data if necessary
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"INFO: Sampled {sample_size} samples for SHAP analysis")
    
    # Extract features
    extractor = FeatureExtractor(
        feature_type='combined',
        morgan_bits=1024,
        morgan_radius=2
    )
    
    # Build SMILES list
    smiles_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = [str(row[col]) for col in smiles_columns if col in df.columns]
        if all(s and s != 'nan' for s in smiles):
            smiles_list.append(smiles)
            valid_indices.append(idx)
    
    if not smiles_list:
        raise ValueError("No valid SMILES found in data")
    
    # Extract features
    features = []
    for smiles in smiles_list:
        try:
            feat = extractor.extract_combination(smiles, feature_type='combined', 
                                                  combination_method='mean')
            features.append(feat)
        except Exception as e:
            print(f"WARNING: Feature extraction failed: {e}")
            continue
    
    X = np.array(features)
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    
    # Generate feature names
    n_features = X.shape[1]
    morgan_size = extractor.morgan_bits if hasattr(extractor, 'morgan_bits') else 1024
    
    feature_names = []
    # Morgan fingerprint bits
    for i in range(min(morgan_size, n_features)):
        feature_names.append(f'MorganFP_{i}')
    # Descriptor features
    if n_features > morgan_size:
        try:
            from rdkit.Chem import Descriptors
            desc_names = [x[0] for x in Descriptors._descList[:n_features - morgan_size]]
            feature_names.extend(desc_names)
        except:
            for i in range(n_features - morgan_size):
                feature_names.append(f'Desc_{i}')
    
    # Pad or truncate to match
    while len(feature_names) < n_features:
        feature_names.append(f'Feature_{len(feature_names)}')
    feature_names = feature_names[:n_features]
    
    return X, df_valid, feature_names


def compute_shap_values(model, X, feature_names, model_name='model'):
    """Compute SHAP values for a model."""
    import shap
    import joblib
    
    print(f"INFO: Computing SHAP values for {model_name}...")
    
    # Load model
    loaded = joblib.load(model)
    
    # Extract the actual predictor
    if hasattr(loaded, 'model'):
        predictor = loaded.model
    elif hasattr(loaded, 'estimator'):
        predictor = loaded.estimator
    else:
        predictor = loaded
    
    # Determine model type and create appropriate explainer
    model_type = type(predictor).__name__.lower()
    
    try:
        if 'xgb' in model_type or 'lgb' in model_type or 'catboost' in model_type:
            # Tree-based models - use TreeExplainer
            explainer = shap.TreeExplainer(predictor)
            shap_values = explainer.shap_values(X)
        elif 'forest' in model_type or 'tree' in model_type or 'gradient' in model_type:
            # Scikit-learn tree models
            explainer = shap.TreeExplainer(predictor)
            shap_values = explainer.shap_values(X)
        else:
            # Generic models - use KernelExplainer (slower)
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.KernelExplainer(predictor.predict, background)
            shap_values = explainer.shap_values(X, nsamples=100)
    except Exception as e:
        print(f"WARNING: TreeExplainer failed, falling back to KernelExplainer: {e}")
        background = shap.sample(X, min(50, len(X)))
        
        def predict_fn(x):
            if hasattr(predictor, 'predict'):
                return predictor.predict(x)
            return predictor(x)
        
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X, nsamples=50)
    
    return shap_values, explainer


def plot_shap_summary(shap_values, X, feature_names, output_path, target_name, label):
    """Generate SHAP beeswarm summary plot with publication-ready styling."""
    import shap
    
    print(f"INFO: Generating SHAP summary plot for {target_name}...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 4))
    
    # Get top features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:][::-1]  # Top 15 features
    
    # Filter to top features
    shap_values_top = shap_values[:, top_indices]
    X_top = X[:, top_indices]
    feature_names_top = [feature_names[i] for i in top_indices]
    
    # Clean up feature names for display
    display_names = []
    for name in feature_names_top:
        if name.startswith('MorganFP_'):
            idx = name.replace('MorganFP_', '')
            display_names.append(f'FP_{idx}')
        else:
            # Truncate long descriptor names
            if len(name) > 12:
                display_names.append(name[:10] + '..')
            else:
                display_names.append(name)
    
    # Create custom beeswarm-style plot
    for i, (feat_idx, name) in enumerate(zip(range(len(feature_names_top)), display_names)):
        shap_vals = shap_values_top[:, feat_idx]
        feat_vals = X_top[:, feat_idx]
        
        # Normalize feature values for coloring
        feat_min, feat_max = feat_vals.min(), feat_vals.max()
        if feat_max > feat_min:
            feat_normalized = (feat_vals - feat_min) / (feat_max - feat_min)
        else:
            feat_normalized = np.zeros_like(feat_vals)
        
        # Add jitter for y-axis
        y_jitter = np.random.normal(0, 0.1, len(shap_vals))
        
        # Scatter plot
        scatter = ax.scatter(shap_vals, [len(feature_names_top) - 1 - i] * len(shap_vals) + y_jitter,
                           c=feat_normalized, cmap='coolwarm', s=8, alpha=0.6,
                           vmin=0, vmax=1)
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Set labels
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names[::-1], fontsize=6)
    ax.set_xlabel(f'SHAP value (impact on {target_name})')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Feature value', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    
    # Add figure label
    ax.text(-0.25, 1.05, label, transform=ax.transAxes, 
           fontsize=12, fontweight='bold', verticalalignment='top')
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['png', 'pdf']:
        save_path = Path(str(output_path).replace('.png', f'.{fmt}'))
        plt.savefig(save_path, dpi=300, format=fmt)
        print(f"  INFO: Saved: {save_path}")
    
    plt.close()


def plot_feature_importance_bar(shap_values, feature_names, output_path, target_name):
    """Generate bar chart of feature importance based on mean |SHAP|."""
    
    print(f"INFO: Generating feature importance bar chart for {target_name}...")
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get top 20 features
    top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
    top_values = mean_abs_shap[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Clean names
    display_names = []
    for name in top_names:
        if name.startswith('MorganFP_'):
            idx = name.replace('MorganFP_', '')
            display_names.append(f'FP_{idx}')
        elif len(name) > 15:
            display_names.append(name[:13] + '..')
        else:
            display_names.append(name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4.5))
    
    # Horizontal bar chart
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_values)))[::-1]
    y_pos = np.arange(len(display_names))
    ax.barh(y_pos, top_values[::-1], color=colors, edgecolor='white', linewidth=0.3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names[::-1], fontsize=6)
    ax.set_xlabel(f'Mean |SHAP value| ({target_name})')
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Save
    for fmt in ['png', 'pdf']:
        save_path = Path(str(output_path).replace('.png', f'.{fmt}'))
        plt.savefig(save_path, dpi=300, format=fmt)
        print(f"  INFO: Saved: {save_path}")
    
    plt.close()


def generate_shap_figures(project_dir, data_path, output_dir, sample_size=200):
    """Generate all SHAP figures."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("SHAP Analysis for Publication-Ready Reporting")
    print("=" * 60)
    
    # Find models
    models = find_best_models(project_dir)
    if not models:
        print("ERROR: No trained models found in project directory")
        return []
    
    print(f"\nFound models:")
    for target, path in models.items():
        print(f"  - {target}: {path}")
    
    # Load data and extract features
    print("\nLoading data and extracting features...")
    X, df, feature_names = load_data_and_features(data_path, sample_size=sample_size)
    print(f"  Features shape: {X.shape}")
    
    generated_files = []
    shap_results = {}
    
    # Generate SHAP plots for each target
    labels = {'wavelength': 'j', 'plqy': 'k'}  # Figure labels
    target_display = {'wavelength': 'Wavelength (nm)', 'plqy': 'PLQY'}
    
    for target, model_path in models.items():
        print(f"\n{'='*40}")
        print(f"Analyzing: {target}")
        print(f"{'='*40}")
        
        try:
            # Compute SHAP values
            shap_values, explainer = compute_shap_values(
                model_path, X, feature_names, model_name=target
            )
            
            # Generate summary plot
            summary_path = output_path / f'figure_{labels.get(target, "x")}_shap_{target}_summary.png'
            plot_shap_summary(
                shap_values, X, feature_names, 
                summary_path, target_display.get(target, target),
                labels.get(target, 'x')
            )
            generated_files.append(summary_path)
            
            # Generate importance bar chart
            bar_path = output_path / f'figure_{labels.get(target, "x")}_shap_{target}_importance.png'
            plot_feature_importance_bar(
                shap_values, feature_names,
                bar_path, target_display.get(target, target)
            )
            generated_files.append(bar_path)
            
            # Store results
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
            shap_results[target] = {
                'top_features': [feature_names[i] for i in top_indices],
                'top_importance': [float(mean_abs_shap[i]) for i in top_indices],
            }
            
        except Exception as e:
            print(f"ERROR: SHAP analysis failed for {target}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results summary
    try:
        with open(output_path / 'shap_analysis_summary.json', 'w') as f:
            json.dump(shap_results, f, indent=2)
        print(f"\nINFO: SHAP summary saved to {output_path / 'shap_analysis_summary.json'}")
    except Exception as e:
        print(f"WARNING: Could not save SHAP summary: {e}")
    
    print("\n" + "=" * 60)
    print("SHAP Analysis Complete")
    print(f"Generated files: {len(generated_files)}")
    print("=" * 60)
    
    return generated_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate SHAP summary plots for publication-ready reporting'
    )
    
    parser.add_argument('--project', '-p', default='.',
                       help='Project directory containing trained models')
    parser.add_argument('--data', '-d', default='data/PhosIrDB.csv',
                       help='Training data file path')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory for figures')
    parser.add_argument('--sample-size', '-n', type=int, default=200,
                       help='Number of samples to use for SHAP analysis')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path(args.project) / 'figures'
    
    # Generate SHAP figures
    files = generate_shap_figures(
        args.project, 
        args.data, 
        output_dir,
        sample_size=args.sample_size
    )
    
    # Print generated files
    if files:
        print("\nGenerated SHAP figure files:")
        print("-" * 40)
        for f in sorted(files):
            print(f"  {f}")


if __name__ == "__main__":
    main()
