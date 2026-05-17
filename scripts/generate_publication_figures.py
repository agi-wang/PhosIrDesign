#!/usr/bin/env python3
"""
Generate publication-ready project figures (C-G, H, I, J)

Publication figure standards:
- Resolution: >=300 DPI (TIF/EPS/PDF)
- Font: Arial/Helvetica, 6-8pt for labels
- Colors: Color-blind friendly palette
- Line width: 0.5-1.0 pt
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import argparse
import json
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Publication-ready plotting configuration
# =============================================================================
# Font configuration - Arial preferred, with fallbacks
try:
    # Check if Arial is available
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Arial' in available_fonts:
        FONT_FAMILY = 'Arial'
    elif 'Helvetica' in available_fonts:
        FONT_FAMILY = 'Helvetica'
    else:
        FONT_FAMILY = 'DejaVu Sans'  # Fallback
except:
    FONT_FAMILY = 'sans-serif'

# Publication-style plotting parameters
PUBLICATION_RCPARAMS = {
    # Figure settings
    'figure.dpi': 300,
    'figure.figsize': (8, 6),
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    
    # Save settings - critical for publication
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    'savefig.transparent': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # Compact journal-style font settings
    'font.family': FONT_FAMILY,
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
    
    # PDF/PS font embedding (critical!)
    'pdf.fonttype': 42,   # TrueType - ensures text is editable
    'ps.fonttype': 42,
    
    # Line settings
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    
    # Grid settings
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    
    # Legend settings
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    
    # Tick settings
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
}

# Apply publication-style configuration
plt.rcParams.update(PUBLICATION_RCPARAMS)

# =============================================================================
# Color-blind friendly palette
# =============================================================================
COLORBLIND_PALETTE = {
    'blue': '#0072B2',      # Strong blue
    'orange': '#E69F00',    # Orange
    'green': '#009E73',     # Bluish green
    'yellow': '#F0E442',    # Yellow
    'sky_blue': '#56B4E9',  # Sky blue
    'vermillion': '#D55E00', # Vermillion (red-orange)
    'purple': '#CC79A7',    # Reddish purple
    'black': '#000000',     # Black
}

# Solvent colors - using colorblind-friendly palette
SOLVENT_COLORS = {
    'CH2Cl2': COLORBLIND_PALETTE['blue'],
    'CH3CN': COLORBLIND_PALETTE['green'],
    'Toluene': COLORBLIND_PALETTE['orange'],
    'Others': COLORBLIND_PALETTE['purple'],
}

# Figure label style
FIGURE_LABEL_STYLE = {
    'fontsize': 12,
    'fontweight': 'bold',
    'verticalalignment': 'top',
    'horizontalalignment': 'left',
}

def load_data(data_file):
    """Load original dataset"""
    df = pd.read_csv(data_file)
    return df

def load_predictions(project_dir, model_name='xgboost'):
    """Load prediction results.
    Preferred search order:
    1) `project_dir/<model>/predictions`
    2) `project_dir/all_models/automl_train/<model>/exports/csv/`
    3) `project_dir/automl_train/<model>/exports/csv/` (if `project_dir` already contains `all_models`)
    4) Fallback to `project_dir/predictions`
    """
    project_path = Path(project_dir)
    model_dir = project_path / model_name
    
    # Default path
    predictions_dir = model_dir / 'predictions'
    
    # If default path does not exist, try AutoML path
    if not model_dir.exists() or not predictions_dir.exists():
        # Check if project_path already contains all_models
        if project_path.name == 'all_models' or 'all_models' in project_path.parts:
            # Already under all_models: look under automl_train directly
            automl_dir = project_path / 'automl_train' / model_name / 'exports' / 'csv'
        else:
            # Otherwise, prepend all_models in the path
            automl_dir = project_path / 'all_models' / 'automl_train' / model_name / 'exports' / 'csv'
        
        if automl_dir.exists():
            predictions_dir = automl_dir
            print(f"INFO: Using AutoML predictions directory: {predictions_dir}")
        else:
            # Fallback: use unified predictions directory
            predictions_dir = project_path / 'predictions'
            if not predictions_dir.exists():
                print(f"WARNING: Prediction directory not found: {model_dir/'predictions'} or {predictions_dir} or {automl_dir}")
                return None
    all_predictions = {}
    target_types = {'wavelength': [], 'PLQY': [], 'tau': []}
    
    csv_files = list(predictions_dir.glob("*.csv"))
    
    # If AutoML directory, only select files containing 'all_predictions'
    if 'automl_train' in str(predictions_dir):
        csv_files = [f for f in csv_files if 'all_predictions' in f.name]
    
    for csv_file in csv_files:
        filename = csv_file.stem
        
        target_type = None
        if 'wavelength' in filename.lower() or 'Max_wavelength' in filename:
            target_type = 'wavelength'
        elif 'plqy' in filename.lower() or 'PLQY' in filename:
            target_type = 'PLQY'
        elif 'tau' in filename.lower():
            target_type = 'tau'
        else:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            
            actual_col = None
            pred_col = None
            
            # Prefer 'true' and 'predicted' columns (AutoML format)
            if 'true' in df.columns and 'predicted' in df.columns:
                actual_col = 'true'
                pred_col = 'predicted'
            else:
                for col in df.columns:
                    if 'true' in col.lower() or 'actual' in col.lower():
                        actual_col = col
                    elif 'predict' in col.lower():
                        pred_col = col
            
            if actual_col and pred_col:
                if 'split' in df.columns:
                    if 'test' in df['split'].values:
                        test_df = df[df['split'] == 'test']
                    elif 'val' in df['split'].values:
                        test_df = df[df['split'] == 'val']
                    else:
                        test_df = df
                else:
                    test_df = df
                
                if len(test_df) > 0:
                    target_types[target_type].append({
                        'actual': test_df[actual_col].values,
                        'predicted': test_df[pred_col].values
                    })
        except Exception as e:
            print(f"WARNING: Failed to read file {csv_file}: {e}")
    
    for target_type in ['wavelength', 'PLQY', 'tau']:
        if target_types[target_type]:
            actual_all = np.concatenate([d['actual'] for d in target_types[target_type]])
            predicted_all = np.concatenate([d['predicted'] for d in target_types[target_type]])
            
            all_predictions[target_type] = {
                'actual': actual_all,
                'predicted': predicted_all
            }
    
    # Extra: try loading test-set predictions (exports/test_predictions_*.csv) to override/supplement
    try:
        exports_dir = project_path / 'exports'
        if exports_dir.exists():
            test_files = list(exports_dir.glob('test_predictions_*.csv'))
            for tf in test_files:
                name = tf.stem.lower()
                target_type = None
                if 'wavelength' in name or 'max_wavelength' in name:
                    target_type = 'wavelength'
                elif 'plqy' in name:
                    target_type = 'PLQY'
                elif 'tau' in name:
                    target_type = 'tau'
                if target_type is None:
                    continue
                try:
                    df = pd.read_csv(tf)
                    # Prediction column
                    pred_col = 'prediction' if 'prediction' in df.columns else None
                    if pred_col is None:
                        continue
                    # Ground-truth column (if present)
                    candidate_actual_cols = [
                        'Max_wavelength(nm)', 'Max_wavelengthnm', 'wavelength',
                        'PLQY', 'tau(s*10^-6)', 'tausx10^-6', 'tau'
                    ]
                    actual_col = next((c for c in candidate_actual_cols if c in df.columns), None)
                    if actual_col is None:
                        # If no ground-truth column, skip this target (cannot plot scatter)
                        continue
                    actual = df[actual_col].values
                    predicted = df[pred_col].values
                    mask = ~(pd.isna(actual) | pd.isna(predicted))
                    actual = actual[mask]
                    predicted = predicted[mask]
                    all_predictions[target_type] = {
                        'actual': actual,
                        'predicted': predicted
                    }
                except Exception as e:
                    print(f"WARNING: Failed to read test predictions {tf}: {e}")
    except Exception:
        pass

    return all_predictions

def plot_figure_c(df, output_dir):
    """
    Figure C: Wavelength-PLQY scatter plot colored by solvent
    
    Enhanced with:
    - Color-blind friendly palette
    - KDE density contours
    - publication-ready formatting
    - Multiple output formats (PNG, PDF, TIF)
    """
    from scipy import stats
    
    print("Generating Figure C: Wavelength-PLQY scatter plot (enhanced)...")
    
    # Single-column figure size (mm to inches: 89mm = 3.5in)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    
    # Find wavelength and PLQY columns
    wavelength_col = None
    plqy_col = None
    
    for col in df.columns:
        if 'wavelength' in col.lower() and 'max' in col.lower():
            wavelength_col = col
        if 'plqy' in col.lower():
            plqy_col = col
    
    if wavelength_col and plqy_col:
        # Get valid data for density estimation
        valid_mask = df[wavelength_col].notna() & df[plqy_col].notna()
        x_all = df.loc[valid_mask, wavelength_col].values
        y_all = df.loc[valid_mask, plqy_col].values
        
        # Add KDE density contours (background layer)
        try:
            # Calculate point density
            xy = np.vstack([x_all, y_all])
            kde = stats.gaussian_kde(xy)
            
            # Create grid for contours
            xgrid = np.linspace(440, 880, 100)
            ygrid = np.linspace(0, 1.0, 100)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
            
            # Plot density contours (light gray background)
            contour = ax.contour(Xgrid, Ygrid, Z, levels=5, colors='gray', 
                                linewidths=0.5, alpha=0.4)
            ax.contourf(Xgrid, Ygrid, Z, levels=5, cmap='Greys', alpha=0.15)
        except Exception as e:
            print(f"  Note: KDE contour generation skipped: {e}")
        
        # Create scatter plot with colorblind-friendly palette
        if 'Solvent' in df.columns:
            # With solvent information - use global SOLVENT_COLORS
            for solvent, color in SOLVENT_COLORS.items():
                mask = df['Solvent'] == solvent
                if mask.sum() > 0:
                    ax.scatter(df.loc[mask, wavelength_col], 
                              df.loc[mask, plqy_col],
                              c=color, label=f'{solvent} (n={mask.sum()})', 
                              alpha=0.7, s=15, marker='o', edgecolors='white', 
                              linewidths=0.3)
        else:
            # No solvent information; use default colorblind-friendly blue
            ax.scatter(df[wavelength_col], df[plqy_col], 
                      alpha=0.7, s=15, c=COLORBLIND_PALETTE['blue'], 
                      marker='o', edgecolors='white', linewidths=0.3)
        
        # Axis labels
        ax.set_xlabel('Emission Wavelength (nm)')
        ax.set_ylabel('PLQY')
        ax.set_xlim(440, 880)
        ax.set_ylim(0, 1.05)
        
        # Set x-axis ticks (cleaner, no 'nm' suffix on tick labels)
        ax.set_xticks([450, 550, 650, 750, 850])
        
        # Compact legend
        ax.legend(loc='upper right', frameon=True, fancybox=False, 
                 edgecolor='0.8', fontsize=6, markerscale=0.8)
        
        # Subtle grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
        
        # Add figure label
        ax.text(-0.12, 1.05, 'c', transform=ax.transAxes, **FIGURE_LABEL_STYLE)
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        plt.tight_layout()
        
        # Save in multiple formats for publication
        base_name = output_dir / 'figure_c_wavelength_plqy'
        for fmt in ['png', 'pdf']:
            save_path = Path(str(base_name) + f'.{fmt}')
            plt.savefig(save_path, dpi=300, format=fmt)
            print(f"  INFO: Saved: {save_path}")
        
        plt.close()

        # Export data used for plotting
        try:
            data_out = df[[wavelength_col, plqy_col]].copy()
            if 'Solvent' in df.columns:
                data_out['Solvent'] = df['Solvent']
            data_out.to_csv(output_dir / 'figure_c_data.csv', index=False)
        except Exception:
            pass

def plot_figure_d(df, output_dir):
    """
    Figure D: PLQY distribution histogram (stacked bar)
    """
    print("Generating Figure D: PLQY distribution histogram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    plqy_col = None
    for col in df.columns:
        if 'plqy' in col.lower():
            plqy_col = col
            break
    
    if plqy_col:
        # Define PLQY ranges
        bins = [-0.001, 0.1, 0.5, 1.001]
        labels = ['<=0.1', '0.1-0.5', '>0.5']
        
        # Count entries in each range
        df['PLQY_range'] = pd.cut(df[plqy_col], bins=bins, labels=labels)
        
        if 'Solvent' in df.columns:
            # Define solvent colors
            solvent_colors = {
                'CH2Cl2': '#2E75B6',
                'CH3CN': '#70AD47',
                'Toluene': '#FFC000',
                'Others': '#7030A0'
            }
            
            # Build stacked data
            data_matrix = []
            solvents = ['CH2Cl2', 'CH3CN', 'Toluene', 'Others']
            
            for label in labels:
                row = []
                for solvent in solvents:
                    count = df[(df['PLQY_range'] == label) & (df['Solvent'] == solvent)].shape[0]
                    row.append(count)
                data_matrix.append(row)
            
            # Plot stacked bar chart
            x = np.arange(len(labels))
            width = 0.6
            bottom = np.zeros(len(labels))
            
            for i, solvent in enumerate(solvents):
                values = [data_matrix[j][i] for j in range(len(labels))]
                ax.bar(x, values, width, bottom=bottom, 
                       label=solvent, color=solvent_colors[solvent])
                bottom += values
        else:
            # Simple histogram
            counts = df['PLQY_range'].value_counts()[labels].fillna(0)
            ax.bar(range(len(labels)), counts.values, color='#2E75B6')
        
        ax.set_xlabel('PLQY Range', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of entries', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 800)
        
        if 'Solvent' in df.columns:
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Add label d
        ax.text(0.02, 0.98, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top')
        
        plt.tight_layout()
        save_path = output_dir / 'figure_d_plqy_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved: {save_path}")

        # Export data used for plotting
        try:
            out_df = df[['PLQY_range']].copy()
            if 'Solvent' in df.columns:
                out_df['Solvent'] = df['Solvent']
            out_df.to_csv(output_dir / 'figure_d_data.csv', index=False)
        except Exception:
            pass

def plot_figure_e_f(predictions, output_dir):
    """
    Figures E and F: Predicted vs Experimental scatter plots
    
    Enhanced with:
    - Residual analysis histograms
    - Colorblind-friendly palette
    - Linear regression fit
    - RMSE metric
    - publication-ready formatting
    """
    from sklearn.linear_model import LinearRegression
    
    print("Generating Figures E and F: Predicted vs Experimental scatter plots (enhanced)...")
    
    if not predictions:
        print("WARNING: No prediction data")
        return
    
    # Create 2x2 layout: top row = scatter plots, bottom row = residual histograms
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.35)
    
    axes_scatter = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes_residual = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    
    def plot_prediction_panel(ax, ax_res, actual, predicted, target_name, color, label_letter, 
                              xlim=None, ylim=None, unit=''):
        """Plot a single prediction scatter with residual histogram."""
        # Remove NaN
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return
        
        # Calculate metrics
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        residuals = predicted - actual
        
        # Linear regression for trendline
        lr = LinearRegression()
        lr.fit(actual.reshape(-1, 1), predicted)
        x_line = np.linspace(actual.min(), actual.max(), 100)
        y_line = lr.predict(x_line.reshape(-1, 1))
        
        # Main scatter plot
        ax.scatter(actual, predicted, alpha=0.5, s=12, c=color, 
                  edgecolors='white', linewidths=0.2)
        
        # Diagonal (perfect prediction) line
        if xlim:
            ax.plot(xlim, xlim, 'k-', lw=0.8, alpha=0.5, label='y=x')
        else:
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k-', lw=0.8, alpha=0.5)
        
        # Regression line
        ax.plot(x_line, y_line, '--', color=COLORBLIND_PALETTE['vermillion'], 
               lw=1.0, alpha=0.8, label='Fit')
        
        # Labels and limits
        ax.set_xlabel(f'Experimental {target_name}{unit}')
        ax.set_ylabel(f'Predicted {target_name}{unit}')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        # Metrics annotation
        metrics_text = f'R2 = {r2:.3f}\nMAE = {mae:.2f}{unit}\nRMSE = {rmse:.2f}{unit}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=6,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.9, edgecolor='0.8', linewidth=0.5))
        
        # Figure label
        ax.text(-0.15, 1.08, label_letter, transform=ax.transAxes, **FIGURE_LABEL_STYLE)
        
        # Grid and spines
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        # Residual histogram (bottom panel)
        ax_res.hist(residuals, bins=30, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
        ax_res.axvline(x=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
        ax_res.axvline(x=residuals.mean(), color=COLORBLIND_PALETTE['vermillion'], 
                      linestyle='--', linewidth=0.8, alpha=0.8)
        ax_res.set_xlabel(f'Residual{unit}')
        ax_res.set_ylabel('Count')
        
        # Residual stats
        ax_res.text(0.95, 0.95, f'mu={residuals.mean():.2f}\nsigma={residuals.std():.2f}',
                   transform=ax_res.transAxes, fontsize=5, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            alpha=0.8, edgecolor='0.8', linewidth=0.3))
        
        for spine in ax_res.spines.values():
            spine.set_linewidth(0.5)
        
        return {'r2': r2, 'mae': mae, 'rmse': rmse, 'n': len(actual)}
    
    results = {}
    
    # Figure E: Wavelength prediction
    if 'wavelength' in predictions:
        results['wavelength'] = plot_prediction_panel(
            axes_scatter[0], axes_residual[0],
            predictions['wavelength']['actual'],
            predictions['wavelength']['predicted'],
            target_name='Wavelength', color=COLORBLIND_PALETTE['blue'],
            label_letter='e', unit=' (nm)'
        )
        
        # Export data
        try:
            actual = predictions['wavelength']['actual']
            predicted = predictions['wavelength']['predicted']
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            pd.DataFrame({
                'actual': actual[mask], 
                'predicted': predicted[mask],
                'residual': predicted[mask] - actual[mask]
            }).to_csv(output_dir / 'figure_e_wavelength_data.csv', index=False)
        except Exception:
            pass
    
    # Figure F: PLQY prediction
    if 'PLQY' in predictions:
        results['PLQY'] = plot_prediction_panel(
            axes_scatter[1], axes_residual[1],
            predictions['PLQY']['actual'],
            predictions['PLQY']['predicted'],
            target_name='PLQY', color=COLORBLIND_PALETTE['orange'],
            label_letter='f', xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), unit=''
        )
        
        # Export data
        try:
            actual = predictions['PLQY']['actual']
            predicted = predictions['PLQY']['predicted']
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            pd.DataFrame({
                'actual': actual[mask], 
                'predicted': predicted[mask],
                'residual': predicted[mask] - actual[mask]
            }).to_csv(output_dir / 'figure_f_plqy_data.csv', index=False)
        except Exception:
            pass
    
    plt.tight_layout()
    
    # Save in multiple formats
    base_name = output_dir / 'figure_e_f_predictions'
    for fmt in ['png', 'pdf']:
        save_path = Path(str(base_name) + f'.{fmt}')
        plt.savefig(save_path, dpi=300, format=fmt)
        print(f"  INFO: Saved: {save_path}")
    
    plt.close()
    
    # Save metrics summary
    try:
        with open(output_dir / 'figure_e_f_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

def plot_figure_g(predictions, output_dir):
    """
    Figure G: PLQY-range prediction accuracy heatmap
    """
    print("Generating Figure G: PLQY-range accuracy heatmap...")
    
    if 'PLQY' not in predictions:
        print("WARNING: No PLQY prediction data")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    actual = predictions['PLQY']['actual']
    predicted = predictions['PLQY']['predicted']
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    # Define PLQY ranges
    bins = [0, 0.1, 0.5, 1.0]
    labels = ['0-0.1', '0.1-0.5', '0.5-1.0']
    
    # Bin actual and predicted values
    actual_binned = pd.cut(actual, bins=bins, labels=labels, include_lowest=True)
    predicted_binned = pd.cut(predicted, bins=bins, labels=labels, include_lowest=True)
    
    # Remove NaNs after binning
    mask2 = ~(actual_binned.isna() | predicted_binned.isna())
    actual_binned = actual_binned[mask2]
    predicted_binned = predicted_binned[mask2]
    
    # Create confusion matrix
    cm = confusion_matrix(actual_binned, predicted_binned, labels=labels)
    
    # Normalize to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
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
    
    # Add label g
    ax.text(-0.15, 1.05, 'g', transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top')
    
    plt.tight_layout()
    save_path = output_dir / 'figure_g_plqy_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved: {save_path}")

    # Export confusion matrix data
    try:
        cm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
        cm_df.to_csv(output_dir / 'figure_g_cm_data.csv')
    except Exception:
        pass

def plot_figure_h(project_dir, output_dir):
    """
    Figure H: Virtual Screening Candidates Visualization
    
    Shows the distribution of predicted PLQY vs wavelength for virtual screening candidates,
    with optimal candidates highlighted (high PLQY, green-to-red emission range).
    """
    print("Generating Figure H: Virtual screening candidates...")
    
    # Try to load virtual predictions
    project_path = Path(project_dir)
    virtual_file = None
    
    # Search for virtual predictions file
    for candidate in [
        project_path / 'virtual_predictions_all.csv',
        project_path / 'virtual_predictions_filtered.csv',
        project_path / 'outputs' / 'virtual_predictions_all.csv',
    ]:
        if candidate.exists():
            virtual_file = candidate
            break
    
    if virtual_file is None:
        print("  INFO: No virtual predictions file found, skipping Figure H")
        return
    
    try:
        df = pd.read_csv(virtual_file)
        print(f"  INFO: Loaded {len(df)} virtual screening candidates")
    except Exception as e:
        print(f"  WARNING: Could not load virtual predictions: {e}")
        return
    
    # Find prediction columns
    wl_col = None
    plqy_col = None
    
    for col in df.columns:
        if 'wavelength' in col.lower() and 'predict' in col.lower():
            wl_col = col
        elif 'plqy' in col.lower() and 'predict' in col.lower():
            plqy_col = col
    
    if wl_col is None or plqy_col is None:
        print("  WARNING: Could not find prediction columns, skipping Figure H")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Get valid data
    valid_mask = df[wl_col].notna() & df[plqy_col].notna()
    wl = df.loc[valid_mask, wl_col].values
    plqy = df.loc[valid_mask, plqy_col].values
    
    # Create 2D histogram for density (background)
    from matplotlib.colors import LogNorm
    h = ax.hist2d(wl, plqy, bins=[50, 50], cmap='Greys', alpha=0.6, 
                 norm=LogNorm(), range=[[440, 880], [0, 1]])
    
    # Highlight optimal candidates (PLQY > 0.8, wavelength 500-700 nm for visible)
    optimal_mask = (plqy > 0.8) & (wl > 500) & (wl < 700)
    if optimal_mask.sum() > 0:
        ax.scatter(wl[optimal_mask], plqy[optimal_mask], 
                  c=COLORBLIND_PALETTE['vermillion'], s=8, alpha=0.8,
                  edgecolors='white', linewidths=0.2, label=f'Optimal (n={optimal_mask.sum()})')
    
    # Add target region box
    from matplotlib.patches import Rectangle
    target_region = Rectangle((520, 0.7), 180, 0.3, 
                               fill=False, edgecolor=COLORBLIND_PALETTE['green'],
                               linewidth=1.5, linestyle='--', alpha=0.8)
    ax.add_patch(target_region)
    ax.text(610, 0.95, 'Target\nRegion', ha='center', va='top', fontsize=6,
           color=COLORBLIND_PALETTE['green'])
    
    # Labels
    ax.set_xlabel('Predicted Wavelength (nm)')
    ax.set_ylabel('Predicted PLQY')
    ax.set_xlim(440, 880)
    ax.set_ylim(0, 1.05)
    
    # Colorbar for density
    cbar = plt.colorbar(h[3], ax=ax, shrink=0.7)
    cbar.set_label('Compound count', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    
    # Legend
    if optimal_mask.sum() > 0:
        ax.legend(loc='lower right', fontsize=6, frameon=True, 
                 edgecolor='0.8', fancybox=False)
    
    # Figure label
    ax.text(-0.15, 1.08, 'h', transform=ax.transAxes, **FIGURE_LABEL_STYLE)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Save
    base_name = output_dir / 'figure_h_virtual_screening'
    for fmt in ['png', 'pdf']:
        save_path = Path(str(base_name) + f'.{fmt}')
        plt.savefig(save_path, dpi=300, format=fmt)
        print(f"  INFO: Saved: {save_path}")
    
    plt.close()
    
    # Export optimal candidates data
    try:
        if optimal_mask.sum() > 0:
            optimal_df = df.loc[valid_mask][optimal_mask].head(100)
            optimal_df.to_csv(output_dir / 'figure_h_optimal_candidates.csv', index=False)
    except Exception:
        pass


def plot_figure_i(df, output_dir):
    """
    Figure I: Ligand Structure-Performance Relationship
    
    Analyzes average PLQY by ligand position (L1, L2, L3) to show
    which positions have the strongest influence on performance.
    """
    print("Generating Figure I: Ligand structure-performance analysis...")
    
    # Find PLQY column
    plqy_col = None
    for col in df.columns:
        if 'plqy' in col.lower():
            plqy_col = col
            break
    
    if plqy_col is None:
        print("  WARNING: No PLQY column found, skipping Figure I")
        return
    
    # Check for ligand columns
    ligand_cols = [col for col in ['L1', 'L2', 'L3'] if col in df.columns]
    if len(ligand_cols) < 2:
        print("  WARNING: Not enough ligand columns, skipping Figure I")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    
    ligand_stats = {}
    
    for idx, (ax, col) in enumerate(zip(axes, ligand_cols)):
        # Group by ligand and calculate mean PLQY
        ligand_plqy = df.groupby(col)[plqy_col].agg(['mean', 'std', 'count'])
        ligand_plqy = ligand_plqy[ligand_plqy['count'] >= 3]  # Filter rare ligands
        ligand_plqy = ligand_plqy.sort_values('mean', ascending=False).head(15)
        
        if ligand_plqy.empty:
            ax.text(0.5, 0.5, f'No data for {col}', ha='center', va='center', 
                   transform=ax.transAxes)
            continue
        
        # Horizontal bar chart
        y_pos = np.arange(len(ligand_plqy))
        colors = plt.cm.RdYlGn(ligand_plqy['mean'].values)
        
        bars = ax.barh(y_pos, ligand_plqy['mean'].values, 
                      color=colors, edgecolor='white', linewidth=0.3,
                      xerr=ligand_plqy['std'].values, error_kw={'linewidth': 0.5, 'capsize': 1})
        
        # Labels - truncate long SMILES
        labels = []
        for smiles in ligand_plqy.index:
            if len(str(smiles)) > 15:
                labels.append(str(smiles)[:12] + '...')
            else:
                labels.append(str(smiles))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=4)
        ax.set_xlabel('Mean PLQY')
        ax.set_title(f'{col} Ligand', fontsize=8)
        ax.set_xlim(0, 1.0)
        
        # Grid
        ax.grid(True, axis='x', alpha=0.2, linewidth=0.3)
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        # Store stats
        ligand_stats[col] = {
            'top_ligand': str(ligand_plqy.index[0])[:50],
            'top_plqy': float(ligand_plqy['mean'].iloc[0]),
            'n_unique': int(len(df[col].unique())),
        }
    
    # Add overall figure label
    axes[0].text(-0.35, 1.15, 'i', transform=axes[0].transAxes, **FIGURE_LABEL_STYLE)
    
    plt.tight_layout()
    
    # Save
    base_name = output_dir / 'figure_i_ligand_analysis'
    for fmt in ['png', 'pdf']:
        save_path = Path(str(base_name) + f'.{fmt}')
        plt.savefig(save_path, dpi=300, format=fmt)
        print(f"  INFO: Saved: {save_path}")
    
    plt.close()
    
    # Export ligand stats
    try:
        with open(output_dir / 'figure_i_ligand_stats.json', 'w') as f:
            json.dump(ligand_stats, f, indent=2)
    except Exception:
        pass


def generate_all_figures(project_dir, data_file, output_dir):
    """Generate all figures (C, D, E, F, G, H, I)"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Generate Publication-Ready Figures")
    print("=" * 60)
    print(f"Output format: 300 DPI, PDF + PNG")
    print(f"Color scheme: Colorblind-friendly (Wong, 2011)")
    
    # Load data
    print("\nLoading data...")
    df = load_data(data_file)
    print(f"INFO: Loaded {len(df)} samples")
    
    # Load predictions
    print("\nLoading predictions...")
    predictions = load_predictions(project_dir)
    if predictions:
        for key, value in predictions.items():
            print(f"INFO: {key}: {len(value['actual'])} predictions")
    
    # Generate figures
    print("\nGenerating figures...")
    print("-" * 40)
    
    # Figure C: Wavelength-PLQY scatter (enhanced with KDE contours)
    plot_figure_c(df, output_path)
    
    # Figure D: PLQY distribution
    plot_figure_d(df, output_path)
    
    # Figures E & F: prediction scatter (enhanced with residual analysis)
    if predictions:
        plot_figure_e_f(predictions, output_path)
        
        # Figure G: PLQY range accuracy
        plot_figure_g(predictions, output_path)
    
    # Figure H: Virtual screening candidates (new)
    plot_figure_h(project_dir, output_path)
    
    # Figure I: Ligand structure-performance relationship (new)
    plot_figure_i(df, output_path)
    
    print("\n" + "=" * 60)
    print("INFO: All figures generated")
    print(f"Saved to: {output_path}")
    print("=" * 60)
    
    # Return generated files (both PNG and PDF)
    files = list(output_path.glob("figure_*.*"))
    return files

def main():
    """Main entry"""
    parser = argparse.ArgumentParser(description='Generate project figures')
    
    parser.add_argument('--project', '-p', default='.',
                       help='Project directory')
    parser.add_argument('--data', '-d', default='data/PhosIrDB.csv',
                       help='Data file')
    parser.add_argument('--output', '-o', help='Output directory')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path(args.project) / 'figures'
    
    # Generate all figures
    files = generate_all_figures(args.project, args.data, output_dir)
    
    # Show generated files
    print("\nGenerated figure files:")
    print("-" * 40)
    for f in sorted(files):
        print(f"  {f}")

if __name__ == "__main__":
    main()
