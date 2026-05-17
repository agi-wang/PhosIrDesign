#!/usr/bin/env python3
"""
Plot virtual database predictions (scatter, interactive, density)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import plotly.graph_objects as go
import plotly.express as px

def create_static_plot(df, output_dir):
    """Create static scatter plot (matplotlib)"""
    print("\nINFO: Creating static scatter plot...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot colored by PLQY
    scatter = ax.scatter(df['Predicted_wavelength'], 
                        df['Predicted_PLQY'],
                        c=df['Predicted_PLQY'],
                        cmap='viridis',
                        alpha=0.3,
                        s=1,
                        edgecolors='none')
    
    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('PLQY', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Predicted Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Predicted PLQY', fontsize=14)
    ax.set_title(f'Virtual Database Predictions ({len(df):,} molecules)', fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Wavelength: {df['Predicted_wavelength'].mean():.1f}+/-{df['Predicted_wavelength'].std():.1f} nm\n"
    stats_text += f"PLQY: {df['Predicted_PLQY'].mean():.3f}+/-{df['Predicted_PLQY'].std():.3f}\n"
    stats_text += f"PLQY>=0.7: {(df['Predicted_PLQY']>=0.7).sum():,} ({100*(df['Predicted_PLQY']>=0.7).mean():.2f}%)\n"
    stats_text += f"PLQY>=0.8: {(df['Predicted_PLQY']>=0.8).sum():,} ({100*(df['Predicted_PLQY']>=0.8).mean():.2f}%)\n"
    stats_text += f"PLQY>=0.9: {(df['Predicted_PLQY']>=0.9).sum():,} ({100*(df['Predicted_PLQY']>=0.9).mean():.2f}%)"
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save image
    output_file = output_dir / 'virtual_predictions_scatter.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"INFO: Saved static plot: {output_file}")
    plt.close()
    
    return output_file

def create_interactive_plot(df, output_dir):
    """Create interactive scatter plot (plotly)"""
    print("\nINFO: Creating interactive scatter plot...")
    
    # Sample for performance if dataset is large
    if len(df) > 50000:
        print(f"INFO: Large dataset ({len(df):,}); sampling 50,000 points for interactive plot")
        df_sample = df.sample(n=50000, random_state=42)
    else:
        df_sample = df
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add all points
    fig.add_trace(go.Scatter(
        x=df_sample['Predicted_wavelength'],
        y=df_sample['Predicted_PLQY'],
        mode='markers',
        marker=dict(
            size=3,
            color=df_sample['Predicted_PLQY'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="PLQY"),
            opacity=0.5,
            line=dict(width=0)
        ),
        text=[f"L1: {row['L1'][:20]}...<br>L2: {row['L2'][:20]}...<br>L3: {row['L3'][:20]}...<br>Wavelength: {row['Predicted_wavelength']:.1f} nm<br>PLQY: {row['Predicted_PLQY']:.4f}" 
              for _, row in df_sample.iterrows()],
        hovertemplate="<b>Molecule</b><br>%{text}<extra></extra>",
        name='All molecules'
    ))
    
    # Highlight high PLQY points (>=0.9)
    high_plqy = df[df['Predicted_PLQY'] >= 0.9]
    if len(high_plqy) > 0:
        fig.add_trace(go.Scatter(
            x=high_plqy['Predicted_wavelength'],
            y=high_plqy['Predicted_PLQY'],
            mode='markers',
            marker=dict(
                size=6,
                color='red',
                symbol='star',
                line=dict(width=1, color='darkred')
            ),
            text=[f"L1: {row['L1'][:20]}...<br>L2: {row['L2'][:20]}...<br>L3: {row['L3'][:20]}...<br>Wavelength: {row['Predicted_wavelength']:.1f} nm<br>PLQY: {row['Predicted_PLQY']:.4f}" 
                  for _, row in high_plqy.iterrows()],
        hovertemplate="<b>High PLQY (>=0.9)</b><br>%{text}<extra></extra>",
        name=f'PLQY>=0.9 ({len(high_plqy):,})'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Virtual Database Predictions ({len(df):,} molecules)",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Predicted Wavelength (nm)",
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title="Predicted PLQY",
            gridcolor='lightgray',
            showgrid=True
        ),
        plot_bgcolor='white',
        hovermode='closest',
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add statistical annotation
    annotations = []
    stats_text = f"<b>Statistics:</b><br>"
    stats_text += f"Wavelength: {df['Predicted_wavelength'].mean():.1f}+/-{df['Predicted_wavelength'].std():.1f} nm<br>"
    stats_text += f"PLQY: {df['Predicted_PLQY'].mean():.3f}+/-{df['Predicted_PLQY'].std():.3f}<br>"
    stats_text += f"PLQY>=0.7: {(df['Predicted_PLQY']>=0.7).sum():,} ({100*(df['Predicted_PLQY']>=0.7).mean():.2f}%)<br>"
    stats_text += f"PLQY>=0.8: {(df['Predicted_PLQY']>=0.8).sum():,} ({100*(df['Predicted_PLQY']>=0.8).mean():.2f}%)<br>"
    stats_text += f"PLQY>=0.9: {(df['Predicted_PLQY']>=0.9).sum():,} ({100*(df['Predicted_PLQY']>=0.9).mean():.2f}%)"
    
    fig.add_annotation(
        text=stats_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        opacity=0.9
    )
    
    # Save HTML file
    output_file = output_dir / 'virtual_predictions_scatter.html'
    fig.write_html(output_file)
    print(f"INFO: Saved interactive plot: {output_file}")
    
    return output_file

def create_density_plot(df, output_dir):
    """Create density plots (2D histogram)"""
    print("\nINFO: Creating density distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 2D density plot
    ax = axes[0, 0]
    h = ax.hist2d(df['Predicted_wavelength'], 
                  df['Predicted_PLQY'],
                  bins=[50, 50],
                  cmap='YlOrRd',
                  density=True)
    plt.colorbar(h[3], ax=ax, label='Density')
    ax.set_xlabel('Predicted Wavelength (nm)')
    ax.set_ylabel('Predicted PLQY')
    ax.set_title('2D Density Distribution')
    
    # 2. Hexbin plot
    ax = axes[0, 1]
    hb = ax.hexbin(df['Predicted_wavelength'], 
                   df['Predicted_PLQY'],
                   gridsize=30,
                   cmap='YlOrRd',
                   mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')
    ax.set_xlabel('Predicted Wavelength (nm)')
    ax.set_ylabel('Predicted PLQY')
    ax.set_title('Hexbin Distribution')
    
    # 3. Wavelength distribution
    ax = axes[1, 0]
    ax.hist(df['Predicted_wavelength'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted Wavelength (nm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Wavelength Distribution (mean={df["Predicted_wavelength"].mean():.1f} nm)')
    ax.axvline(df['Predicted_wavelength'].mean(), color='red', linestyle='--', label='Mean')
    ax.legend()
    
    # 4. PLQY distribution
    ax = axes[1, 1]
    ax.hist(df['Predicted_PLQY'], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted PLQY')
    ax.set_ylabel('Count')
    ax.set_title(f'PLQY Distribution (mean={df["Predicted_PLQY"].mean():.3f})')
    ax.axvline(df['Predicted_PLQY'].mean(), color='red', linestyle='--', label='Mean')
    ax.axvline(0.7, color='orange', linestyle='--', label='PLQY=0.7')
    ax.axvline(0.9, color='darkgreen', linestyle='--', label='PLQY=0.9')
    ax.legend()
    
    plt.suptitle(f'Virtual Database Predictions Analysis ({len(df):,} molecules)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save image
    output_file = output_dir / 'virtual_predictions_density.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"INFO: Saved density plot: {output_file}")
    plt.close()
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Plot virtual database prediction results')
    parser.add_argument('--input', '-i', required=True,
                       help='Prediction CSV file path')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory')
    parser.add_argument('--sample', type=int,
                       help='Sample size (for testing)')
    parser.add_argument('--formats', nargs='+', 
                       default=['static', 'interactive', 'density'],
                       choices=['static', 'interactive', 'density'],
                       help='Plot formats to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Virtual database prediction visualization")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {output_dir}")
    
    # Read data
    print("\nINFO: Reading predictions...")
    df = pd.read_csv(args.input)
    print(f"INFO: Loaded {len(df):,} prediction rows")
    
    # Sample if specified
    if args.sample and args.sample < len(df):
        print(f"INFO: Sampling {args.sample:,} rows for visualization")
        df = df.sample(n=args.sample, random_state=42)
    
    # Show basic statistics
    print("\nBasic statistics:")
    print(f"  Wavelength range: {df['Predicted_wavelength'].min():.1f} - {df['Predicted_wavelength'].max():.1f} nm")
    print(f"  Wavelength mean: {df['Predicted_wavelength'].mean():.1f} +/- {df['Predicted_wavelength'].std():.1f} nm")
    print(f"  PLQY range: {df['Predicted_PLQY'].min():.4f} - {df['Predicted_PLQY'].max():.4f}")
    print(f"  PLQY mean: {df['Predicted_PLQY'].mean():.4f} +/- {df['Predicted_PLQY'].std():.4f}")
    print(f"  PLQY>=0.7: {(df['Predicted_PLQY']>=0.7).sum():,} ({100*(df['Predicted_PLQY']>=0.7).mean():.2f}%)")
    print(f"  PLQY>=0.8: {(df['Predicted_PLQY']>=0.8).sum():,} ({100*(df['Predicted_PLQY']>=0.8).mean():.2f}%)")
    print(f"  PLQY>=0.9: {(df['Predicted_PLQY']>=0.9).sum():,} ({100*(df['Predicted_PLQY']>=0.9).mean():.2f}%)")
    
    # Generate plots
    generated_files = []
    
    if 'static' in args.formats:
        generated_files.append(create_static_plot(df, output_dir))
    
    if 'interactive' in args.formats:
        generated_files.append(create_interactive_plot(df, output_dir))
    
    if 'density' in args.formats:
        generated_files.append(create_density_plot(df, output_dir))
    
    print("\n" + "="*80)
    print("INFO: Visualization completed!")
    print("="*80)
    print("Generated files:")
    for f in generated_files:
        print(f"  - {f}")

if __name__ == "__main__":
    main()
