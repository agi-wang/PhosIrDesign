#!/usr/bin/env python3
"""
Training curve saving and visualization module
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TrainingCurveRecorder:
    """Training curve recorder"""
    
    def __init__(self, save_dir: Path, model_name: str, target: str):
        """
        Initialize recorder
        
        Args:
            save_dir: Save directory
            model_name: Model name
            target: Target variable name
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.target = target
        
        # Store training data
        self.fold_data = []
        self.current_fold = None
        
        # Create subdirectory
        self.curves_dir = self.save_dir / "training_curves"
        self.curves_dir.mkdir(exist_ok=True)
        
    def start_fold(self, fold_idx: int):
        """Start a new fold"""
        self.current_fold = {
            'fold': fold_idx,
            'train_scores': [],
            'val_scores': [],
            'iterations': [],
            'train_metrics': {},
            'val_metrics': {}
        }
    
    def add_iteration(self, iteration: int, train_score: float, val_score: float):
        """Add one iteration score"""
        if self.current_fold is not None:
            self.current_fold['iterations'].append(iteration)
            self.current_fold['train_scores'].append(train_score)
            self.current_fold['val_scores'].append(val_score)
    
    def end_fold(self, train_metrics: Dict, val_metrics: Dict):
        """End current fold"""
        if self.current_fold is not None:
            self.current_fold['train_metrics'] = train_metrics
            self.current_fold['val_metrics'] = val_metrics
            self.fold_data.append(self.current_fold)
            self.current_fold = None
    
    def save_curves(self):
        """Save training curve data and charts"""
        if not self.fold_data:
            return
        
        # Save raw data as JSON
        data_file = self.curves_dir / f"{self.model_name}_{self.target}_curves.json"
        with open(data_file, 'w') as f:
            json.dump({
                'model': self.model_name,
                'target': self.target,
                'folds': self.fold_data
            }, f, indent=2)
        
        # Generate and save visualization charts
        self._generate_plots()
        
        return data_file
    
    def _generate_plots(self):
        """Generate training curve charts"""
        if not self.fold_data:
            return
        
        # Create subplots: one per fold
        n_folds = len(self.fold_data)
        fig = make_subplots(
            rows=(n_folds + 1) // 2,
            cols=2,
            subplot_titles=[f"Fold {d['fold']}" for d in self.fold_data],
            vertical_spacing=0.1
        )
        
        # Add curves for each fold
        for idx, fold_data in enumerate(self.fold_data):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            if fold_data['iterations']:
                # Train curve
                fig.add_trace(
                    go.Scatter(
                        x=fold_data['iterations'],
                        y=fold_data['train_scores'],
                        mode='lines',
                        name=f'Train (Fold {fold_data["fold"]})',
                        line=dict(color='blue', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
                
                # Validation curve
                fig.add_trace(
                    go.Scatter(
                        x=fold_data['iterations'],
                        y=fold_data['val_scores'],
                        mode='lines',
                        name=f'Val (Fold {fold_data["fold"]})',
                        line=dict(color='red', width=2),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title=f"{self.model_name} - {self.target} Training Curves",
            height=300 * ((n_folds + 1) // 2),
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text="Score")
        
        # Save HTML
        html_file = self.curves_dir / f"{self.model_name}_{self.target}_curves.html"
        fig.write_html(str(html_file))
        
        # Also generate aggregate plot (average curve)
        self._generate_average_plot()
    
    def _generate_average_plot(self):
        """Generate average training curve"""
        # Collect data for all folds
        all_iterations = []
        all_train_scores = []
        all_val_scores = []
        
        for fold_data in self.fold_data:
            if fold_data['iterations']:
                all_iterations.append(fold_data['iterations'])
                all_train_scores.append(fold_data['train_scores'])
                all_val_scores.append(fold_data['val_scores'])
        
        if not all_iterations:
            return
        
        # Find minimum length
        min_len = min(len(iters) for iters in all_iterations)
        
        # Truncate to same length and compute averages
        train_mean = np.mean([scores[:min_len] for scores in all_train_scores], axis=0)
        train_std = np.std([scores[:min_len] for scores in all_train_scores], axis=0)
        val_mean = np.mean([scores[:min_len] for scores in all_val_scores], axis=0)
        val_std = np.std([scores[:min_len] for scores in all_val_scores], axis=0)
        iterations = all_iterations[0][:min_len]
        
        # Create figure
        fig = go.Figure()
        
        # Train curve (with confidence band)
        fig.add_trace(go.Scatter(
            x=iterations,
            y=train_mean,
            mode='lines',
            name='Train (mean)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=iterations + iterations[::-1],
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Train std'
        ))
        
        # Validation curve (with confidence band)
        fig.add_trace(go.Scatter(
            x=iterations,
            y=val_mean,
            mode='lines',
            name='Validation (mean)',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=iterations + iterations[::-1],
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(200,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Val std'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{self.model_name} - {self.target} Average Training Curves",
            xaxis_title="Iteration",
            yaxis_title="Score",
            hovermode='x unified',
            showlegend=True
        )
        
        # Save
        html_file = self.curves_dir / f"{self.model_name}_{self.target}_avg_curves.html"
        fig.write_html(str(html_file))


class TrainingCurveAggregator:
    """Training curve aggregator - compare multiple models"""
    
    @staticmethod
    def aggregate_curves(curves_dir: Path, output_file: Optional[Path] = None):
        """
        Aggregate training curves for all models
        
        Args:
            curves_dir: Curves data directory
            output_file: Output file path
        """
        if output_file is None:
            output_file = curves_dir / "all_models_comparison.html"
        
        # Read all JSON files
        json_files = list(curves_dir.glob("*_curves.json"))
        if not json_files:
            return
        
        # Create comparison plot
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["All Models Comparison"]
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        for idx, json_file in enumerate(json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            model_name = data['model']
            target = data['target']
            color = colors[idx % len(colors)]
            
            # Compute average curve
            all_val_scores = []
            for fold_data in data['folds']:
                if fold_data['val_scores']:
                    all_val_scores.append(fold_data['val_scores'])
            
            if all_val_scores:
                min_len = min(len(scores) for scores in all_val_scores)
                val_mean = np.mean([scores[:min_len] for scores in all_val_scores], axis=0)
                iterations = list(range(min_len))
                
                fig.add_trace(go.Scatter(
                    x=iterations,
                    y=val_mean,
                    mode='lines',
                    name=f"{model_name}_{target}",
                    line=dict(color=color, width=2)
                ))
        
        # Update layout
        fig.update_layout(
            title="Model Training Curves Comparison",
            xaxis_title="Iteration",
            yaxis_title="Validation Score",
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        # Save
        fig.write_html(str(output_file))
        
        return output_file
