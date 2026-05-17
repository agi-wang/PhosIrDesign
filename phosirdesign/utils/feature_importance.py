#!/usr/bin/env python3
"""
Feature importance saving and visualization module
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import plotly.graph_objects as go


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class FeatureImportanceRecorder:
    """Feature importance recorder"""
    
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
        
        # Create subdirectory
        self.importance_dir = self.save_dir / "feature_importance"
        self.importance_dir.mkdir(exist_ok=True)
        
        # Stored data
        self.importance_data = []
        
    def add_fold_importance(self, fold_idx: int, importance_dict: Dict[str, float], 
                           feature_names: Optional[List[str]] = None):
        """
        Add feature importance for one fold
        
        Args:
            fold_idx: Fold index
            importance_dict: Feature importance dict or array
            feature_names: List of feature names
        """
        if isinstance(importance_dict, np.ndarray):
            # If array, convert to dict
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance_dict))]
            # Convert to Python native types
            importance_dict = {
                name: float(val) for name, val in zip(feature_names, importance_dict)
            }
        else:
            # Ensure dict values are Python native types
            importance_dict = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in importance_dict.items()
            }
        
        self.importance_data.append({
            'fold': int(fold_idx),
            'importance': importance_dict
        })
    
    def save_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Save feature importance data and plots
        
        Args:
            feature_names: Feature names (if not provided earlier)
            
        Returns:
            Dict of saved file paths
        """
        if not self.importance_data:
            return {}
        
        saved_files = {}
        
        # Compute average feature importance
        avg_importance = self._calculate_average_importance()
        
        # Save as JSON (using custom encoder for NumPy types)
        json_file = self.importance_dir / f"{self.model_name}_{self.target}_importance.json"
        with open(json_file, 'w') as f:
            json.dump({
                'model': self.model_name,
                'target': self.target,
                'average_importance': avg_importance,
                'fold_importance': self.importance_data
            }, f, indent=2, ensure_ascii=True, cls=NumpyEncoder)
        saved_files['json'] = json_file
        
        # Save as CSV
        csv_file = self.importance_dir / f"{self.model_name}_{self.target}_importance.csv"
        df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in avg_importance.items()
        ])
        df = df.sort_values('importance', ascending=False)
        df.to_csv(csv_file, index=False)
        saved_files['csv'] = csv_file
        
        # Generate visualization
        html_file = self._generate_plot(avg_importance)
        if html_file:
            saved_files['html'] = html_file
        
        print("   Feature importance saved:")
        print(f"      - CSV: {csv_file.name}")
        print(f"      - JSON: {json_file.name}")
        if html_file:
            print(f"      - Plot: {html_file.name}")
        
        return saved_files
    
    def _calculate_average_importance(self) -> Dict[str, float]:
        """Compute average feature importance"""
        if not self.importance_data:
            return {}
        
        # Collect all features
        all_features = set()
        for data in self.importance_data:
            all_features.update(data['importance'].keys())
        
        # Compute averages
        avg_importance = {}
        for feature in all_features:
            values = []
            for data in self.importance_data:
                if feature in data['importance']:
                    values.append(data['importance'][feature])
            if values:
                avg_importance[feature] = np.mean(values)
        
        return avg_importance
    
    def _generate_plot(self, importance_dict: Dict[str, float], 
                      top_n: int = 20) -> Optional[Path]:
        """
        Generate feature importance plot
        
        Args:
            importance_dict: Feature importance dict
            top_n: Top-N features to display
            
        Returns:
            Plot file path
        """
        if not importance_dict:
            return None
        
        # Sort and take top-N
        sorted_items = sorted(importance_dict.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:top_n]
        
        if not sorted_items:
            return None
        
        features = [item[0] for item in sorted_items]
        importances = [item[1] for item in sorted_items]
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{v:.4f}' for v in importances],
            textposition='outside'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{self.model_name} - {self.target} Feature Importance (Top {len(features)})",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(features) * 25),
            margin=dict(l=200),
            showlegend=False,
            yaxis=dict(autorange="reversed")  # most important on top
        )
        
        # Save HTML
        html_file = self.importance_dir / f"{self.model_name}_{self.target}_importance.html"
        fig.write_html(str(html_file))
        
        return html_file
    
    @staticmethod
    def extract_importance_from_model(model: Any) -> Optional[np.ndarray]:
        """
        Extract feature importance from a trained model
        
        Args:
            model: Trained model
            
        Returns:
            Feature importance array, or None if not supported
        """
        # Check various possible attribute names
        importance_attrs = [
            'feature_importances_',  # sklearn tree models
            'feature_importance',     # LightGBM
            'get_feature_importance', # CatBoost method
            'feature_importances',    # some custom models
        ]
        
        for attr in importance_attrs:
            if hasattr(model, attr):
                importance = getattr(model, attr)
                if callable(importance):
                    # If method, call it
                    try:
                        return importance()
                    except:
                        continue
                else:
                    # If attribute, return directly
                    return importance
        
        # XGBoost special handling
        if hasattr(model, 'get_score'):
            try:
                scores = model.get_score(importance_type='gain')
                if scores:
                    # Convert to array format
                    max_idx = max(int(k[1:]) for k in scores.keys())
                    importance = np.zeros(max_idx + 1)
                    for k, v in scores.items():
                        idx = int(k[1:])  # 'f0' -> 0
                        importance[idx] = v
                    return importance
            except:
                pass
        
        return None


class FeatureImportanceAggregator:
    """Feature importance aggregator - compare multiple models"""
    
    @staticmethod
    def compare_models(importance_dir: Path, output_file: Optional[Path] = None):
        """
        Compare feature importance across multiple models
        
        Args:
            importance_dir: Directory of feature importance data
            output_file: Output file path
        """
        if output_file is None:
            output_file = importance_dir / "models_importance_comparison.html"
        
        # Read all CSV files
        csv_files = list(importance_dir.glob("*_importance.csv"))
        if not csv_files:
            return None
        
        # Collect data
        all_data = {}
        for csv_file in csv_files:
            # Extract model and target from filename
            parts = csv_file.stem.replace('_importance', '').split('_')
            model_name = parts[0]
            target = '_'.join(parts[1:])
            
            df = pd.read_csv(csv_file)
            # Take top 10 features
            df_top = df.head(10)
            
            key = f"{model_name}_{target}"
            all_data[key] = df_top
        
        if not all_data:
            return None
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        n_models = len(all_data)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=list(all_data.keys()),
            horizontal_spacing=0.1
        )
        
        # Add a bar chart for each model
        for idx, (key, df) in enumerate(all_data.items(), 1):
            fig.add_trace(
                go.Bar(
                    x=df['importance'].values,
                    y=df['feature'].values,
                    orientation='h',
                    name=key,
                    showlegend=False,
                    marker=dict(color='lightblue')
                ),
                row=1, col=idx
            )
        
        # Update layout
        fig.update_layout(
            title="Feature Importance Comparison Across Models",
            height=500,
            showlegend=False
        )
        
        # Update x-axis labels
        for i in range(1, n_models + 1):
            fig.update_xaxes(title_text="Importance", row=1, col=i)
            if i == 1:
                fig.update_yaxes(title_text="Feature", row=1, col=i)
        
        # Save
        fig.write_html(str(output_file))
        
        return output_file
