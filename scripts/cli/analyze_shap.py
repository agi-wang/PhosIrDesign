#!/usr/bin/env python3
"""
SHAP interpretability analysis tool - post-processing script
Analyzes trained models without modifying the project

Usage:
    python analyze_shap.py Project_Output_YYYYMMDD_HHMMSS
    python analyze_shap.py Project_Output_YYYYMMDD_HHMMSS --models xgboost lightgbm
    python analyze_shap.py Project_Output_YYYYMMDD_HHMMSS --sample-size 200
"""

import sys
import os
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
import json
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Ensure project root on sys.path so internal modules resolve.
ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT_DIR)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

# Use SHAP library directly, independent of internal modules
print("INFO: SHAP analysis tool loaded")


class ModelShapAnalyzer:
    """SHAP analyzer for trained models"""

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.models_dir = self.project_dir / 'all_models' / 'automl_train'
        self.output_dir = self.project_dir / 'shap_analysis'
        self.output_dir.mkdir(exist_ok=True)

        # KernelExplainer acceleration parameters (can be overridden in main)
        self.kernel_k = 20           # Background cluster count
        self.kernel_nsamples = 200   # Max samples per explanation
        self.kernel_max_samples = 40 # Max sample size for kernel type

        # Load data
        self.load_data()

    def load_data(self):
        """Load training data for SHAP background"""
        print("\nLoading data...")

        # Try loading data from multiple locations
        data_paths = [
            # Prefer project-local dataset paths
            self.project_dir / 'data' / 'PhosIrDB.csv',
            Path('data/PhosIrDB.csv')
        ]

        for data_path in data_paths:
            if data_path.exists():
                self.df = pd.read_csv(data_path)
                print(f"INFO: Data loaded: {data_path}")
                print(f"INFO: Data shape: {self.df.shape}")
                return

        raise FileNotFoundError("ERROR: Training data file not found")

    def extract_features(self, smiles_list):
        """Extract molecular features"""
        # Match training pipeline (1024-bit Morgan + 85 descriptors)
        from phosirdesign.core.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor(feature_type="combined", morgan_bits=1024, morgan_radius=2, descriptor_count=85)

        features_list = []
        for smiles in smiles_list:
            if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
                continue
            feat = extractor.extract_from_smiles(smiles, feature_type="combined")
            features_list.append(feat)

        if not features_list:
            return np.array([])

        return np.vstack(features_list)

    def get_feature_names(self):
        from phosirdesign.core.feature_extractor import DESCRIPTOR_NAMES
        fp_names = [f'Morgan_{i}' for i in range(1024)]
        desc_names = DESCRIPTOR_NAMES if DESCRIPTOR_NAMES else []
        return fp_names + desc_names

    def find_models(self, model_filter=None):
        """Find all trained models"""
        print("\nSearching for trained models...")

        models_info = []

        if not self.models_dir.exists():
            print(f"ERROR: Model directory not found: {self.models_dir}")
            return models_info

        # Iterate all model directories
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Filter models
            if model_filter and model_name not in model_filter:
                continue

            # Find model files
            model_files_dir = model_dir / 'models'
            if not model_files_dir.exists():
                continue

            for model_file in model_files_dir.glob('*.joblib'):
                # Parse target name
                filename = model_file.stem
                if 'Max_wavelength' in filename:
                    target = 'Max_wavelength(nm)'
                    target_clean = 'Wavelength'
                elif 'PLQY' in filename:
                    target = 'PLQY'
                    target_clean = 'PLQY'
                else:
                    continue

                models_info.append({
                    'model_name': model_name,
                    'model_path': model_file,
                    'target': target,
                    'target_clean': target_clean
                })

        print(f"INFO: Found {len(models_info)} models")
        for info in models_info:
            print(f"     - {info['model_name']:20s} | {info['target_clean']}")

        return models_info

    def _resolve_predictor(self, loaded_model):
        """Parse model object, return prediction function and underlying model for SHAP"""
        # Use directly a model that implements predict
        if hasattr(loaded_model, 'predict'):
            return loaded_model.predict, loaded_model

        # Dictionary wrapper (includes scaler/target_scaler)
        if isinstance(loaded_model, dict):
            inner = loaded_model.get('model', None)
            scaler = loaded_model.get('scaler', None)
            target_scaler = loaded_model.get('target_scaler', None)

            if inner is None:
                raise ValueError('Dictionary model missing "model" key')

            # Assemble prediction function with preprocessing
            def predict_fn(X):
                X_in = X
                if scaler is not None:
                    X_in = scaler.transform(X_in)
                y_pred = inner.predict(X_in)
                if target_scaler is not None:
                    y_pred = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).ravel()
                return y_pred

            return predict_fn, inner

        # Other wrapper types (e.g., pipeline)
        try:
            predict = getattr(loaded_model, 'predict')
            return predict, loaded_model
        except Exception:
            raise AttributeError('Cannot resolve predict function: model object does not support predict')

    def analyze_model(self, model_info, sample_size=100):
        """Analyze a single model"""
        model_name = model_info['model_name']
        target = model_info['target']
        target_clean = model_info['target_clean']

        print(f"\n{'='*70}")
        print(f"Analyzing model: {model_name} - {target_clean}")
        print(f"{'='*70}")

        # Load model
        try:
            model = joblib.load(model_info['model_path'])
            print(f"INFO: Model loaded")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            return None

        # Prepare data
        print(f"INFO: Preparing feature data...")
        valid_df = self.df.dropna(subset=[target])
        smiles_cols = ['L1', 'L2', 'L3']
        print(f"     Extracting molecular features...")
        from phosirdesign.core.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor(feature_type="combined", morgan_bits=1024, morgan_radius=2, descriptor_count=85)
        X = extractor.extract_from_dataframe(valid_df, smiles_columns=smiles_cols, feature_type="combined")

        if len(X) == 0:
            print(f"ERROR: Feature extraction failed")
            return None

        print(f"     INFO: Feature shape: {X.shape}")

        # Sampling (SHAP is slow)
        if len(X) > sample_size:
            print(f"INFO: Sampling {sample_size} examples for SHAP")
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X

        # Create SHAP analyzer
        print(f"INFO: Computing SHAP values...")
        try:
            # Determine model type
            model_type_map = {
                'xgboost': 'tree',
                'lightgbm': 'tree',
                'catboost': 'tree',
                'random_forest': 'tree',
                'gradient_boosting': 'tree',
                'decision_tree': 'tree',
                'adaboost': 'tree',
                'ridge': 'linear',
                'lasso': 'linear',
                'elastic_net': 'linear'
            }

            shap_model_type = model_type_map.get(model_name, 'kernel')

            # Create explainer
            if shap_model_type == 'tree':
                explainer = shap.TreeExplainer(model)
            elif shap_model_type == 'linear':
                explainer = shap.LinearExplainer(model, X_sample)
            else:
                print(f"INFO: Kernel fast mode: kmeans={self.kernel_k}, nsamples={self.kernel_nsamples}")
                predict_fn, _ = self._resolve_predictor(model)
                # Limit sample size
                if len(X_sample) > self.kernel_max_samples:
                    sample_idx = np.random.choice(len(X_sample), self.kernel_max_samples, replace=False)
                    X_sample = X_sample[sample_idx]
                    print(f"WARNING: Limited sample size to {len(X_sample)} for Kernel explainer")
                # Use kmeans summary as background
                try:
                    background = shap.kmeans(X_sample, self.kernel_k)
                except Exception:
                    # Fall back to random sampling
                    k = min(self.kernel_k, len(X_sample))
                    background = shap.sample(X_sample, k)
                explainer = shap.KernelExplainer(predict_fn, background)

            # Compute SHAP values
            # For kernel branch, limit nsamples for speed; other explainers compute normally
            if shap_model_type == 'kernel':
                shap_values = explainer.shap_values(X_sample, nsamples=self.kernel_nsamples)
            else:
                shap_values = explainer.shap_values(X_sample)

            print(f"INFO: SHAP values computed")

        except Exception as e:
            print(f"ERROR: SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Analyze feature importance
        feature_names = self.get_feature_names()

        # Ensure feature dimension matches
        if len(feature_names) > X_sample.shape[1]:
            feature_names = feature_names[:X_sample.shape[1]]
        elif len(feature_names) < X_sample.shape[1]:
            feature_names = feature_names + [f'Feature_{i}' for i in range(len(feature_names), X_sample.shape[1])]

        # Compute feature importance
        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Save results
        output_subdir = self.output_dir / model_name / target_clean
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save feature importance CSV
        csv_path = output_subdir / 'shap_feature_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        print(f"INFO: Feature importance saved: {csv_path}")

        # Save Top 30 features
        top30 = importance_df.head(30)

        # Create visualizations
        print(f"INFO: Generating visualizations...")

        # 1. Feature importance bar chart
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top30)), top30['importance'].values)
        plt.yticks(range(len(top30)), top30['feature'].values)
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top 30 Feature Importance - {model_name} ({target_clean})')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        fig_path = output_subdir / 'feature_importance_bar.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"INFO: Bar chart: {fig_path.name}")

        # 2. SHAP summary plot
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                max_display=30,
                show=False
            )
            summary_path = output_subdir / 'shap_summary_plot.png'
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"INFO: Summary plot: {summary_path.name}")
        except Exception as e:
            print(f"WARNING: Failed to generate summary plot: {e}")

        # Save metadata
        metadata = {
            'model_name': model_name,
            'target': target_clean,
            'sample_size': len(X_sample),
            'n_features': X_sample.shape[1],
            'top_10_features': top30.head(10).to_dict('records')
        }

        json_path = output_subdir / 'shap_metadata.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)

        print(f"INFO: Analysis completed")

        return {
            'model': model_name,
            'target': target_clean,
            'top_features': top30.head(10)
        }

    def generate_summary_report(self, results):
        """Generate summary report"""
        print(f"\n{'='*70}")
        print(f"Generate summary report")
        print(f"{'='*70}")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SHAP Interpretability Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            padding-left: 10px;
            border-left: 4px solid #3498db;
        }}
        .model-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .model-title {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .feature-table th {{
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .feature-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .feature-table tr:hover {{
            background: #f0f0f0;
        }}
        .images {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .images img {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .summary-box {{
            background: #e8f4f8;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .file-link {{
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }}
        .file-link:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SHAP Interpretability Analysis Report</h1>

        <div class="summary-box">
            <p><strong>Project directory:</strong> {self.project_dir.name}</p>
            <p><strong>Models analyzed:</strong> {len(results)}</p>
            <p><strong>Note:</strong> SHAP values indicate each feature's contribution to the model prediction. Larger values mean stronger influence on the prediction.</p>
        </div>

        <h2>Analysis Results</h2>
"""

        for result in results:
            if result is None:
                continue

            model = result['model']
            target = result['target']
            top_features = result['top_features']

            html += f"""
        <div class="model-section">
            <div class="model-title">{model} - {target}</div>

            <h3>Top 10 Important Features</h3>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance (Mean |SHAP|)</th>
                    </tr>
                </thead>
                <tbody>
"""

            for idx, row in enumerate(top_features.iterrows(), 1):
                feature = row[1]['feature']
                importance = row[1]['importance']
                html += f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{feature}</td>
                        <td>{importance:.6f}</td>
                    </tr>
"""

            html += """
                </tbody>
            </table>

            <h3>Visualizations</h3>
            <div class="images">
"""

            # Add image links
            img_dir = f"{model}/{target}"
            html += f"""
                <div>
                    <p><strong>Feature Importance Bar Chart</strong></p>
                    <a href="{img_dir}/feature_importance_bar.png" target="_blank">
                        <img src="{img_dir}/feature_importance_bar.png" alt="Feature Importance">
                    </a>
                </div>
                <div>
                    <p><strong>SHAP Summary Plot</strong></p>
                    <a href="{img_dir}/shap_summary_plot.png" target="_blank">
                        <img src="{img_dir}/shap_summary_plot.png" alt="SHAP Summary">
                    </a>
                </div>
"""

            html += """
            </div>

            <p style="margin-top: 15px;">
                <a class="file-link" href="{}/{}/shap_feature_importance.csv">Download full feature importance (CSV)</a> |
                <a class="file-link" href="{}/{}/shap_metadata.json">Metadata (JSON)</a>
            </p>
        </div>
""".format(model, target, model, target)

        html += """
        <div class="summary-box" style="margin-top: 40px;">
            <h3>How to use these results</h3>
            <ol>
                <li><strong>Identify key features:</strong> Top features reveal which molecular properties matter most</li>
                <li><strong>Guide molecular design:</strong> Focus on important features to optimize structures</li>
                <li><strong>Model diagnostics:</strong> Check whether the model focuses on reasonable chemical features</li>
                <li><strong>Reporting:</strong> Explain the model's predictive basis with supporting evidence</li>
            </ol>
        </div>
    </div>
</body>
</html>
"""

        report_path = self.output_dir / 'shap_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"INFO: Report generated: {report_path}")
        print(f"Open in browser: file://{report_path.absolute()}")

        return report_path


def main():
    parser = argparse.ArgumentParser(
        description='SHAP interpretability analysis tool - analyze trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze all models
    python analyze_shap.py Project_Output_YYYYMMDD_HHMMSS

    # Analyze XGBoost and LightGBM
    python analyze_shap.py Project_Output_YYYYMMDD_HHMMSS --models xgboost lightgbm

    # Use a larger sample size
    python analyze_shap.py Project_Output_YYYYMMDD_HHMMSS --sample-size 200
        """
    )

    parser.add_argument('project_dir', help='Project output directory (e.g., Project_Output_YYYYMMDD_HHMMSS)')
    parser.add_argument('--models', nargs='+', help='Models to analyze (e.g., xgboost lightgbm)')
    parser.add_argument('--sample-size', type=int, default=100, help='Sample size for SHAP analysis (default: 100)')
    parser.add_argument('--kernel-k', type=int, default=20, help='Background k-means clusters for KernelExplainer (default: 20)')
    parser.add_argument('--kernel-nsamples', type=int, default=200, help='Maximum nsamples per explanation for KernelExplainer (default: 200)')
    parser.add_argument('--kernel-max-samples', type=int, default=40, help='Maximum sample count used by Kernel explainer (default: 40)')

    args = parser.parse_args()

    print("="*70)
    print("SHAP Interpretability Analysis Tool")
    print("="*70)

    # Create analyzer
    analyzer = ModelShapAnalyzer(args.project_dir)
    # Override Kernel fast parameters
    analyzer.kernel_k = max(5, int(args.kernel_k))
    analyzer.kernel_nsamples = max(50, int(args.kernel_nsamples))
    analyzer.kernel_max_samples = max(10, int(args.kernel_max_samples))

    # Find models
    models = analyzer.find_models(model_filter=args.models)

    if not models:
        print("\nERROR: No model files found")
        return 1

    # Analyze each model
    results = []
    for model_info in models:
        result = analyzer.analyze_model(model_info, sample_size=args.sample_size)
        results.append(result)

    # Generate summary report
    analyzer.generate_summary_report(results)

    print("\n" + "="*70)
    print("INFO: All analyses completed!")
    print(f"Results saved to: {analyzer.output_dir}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
