"""
PhosIrDesign machine-learning toolkit for Ir(III) emitter design.

Usage:
    from phosirdesign import AutoML

    model = AutoML("xgboost")
    model.train("data.csv")
    results = model.predict(test_data)
"""

# Main interfaces
try:
    from .utils.automl_model import AutoML, load_model
except ImportError:
    from phosirdesign.utils.automl_model import AutoML, load_model

# Backward-compatible aliases for the historical public API.
XGBoost = AutoML
LightGBM = AutoML
CatBoost = AutoML
RandomForest = AutoML

# Version info
__version__ = '3.0.0'
__author__ = 'PhosIrDesign contributors'

# Exported public API
__all__ = [
    'AutoML',
    'XGBoost',
    'LightGBM',
    'CatBoost', 
    'RandomForest',
    'load_model'
]

# Quick-use functions
def quick_train(data_path: str, model_type: str = 'xgboost', **kwargs):
    """Quickly train a model"""
    model = AutoML(model_type)
    model.train(data_path, **kwargs)
    return model

def quick_predict(model_path: str, smiles_list: list):
    """Quick prediction"""
    model = load_model(model_path)
    return model.predict(smiles_list)
