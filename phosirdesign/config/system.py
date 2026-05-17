#!/usr/bin/env python3
"""
Configuration system for defining the full training workflow
Define and manage experiments via configuration files
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
import copy


# ========================================
#           Configuration Dataclasses
# ========================================

@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str = "data/PhosIrDB.csv"
    smiles_columns: List[str] = field(default_factory=lambda: ['L1', 'L2', 'L3'])
    target_columns: List[str] = field(default_factory=lambda: ['Max_wavelength(nm)', 'PLQY', 'tau(s*10^-6)'])
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    random_seed: int = 42
    # Optional: external test CSV path; if provided, evaluate after full training
    test_data_path: Optional[str] = None
    
    # Missing value handling strategy
    # Options: 'skip' (drop rows with NaN), 'mean', 'median',
    #          'zero', 'forward', 'interpolate'
    nan_handling: str = "skip"
    
    # Detailed missing value handling configuration
    nan_threshold: float = 0.5  # Skip rows when missing ratio exceeds threshold
    feature_nan_strategy: str = "zero"  # Feature missing handling (when nan_handling != 'skip')
    target_nan_strategy: str = "skip"   # Target missing handling
    
    # Multi-target data selection strategy
    # 'intersection': use samples where all targets are present (strictest, least data)
    # 'independent': use valid data per target independently (default, higher utilization)
    # 'union': use all data with imputation (loosest, use with nan_handling)
    multi_target_strategy: str = "independent"
    
    # Data sampling
    sample_size: Optional[int] = None  # If set, use first N samples
    
    def validate(self):
        """Validate configuration"""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, "Sum of split ratios must be 1"
        assert len(self.smiles_columns) > 0, "At least one SMILES column is required"
        assert len(self.target_columns) > 0, "At least one target column is required"
        assert self.nan_handling in ["skip", "mean", "median", "zero", "forward", "interpolate"], \
            f"Unsupported missing value handling method: {self.nan_handling}"
        assert self.multi_target_strategy in ["intersection", "independent", "union"], \
            f"Unsupported multi-target strategy: {self.multi_target_strategy}"


@dataclass
class FeatureConfig:
    """Feature configuration"""
    feature_type: str = "combined"  # morgan, descriptors, combined
    morgan_bits: int = 1024
    morgan_radius: int = 2
    combination_method: str = "mean"  # mean, sum, concat
    use_cache: bool = True
    cache_dir: str = "feature_cache"
    descriptor_count: int = 85
    
    def validate(self):
        """Validate configuration"""
        assert self.feature_type in ["morgan", "descriptors", "combined", "tabular", "auto"], \
            f"Unsupported feature type: {self.feature_type}"
        assert self.combination_method in ["mean", "sum", "concat"], \
            f"Unsupported combination method: {self.combination_method}"
        assert isinstance(self.descriptor_count, int) and self.descriptor_count > 0, \
            f"descriptor_count must be a positive integer: {self.descriptor_count}"


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = "xgboost"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization"""
        # Set default hyperparameters
        if not self.hyperparameters:
            self.hyperparameters = self.get_default_params()
    
    def get_default_params(self) -> Dict:
        """Get default parameters"""
        from phosirdesign.models import MODEL_PARAMS
        return MODEL_PARAMS.get(self.model_type, {}).copy()
    
    def validate(self):
        """Validate configuration"""
        from phosirdesign.models import ModelFactory
        assert self.model_type in ModelFactory.get_supported_models(), \
            f"Unsupported model type: {self.model_type}"


@dataclass
class TrainingConfig:
    """Training configuration"""
    n_folds: int = 10
    metrics: List[str] = field(default_factory=lambda: ["rmse", "mae", "r2", "mape"])
    early_stopping: bool = False
    early_stopping_rounds: int = 10
    verbose: int = 1
    save_fold_models: bool = True
    save_final_model: bool = True
    save_training_curves: bool = True  # Save training curves (enabled by default)
    save_feature_importance: bool = True  # Save feature importance (enabled by default)
    model_selection: Optional[str] = None  # Model selection strategy (for AutoML): best_r2, best_rmse, etc.
    
    def validate(self):
        """Validate configuration"""
        assert self.n_folds > 1, "Number of cross-validation folds must be greater than 1"
        valid_metrics = ["rmse", "mae", "r2", "mape", "mse"]
        for metric in self.metrics:
            assert metric in valid_metrics, f"Unsupported metric: {metric}"


@dataclass
class ComparisonConfig:
    """Model comparison configuration"""
    enable: bool = False  # Whether to enable comparison table generation
    formats: List[str] = field(default_factory=lambda: ["markdown", "html", "latex", "csv"])
    highlight_best: bool = True  # Highlight best models
    include_std: bool = True  # Include standard deviation
    save_to_file: bool = True  # Save to file
    output_dir: Optional[str] = None  # Output directory (None uses training directory)
    
    # Numeric precision configuration
    decimal_places: Dict[str, int] = field(default_factory=lambda: {
        'r2': 4,
        'rmse': 4,
        'mae': 4
    })
    
    def validate(self):
        """Validate configuration"""
        valid_formats = ["markdown", "html", "latex", "csv", "excel"]
        for fmt in self.formats:
            assert fmt in valid_formats, f"Unsupported table format: {fmt}"


@dataclass
class ExportConfig:
    """Export configuration"""
    enable: bool = True
    formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    include_predictions: bool = True
    include_feature_importance: bool = True
    include_cv_details: bool = True
    generate_plots: bool = True
    generate_report: bool = True
    stratified_analysis: bool = False  # Generate stratified performance analysis (e.g., PLQY range confusion matrix)
    
    def validate(self):
        """Validate configuration"""
        valid_formats = ["json", "csv", "excel", "pickle"]
        for fmt in self.formats:
            assert fmt in valid_formats, f"Unsupported export format: {fmt}"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    project_name: str = "ml_experiment"
    base_dir: str = "training_logs"
    auto_save: bool = True
    save_plots: bool = True
    generate_report: bool = True
    export_for_publication: bool = False
    log_level: str = "INFO"
    
    def validate(self):
        """Validate configuration"""
        assert self.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"], \
            f"Unsupported log level: {self.log_level}"


@dataclass
class ExperimentConfig:
    """Experiment configuration - main config class"""
    name: str = "default_experiment"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # AutoML-specific configuration (for automl templates)
    automl_models: Optional[List[str]] = None  # Deprecated; use models_to_train
    automl_model_configs: Optional[Dict[str, Dict]] = None  # Per-model configuration
    models_to_train: Optional[List[str]] = None  # Model list for multi-model training
    
    # Metadata
    run_sequence: int = 0
    config_path: Optional[str] = None
    
    def validate(self):
        """Validate all sub-configurations"""
        self.data.validate()
        self.feature.validate()
        self.model.validate()
        self.training.validate()
        
        # Handle cases where deep copy may become dict
        if isinstance(self.comparison, dict):
            self.comparison = ComparisonConfig(**self.comparison)
        self.comparison.validate()
        
        if isinstance(self.export, dict):
            self.export = ExportConfig(**self.export)
        self.export.validate()
        
        self.logging.validate()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_yaml(self, path: Optional[str] = None) -> str:
        """Convert to YAML"""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            with open(path, 'w') as f:
                f.write(yaml_str)
        return yaml_str
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Convert to JSON"""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, config_path=path)
    
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, config_path=path)
    
    @classmethod
    def from_dict(cls, data: Dict, config_path: Optional[str] = None) -> 'ExperimentConfig':
        """Create configuration from dictionary"""
        # Create sub-configurations
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        if 'feature' in data and isinstance(data['feature'], dict):
            data['feature'] = FeatureConfig(**data['feature'])
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        if 'comparison' in data and isinstance(data['comparison'], dict):
            data['comparison'] = ComparisonConfig(**data['comparison'])
        if 'export' in data and isinstance(data['export'], dict):
            data['export'] = ExportConfig(**data['export'])
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig(**data['logging'])
        
        config = cls(**data)
        config.config_path = config_path
        return config
    
    def copy(self) -> 'ExperimentConfig':
        """Deep copy configuration"""
        return copy.deepcopy(self)
    
    def update(self, updates: Dict) -> 'ExperimentConfig':
        """Update configuration"""
        new_config = self.copy()
        
        for key, value in updates.items():
            if '.' in key:  # Support nested updates, e.g., "model.hyperparameters.n_estimators"
                parts = key.split('.')
                obj = new_config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(new_config, key, value)
        
        return new_config


# ========================================
#           Configuration Manager
# ========================================

class ConfigManager:
    """Configuration manager - loads templates from YAML files in config/m/"""

    # Short aliases: {alias: canonical_stem}
    _ALIASES = {
        'random_forest': 'random_forest_standard',
        'gradient_boosting': 'gradient_boosting_standard',
        'adaboost': 'ada_boost_standard',
        'ada_boost': 'ada_boost_standard',
        'extra_trees': 'extra_trees_standard',
        'svr': 'svr_standard',
        'svr_rbf': 'svr_standard',
        'knn': 'knn_standard',
        'decision_tree': 'decision_tree_standard',
        'ridge': 'ridge_standard',
        'lasso': 'lasso_standard',
        'elastic_net': 'elastic_net_standard',
        'elasticnet': 'elastic_net_standard',
        'elasticnet_fast': 'elastic_net_fast',
        'elasticnet_standard': 'elastic_net_standard',
        'elasticnet_large': 'elastic_net_large',
        'lightgbm': 'lightgbm_standard',
        'catboost': 'catboost_standard',
        'xgboost': 'xgboost_standard',
        'ensemble': 'random_forest_standard',
    }

    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.

        Args:
            config_dir: legacy parameter kept for compatibility (unused; YAML
                        files are always loaded from the package's own m/ directory)
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.templates: Dict[str, ExperimentConfig] = {}
        self.load_templates()

    def load_templates(self):
        """Scan the package's m/ directory and load all YAML config files as templates."""
        m_dir = Path(__file__).resolve().parent / "m"
        if not m_dir.exists():
            return

        for yaml_file in sorted(m_dir.glob("**/*.yaml")):
            try:
                config = ExperimentConfig.from_yaml(str(yaml_file))
                stem = yaml_file.stem  # e.g. "xgboost_standard"
                self.templates[stem] = config
            except Exception as exc:
                print(f"WARNING: Failed to load config {yaml_file}: {exc}")

        # Register short-name aliases for backwards compatibility
        for alias, canonical in self._ALIASES.items():
            if canonical in self.templates and alias not in self.templates:
                self.templates[alias] = self.templates[canonical]
    
    def get_template(self, template_name: str) -> ExperimentConfig:
        """
        Get template configuration
        
        Args:
            template_name: template name
        
        Returns:
            configuration object
        """
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}. Available templates: {list(self.templates.keys())}")
        return self.templates[template_name].copy()
    
    def list_templates(self) -> List[str]:
        """List all available templates"""
        return list(self.templates.keys())
    
    def save_config(self, config: ExperimentConfig, filename: str, format: str = "yaml"):
        """
        Save configuration file
        
        Args:
            config: configuration object
            filename: file name (without extension)
            format: format (yaml/json)
        """
        if format == "yaml":
            path = self.config_dir / f"{filename}.yaml"
            config.to_yaml(str(path))
        elif format == "json":
            path = self.config_dir / f"{filename}.json"
            config.to_json(str(path))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Configuration saved: {path}")
        return path
    
    def load_config(self, filename: str) -> ExperimentConfig:
        """
        Load configuration file
        
        Args:
            filename: file name or path
        
        Returns:
            configuration object
        """
        # Try different paths and formats
        paths_to_try = [
            Path(filename),
            self.config_dir / filename,
            self.config_dir / f"{filename}.yaml",
            self.config_dir / f"{filename}.json"
        ]
        
        for path in paths_to_try:
            if path.exists():
                if path.suffix == '.yaml' or path.suffix == '.yml':
                    return ExperimentConfig.from_yaml(str(path))
                elif path.suffix == '.json':
                    return ExperimentConfig.from_json(str(path))
        
        raise FileNotFoundError(f"Configuration file not found: {filename}")
    
    def create_from_wizard(self) -> ExperimentConfig:
        """Create configuration via wizard"""
        print("\nConfiguration Wizard")
        print("=" * 50)
        
        # Select template
        print("\nAvailable templates:")
        for i, template in enumerate(self.templates.keys(), 1):
            desc = self.templates[template].description
            print(f"  {i}. {template}: {desc}")
        
        choice = input("\nSelect template (number or name; press Enter for default): ").strip()
        
        if choice.isdigit():
            template_name = list(self.templates.keys())[int(choice) - 1]
        elif choice in self.templates:
            template_name = choice
        else:
            template_name = 'xgboost_quick'
        
        config = self.get_template(template_name)
        
        # Customize configuration
        name = input(f"Experiment name [{config.name}]: ").strip() or config.name
        config.name = name
        
        description = input(f"Experiment description [{config.description}]: ").strip() or config.description
        config.description = description
        
        # Model parameters
        n_folds = input(f"Cross-validation folds [{config.training.n_folds}]: ").strip()
        if n_folds.isdigit():
            config.training.n_folds = int(n_folds)
        
        # Feature type
        feature_type = input(f"Feature type (morgan/descriptors/combined) [{config.feature.feature_type}]: ").strip()
        if feature_type in ["morgan", "descriptors", "combined"]:
            config.feature.feature_type = feature_type
        
        print("\nConfiguration created!")
        return config


# ========================================
#           Batch Experiment Configuration
# ========================================

@dataclass
class BatchExperimentConfig:
    """Batch experiment configuration"""
    base_config: ExperimentConfig
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_experiment(self, name: str, updates: Dict):
        """
        Add an experiment
        
        Args:
            name: experiment name
            updates: configuration updates
        """
        self.experiments.append({
            'name': name,
            'updates': updates
        })
    
    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations"""
        configs = []
        for exp in self.experiments:
            config = self.base_config.copy()
            config.name = exp['name']
            config = config.update(exp['updates'])
            configs.append(config)
        return configs
    
    @classmethod
    def create_grid_search(cls, 
                          base_config: ExperimentConfig,
                          param_grid: Dict[str, List]) -> 'BatchExperimentConfig':
        """
        Create grid-search configuration
        
        Args:
            base_config: base configuration
            param_grid: parameter grid
        
        Returns:
            batch experiment configuration
        """
        batch = cls(base_config=base_config)
        
        # Generate all parameter combinations
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        for i, combination in enumerate(itertools.product(*values)):
            updates = dict(zip(keys, combination))
            name = f"{base_config.name}_grid_{i+1}"
            batch.add_experiment(name, updates)
        
        return batch


# ========================================
#           Configuration Validator
# ========================================

class ConfigValidator:
    """Configuration validator"""
    
    @staticmethod
    def validate_file_exists(config: ExperimentConfig) -> bool:
        """Validate that data file exists"""
        data_path = Path(config.data.data_path)
        if not data_path.exists():
            print(f"WARNING: Data file not found: {data_path}")
            return False
        return True
    
    @staticmethod
    def validate_dependencies(config: ExperimentConfig) -> bool:
        """Validate that required dependencies are installed"""
        base_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
        model_packages = {
            'xgboost': ['xgboost'],
            'lightgbm': ['lightgbm'],
            'catboost': ['catboost'],
            'random_forest': []
        }
        feature_requires_rdkit = getattr(config.feature, 'feature_type', None) in ['morgan', 'descriptors', 'combined']
        packages_to_check = list(base_packages)
        packages_to_check.extend(model_packages.get(config.model.model_type, []))
        if feature_requires_rdkit:
            packages_to_check.append('rdkit')
        missing = []
        for package in packages_to_check:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        if missing:
            print(f"WARNING: Missing dependencies: {missing}")
            return False
        return True
    
    @staticmethod
    def validate_all(config: ExperimentConfig) -> bool:
        """Run all validations"""
        try:
            # Internal configuration validation
            config.validate()
            
            # File validation
            if not ConfigValidator.validate_file_exists(config):
                return False
            
            # Dependency validation
            if not ConfigValidator.validate_dependencies(config):
                return False
            
            print("INFO: Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"ERROR: Configuration validation failed: {e}")
            return False


# ========================================
#           Convenience Functions
# ========================================

def create_default_config(model_type: str = "xgboost") -> ExperimentConfig:
    """Create default configuration"""
    return ExperimentConfig(
        name=f"{model_type}_experiment",
        model=ModelConfig(model_type=model_type)
    )


def load_config(path: str) -> ExperimentConfig:
    """Load configuration file"""
    if path.endswith('.yaml') or path.endswith('.yml'):
        return ExperimentConfig.from_yaml(path)
    elif path.endswith('.json'):
        return ExperimentConfig.from_json(path)
    else:
        raise ValueError(f"Unsupported configuration file format: {path}")


def save_config(config: ExperimentConfig, path: str):
    """Save configuration file"""
    if path.endswith('.yaml') or path.endswith('.yml'):
        config.to_yaml(path)
    elif path.endswith('.json'):
        config.to_json(path)
    else:
        raise ValueError(f"Unsupported configuration file format: {path}")
