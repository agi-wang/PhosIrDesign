#!/usr/bin/env python3
from __future__ import annotations

"""
Dynamic configuration manager - load configurations from YAML files
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
import copy
import glob

# Import configuration data classes
from phosirdesign.config.system import (
    DataConfig, FeatureConfig, ModelConfig, 
    TrainingConfig, LoggingConfig,
    ExperimentConfig
)


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent


class DynamicConfigManager:
    """Dynamic configuration manager - load configurations from filesystem"""
    
    def __init__(self, config_dir: str | Path = DEFAULT_CONFIG_DIR):
        """
        Initialize configuration manager
        
        Args:
            config_dir: root directory of configuration files
        """
        self.config_dir = Path(config_dir)
        self.templates = {}
        self.config_cache = {}
        
        # Automatically scan and load all configurations
        self.scan_and_load_configs()
    
    def scan_and_load_configs(self):
        """Scan configuration directory and load all YAML configs"""
        if not self.config_dir.exists():
            print(f"Config directory not found: {self.config_dir}")
            return
        
        # Find all YAML files
        yaml_files = list(self.config_dir.glob("**/*.yaml")) + list(self.config_dir.glob("**/*.yml"))
        
        loaded_count = 0
        for yaml_file in yaml_files:
            try:
                # Generate config key (relative path without extension)
                relative_path = yaml_file.relative_to(self.config_dir)
                config_key = str(relative_path.with_suffix(''))
                
                # Also generate short name (filename only)
                short_key = yaml_file.stem
                
                # Load configuration
                config = self.load_config_file(yaml_file)
                
                # Store configuration under two keys
                self.templates[config_key] = config
                
                # Use short key when not conflicting
                if short_key not in self.templates:
                    self.templates[short_key] = config
                
                loaded_count += 1
                
            except Exception as e:
                print(f"Failed to load config {yaml_file}: {e}")
        
        print(f"Loaded {loaded_count} configuration templates")
    
    def load_config_file(self, file_path: Path) -> ExperimentConfig:
        """Load a single configuration file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Convert to configuration object
        return self._dict_to_config(data, config_path=str(file_path))
    
    def _dict_to_config(self, data: Dict, config_path: Optional[str] = None) -> ExperimentConfig:
        """Convert a dictionary to a configuration object"""
        # Handle nested configuration objects
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        else:
            data['data'] = DataConfig()
        
        if 'feature' in data and isinstance(data['feature'], dict):
            data['feature'] = FeatureConfig(**data['feature'])
        else:
            data['feature'] = FeatureConfig()
        
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        else:
            data['model'] = ModelConfig()
        
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        else:
            data['training'] = TrainingConfig()
        
        # Optimization has been removed from the codebase
        if 'optimization' in data:
            # Skip optimization config as it's no longer supported
            data.pop('optimization', None)
        
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig(**data['logging'])
        else:
            data['logging'] = LoggingConfig()
        
        # Handle AutoML-specific configuration
        if 'models' in data:
            # AutoML configuration with multiple models
            data['automl_models'] = data.pop('models')
        
        if 'model_configs' in data:
            data['automl_model_configs'] = data.pop('model_configs')
        
        config = ExperimentConfig(**data)
        config.config_path = config_path
        return config
    
    def get_config(self, name: str) -> Optional[ExperimentConfig]:
        """
        Get configuration
        
        Args:
            name: configuration name or path
        
        Returns:
            configuration object, or None if not found
        """
        # Check cached templates first
        if name in self.templates:
            return self.templates[name].copy()
        
        # Try loading as a file path
        config_path = Path(name)
        if config_path.exists() and config_path.suffix in ['.yaml', '.yml']:
            try:
                return self.load_config_file(config_path)
            except Exception as e:
                print(f"Failed to load config file {config_path}: {e}")
                return None
        
        # Try locating within the configuration directory
        possible_paths = [
            self.config_dir / f"{name}.yaml",
            self.config_dir / f"{name}.yml",
            self.config_dir / f"**/{name}.yaml",
            self.config_dir / f"**/{name}.yml"
        ]
        
        for pattern in possible_paths:
            if '*' in str(pattern):
                matches = list(self.config_dir.glob(str(pattern.relative_to(self.config_dir))))
                if matches:
                    try:
                        return self.load_config_file(matches[0])
                    except Exception as e:
                        print(f"Failed to load config file {matches[0]}: {e}")
            elif pattern.exists():
                try:
                    return self.load_config_file(pattern)
                except Exception as e:
                    print(f"Failed to load config file {pattern}: {e}")
        
        return None

    def _lookup_candidates(self, name: str) -> List[Path]:
        """Return the file paths/patterns checked for a config name."""
        return [
            Path(name),
            self.config_dir / f"{name}.yaml",
            self.config_dir / f"{name}.yml",
            self.config_dir / f"**/{name}.yaml",
            self.config_dir / f"**/{name}.yml",
        ]

    def describe_lookup_failure(self, name: str, limit: int = 20) -> str:
        """Build a detailed lookup error for missing configuration names."""
        lines = [
            f"ERROR: Config file or template not found: {name}",
            f"Config root: {self.config_dir.resolve()}",
            "Tried paths/patterns:",
        ]
        for candidate in self._lookup_candidates(name):
            lines.append(f"  - {candidate}")

        available = self.list_configs()
        if available:
            matches = [config_name for config_name in available if name.lower() in config_name.lower()]
            if matches:
                lines.append("Matching configs:")
                for config_name in matches[:limit]:
                    lines.append(f"  - {config_name}")
                if len(matches) > limit:
                    lines.append(f"  ... plus {len(matches) - limit} more matches")
            lines.append("Available configs include:")
            for config_name in available[:limit]:
                lines.append(f"  - {config_name}")
            if len(available) > limit:
                lines.append(f"  ... plus {len(available) - limit} more")
        else:
            lines.append("No YAML config files were loaded from the config root.")

        lines.append("Tip: if configs were moved into a subdirectory, use either the short name shown above or the relative key such as m/xgboost/xgboost_standard.")
        return "\n".join(lines)
    
    def list_configs(self) -> List[str]:
        """List all available configurations"""
        return sorted(list(self.templates.keys()))
    
    def get_config_info(self, name: str) -> Optional[Dict]:
        """Get configuration information"""
        config = self.get_config(name)
        if config:
            return {
                'name': config.name,
                'description': config.description,
                'model': config.model.model_type,
                'feature': config.feature.feature_type,
                'n_folds': config.training.n_folds,
                # 'optimization': removed from codebase
            }
        return None
    
    def save_config(self, config: ExperimentConfig, path: str):
        """Save configuration to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            config.to_json(str(path))
        else:
            config.to_yaml(str(path))
    
    def create_config_from_params(self, **params) -> ExperimentConfig:
        """Create configuration from parameters"""
        # If a base configuration is specified, load it first
        base_config = None
        if 'config' in params:
            base_config = self.get_config(params.pop('config'))
        
        if base_config:
            config = base_config.copy()
        else:
            config = ExperimentConfig()
        
        # Apply parameter updates
        config = self._apply_params_to_config(config, params)
        
        return config
    
    def _apply_params_to_config(self, config: ExperimentConfig, params: Dict) -> ExperimentConfig:
        """Apply parameters to configuration"""
        for key, value in params.items():
            if '.' in key:
                # Handle nested parameters, e.g., model.hyperparameters.n_estimators
                parts = key.split('.')
                obj = config
                
                # Navigate to the target attribute's parent object
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        # If attribute does not exist, create a dictionary
                        setattr(obj, part, {})
                        obj = getattr(obj, part)
                
                # Set final value
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
                elif isinstance(obj, dict):
                    obj[parts[-1]] = value
            else:
                # Handle top-level parameters
                if hasattr(config, key):
                    setattr(config, key, value)
                # Special handling for common parameters
                elif key == 'model' and hasattr(config.model, 'model_type'):
                    config.model.model_type = value
                elif key == 'n_folds' and hasattr(config.training, 'n_folds'):
                    config.training.n_folds = value
                elif key == 'feature' and hasattr(config.feature, 'feature_type'):
                    config.feature.feature_type = value
                elif key == 'multi_target' and hasattr(config.data, 'multi_target_strategy'):
                    config.data.multi_target_strategy = value
                elif key == 'nan_handling' and hasattr(config.data, 'nan_handling'):
                    config.data.nan_handling = value
                # Optimization parameters removed from codebase
                # elif key == 'optimization': removed
                # elif key == 'n_trials': removed
        
        return config
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("Available configuration templates")
        print("="*60)
        
        # Group by model type
        configs_by_model = {}
        for name, config in self.templates.items():
            if '/' not in name:  # Show short names only
                model_type = config.model.model_type
                if model_type not in configs_by_model:
                    configs_by_model[model_type] = []
                configs_by_model[model_type].append((name, config))
        
        for model_type in sorted(configs_by_model.keys()):
            print(f"\n{model_type.upper()}")
            print("-" * 40)
            
            for name, config in sorted(configs_by_model[model_type]):
                desc = config.description[:40] + "..." if len(config.description) > 40 else config.description
                print(f"  - {name:<20} {desc}")
        
        print("\n" + "="*60)
        print(f"Total: {len(self.templates)} configuration templates")
        print("Usage: python automl.py train config=<template_name>")
        print("="*60 + "\n")


config_manager: Optional[DynamicConfigManager] = None


def _get_config_manager() -> DynamicConfigManager:
    """Create the global config manager only when template access is needed."""
    global config_manager
    if config_manager is None:
        config_manager = DynamicConfigManager()
    return config_manager


def get_config(name_or_params: Union[str, Dict]) -> ExperimentConfig:
    """
    Convenience function to get a configuration
    
    Args:
        name_or_params: configuration name or parameter dict
    
    Returns:
        configuration object
    """
    manager = _get_config_manager()
    if isinstance(name_or_params, str):
        return manager.get_config(name_or_params)
    else:
        return manager.create_config_from_params(**name_or_params)


def list_configs() -> List[str]:
    """List all available configurations"""
    return _get_config_manager().list_configs()


def save_config(config: ExperimentConfig, path: str):
    """Save configuration"""
    _get_config_manager().save_config(config, path)
