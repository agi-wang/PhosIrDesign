#!/usr/bin/env python3
"""
AutoML - Command-line interface for automated machine learning

Usage:
    automl train model=xgboost data=mydata.csv config=config.yaml
    automl predict model=saved_model.joblib data=test.csv
    automl validate config=config.yaml
    automl export model=xgboost target=wavelength format=onnx
"""

import sys
import os
from pathlib import Path
import argparse
import json
import yaml
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add CLI directory and project root to sys.path.
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT_DIR))

from phosirdesign.config.system import ExperimentConfig, ConfigValidator, ConfigManager
from phosirdesign.config.manager import DynamicConfigManager, get_config, list_configs, save_config
from phosirdesign.training.pipeline import TrainingPipeline
from phosirdesign.models.base import load_model
from phosirdesign.utils.run_manager import RunManager
from phosirdesign.utils.analysis import ResultsAnalyzer
from phosirdesign.utils.training_progress import monitor_context, should_use_tui
from phosirdesign.utils.workflow_progress import write_progress_event
import joblib


# ========================================
#           Argument Parser
# ========================================

class MLArgumentParser:
    """ML CLI argument parser"""
    
    @staticmethod
    def parse_args_string(args_string: str) -> Dict[str, Any]:
        """
        Parse a key=value formatted argument string
        
        Args:
            args_string: Argument string, e.g. "model=xgboost data=file.csv"
        
        Returns:
            Parsed argument dictionary
        """
        params = {}
        
        # Split arguments
        parts = args_string.split()
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                
                # Remove outer quotes if present
                if (value.startswith("'") and value.endswith("'")) or \
                   (value.startswith('"') and value.endswith('"')):
                    value = value[1:-1]
                
                # Try to infer value types
                # Special keys: name and project should stay strings
                if key in ['name', 'project']:
                    # Keep as string
                    pass
                # Boolean
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                # Number
                elif value.replace('.', '').replace('-', '').isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                # List
                elif value.startswith('[') and value.endswith(']'):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        # Try to fix single-quoted JSON
                        try:
                            fixed_value = value.replace("'", '"')
                            value = json.loads(fixed_value)
                        except:
                            # Fallback: comma-separated list
                            inner = value[1:-1].strip()
                            if inner:
                                value = [v.strip().strip("'\"") for v in inner.split(',')]
                            else:
                                value = []
                # Dict
                elif value.startswith('{') and value.endswith('}'):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        # Try to fix single-quoted JSON
                        try:
                            fixed_value = value.replace("'", '"')
                            value = json.loads(fixed_value)
                        except:
                            pass  # keep original
                # Special handling for models: allow comma-separated format
                elif key == 'models' and ',' in value:
                    value = [m.strip() for m in value.split(',')]
                
                params[key] = value
        
        return params
    
    @staticmethod
    def _parse_bool(value) -> bool:
        """Parse boolean values"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['true', 'yes', '1', 'on']
        return bool(value)
    
    @staticmethod
    def merge_params_to_config(config: ExperimentConfig, params: Dict[str, Any]) -> ExperimentConfig:
        """
        Merge parsed parameters into an ExperimentConfig
        
        Args:
            config: Base configuration
            params: Parameters to merge
        
        Returns:
            Updated configuration
        """
        for key, value in params.items():
            # Handle dotted nested keys
            if '.' in key:
                parts = key.split('.')
                obj = config
                
                # Navigate to nested object
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        print(f"WARNING: Unknown config key: {key}")
                        continue
                
                # Set value
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
            else:
                # Handle common top-level params
                if key == 'model':
                    config.model.model_type = value
                elif key == 'data':
                    config.data.data_path = value
                elif key == 'feature':
                    config.feature.feature_type = value
                elif key == 'folds':
                    config.training.n_folds = int(value) if isinstance(value, str) else value
                elif key == 'project':
                    config.logging.project_name = value
                elif key == 'target':
                    if isinstance(value, str):
                        config.data.target_columns = [value]
                    else:
                        config.data.target_columns = value
                elif key == 'save_curves':
                    # Save training curves
                    config.training.save_training_curves = MLArgumentParser._parse_bool(value)
                elif key == 'save_importance':
                    # Save feature importance
                    config.training.save_feature_importance = MLArgumentParser._parse_bool(value)
                elif key in ['test_data', 'test_data_path']:
                    # Evaluate on configured test data after training
                    config.data.test_data_path = value
                    print(f"   INFO: Test dataset set: {value}")
                elif key in ['nan_handling', 'nan', 'missing']:
                    # Missing value handling strategy
                    config.data.nan_handling = value
                    print(f"   INFO: Missing value handling: {value}")
                elif key in ['multi_target', 'multi_target_strategy', 'target_strategy']:
                    # Multi-target data selection strategy
                    config.data.multi_target_strategy = value
                    print(f"   INFO: Multi-target strategy: {value}")
                elif hasattr(config, key):
                    setattr(config, key, value)
        
        return config


# ========================================
#           Train Command
# ========================================

def train_command(args: List[str]):
    """Train command"""
    print("\n" + "="*60)
    print("AutoML Training System")
    print("="*60)
    from time import perf_counter as _pc
    _t0 = _pc()
    
    # Parse arguments (with type inference)
    parser = MLArgumentParser()
    params = parser.parse_args_string(' '.join(args))
    config_path = params.pop('config', None)
    name = params.get('name')
    project = params.get('project')
    # Detect train-all flag
    all_flag = any(flag in args for flag in ['-all', '--all'])
    
    # Parse NUMA and parallel parameters
    numa_enabled = parser._parse_bool(params.get('numa', False))
    cores_per_task = int(params.get('cores', 4)) if 'cores' in params else None
    parallel_tasks = int(params.get('parallel', 1)) if 'parallel' in params else 1
    bind_cpu = parser._parse_bool(params.get('bind_cpu', False))
    
    # Load or create configuration
    _t_conf_start = _pc()
    manager = None
    
    if config_path:
        # Only initialize config manager when a config is explicitly provided
        manager = DynamicConfigManager()
        config = manager.get_config(config_path)
        if config:
            print(f"INFO: Using config: {config_path}")
        else:
            print(manager.describe_lookup_failure(config_path))
            return 1
    else:
        # Use built-in default configuration without scanning YAML templates
        config = ExperimentConfig()
        print("INFO: Using built-in default config")
    
    # Merge CLI params
    config = parser.merge_params_to_config(config, params)
    _t_conf_end = _pc(); conf_secs = _t_conf_end - _t_conf_start
    
    # Check multi-model training
    models_to_train = []
    if all_flag:
        # Train all supported models
        from phosirdesign.models import ModelFactory
        models_to_train = ModelFactory.get_supported_models()
        print("INFO: Enabled train-all mode")
        print(f"   Models to train: {len(models_to_train)} -> {models_to_train}")
    elif 'models' in params and params['models']:
        # Get model list from CLI
        if isinstance(params['models'], list):
            models_to_train = params['models']
        elif isinstance(params['models'], str):
            # Support comma-separated list
            models_to_train = [m.strip() for m in params['models'].split(',')]
        print("INFO: Multi-model training mode")
        print(f"   Models to train: {len(models_to_train)} -> {models_to_train}")
    
    # Save model list into config (used for subsequent training)
    if models_to_train:
        config.models_to_train = models_to_train
    
    # Create run directory
    _t_run_dir_start = _pc()
    # If project specified, use it as base dir; otherwise default runs
    if project:
        run_manager = RunManager(base_dir=project, task="train")
        run_dir = run_manager.get_next_run_dir(name=name, project=None)  # project already used as base_dir
        # Keep full directory structure for project
        config.logging.base_dir = str(run_dir.parent)
        config.logging.project_name = run_dir.name
    else:
        run_manager = RunManager(task="train")
        run_dir = run_manager.get_next_run_dir(name=name, project=None)
        config.logging.base_dir = str(run_dir.parent)
        config.logging.project_name = run_dir.name
    
    # Show configuration summary
    print(f"\n" + "="*60)
    print("Configuration")
    print("="*60)
    
    # Data configuration
    print("\nData Configuration:")
    print(f"   Training data: {config.data.data_path}")
    data_path = Path(config.data.data_path)
    if data_path.exists():
        print(f"   INFO: Training data found ({data_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"   ERROR: Training data not found!")
    
    # Test data configuration
    if hasattr(config.data, 'test_data_path') and config.data.test_data_path:
        print(f"   Test data: {config.data.test_data_path}")
        test_path = Path(config.data.test_data_path)
        if test_path.exists():
            print(f"   INFO: Test data found ({test_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"   WARNING: Invalid test data path: {test_path}")
            # Try alternative paths
            alt_paths = [
                Path(test_path.name),
                Path("../data") / test_path.name,
                Path("data") / test_path.name
            ]
            for alt in alt_paths:
                if alt.exists():
                    print(f"   TIP: Found file at: {alt}")
                    config.data.test_data_path = str(alt)
                    break
    else:
        print("   Test data: not specified")
    
    print(f"   Target columns: {config.data.target_columns}")
    print(f"   Multi-target strategy: {config.data.multi_target_strategy}")
    if config.data.multi_target_strategy == "intersection":
        print(f"     -> Use rows where all targets have values (strict)")
    elif config.data.multi_target_strategy == "independent":
        print(f"     -> Each target uses its own valid rows (default)")
    elif config.data.multi_target_strategy == "union":
        print(f"     -> Use all rows with missing values filled")
    print(f"   Missing value handling: {config.data.nan_handling}")
    if config.data.nan_handling != "skip":
        print(f"     - Feature NaN strategy: {config.data.feature_nan_strategy}")
        print(f"     - Target NaN strategy: {config.data.target_nan_strategy}")
    
    # Model configuration
    print("\nModel Configuration:")
    print(f"   Model type: {config.model.model_type}")
    print(f"   Cross validation: {config.training.n_folds}-fold")
    if config.model.hyperparameters:
        print("   Hyperparameters:")
        for key, value in config.model.hyperparameters.items():
            print(f"     - {key}: {value}")
    
    # Feature configuration
    print("\nFeature Configuration:")
    print(f"   Feature type: {config.feature.feature_type}")
    if hasattr(config.feature, 'morgan_bits'):
        print(f"   Morgan bits: {config.feature.morgan_bits}")
    if hasattr(config.feature, 'morgan_radius'):
        print(f"   Morgan radius: {config.feature.morgan_radius}")
    print(f"   Cache: {'enabled' if config.feature.use_cache else 'disabled'}")
    
    # Output configuration
    print("\nOutput Configuration:")
    print(f"   Project directory: {run_dir}")
    print(f"   Models: {run_dir}/models/")
    print(f"   Exports: {run_dir}/exports/")
    print(f"   Feature importance: {run_dir}/feature_importance/")
    
    print("\n" + "="*60)
    if hasattr(config, 'models_to_train') and config.models_to_train:
        print(f"   Multi-model training: enabled")
        print(f"   Models to train: {len(config.models_to_train)}")
        print(f"   Model list: {', '.join(config.models_to_train[:5])}{'...' if len(config.models_to_train) > 5 else ''}")
    if numa_enabled:
        print(f"   NUMA optimization: enabled")
        print(f"   Parallel tasks: {parallel_tasks}")
        if cores_per_task:
            print(f"   Cores per task: {cores_per_task}")
    print(f"   Run directory: {run_dir}")
    
    # Validate config
    _t_validate_start = _pc()
    if not ConfigValidator.validate_all(config):
        return 1
    _t_validate_end = _pc(); validate_secs = _t_validate_end - _t_validate_start
    
    # Run training
    _t_train_start = _pc()
    monitor_models = config.models_to_train if getattr(config, 'models_to_train', None) else [config.model.model_type]
    monitor_targets = list(config.data.target_columns)
    progress_file = os.environ.get("TRAIN_PROGRESS_FILE")
    progress_monitor = monitor_context(
        monitor_models,
        monitor_targets,
        log_path=os.environ.get("TRAIN_LOG_FILE"),
        enabled=should_use_tui() and not progress_file,
    )
    progress_monitor.__enter__()
    try:
        # Check if multiple models need to be trained
        if hasattr(config, 'models_to_train') and config.models_to_train:
            # Multi-model mode
            if parallel_tasks > 1:
                print(f"\nINFO: Starting parallel training with {parallel_tasks} concurrent tasks")
                results = parallel_train_models(
                    config, run_dir, 
                    numa_enabled, cores_per_task, parallel_tasks, bind_cpu
                )
            else:
                print(f"\nINFO: Serial training of {len(config.models_to_train)} models...")
                all_results = []
                
                for i, model_type in enumerate(config.models_to_train, 1):
                    model_start = _pc()
                    progress_monitor.model_started(model_type, i, len(config.models_to_train))
                    write_progress_event(progress_file, {
                        "type": "model_started",
                        "model": model_type,
                        "index": i,
                        "total": len(config.models_to_train),
                    })
                    completed = [r['model'] for r in all_results if r.get('success')]
                    failed = [r['model'] for r in all_results if not r.get('success')]
                    print(f"\n[{i}/{len(config.models_to_train)}] Training model: {model_type}", flush=True)
                    print(f"Completed models: {completed if completed else 'none'}", flush=True)
                    print(f"Failed models: {failed if failed else 'none'}", flush=True)
                    print("-" * 40, flush=True)
                    
                    # Create model-specific config (deep copy)
                    import copy
                    model_config = copy.deepcopy(config)
                    model_config.model.model_type = model_type
                    
                    # Reset hypers to model-specific defaults to avoid mixing params
                    from phosirdesign.models.base import MODEL_PARAMS
                    if model_type in MODEL_PARAMS:
                        model_config.model.hyperparameters = MODEL_PARAMS[model_type].copy()
                    else:
                        model_config.model.hyperparameters = {}
                    
                    # Fix config objects after deep copy
                    from phosirdesign.config.system import ComparisonConfig, ExportConfig
                    if isinstance(model_config.comparison, dict):
                        model_config.comparison = ComparisonConfig(**model_config.comparison)
                    if isinstance(model_config.export, dict):
                        model_config.export = ExportConfig(**model_config.export)
                    
                    # Create unified AutoML directory structure
                    automl_dir = run_dir / "automl_train"
                    automl_dir.mkdir(parents=True, exist_ok=True)
                    model_run_dir = automl_dir / model_type
                    model_run_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Update logging config to use parent directory
                    model_config.logging.base_dir = str(run_dir.parent)
                    model_config.logging.project_name = f"{run_dir.name}/automl_train/{model_type}"
                    
                    try:
                        # Set CPU affinity if enabled
                        if numa_enabled and cores_per_task:
                            setup_cpu_affinity(0, cores_per_task, bind_cpu)
                            if 'n_jobs' in model_config.model.hyperparameters:
                                model_config.model.hyperparameters['n_jobs'] = cores_per_task
                        
                        # Train model
                        pipeline = TrainingPipeline(model_config)
                        results = pipeline.run()
                        all_results.append({'model': model_type, 'success': True, 'results': results})
                        model_elapsed = _pc() - model_start
                        progress_monitor.model_completed(model_type, model_elapsed)
                        write_progress_event(progress_file, {
                            "type": "model_completed",
                            "model": model_type,
                            "elapsed": model_elapsed,
                        })
                        print(
                            f"SUCCESS: {model_type} training completed in {_pc() - model_start:.2f}s",
                            flush=True,
                        )
                        
                    except Exception as e:
                        progress_monitor.model_failed(model_type, str(e))
                        write_progress_event(progress_file, {
                            "type": "model_failed",
                            "model": model_type,
                            "error": str(e),
                        })
                        print(
                            f"ERROR: {model_type} training failed after {_pc() - model_start:.2f}s: {e}",
                            flush=True,
                        )
                        all_results.append({'model': model_type, 'success': False, 'error': str(e)})
                
                # Aggregate results
                results = all_results
        
        else:
            # Single-model training
            progress_monitor.model_started(config.model.model_type, 1, 1)
            write_progress_event(progress_file, {
                "type": "model_started",
                "model": config.model.model_type,
                "index": 1,
                "total": 1,
            })
            if numa_enabled and cores_per_task:
                # Set CPU affinity
                setup_cpu_affinity(0, cores_per_task, bind_cpu)
                # Update n_jobs parameter
                if 'n_jobs' in config.model.hyperparameters:
                    config.model.hyperparameters['n_jobs'] = cores_per_task
            
            pipeline = TrainingPipeline(config)
            results = pipeline.run()
            model_elapsed = _pc() - _t_train_start
            progress_monitor.model_completed(config.model.model_type, model_elapsed)
            write_progress_event(progress_file, {
                "type": "model_completed",
                "model": config.model.model_type,
                "elapsed": model_elapsed,
            })
        
        _t_train_end = _pc(); train_secs = _t_train_end - _t_train_start
        total_secs = _pc() - _t0
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        
        # If comparison table generation is enabled
        _t_table_start = _pc()
        if (hasattr(config, 'comparison') and hasattr(config.comparison, 'enable') and config.comparison.enable and
            hasattr(config, 'models_to_train') and config.models_to_train):
            print("\nGenerating model comparison table...")
            try:
                from phosirdesign.utils.comparison_table import ComparisonTableGenerator
                
                # Create table generator
                generator = ComparisonTableGenerator(str(run_dir))
                
                # Collect all results
                df_comparison = generator.collect_all_results()
                
                if not df_comparison.empty:
                    # Export all formats
                    formats = config.comparison.formats if hasattr(config.comparison, 'formats') else ['markdown', 'csv']
                    output_files = generator.export_all_formats(
                        output_dir=str(run_dir),
                        formats=formats
                    )
                    
                    print("INFO: Comparison tables generated:")
                    for fmt, path in output_files.items():
                        print(f"   - {fmt}: {Path(path).name}")
                else:
                    print("WARNING: Not enough results to generate comparison table")
                    
            except Exception as e:
                print(f"ERROR: Failed to generate comparison table: {e}")
        _t_table_end = _pc(); table_secs = _t_table_end - _t_table_start
        
        # If test set provided, show summary
        if hasattr(config.data, 'test_data_path') and config.data.test_data_path:
            print("\nTest set evaluation summary:")
            print("   Test file: " + Path(config.data.test_data_path).name)
            print("   Note: See per-target evaluation above for details")
        
        # Save run info
        run_manager.save_run_info(
            run_dir, 
            config.to_dict(),
            command=' '.join(['automl', 'train'] + args)
        )
        
        # Create symlink to latest run
        RunManager.create_symlink(run_dir, "last")
        
        # Save config
        config_save_path = run_dir / "config.yaml"
        config.to_yaml(str(config_save_path))
        print(f"INFO: Results saved in: {run_dir}")
        print(f"   See exports: {run_dir}/exports/")
        print(f"   See reports: {run_dir}/exports/*.html")
        print(f"   See models: {run_dir}/models/")

        # Training phase timing (summary + detail)
        try:
            # Attempt to write timing to logger (if available)
            if 'training' in locals() or 'pipeline' in locals():
                # Internal pipeline logger exists during run; append global timing to summary file only
                pass
            timing_summary = {
                'startup_to_end': total_secs,
                'config_prepare': conf_secs,
                'validate': validate_secs,
                'training_all': train_secs,
                'comparison_tables': table_secs,
            }
            import json as __json
            with open(_Path(run_dir) / 'timing_summary.json', 'w') as f:
                __json.dump(timing_summary, f, indent=2)
            print(f"   INFO: Timing summary saved: {run_dir}/timing_summary.json")

            # Fine-grained: aggregate each experiment into timing_detail.json (if logger-exported JSON exists)
            try:
                detail = {}
                exp_dir = _Path(run_dir) / 'training_logs' / run_dir.name / 'experiments'
                if exp_dir.exists():
                    for p in exp_dir.glob('*_complete.json'):
                        try:
                            with open(p, 'r') as f:
                                exp = __json.load(f)
                            exp_id = exp.get('experiment_id', p.stem.replace('_complete', ''))
                            detail[exp_id] = exp.get('timing', {})
                        except Exception:
                            continue
                with open(_Path(run_dir) / 'timing_detail.json', 'w') as f:
                    __json.dump(detail, f, indent=2, ensure_ascii=True)
                print(f"   INFO: Fine-grained timing saved: {run_dir}/timing_detail.json")
            except Exception:
                pass
        except Exception:
            pass

        # Report package assembly (enabled for comparison workflows)
        try:
            is_report_mode = (config.name.lower().startswith('report_comparison') if hasattr(config, 'name') else False)
        except Exception:
            is_report_mode = False

        should_make_report_package = False
        try:
            if hasattr(config, 'comparison') and hasattr(config.comparison, 'enable'):
                should_make_report_package = bool(config.comparison.enable)
        except Exception:
            pass
        should_make_report_package = should_make_report_package or is_report_mode

        if should_make_report_package:
            try:
                from phosirdesign.utils.comparison_table import ComparisonTableGenerator
                from pathlib import Path as _Path
                import shutil as _shutil
                import json as _json
                report_dir = _Path(run_dir) / 'report_package'
                report_dir.mkdir(parents=True, exist_ok=True)

                # 1) Export tables (four formats)
                generator = ComparisonTableGenerator(str(run_dir))
                exported = generator.export_all_formats(output_dir=str(report_dir), formats=['markdown','html','latex','csv'])

                # 2) Generate publication-style figures (with data)
                try:
                    from scripts.generate_publication_figures import generate_all_figures
                    data_path = config.data.data_path if hasattr(config, 'data') else 'data/PhosIrDB.csv'
                    generate_all_figures(str(run_dir), data_path, str(report_dir))
                except Exception as e:
                    print(f"WARNING: Failed to generate publication-style figures: {e}")

                # 3) Preserve test set raw predictions and ground truth (if available)
                try:
                    from pathlib import Path as __Path
                    exports_dir = __Path(run_dir) / 'exports'
                    if exports_dir.exists():
                        for f in exports_dir.glob('test_predictions_*.csv'):
                            _shutil.copy(f, report_dir / f.name)
                        for f in exports_dir.glob('test_metrics_*.json'):
                            _shutil.copy(f, report_dir / f.name)
                except Exception:
                    pass

                # 4) Summarize files and configuration
                import numpy as _np

                # Define a JSON encoder to handle numpy types
                class NumpyEncoder(_json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, (_np.integer, _np.int64)):
                            return int(obj)
                        elif isinstance(obj, (_np.floating, _np.float64)):
                            return float(obj)
                        elif isinstance(obj, _np.ndarray):
                            return obj.tolist()
                        return super().default(obj)

                summary = {
                    'project': str(run_dir.name),
                    'path': str(run_dir),
                    'comparison_tables': {k: _Path(v).name for k, v in exported.items()},
                    'best_models': generator.get_best_models() if exported else {},
                }
                with open(report_dir / 'summary.json', 'w', encoding='utf-8') as f:
                    _json.dump(summary, f, indent=2, ensure_ascii=True, cls=NumpyEncoder)

                # Save final config copy
                try:
                    _shutil.copy(_Path(run_dir) / 'config.yaml', report_dir / 'config.yaml')
                except Exception:
                    pass

                # 5) Optional homepage index.html
                try:
                    index_path = report_dir / 'index.html'
                    index_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head><meta charset='utf-8'><title>Report Package</title></head>
                    <body>
                      <h1>Report Package</h1>
                      <ul>
                        <li><a href="{_Path(exported.get('html','')).name if exported else ''}">Comparison Table (HTML)</a></li>
                        <li><a href="summary.json">Summary (JSON)</a></li>
                        <li><a href="../timing_summary.json">Timing Summary</a></li>
                        <li><a href="../timing_detail.json">Timing Detail</a></li>
                        <li><a href="figure_c_wavelength_plqy.png">Figure C</a></li>
                        <li><a href="figure_d_plqy_distribution.png">Figure D</a></li>
                        <li><a href="figure_e_f_predictions.png">Figure E-F</a></li>
                        <li><a href="figure_g_plqy_accuracy.png">Figure G</a></li>
                      </ul>
                    </body>
                    </html>
                    """
                    with open(index_path, 'w', encoding='utf-8') as f:
                        f.write(index_html)
                except Exception:
                    pass

                # Append timing to summary
                try:
                    import json as ___json
                    import numpy as _np
                    
                    # Define a JSON encoder to handle numpy types
                    class NumpyEncoder(___json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, (_np.integer, _np.int64)):
                                return int(obj)
                            elif isinstance(obj, (_np.floating, _np.float64)):
                                return float(obj)
                            elif isinstance(obj, _np.ndarray):
                                return obj.tolist()
                            return super().default(obj)
                    
                    s_path = report_dir / 'summary.json'
                    if s_path.exists():
                        data = ___json.load(open(s_path, 'r'))
                    else:
                        data = {}
                    data['timing'] = timing_summary if 'timing_summary' in locals() else {}
                    with open(s_path, 'w') as f:
                        ___json.dump(data, f, indent=2, ensure_ascii=True, cls=NumpyEncoder)
                except Exception:
                    pass

                print(f"\nINFO: Report package generated: {report_dir}")
            except Exception as e:
                print(f"WARNING: Failed to create report package: {e}")

        
        # Print example prediction commands (for all models produced in this run: single-ligand/multi-ligand)
        try:
            models_dir = run_dir / "models"
            model_paths = []
            if models_dir.exists():
                model_paths = sorted(
                    [p for p in models_dir.glob("*.joblib")],
                    key=lambda p: p.stat().st_mtime
                )
            # Fallback: search for all joblib files under run_dir
            if not model_paths:
                model_paths = sorted(
                    [p for p in run_dir.glob("**/*.joblib")],
                    key=lambda p: p.stat().st_mtime
                )
            if model_paths:
                print("\nExample prediction commands (copy to run; listed by model):")
                for mp in model_paths:
                    print(f"  # {mp.name}")
                    # Quote path if it contains special characters
                    model_param = f"model={mp}"
                    if any(char in str(mp) for char in ['(', ')', '[', ']', '{', '}', ' ', '*', '?']):
                        model_param = f'"model={mp}"'
                    
                    # Single sample: use real example ligands L1/L2/L3
                    print(f"  python automl.py predict {model_param} input='[[\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"C1=CN=C(C2=CN(CCCCCCN3C4=CC=CC=C4C4=C3C=CC=C4)N=N2)C=C1\"]]' feature=combined")
                    # Two samples: repeat the triple as the second sample
                    print(f"  python automl.py predict {model_param} input='[[\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"C1=CN=C(C2=CN(CCCCCCN3C4=CC=CC=C4C4=C3C=CC=C4)N=N2)C=C1\"],[\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"C1=CN=C(C2=CN(CCCCCCN3C4=CC=CC=C4C4=C3C=CC=C4)N=N2)C=C1\"]]' feature=combined")
        except Exception:
            pass
        
        progress_monitor.__exit__(None, None, None)
        return 0
        
    except KeyboardInterrupt:
        progress_monitor.__exit__(*sys.exc_info())
        print("\nINFO: Interrupted; saving completed parts and printing example commands...")
        try:
            # Try to save run info and configuration
            run_manager.save_run_info(
                run_dir,
                config.to_dict(),
                command=' '.join(['automl', 'train'] + args)
            )
            RunManager.create_symlink(run_dir, "last")
            config_save_path = run_dir / "config.yaml"
            config.to_yaml(str(config_save_path))
        except Exception:
            pass
        # Try to print example prediction commands for currently available models
        try:
            models_dir = run_dir / "models"
            model_paths = []
            if models_dir.exists():
                model_paths = sorted(
                    [p for p in models_dir.glob("*.joblib")],
                    key=lambda p: p.stat().st_mtime
                )
            if model_paths:
                print("\nINFO: Example prediction commands for completed models:")
                for mp in model_paths:
                    print(f"  # {mp.name}")
                    print(f"  python automl.py predict model={mp} input='[[\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"[C-]1=C(C2=NC=CC3=CC=CC=C23)C=CC=C1\",\"C1=CN=C(C2=CN(CCCCCCN3C4=CC=CC=C4C4=C3C=CC=C4)N=N2)C=C1\"]]' feature=combined")
            else:
                print("WARNING: No model files have been produced yet.")
        except Exception:
            pass
        return 130  # Common interrupt exit code

    except Exception as e:
        progress_monitor.__exit__(*sys.exc_info())
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ========================================
#           Prediction Command
# ========================================

def predict_command(args: List[str]):
    """Prediction command"""
    print("\n" + "="*60)
    print("AutoML Prediction System")
    print("="*60)
    
    # Parse arguments
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    
    # Check required params
    if 'model' not in params:
        print("ERROR: Missing model parameter: model=path/to/model.joblib")
        return 1
    
    if 'data' not in params and 'input' not in params:
        print("ERROR: Provide data via data=path/to/data.csv or input=['CCO','c1ccccc1']")
        return 1
    
    # Load model
    print(f"\nLoading model: {params['model']}")
    try:
        model = load_model(params['model'])
    except Exception as e:
        print(f"ERROR: Model loading failed: {e}")
        return 1
    
    # Infer training config (auto-align feature type and SMILES columns)
    training_feature_type = None
    training_smiles_columns = None
    training_morgan_bits = None
    training_morgan_radius = None
    training_descriptor_count = None
    try:
        model_path = Path(params['model']).resolve()
        # Common save location: runs/.../models/*.joblib -> runs/.../config.yaml
        run_dir = model_path.parent.parent if model_path.parent.name == 'models' else model_path.parent
        config_candidates = [run_dir / 'config.yaml', run_dir / 'experiment_config.yaml']
        for cfg in config_candidates:
            if cfg.exists():
                try:
                    from phosirdesign.config.system import ExperimentConfig
                    cfg_obj = ExperimentConfig.from_yaml(str(cfg)) if cfg.suffix in ['.yml', '.yaml'] else ExperimentConfig.from_json(str(cfg))
                    training_feature_type = str(getattr(cfg_obj.feature, 'feature_type', None)).lower()
                    training_smiles_columns = list(getattr(cfg_obj.data, 'smiles_columns', []))
                    training_morgan_bits = getattr(cfg_obj.feature, 'morgan_bits', None)
                    training_morgan_radius = getattr(cfg_obj.feature, 'morgan_radius', None)
                    training_descriptor_count = getattr(cfg_obj.feature, 'descriptor_count', None)
                    break
                except Exception:
                    pass
    except Exception:
        pass
    
    # Resolve feature type
    feature_param = params.get('feature')
    if feature_param is None or str(feature_param).lower() == 'auto':
        feature_type = (training_feature_type or 'combined').lower()
        if training_feature_type:
            print(f"INFO: Auto-set feature type from training config: {feature_type}")
    else:
        feature_type = str(feature_param).lower()
    
    # Resolve SMILES columns
    smiles_param = params.get('smiles_columns')
    if smiles_param:
        resolved_smiles_cols = [c.strip() for c in smiles_param.split(',') if c.strip()]
        print(f"INFO: Using specified SMILES columns: {','.join(resolved_smiles_cols)}")
    else:
        resolved_smiles_cols = training_smiles_columns or ['L1', 'L2', 'L3']
        if training_smiles_columns:
            print(f"INFO: Auto-set SMILES columns from training config: {','.join(resolved_smiles_cols)}")
    expected_ligand_count = len(resolved_smiles_cols)
    
    # Output column name
    output_column = params.get('output_column', 'Prediction')
    
    # Batch parameters
    batch_size = int(params.get('batch_size', '1000'))
    show_progress = params.get('show_progress', 'true').lower() in ['true', '1', 'yes']
    skip_errors = params.get('skip_errors', 'true').lower() in ['true', '1', 'yes']
    
    # Prepare features
    print("\nPreparing features...")
    from phosirdesign.core.feature_extractor import FeatureExtractor
    X = None
    df = None
    
    # Allow specifying morgan_bits/morgan_radius (aliases bits/radius)
    morgan_bits = params.get('morgan_bits', params.get('bits'))
    morgan_radius = params.get('morgan_radius', params.get('radius'))
    try:
        morgan_bits = int(morgan_bits) if morgan_bits is not None else None
    except ValueError:
        morgan_bits = None
    try:
        morgan_radius = int(morgan_radius) if morgan_radius is not None else None
    except ValueError:
        morgan_radius = None
    # If not provided, auto-set from training config
    if morgan_bits is None and training_morgan_bits is not None:
        morgan_bits = int(training_morgan_bits)
        print(f"INFO: Auto-set morgan_bits from training config: {morgan_bits}")
    if morgan_radius is None and training_morgan_radius is not None:
        morgan_radius = int(training_morgan_radius)
        print(f"INFO: Auto-set morgan_radius from training config: {morgan_radius}")
    feature_extractor = FeatureExtractor(use_cache=True, morgan_bits=morgan_bits, morgan_radius=morgan_radius, descriptor_count=training_descriptor_count)
    
    if 'input' in params:
        raw_input = params['input']
        user_input = None
        if isinstance(raw_input, list):
            user_input = raw_input
        else:
            try:
                user_input = json.loads(raw_input)
            except Exception:
                # Fallback: split string by comma
                user_input = [s.strip() for s in str(raw_input).split(',') if s.strip()]
        
        print("INFO: Using inline input for prediction")
        if feature_type in ['morgan', 'descriptors', 'combined']:
            # Normalize to a SMILES list per sample
            samples = []
            if all(isinstance(x, str) for x in user_input):
                samples = [[s] for s in user_input]
            elif all(isinstance(x, (list, tuple)) for x in user_input):
                samples = [list(sample) for sample in user_input]
            else:
                print("ERROR: Unsupported input format. For molecular features, use ['SMI', ...] or [['L1','L2'], ...]")
                return 1
            
            # Auto pad/truncate ligand count based on training requirement
            if expected_ligand_count > 0:
                adjusted = False
                for i in range(len(samples)):
                    if len(samples[i]) < expected_ligand_count:
                        samples[i] = samples[i] + [None] * (expected_ligand_count - len(samples[i])
                        )
                        adjusted = True
                    elif len(samples[i]) > expected_ligand_count:
                        samples[i] = samples[i][:expected_ligand_count]
                        adjusted = True
                if adjusted:
                    print(f"INFO: Ligand count aligned to training config: expected {expected_ligand_count}; auto padded/truncated")

            features = []
            for smiles_list in samples:
                feat = feature_extractor.extract_combination(
                    smiles_list,
                    feature_type=feature_type,
                    combination_method='mean'
                )
                features.append(feat)
            X = np.array(features)
            # Keep a minimal df for export
            df = pd.DataFrame({'L1_L2_L3': [','.join([s for s in sm if s is not None]) for sm in samples]})
        else:
            # tabular/auto: use numeric/array values directly
            arr = np.array(user_input, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X = arr
            df = pd.DataFrame({'row': list(range(len(X)))}
            )
    else:
        # Read from CSV - use batch optimization
        print(f"Loading data: {params['data']}")
        try:
            df = pd.read_csv(params['data'])
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        except Exception as e:
            print(f"ERROR: Data loading failed: {e}")
            return 1
        
        # Check SMILES columns existence
        missing_cols = [col for col in resolved_smiles_cols if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Missing columns {missing_cols}; using None values")
        
        if feature_type in ['morgan', 'descriptors', 'combined']:
            # Enhanced batch prediction (with file cache)
            print(f"\nINFO: Using batch mode (batch_size={batch_size})")
        
            # Check whether to use file cache
            use_file_cache = params.get('use_file_cache', 'true').lower() in ['true', '1', 'yes']
            file_cache_dir = params.get('file_cache_dir', 'file_feature_cache')
            
            # Use V2 batch predictor
            from phosirdesign.utils.batch_predictor_v2 import BatchPredictorV2
            
            predictor = BatchPredictorV2(
                batch_size=batch_size,
                show_progress=show_progress,
                skip_errors=skip_errors,
                use_file_cache=use_file_cache,
                file_cache_dir=file_cache_dir
            )
            
            predictions, failed_indices = predictor.predict_with_cache(
                df=df,
                model=model,
                feature_extractor=feature_extractor,
                smiles_columns=resolved_smiles_cols,
                feature_type=feature_type,
                combination_method='mean',
                input_file=params['data']  # Pass file path for caching
            )
            
            # Append prediction column to original dataframe
            df[output_column] = predictions
            
            # Show statistics
            stats = predictor.get_statistics(predictions)
            print(f"\nPrediction statistics:")
            print(f"   Success: {stats['count']} / {len(df)} ({stats['success_rate']:.1f}%)")
            if stats['count'] > 0:
                print(f"   Min: {stats['min']:.4f}")
                print(f"   Max: {stats['max']:.4f}")
                print(f"   Mean: {stats['mean']:.4f}")
                print(f"   Std: {stats['std']:.4f}")
            
            # Save error log
            if failed_indices and skip_errors:
                error_file = params.get('output', 'predictions.csv').replace('.csv', '_errors.log')
                predictor.save_error_log(error_file)
            
            # Skip subsequent prediction steps; save directly
            output_path = params.get('output', None)
            
            # If output not specified, use default filename and overwrite
            if output_path is None:
                output_path = 'predictions.csv'
            
            df.to_csv(output_path, index=False)
            
            # Get absolute path
            from pathlib import Path
            abs_path = Path(output_path).absolute()
            
            print(f"\nPrediction results saved:")
            print(f"   File: {output_path}")
            print(f"   Absolute path: {abs_path}")
            print(f"   Preserved all {len(df.columns)} columns")
            
            # Show preview
            print(f"\nPrediction preview:")
            preview_df = df.copy()
            
            # Limit SMILES display length
            for col in resolved_smiles_cols:
                if col in preview_df.columns:
                    preview_df[col] = preview_df[col].apply(
                        lambda x: str(x)[:30] + '...' if isinstance(x, str) and len(str(x)) > 30 else x
                    )
            
            # Show head and tail
            print("-" * 80)
            if len(preview_df) <= 20:
                print(preview_df.to_string(index=False))
            else:
                print("Head (first 5):")
                print(preview_df.head(5).to_string(index=False))
                print("\nTail (last 5):")
                print(preview_df.tail(5).to_string(index=False))
                print(f"\n(total {len(preview_df)} rows)")
            print("-" * 80)
            
            return 0
        else:
            # tabular or auto mode
            target_cols = []
            if 'target' in params:
                target_cols = [t.strip() for t in str(params['target']).split(',') if t.strip()]
            X = feature_extractor.extract_from_dataframe(
                df,
                target_columns=target_cols or None,
                feature_type=feature_type
            )
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"   Feature shape: {X.shape}")
    
    # Predict
    print("\nRunning prediction...")
    try:
        predictions = model.predict(X)
        print(f"   Predictions complete: {len(predictions)} samples")
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        return 1
    
    # Save results - keep original columns
    output_path = params.get('output', None)
    
    # If output not specified, use default filename and overwrite
    if output_path is None:
        output_path = 'predictions.csv'
    
    if df is None:
        df = pd.DataFrame()
    
    # Use user-specified output column name
    df[output_column] = predictions
    df.to_csv(output_path, index=False)
    
    # Get absolute path
    from pathlib import Path
    abs_path = Path(output_path).absolute()
    
    print(f"\nPrediction results saved:")
    print(f"   File: {output_path}")
    print(f"   Absolute path: {abs_path}")
    print(f"   Preserved all {len(df.columns)} columns")
    
    # Show statistics
    print(f"\nPrediction statistics:")
    print(f"   Min: {predictions.min():.4f}")
    print(f"   Max: {predictions.max():.4f}")
    print(f"   Mean: {predictions.mean():.4f}")
    print(f"   Std: {predictions.std():.4f}")
    
    # Show predictions preview
    print(f"\nPrediction preview:")
    # If original data identifiers exist, display them as well
    preview_df = df.copy() if df is not None else pd.DataFrame()
    
    # Select columns to display if present
    display_cols = []
    for col in ['Unnamed: 0', 'Abbreviation_in_the_article', 'L1', 'L2', 'L3']:
        if col in preview_df.columns:
            display_cols.append(col)
    
    # Limit SMILES display length
    if display_cols:
        preview_df = preview_df[display_cols].copy()
        for col in ['L1', 'L2', 'L3']:
            if col in preview_df.columns:
                preview_df[col] = preview_df[col].apply(lambda x: str(x)[:30] + '...' if isinstance(x, str) and len(str(x)) > 30 else x)
    
    preview_df['Prediction'] = predictions
    preview_df['Prediction'] = preview_df['Prediction'].round(4)
    
    # Print table
    print("-" * 80)
    if len(preview_df) <= 20:
        print(preview_df.to_string(index=False))
    else:
        print("Head (first 10):")
        print(preview_df.head(10).to_string(index=False))
        print("\nTail (last 10):")
        print(preview_df.tail(10).to_string(index=False))
        print(f"\n(total {len(preview_df)} rows)")
    print("-" * 80)
    
    return 0


# ========================================
#           Validate Command
# ========================================

def validate_command(args: List[str]):
    """Validate command - validate config files or data files"""
    print("\n" + "="*60)
    print("AutoML Validator")
    print("="*60)
    
    # Parse arguments
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    
    # Determine whether to validate data or config
    data_path = params.get('data')
    config_path = params.get('config')
    
    if data_path:
        # Validate data file
        print(f"\nValidating data file: {data_path}")
        
        # Check file exists
        if not Path(data_path).exists():
            print(f"ERROR: Data file not found: {data_path}")
            return 1
        
        try:
            # Load data
            import pandas as pd
            df = pd.read_csv(data_path)
            print(f"INFO: Data loaded successfully")
            
            # Show data info
            print("\nData info:")
            print("-" * 40)
            print(f"Rows: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            print(f"Column names: {', '.join(df.columns[:10])}")
            if len(df.columns) > 10:
                print(f"      ... plus {len(df.columns)-10} more columns")
            
            # Check required columns
            smiles_cols = ['L1', 'L2', 'L3']
            target_cols = ['Max_wavelength(nm)', 'PLQY', 'tau(s*10^-6)']
            
            print("\nChecking required columns...")
            
            # Check SMILES columns
            has_smiles = any(col in df.columns for col in smiles_cols)
            if has_smiles:
                found_smiles = [col for col in smiles_cols if col in df.columns]
                print(f"INFO: SMILES columns: {', '.join(found_smiles)}")
            else:
                print(f"WARNING: No SMILES columns found (expected: {', '.join(smiles_cols)})")
            
            # Check target columns
            has_targets = any(col in df.columns for col in target_cols)
            if has_targets:
                found_targets = [col for col in target_cols if col in df.columns]
                print(f"INFO: Target columns: {', '.join(found_targets)}")
            else:
                print(f"WARNING: No target columns found (expected: {', '.join(target_cols)})")
            
            # Check data quality
            print("\nData quality check:")
            print(f"Missing values total: {df.isnull().sum().sum()}")
            print(f"Duplicate rows: {df.duplicated().sum()}")
            
            # If SMILES columns exist, validate SMILES format
            if has_smiles:
                try:
                    from rdkit import Chem
                    invalid_count = 0
                    for col in found_smiles:
                        if col in df.columns:
                            # Sample up to 100 rows
                            sample = df[col].dropna().head(100)
                            for smiles in sample:
                                if pd.notna(smiles) and smiles != '':
                                    mol = Chem.MolFromSmiles(str(smiles))
                                    if mol is None:
                                        invalid_count += 1
                    if invalid_count > 0:
                        print(f"WARNING: Found {invalid_count} invalid SMILES")
                    else:
                        print(f"INFO: SMILES format check passed")
                except ImportError:
                    print("INFO: RDKit not installed; skipping SMILES validation")
            
            print("\nData validation completed!")
            return 0
            
        except Exception as e:
            print(f"ERROR: Data validation failed: {e}")
            return 1
    
    elif config_path:
        # Validate configuration file
        print(f"\nValidating configuration file: {config_path}")
        
        if not Path(config_path).exists():
            manager = DynamicConfigManager()
            print(manager.describe_lookup_failure(config_path))
            return 1
        
        try:
            if config_path.endswith('.yaml'):
                config = ExperimentConfig.from_yaml(config_path)
            else:
                config = ExperimentConfig.from_json(config_path)
            
            # Show configuration
            print("\nConfig content:")
            print("-" * 40)
            print(f"Name: {config.name}")
            print(f"Description: {config.description}")
            print(f"Model: {config.model.model_type}")
            print(f"Feature: {config.feature.feature_type}")
            print(f"Data: {config.data.data_path}")
            print(f"Targets: {config.data.target_columns}")
            print(f"Cross validation: {config.training.n_folds}-fold")
            
            # Validate configuration
            print("\nValidating configuration...")
            if ConfigValidator.validate_all(config):
                print("INFO: Configuration validation passed!")
                return 0
            else:
                print("ERROR: Configuration validation failed!")
                return 1
                
        except Exception as e:
            print(f"ERROR: Failed to load configuration: {e}")
            return 1
    
    else:
        # Default: look for config files
        if Path('config.yaml').exists():
            return validate_command(['config=config.yaml'])
        elif Path('config.json').exists():
            return validate_command(['config=config.json'])
        else:
            print("ERROR: Please specify a file to validate:")
            print("   Validate data: automl validate data=<data_file>")
            print("   Validate config: automl validate config=<config_file>")
            return 1


# ========================================
#           Export Command
# ========================================

def export_command(args: List[str]):
    """Export command"""
    print("\n" + "="*60)
    print("AutoML Model Export System")
    print("="*60)
    
    # Parse arguments
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    
    model_path = params.get('model')
    format_type = params.get('format', 'onnx')
    output_path = params.get('output', 'exported_model')
    
    if not model_path:
        print("ERROR: Missing model parameter: model=path/to/model.joblib")
        return 1
    
    # Load model
    print(f"\nLoading model: {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"ERROR: Model loading failed: {e}")
        return 1
    
    # Export model
    print(f"Exporting as {format_type} format...")
    
    if format_type == 'onnx':
        try:
            import skl2onnx
            from skl2onnx import convert_sklearn
            
            # Input shape information is required
            n_features = int(params.get('n_features', 1109))
            initial_type = [('float_input', skl2onnx.common.data_types.FloatTensorType([None, n_features]))]
            
            onx = convert_sklearn(model, initial_types=initial_type)
            
            with open(f"{output_path}.onnx", "wb") as f:
                f.write(onx.SerializeToString())
            
            print(f"INFO: Model exported: {output_path}.onnx")
            
        except ImportError:
            print("ERROR: skl2onnx is required: pip install skl2onnx")
            return 1
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
            return 1
    
    elif format_type == 'pmml':
        print("ERROR: PMML export not implemented")
        return 1
    
    elif format_type == 'pickle':
        import pickle
        with open(f"{output_path}.pkl", 'wb') as f:
            pickle.dump(model, f)
        print(f"INFO: Model exported: {output_path}.pkl")
    
    else:
        print(f"ERROR: Unsupported format: {format_type}")
        return 1
    
    return 0


# ========================================
#           Analysis Command
# ========================================

def analyze_command(args: List[str]):
    """Analyze experiment results"""
    print("\n" + "="*60)
    print("AutoML Results Analysis")
    print("="*60)
    
    # Parse arguments
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            params[key] = value
    
    # Get run directory
    run_dir = params.get('run_dir', params.get('dir', 'runs/train'))
    output_format = params.get('format', 'text')
    output_path = params.get('output')
    print_results = params.get('print', 'true').lower() == 'true'
    
    # Convert to Path
    run_dir = Path(run_dir)
    
    # If using 'last', find most recent run
    if str(run_dir) == 'last':
        # Find latest run directory
        if Path('runs/train').exists():
            run_dirs = sorted([d for d in Path('runs/train').iterdir() if d.is_dir() and d.name != 'last'])
            if run_dirs:
                run_dir = run_dirs[-1]
            else:
                print("ERROR: No training runs found")
                return 1
        else:
            print("ERROR: No training runs found")
            return 1
    
    # Check directory exists
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        print("\nAvailable run directories:")
        
        # List available run directories
        for base_dir in ['runs', '.']:
            base_path = Path(base_dir)
            if base_path.exists():
                for task_dir in base_path.iterdir():
                    if task_dir.is_dir() and not task_dir.name.startswith('.'):
                        sub_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name != 'last']
                        if sub_dirs:
                            print(f"  {task_dir}:")
                            for d in sorted(sub_dirs)[-5:]:  # show latest 5
                                print(f"    - {d}")
        return 1
    
    print(f"\nAnalyzing directory: {run_dir}")
    
    # Create analyzer
    try:
        analyzer = ResultsAnalyzer(run_dir)
    except Exception as e:
        print(f"ERROR: Failed to create analyzer: {e}")
        return 1
    
    # Generate report
    print(f"Generating {output_format.upper()} report...")
    
    try:
        # Save report
        if output_path:
            output_path = Path(output_path)
        analyzer.save_report(output_path=output_path, output_format=output_format)
        
        # Print to console
        if print_results:
            print("\n" + "="*60)
            print(analyzer.generate_report('text'))
            print("="*60)
        
        print("\nAnalysis completed!")
        return 0
        
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ========================================
#           Info Command
# ========================================

def info_command(args: List[str]):
    """Show system information"""
    print("\n" + "="*60)
    print("AutoML System Information")
    print("="*60)
    
    # System information
    print("\nSystem info:")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   Platform: {sys.platform}")
    
    # Available models
    from phosirdesign.models import ModelFactory
    print("\nAvailable models:")
    for model in ModelFactory.get_supported_models():
        print(f"   - {model}")
    
    # Available templates
    manager = ConfigManager()
    print("\nConfig templates:")
    for template in manager.list_templates():
        desc = manager.templates[template].description
        print(f"   - {template}: {desc}")
    
    # Feature types
    print("\nFeature types:")
    print("   - morgan: Morgan fingerprint")
    print("   - descriptors: molecular descriptors")
    print("   - combined: combined features")
    
    # Usage examples
    print("\nUsage examples:")
    print("   train: automl train model=xgboost data=data.csv config=config.yaml")
    print("   analyze: automl analyze dir=run_directory format=html")
    print("   predict: automl predict model=model.joblib data=test.csv")
    print("   validate: automl validate config=config.yaml")
    print("   export: automl export model=model.joblib format=onnx")
    
    return 0


# ========================================
#           NUMA and Parallel Support
# ========================================

def setup_cpu_affinity(task_id: int, cores_per_task: int, bind_cpu: bool = False):
    """
    Set CPU affinity and optional NUMA binding
    
    Args:
        task_id: Task ID
        cores_per_task: Number of cores per task
        bind_cpu: Whether to bind CPU
    """
    if not bind_cpu:
        return
    
    try:
        # Get system CPU info
        cpu_count = psutil.cpu_count(logical=True)
        
        # Compute core range
        core_start = (task_id * cores_per_task) % cpu_count
        core_end = min(core_start + cores_per_task, cpu_count)
        cores = list(range(core_start, core_end))
        
        # Set CPU affinity
        p = psutil.Process()
        p.cpu_affinity(cores)
        
        print(f"   INFO: CPU affinity set: task {task_id} -> cores {cores}")
        
    except Exception as e:
        print(f"   WARNING: Failed to set CPU affinity: {e}")


def get_numa_info():
    """Get NUMA information"""
    try:
        import subprocess
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'available:' in line and 'nodes' in line:
                    numa_nodes = int(line.split()[1])
                    return numa_nodes
    except:
        pass
    return 1


# ========================================
#           Warmup Cache Command
# ========================================

def warmup_command(args: List[str]):
    """Precompute and write feature cache (molecular/tabular) to reduce training-time overhead"""
    print("\n" + "="*60)
    print("AutoML Cache Warmup")
    print("="*60)

    # Parse key=value arguments
    params = {}
    for arg in args:
        if '=' in arg:
            k, v = arg.split('=', 1)
            params[k] = v

    # Required params
    data_path = params.get('data')
    if not data_path:
        print("ERROR: Missing parameter: data=path/to.csv")
        return 1

    feature_type = str(params.get('feature', 'auto')).lower()
    smiles_columns = params.get('smiles_columns')
    if smiles_columns:
        smiles_columns = [c.strip() for c in smiles_columns.split(',') if c.strip()]
    morgan_bits = params.get('morgan_bits', params.get('bits'))
    morgan_radius = params.get('morgan_radius', params.get('radius'))
    try:
        morgan_bits = int(morgan_bits) if morgan_bits is not None else None
        morgan_radius = int(morgan_radius) if morgan_radius is not None else None
    except Exception:
        morgan_bits = None
        morgan_radius = None

    # Concurrency params (warmup runs serially to avoid contention; n_jobs for intra-row parallel)
    n_jobs = int(params.get('n_jobs', 0))

    # Load data
    import pandas as pd
    import numpy as np
    from phosirdesign.core.feature_extractor import FeatureExtractor

    print(f"\nLoading data: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"ERROR: Data loading failed: {e}")
        return 1
    print(f"   Shape: {df.shape}")

    # Build feature extractor
    extractor = FeatureExtractor(
        feature_type=feature_type,
        use_cache=True,
        morgan_bits=morgan_bits,
        morgan_radius=morgan_radius
    )

    # Auto-detect SMILES columns
    if feature_type in ['morgan', 'descriptors', 'combined', 'auto']:
        if not smiles_columns:
            # For auto/molecular, guess from DF
            guessed = [col for col in df.columns if any(ind in col.lower() for ind in ['smiles','l1','l2','l3'])]
            smiles_columns = guessed or ['L1','L2','L3']

    print(f"   Feature type: {feature_type}")
    if smiles_columns:
        print(f"   SMILES columns: {','.join(smiles_columns)}")
    if morgan_bits:
        print(f"   morgan_bits: {morgan_bits}")
    if morgan_radius:
        print(f"   morgan_radius: {morgan_radius}")

    # Warmup: row-wise extraction
    from tqdm import tqdm
    total = len(df)
    errors = 0

    if feature_type in ['morgan', 'descriptors', 'combined'] or (
        feature_type == 'auto' and extractor.detect_data_type(df) == 'molecular'
    ):
        # Molecular path
        for _, row in tqdm(df.iterrows(), total=total, desc='Warm up molecular feature cache'):
            smiles_list = [row[col] if col in row and pd.notna(row[col]) else None for col in smiles_columns]
            try:
                _ = extractor.extract_combination(smiles_list, feature_type=feature_type if feature_type!='auto' else 'combined')
            except Exception:
                errors += 1
                continue
    else:
        # Tabular path: single-shot write (caches column-level feature names)
        try:
            _ = extractor.extract_from_dataframe(df, target_columns=[] if 'target' not in params else [params['target']])
        except Exception:
            errors += 1

    print(f"\nWarmup completed: {total - errors}/{total} records cached/hit")
    return 0

def train_single_model_parallel(args):
    """
    Worker function to train a single model in parallel
    
    Args:
        args: (config, model_type, task_id, numa_enabled, cores_per_task, bind_cpu)
    """
    config, model_type, task_id, numa_enabled, cores_per_task, bind_cpu = args
    
    # Set CPU affinity
    if numa_enabled and cores_per_task:
        setup_cpu_affinity(task_id, cores_per_task, bind_cpu)
    
    # Rebuild config object (from dict or config object)
    from phosirdesign.config.system import ExperimentConfig
    if isinstance(config, dict):
        config = ExperimentConfig.from_dict(config)
    else:
        config = ExperimentConfig.from_dict(config.to_dict())  # deep copy
    
    config.model.model_type = model_type
    
    # Reset hyperparameters to model-specific defaults to avoid cross-contamination
    from phosirdesign.models.base import MODEL_PARAMS
    if model_type in MODEL_PARAMS:
        config.model.hyperparameters = MODEL_PARAMS[model_type].copy()
    else:
        config.model.hyperparameters = {}
    
    config.logging.project_name = f"{config.logging.project_name}_{model_type}"
    
    # Set n_jobs
    if cores_per_task and 'n_jobs' in config.model.hyperparameters:
        config.model.hyperparameters['n_jobs'] = cores_per_task
    
    # Run training
    try:
        from phosirdesign.training.pipeline import TrainingPipeline
        pipeline = TrainingPipeline(config)
        results = pipeline.run()
        return {'model': model_type, 'success': True, 'results': results}
    except Exception as e:
        return {'model': model_type, 'success': False, 'error': str(e)}


def parallel_train_models(config, run_dir, numa_enabled=False, 
                         cores_per_task=None, parallel_tasks=8, bind_cpu=False):
    """
    Train multiple models in parallel
    
    Args:
        config: Experiment configuration
        run_dir: Run directory
        numa_enabled: Whether to enable NUMA optimization
        cores_per_task: Number of cores per task
        parallel_tasks: Number of parallel tasks
        bind_cpu: Whether to bind CPU
    """
    models = config.models_to_train if hasattr(config, 'models_to_train') else []
    
    # Prepare task parameters (serialize config to dict)
    tasks = []
    config_dict = config.to_dict()  # convert to dict for serialization
    for i, model in enumerate(models):
        task_args = (config_dict, model, i, numa_enabled, cores_per_task, bind_cpu)
        tasks.append(task_args)
    
    # Show NUMA info
    if numa_enabled:
        numa_nodes = get_numa_info()
        print(f"   NUMA nodes: {numa_nodes}")
        print(f"   CPU total cores: {psutil.cpu_count(logical=True)}")
    
    # Execute in parallel
    results = []
    with ProcessPoolExecutor(max_workers=parallel_tasks) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(train_single_model_parallel, task): task[1]
            for task in tasks
        }
        
        # Collect results
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
                if result['success']:
                    print(f"   SUCCESS: {model} training completed")
                else:
                    print(f"   ERROR: {model} training failed: {result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"   ERROR: {model} execution exception: {e}")
                results.append({'model': model, 'success': False, 'error': str(e)})
    
    # Summarize results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nParallel training results:")
    print(f"   Success: {len(successful)}/{len(models)}")
    if failed:
        print(f"   Failed: {', '.join([r['model'] for r in failed])}")
    
    return results


# ========================================
#           Main Entry
# ========================================

def config_command(args: List[str]):
    """Configuration management command"""
    print("\n" + "="*60)
    print("AutoML Configuration Manager")
    print("="*60)
    
    # Parse subcommands
    if not args or args[0] == 'list':
        # List all available configs
        manager = DynamicConfigManager()
        manager.print_config_summary()
        return 0
    
    elif args[0] == 'show':
        # Show specific config details
        if len(args) < 2:
            print("ERROR: Please specify config name: config show <name>")
            return 1
        
        config_name = args[1]
        manager = DynamicConfigManager()
        config = manager.get_config(config_name)
        
        if not config:
            print(f"ERROR: Config not found: {config_name}")
            return 1
        
        print(f"\nConfig: {config_name}")
        print("-" * 40)
        print(f"Description: {config.description}")
        print(f"Model: {config.model.model_type}")
        print(f"Feature: {config.feature.feature_type}")
        print(f"Folds: {config.training.n_folds}")
        if config.model.hyperparameters:
            print("\nHyperparameters:")
            for k, v in config.model.hyperparameters.items():
                print(f"  {k}: {v}")
        
        return 0
    
    else:
        print(f"ERROR: Unknown subcommand: {args[0]}")
        print("Available subcommands: list, show")
        return 1


def cache_command(args: List[str]):
    """Cache management command"""
    print("\n" + "="*60)
    print("Cache Management System")
    print("="*60)
    
    # Import cache manager
    from phosirdesign.utils.file_feature_cache import FileFeatureCache
    
    # Parse subcommand
    if not args or args[0] == 'stats':
        # Show cache statistics
        cache = FileFeatureCache()
        stats = cache.get_cache_stats()
        
        print("\nCache statistics:")
        print(f"   Cache dir: {stats['cache_dir']}")
        print(f"   Cache files: {stats['total_files']}")
        print(f"   Total size: {stats['total_size_mb']:.2f} MB")
        print(f"   Total accesses: {stats['total_accesses']}")
        
        if stats['most_accessed']:
            print("\nMost accessed:")
            for item in stats['most_accessed']:
                print(f"   - {item['file']}: {item['accesses']} times ({item['feature_type']})")
        
        if stats['largest_files']:
            print("\nLargest files:")
            for item in stats['largest_files']:
                print(f"   - {item['file']}: {item['size_mb']:.2f} MB ({item['feature_type']})")
        
        return 0
    
    elif args[0] == 'clear':
        # Clear cache
        cache = FileFeatureCache()
        
        # Check for parameter
        if len(args) > 1 and args[1].isdigit():
            days = int(args[1])
            print(f"\nClearing cache older than {days} days...")
            count, size = cache.clear_cache(older_than_days=days)
        else:
            print("\nClearing ALL cache...")
            confirm = input("Confirm clearing all cache? (y/n): ")
            if confirm.lower() != 'y':
                print("Cancelled clearing")
                return 0
            count, size = cache.clear_cache()
        
        print(f"INFO: Cleared {count} files ({size / 1024 / 1024:.2f} MB)")
        return 0
    
    elif args[0] == 'verify':
        # Verify cache integrity
        cache = FileFeatureCache()
        print("\nVerifying cache integrity...")
        valid, invalid = cache.verify_cache()
        print(f"   Valid: {valid} files")
        print(f"   Invalid: {invalid} files")
        if invalid > 0:
            print(f"   Automatically cleaned invalid cache")
        return 0
    
    else:
        print(f"ERROR: Unknown subcommand: {args[0]}")
        print("\nAvailable subcommands:")
        print("  stats  - show cache statistics")
        print("  clear  - clear cache")
        print("  verify - verify cache integrity")
        print("\nExamples:")
        print("  automl cache stats")
        print("  automl cache clear")
        print("  automl cache clear 30  # clear cache older than 30 days")
        print("  automl cache verify")
        return 1


def project_command(args: List[str]):
    """
    Project management command
    
    Examples:
        automl project list                        # list all projects
        automl project info project=test           # project details
        automl project predict project=test data=test.csv mode=best  # batch prediction
        automl project export project=test format=zip  # export project
    """
    if not args:
        print("Project Management")
        print("\nSubcommands:")
        print("  list    - list all projects")
        print("  info    - show project info")
        print("  predict - batch prediction using project models")
        print("  export  - export project")
        print("  report  - generate project report")
        print("\nExamples:")
        print("  automl project list")
        print("  automl project info project=TestReportComparison")
        print("  automl project predict project=test data=test.csv mode=best")
        print("  automl project export project=test format=zip")
        return 0
    
    subcommand = args[0].lower()
    params = MLArgumentParser.parse_args_string(' '.join(args[1:]))
    
    # Import project manager
    from phosirdesign.utils.project_manager import ProjectManager
    from phosirdesign.utils.project_predictor import ProjectPredictor
    
    manager = ProjectManager()
    
    if subcommand == 'list':
        # List all projects
        projects = manager.list_projects()
        if projects:
            print("\nProject list:")
            for p in projects:
                print(f"\n  Project: {p['name']}")
                print(f"     Path: {p['path']}")
                print(f"     Models: {p['models']}, Runs: {p['runs']}")
        else:
            print("ERROR: No projects found")
        return 0
    
    elif subcommand == 'info':
        # Show project info
        project = params.get('project')
        if not project:
            print("ERROR: Please specify project: project=<name>")
            return 1
        
        try:
            info = manager.get_project_info(project)
            predictor = ProjectPredictor(project, verbose=False)
            
            print(f"\nProject info: {info['project_name']}")
            print(f"   Path: {info['path']}")
            
            # Show model list
            df = predictor.list_models()
            
            # Show best models
            if info.get('best_models'):
                print("\nBest models:")
                for target, best in info['best_models'].items():
                    print(f"   {target}: {best['model']} (R^2={best['r2']:.4f})")
            
        except Exception as e:
            print(f"ERROR: Failed to get project info: {e}")
            return 1
        
        return 0
    
    elif subcommand == 'predict':
        # Batch prediction
        project = params.get('project')
        data = params.get('data')
        mode = params.get('mode', 'all')  # all, best, ensemble
        output = params.get('output')
        
        if not project:
            print("ERROR: Please specify project: project=<name>")
            return 1
        if not data:
            print("ERROR: Please specify data file: data=<file>")
            return 1
        
        try:
            predictor = ProjectPredictor(project)
            
            if mode == 'all':
                predictor.predict_all_models(
                    data_path=data,
                    output_dir=output
                )
            elif mode == 'best':
                predictor.predict_best_models(
                    data_path=data,
                    output_path=output
                )
            elif mode == 'ensemble':
                method = params.get('method', 'mean')
                predictor.predict_ensemble(
                    data_path=data,
                    output_path=output,
                    method=method
                )
            else:
                print(f"ERROR: Unknown predict mode: {mode}")
                print("   Available modes: all, best, ensemble")
                return 1
                
        except Exception as e:
            print(f"ERROR: Prediction failed: {e}")
            return 1
        
        return 0
    
    elif subcommand == 'export':
        # Export project
        project = params.get('project')
        output = params.get('output')
        format = params.get('format', 'zip')
        
        if not project:
            print("ERROR: Please specify project: project=<name>")
            return 1
        
        try:
            manager.export_project(project, output, format)
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
            return 1
        
        return 0
    
    elif subcommand == 'report':
        # Generate project report
        project = params.get('project')
        output = params.get('output')
        
        if not project:
            print("ERROR: Please specify project: project=<name>")
            return 1
        
        try:
            manager.generate_project_report(project, output)
        except Exception as e:
            print(f"ERROR: Report generation failed: {e}")
            return 1
        
        return 0
    
    else:
        print(f"ERROR: Unknown subcommand: {subcommand}")
        return 1


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("AutoML - Command-line tool for automated machine learning")
        print("\nUsage:")
        print("  automl <command> [options]")
        print("\nAvailable commands:")
        print("  train       - train models")
        print("  analyze     - analyze experiment results")
        print("  predict     - run predictions")
        print("  project     - project management (batch prediction)")
        print("  validate    - validate configuration")
        print("  config      - manage config templates")
        print("  cache       - manage feature cache")
        print("  export      - export models")
        print("  warmup      - precompute and write feature cache")
        print("  info        - show system information")
        print("\nExamples:")
        print("  automl train model=xgboost data=data.csv")
        print("  automl analyze dir=runs/train format=html")
        print("  automl project list")
        print("  automl project predict project=test data=test.csv mode=best")
        print("  automl config list")
        print("  automl train config=xgboost_standard")
        print("  automl predict model=model.joblib data=test.csv")
        print("\nMore info: automl info")
        return 0
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]
    
    # Route to corresponding command
    if command == 'train':
        return train_command(args)
    elif command == 'analyze':
        return analyze_command(args)
    elif command == 'predict':
        return predict_command(args)
    elif command == 'project':
        return project_command(args)
    elif command == 'validate':
        return validate_command(args)
    elif command == 'config':
        return config_command(args)
    elif command == 'cache':
        return cache_command(args)
    elif command == 'export':
        return export_command(args)
    elif command == 'warmup':
        return warmup_command(args)
    elif command == 'info':
        return info_command(args)
    else:
        print(f"ERROR: Unknown command: {command}")
        print("Use 'automl info' for help")
        return 1


if __name__ == "__main__":
    sys.exit(main())
