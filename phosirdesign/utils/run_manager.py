#!/usr/bin/env python3
"""
Run manager for incremental experiment directories.

Automatically creates runs/train, runs/train2, runs/train3 directories.
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple
import yaml
import json


class RunManager:
    """Run manager for experiment directories"""
    
    def __init__(self, base_dir: str = "runs", task: str = "train"):
        """
        Initialize run manager
        
        Args:
            base_dir: Base directory (default: runs)
            task: Task type (train/predict/validate)
        """
        self.base_dir = Path(base_dir)
        self.task = task
        self.base_dir.mkdir(exist_ok=True)
    
    def get_next_run_dir(self, name: Optional[str] = None, project: Optional[str] = None) -> Path:
        """
        Get the next run directory
        
        Incremental naming:
        - Default: runs/train, runs/train2, runs/train3, ...
        - With name: runs/train/my_experiment
        - With project: my_project/train, my_project/train2, ...
        - With both: my_project/my_experiment
        
        Args:
            name: Experiment name (optional)
            project: Project name (optional)
        
        Returns:
            Run directory path
        """
        # Determine base path
        if project:
            base_path = Path(project)
        else:
            base_path = self.base_dir
        
        # Use name directly if specified
        if name:
            run_dir = base_path / name
        else:
            run_dir = base_path / self.task
        
        # Create directory
        run_dir.mkdir(parents=True, exist_ok=True)
        
        return run_dir
    
    def _get_increment_dir(self, base_path: Path, prefix: str) -> Path:
        """
        Get incremented directory
        
        Args:
            base_path: Base path
            prefix: Prefix (e.g., train)
        
        Returns:
            Incremented directory path
        """
        # Find existing run directories
        existing_runs = []
        
        # Matching pattern: prefix, prefix2, prefix3, ...
        pattern = re.compile(f"^{re.escape(prefix)}(\\d*)$")
        
        # Scan directories
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir():
                    match = pattern.match(item.name)
                    if match:
                        num_str = match.group(1)
                        if num_str == "":
                            existing_runs.append(1)
                        else:
                            existing_runs.append(int(num_str))
        
        # Determine next index
        if not existing_runs:
            # First run, no number
            next_dir = base_path / prefix
        else:
            # Find max index and add 1
            max_num = max(existing_runs)
            if max_num == 1 and 1 in existing_runs:
                # If prefix exists (equivalent to prefix1), next is prefix2
                next_dir = base_path / f"{prefix}2"
            else:
                next_dir = base_path / f"{prefix}{max_num + 1}"
        
        return next_dir
    
    @staticmethod
    def parse_run_path(path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse run path, extract project and name
        
        Args:
            path: Path string
        
        Returns:
            (project, name) tuple
        """
        parts = Path(path).parts
        
        if len(parts) == 0:
            return None, None
        elif len(parts) == 1:
            # Only name or task
            return None, parts[0]
        else:
            # project/name format
            if parts[0] == "runs":
                # runs/train format
                return None, parts[-1] if len(parts) > 1 else None
            else:
                # project/name format
                return parts[0], parts[-1]
    
    def save_run_info(self, run_dir: Path, config: dict, command: str = None):
        """
        Save run information
        
        Args:
            run_dir: Run directory
            config: Configuration dict
            command: Run command
        """
        run_info = {
            'run_dir': str(run_dir),
            'run_sequence': self._sequence_from_name(run_dir.name),
            'command': command,
            'config': config
        }
        
        # Save as YAML
        info_file = run_dir / "run_info.yaml"
        with open(info_file, 'w') as f:
            yaml.dump(run_info, f, default_flow_style=False)
        
        # Also save as JSON
        json_file = run_dir / "run_info.json"
        with open(json_file, 'w') as f:
            json.dump(run_info, f, indent=2)
    
    @staticmethod
    def create_symlink(run_dir: Path, link_name: str = "last"):
        """
        Create a symlink pointing to the latest run
        
        Args:
            run_dir: Run directory
            link_name: Link name (default: last)
        """
        # Create link in parent directory
        parent = run_dir.parent
        link_path = parent / link_name
        
        # Remove old link
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        
        # Create new link (relative path)
        try:
            link_path.symlink_to(run_dir.name)
        except Exception:
            pass

    @staticmethod
    def _sequence_from_name(name: str) -> int:
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else 1
    
    def get_latest_run(self, project: Optional[str] = None) -> Optional[Path]:
        """
        Get the latest run directory
        
        Args:
            project: Project name (optional)
        
        Returns:
            Latest run directory path
        """
        if project:
            base_path = Path(project)
        else:
            base_path = self.base_dir
        
        if not base_path.exists():
            return None
        
        # Find all run directories
        run_dirs = []
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Get modification time
                run_dirs.append((item, item.stat().st_mtime))
        
        if not run_dirs:
            return None
        
        # Sort by time, return latest
        run_dirs.sort(key=lambda x: x[1], reverse=True)
        return run_dirs[0][0]
    
    def list_runs(self, project: Optional[str] = None, limit: int = 10):
        """
        List run history
        
        Args:
            project: Project name (optional)
            limit: Max number of runs to show
        
        Returns:
            List of run directories
        """
        if project:
            base_path = Path(project)
        else:
            base_path = self.base_dir
        
        if not base_path.exists():
            return []
        
        # Collect all runs
        runs = []
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Read run info
                info_file = item / "run_info.yaml"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = yaml.safe_load(f)
                else:
                    info = {
                        'run_dir': str(item),
                        'run_sequence': self._sequence_from_name(item.name)
                    }
                
                runs.append({
                    'path': item,
                    'name': item.name,
                    'run_sequence': info.get('run_sequence', self._sequence_from_name(item.name)),
                    'config': info.get('config', {})
                })
        
        runs.sort(key=lambda x: x['run_sequence'], reverse=True)
        
        # Limit number
        if limit:
            runs = runs[:limit]
        
        return runs
    
    def clean_old_runs(self, project: Optional[str] = None, keep: int = 5):
        """
        Clean old run directories
        
        Args:
            project: Project name (optional)
            keep: Number of runs to keep
        """
        runs = self.list_runs(project, limit=None)
        
        if len(runs) <= keep:
            return
        
        # Delete old runs
        for run in runs[keep:]:
            import shutil
            shutil.rmtree(run['path'])
            print(f"Deleted old run: {run['path']}")


class ExperimentTracker:
    """Experiment tracker - record and manage experiments"""
    
    def __init__(self, run_dir: Path):
        """
        Initialize experiment tracker
        
        Args:
            run_dir: Run directory
        """
        self.run_dir = run_dir
        self.metrics_file = run_dir / "metrics.json"
        self.log_file = run_dir / "experiment.log"
        
        # Create subdirectories
        (run_dir / "weights").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "predictions").mkdir(exist_ok=True)
        (run_dir / "exports").mkdir(exist_ok=True)
        
        # Initialize metrics record
        self.metrics = {
            'epochs': [],
            'train': {},
            'val': {},
            'test': {},
            'log_entries': []
        }
    
    def log_metrics(self, epoch: int, metrics: dict, split: str = 'train'):
        """
        Log metrics
        
        Args:
            epoch: Epoch number
            metrics: Metrics dict
            split: Dataset split (train/val/test)
        """
        # Update metrics in memory
        if split not in self.metrics:
            self.metrics[split] = {}
        
        for key, value in metrics.items():
            if key not in self.metrics[split]:
                self.metrics[split][key] = []
            self.metrics[split][key].append(value)
        
        if epoch not in self.metrics['epochs']:
            self.metrics['epochs'].append(epoch)
        
        # Save to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_model(self, model, name: str = "best", format: str = "joblib"):
        """
        Save model
        
        Args:
            model: Model object
            name: Model name
            format: Save format
        """
        weights_dir = self.run_dir / "weights"
        
        if format == "joblib":
            import joblib
            model_path = weights_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
        elif format == "pickle":
            import pickle
            model_path = weights_dir / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return model_path
    
    def log(self, message: str, level: str = "INFO"):
        """
        Write log entry
        
        Args:
            message: Log message
            level: Log level
        """
        index = len(self.metrics.get("log_entries", [])) + 1
        self.metrics.setdefault("log_entries", []).append({"index": index, "level": level})
        log_entry = f"[{index:04d}] [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # Also print to console
        print(f"[{level}] {message}")
    
    def get_summary(self) -> dict:
        """Get experiment summary"""
        if not self.metrics_file.exists():
            return {}
        
        with open(self.metrics_file, 'r') as f:
            metrics = json.load(f)
        
        summary = {
            'run_dir': str(self.run_dir),
            'n_epochs': len(metrics.get('epochs', [])),
            'metrics': {}
        }
        
        # Compute final metrics
        for split in ['train', 'val', 'test']:
            if split in metrics and metrics[split]:
                summary['metrics'][split] = {}
                for key, values in metrics[split].items():
                    if values:
                        summary['metrics'][split][key] = {
                            'final': values[-1],
                            'best': min(values) if 'loss' in key or 'rmse' in key else max(values),
                            'mean': sum(values) / len(values)
                        }
        
        return summary


# ========================================
#           Convenience functions
# ========================================

def get_run_dir(name: Optional[str] = None, 
                project: Optional[str] = None,
                task: str = "train") -> Path:
    """
    Convenience function to get run directory
    
    Args:
        name: Experiment name
        project: Project name
        task: Task type
    
    Returns:
        Run directory path
    """
    manager = RunManager(task=task)
    return manager.get_next_run_dir(name=name, project=project)


def setup_experiment(name: Optional[str] = None,
                    project: Optional[str] = None,
                    config: dict = None) -> Tuple[Path, ExperimentTracker]:
    """
    Set up experiment environment
    
    Args:
        name: Experiment name
        project: Project name
        config: Configuration dict
    
    Returns:
        (run directory, experiment tracker)
    """
    # Get run directory
    run_dir = get_run_dir(name=name, project=project)
    
    # Create tracker
    tracker = ExperimentTracker(run_dir)
    
    # Save config
    if config:
        config_path = run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Create symlink
    RunManager.create_symlink(run_dir, "last")
    
    print(f"Experiment directory: {run_dir}")
    
    return run_dir, tracker
