#!/usr/bin/env python3
"""
Project Manager
Manage the full lifecycle of AutoML projects
"""

import json
import yaml
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from .comparison_table import ComparisonTableGenerator

class ProjectManager:
    """Project Manager"""
    
    def __init__(self, base_dir: str = "."):
        """
        Initialize project manager
        
        Args:
            base_dir: Base directory
        """
        self.base_dir = Path(base_dir)
    
    def create_project_metadata(self, project_dir: str) -> Dict:
        """
        Create project metadata
        
        Args:
            project_dir: Project directory
        
        Returns:
            Metadata dict
        """
        project_path = Path(project_dir)
        if not project_path.exists():
            raise ValueError(f"Project directory does not exist: {project_dir}")
        
        metadata = {
            'project_name': project_path.name,
            'path': str(project_path.absolute()),
            'models_trained': [],
            'targets': [],
            'best_models': {},
            'data_info': {},
            'comparison_tables': [],
            'training_runs': []
        }
        
        # Scan training runs
        for run_dir in project_path.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith('.'):
                run_info = self._analyze_run(run_dir)
                if run_info:
                    metadata['training_runs'].append(run_info)
                    
                    # Collect model types
                    if run_info['model'] not in metadata['models_trained']:
                        metadata['models_trained'].append(run_info['model'])
                    
                    # Collect targets
                    for target in run_info.get('targets', []):
                        if target not in metadata['targets']:
                            metadata['targets'].append(target)
                    
                    # Update best models
                    for target, perf in run_info.get('performance', {}).items():
                        if target not in metadata['best_models'] or \
                           perf.get('r2', 0) > metadata['best_models'][target].get('r2', 0):
                            metadata['best_models'][target] = {
                                'model': run_info['model'],
                                'run': run_info['name'],
                                'r2': perf.get('r2', 0),
                                'rmse': perf.get('rmse', 0),
                                'path': str(run_dir / 'models')
                            }
        
        # Find comparison tables
        for table_file in project_path.glob('comparison_table_*'):
            metadata['comparison_tables'].append(table_file.name)
        
        # Save metadata
        metadata_file = project_path / 'project_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Project metadata created: {metadata_file}")
        
        return metadata
    
    def _analyze_run(self, run_dir: Path) -> Optional[Dict]:
        """Analyze a single training run"""
        run_info = {
            'name': run_dir.name,
            'path': str(run_dir),
            'model': None,
            'targets': [],
            'performance': {}
        }
        
        # Read config
        config_file = run_dir / 'config.yaml'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                run_info['model'] = config.get('model', {}).get('model_type')
                run_info['targets'] = config.get('data', {}).get('target_columns', [])
        
        # Read run info
        run_info_file = run_dir / 'run_info.json'
        if run_info_file.exists():
            with open(run_info_file, 'r') as f:
                info = json.load(f)
                run_info['command'] = info.get('command')
        
        # Collect performance metrics
        for summary_file in (run_dir / 'exports').glob('*_summary.json'):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                target = summary.get('target')
                if target:
                    run_info['performance'][target] = {
                        'r2': summary.get('mean_r2', 0),
                        'rmse': summary.get('mean_rmse', 0),
                        'mae': summary.get('mean_mae', 0)
                    }
        
        return run_info if run_info['model'] else None
    
    def generate_comparison_table(self, project_name: str, output_dir: Optional[str] = None,
                                  formats: Optional[List[str]] = None,
                                  decimal_places: Optional[Dict[str, int]] = None) -> Dict[str, str]:
        """
        Generate model performance comparison tables (scan *_summary.json)
        
        Args:
            project_name: Project name or path
            output_dir: Output directory (default: project root)
            formats: Output formats (markdown, html, latex, csv)
            decimal_places: Decimal places dict, e.g., {'r2': 4, 'rmse': 4, 'mae': 4}
        
        Returns:
            Dict of output paths per format
        """
        # Resolve project path
        project_path = Path(project_name)
        if not project_path.exists():
            project_path = self.base_dir / project_name
        if not project_path.exists():
            raise ValueError(f"Project not found: {project_name}")

        # Create generator and export
        generator = ComparisonTableGenerator(results_dir=str(project_path))
        exported_files = generator.export_all_formats(
            output_dir=output_dir if output_dir else str(project_path),
            formats=formats if formats else ['markdown', 'html', 'latex', 'csv'],
            decimal_places=decimal_places
        )

        # Update/write comparison_tables list in project metadata
        metadata_file = project_path / 'project_metadata.json'
        metadata = None
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = None
        if not metadata:
            metadata = self.create_project_metadata(str(project_path))

        # Merge new table file names (store file names only to avoid absolute path differences)
        new_names = [Path(p).name for p in exported_files.values()]
        existing = set(metadata.get('comparison_tables', []))
        for name in new_names:
            if name not in existing:
                metadata['comparison_tables'].append(name)
                existing.add(name)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)

        print(f"INFO: Comparison tables generated, count: {len(exported_files)}")
        for k, v in exported_files.items():
            print(f"   - {k}: {v}")

        return exported_files

    def list_projects(self) -> List[Dict]:
        """
        List all projects
        
        Returns:
            List of projects
        """
        projects = []
        
        # Find all directories containing project_metadata.json
        for metadata_file in self.base_dir.rglob('project_metadata.json'):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    projects.append({
                        'name': metadata['project_name'],
                        'path': metadata_file.parent,
                        'models': len(metadata.get('models_trained', [])),
                        'runs': len(metadata.get('training_runs', []))
                    })
            except Exception as e:
                continue
        
        # Also find directories without metadata but with models
        for model_file in self.base_dir.rglob('*.joblib'):
            project_dir = model_file.parent
            while project_dir != self.base_dir and project_dir.parent != self.base_dir:
                project_dir = project_dir.parent
            
            if not (project_dir / 'project_metadata.json').exists():
                # Check if already added
                if not any(p['path'] == project_dir for p in projects):
                    projects.append({
                        'name': project_dir.name,
                        'path': project_dir,
                        'models': '?',
                        'runs': '?'
                    })
        
        return projects
    
    def get_project_info(self, project_name: str) -> Dict:
        """
        Get project details
        
        Args:
            project_name: Project name or path
        
        Returns:
            Project info dict
        """
        # Find project
        project_path = Path(project_name)
        if not project_path.exists():
            # Try to find in base_dir
            project_path = self.base_dir / project_name
        
        if not project_path.exists():
            raise ValueError(f"Project not found: {project_name}")
        
        # Check metadata
        metadata_file = project_path / 'project_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            # Dynamically generate metadata
            return self.create_project_metadata(str(project_path))
    
    def export_project(self, project_name: str, output_path: str = None,
                      format: str = 'zip') -> str:
        """
        Export project
        
        Args:
            project_name: Project name or path
            output_path: Output path
            format: Export format ('zip', 'tar')
        
        Returns:
            Exported file path
        """
        # Find project
        project_path = Path(project_name)
        if not project_path.exists():
            project_path = self.base_dir / project_name
        
        if not project_path.exists():
            raise ValueError(f"Project not found: {project_name}")
        
        # Ensure metadata exists
        self.create_project_metadata(str(project_path))
        
        # Set output path
        if output_path is None:
            output_path = f"{project_path.name}.{format}"
        
        # Export
        if format == 'zip':
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in project_path.rglob('*'):
                    if file.is_file():
                        arcname = file.relative_to(project_path.parent)
                        zipf.write(file, arcname)
        elif format == 'tar':
            import tarfile
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(project_path, arcname=project_path.name)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"INFO: Project exported: {output_path}")
        return output_path
    
    def clean_project(self, project_name: str, keep_models: bool = True,
                     keep_results: bool = True) -> None:
        """
        Clean project (delete intermediate files)
        
        Args:
            project_name: Project name or path
            keep_models: Keep model files
            keep_results: Keep result files
        """
        # Locate project
        project_path = Path(project_name)
        if not project_path.exists():
            project_path = self.base_dir / project_name
        
        if not project_path.exists():
            raise ValueError(f"Project not found: {project_name}")
        
        # Cleanup rules
        patterns_to_delete = [
            '**/checkpoints/*',
            '**/predictions/*',
            '**/feature_importance/*' if not keep_results else None,
            '**/plots/*' if not keep_results else None,
            '**/models/*' if not keep_models else None,
        ]
    
        deleted_count = 0
        for pattern in patterns_to_delete:
            if pattern:
                for file in project_path.glob(pattern):
                    if file.is_file():
                        file.unlink()
                        deleted_count += 1
        
        print(f"INFO: Cleanup completed, deleted {deleted_count} files")
    
    def generate_project_report(self, project_name: str, output_path: str = None) -> str:
        """
        Generate project report
        
        Args:
            project_name: Project name or path
            output_path: Output path
        
        Returns:
            Report file path
        """
        # Get project info
        info = self.get_project_info(project_name)
        
        # Generate Markdown report
        report = f"# Project Report: {info['project_name']}\n\n"
        
        # Model information
        report += "## Trained Models\n\n"
        if info.get('models_trained'):
            for model in info['models_trained']:
                report += f"- {model}\n"
        else:
            report += "No model information\n"
        
        report += "\n## Prediction Targets\n\n"
        if info.get('targets'):
            for target in info['targets']:
                report += f"- {target}\n"
                if target in info.get('best_models', {}):
                    best = info['best_models'][target]
                    report += f"  - Best model: {best['model']} (R^2={best.get('r2', 'N/A'):.4f})\n"
        
        report += "\n## Training Runs\n\n"
        if info.get('training_runs'):
            report += "| Run | Model | Targets | Avg R^2 |\n"
            report += "|-----|-------|---------|---------|\n"
            for run in info['training_runs']:
                avg_r2 = 0
                if run.get('performance'):
                    r2_values = [p.get('r2', 0) for p in run['performance'].values()]
                    avg_r2 = sum(r2_values) / len(r2_values) if r2_values else 0
                report += f"| {run['name']} | {run.get('model', 'Unknown')} | "
                report += f"{len(run.get('targets', []))} | {avg_r2:.4f} |\n"
        
        # Save report
        if output_path is None:
            project_path = Path(info['path'])
            output_path = project_path / "project_report.md"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"INFO: Project report generated: {output_path}")
        return str(output_path)


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Project Manager')
    subparsers = parser.add_subparsers(dest='command', help='command')
    
    # list command
    list_parser = subparsers.add_parser('list', help='List all projects')
    
    # info command
    info_parser = subparsers.add_parser('info', help='Get project info')
    info_parser.add_argument('project', help='Project name or path')
    
    # export command
    export_parser = subparsers.add_parser('export', help='Export project')
    export_parser.add_argument('project', help='Project name or path')
    export_parser.add_argument('--output', help='Output path')
    export_parser.add_argument('--format', default='zip', choices=['zip', 'tar'])
    
    # clean command
    clean_parser = subparsers.add_parser('clean', help='Clean project')
    clean_parser.add_argument('project', help='Project name or path')
    clean_parser.add_argument('--keep-models', action='store_true')
    clean_parser.add_argument('--keep-results', action='store_true')
    
    # report command
    report_parser = subparsers.add_parser('report', help='Generate project report')
    report_parser.add_argument('project', help='Project name or path')
    report_parser.add_argument('--output', help='Output path')

    # table command
    table_parser = subparsers.add_parser('table', help='Generate model comparison table')
    table_parser.add_argument('project', help='Project name or path')
    table_parser.add_argument('--output', help='Output directory (default: project root)')
    table_parser.add_argument('--formats', nargs='+', default=['markdown','html','latex','csv'],
                              help='Output formats')
    
    args = parser.parse_args()
    
    manager = ProjectManager()
    
    if args.command == 'list':
        projects = manager.list_projects()
        if projects:
            print("\nProject list:")
            for p in projects:
                print(f"  - {p['name']} ({p['path']})")
                print(f"    Models: {p['models']}, Runs: {p['runs']}")
        else:
            print("No projects found")
    
    elif args.command == 'info':
        info = manager.get_project_info(args.project)
        print(json.dumps(info, indent=2, ensure_ascii=False))
    
    elif args.command == 'export':
        manager.export_project(args.project, args.output, args.format)
    
    elif args.command == 'clean':
        manager.clean_project(args.project, args.keep_models, args.keep_results)
    
    elif args.command == 'report':
        manager.generate_project_report(args.project, args.output)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
