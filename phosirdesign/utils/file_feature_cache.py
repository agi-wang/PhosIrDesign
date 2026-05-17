#!/usr/bin/env python3
"""
File-Level Feature Cache System
Caches entire feature matrices for input files to avoid recomputation
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pickle
import os
import time
import logging

logger = logging.getLogger(__name__)


class FileFeatureCache:
    """
    File-level feature caching system
    Caches entire feature matrices based on file content and extraction parameters
    """
    
    def __init__(self, cache_dir: str = "file_feature_cache"):
        """
        Initialize file feature cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.by_file_dir = self.cache_dir / "by_file"
        self.by_file_dir.mkdir(exist_ok=True)
        
        # Load or create metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": "1.0",
            "cache_entries": {}
        }
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """
        Generate hash for file based on path, size, and modification time
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash string
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file stats
        stat = file_path.stat()
        file_size = stat.st_size
        file_mtime = stat.st_mtime
        
        # Create hash from file metadata
        hash_input = f"{file_path.absolute()}_{file_size}_{file_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def get_cache_key(self, 
                     file_path: str,
                     feature_type: str,
                     morgan_bits: int = 1024,
                     morgan_radius: int = 2,
                     smiles_columns: list = None,
                     combination_method: str = "mean",
                     descriptor_count: int = 85) -> str:
        """
        Generate cache key for file + feature parameters
        
        Args:
            file_path: Input CSV file path
            feature_type: Type of features (morgan/descriptors/combined)
            morgan_bits: Number of bits for Morgan fingerprint
            morgan_radius: Radius for Morgan fingerprint
            smiles_columns: List of SMILES column names
            combination_method: Method to combine multiple ligands
            
        Returns:
            Cache key string
        """
        if smiles_columns is None:
            smiles_columns = ['L1', 'L2', 'L3']
        
        # Get file hash
        file_hash = self.get_file_hash(file_path)
        
        # Create parameter string
        params = {
            "feature_type": feature_type,
            "morgan_bits": morgan_bits,
            "morgan_radius": morgan_radius,
            "smiles_columns": sorted(smiles_columns),
            "combination_method": combination_method,
            "descriptor_count": descriptor_count,
            "feature_schema": self._feature_schema(feature_type, descriptor_count),
        }
        
        # Generate hash from parameters
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        # Combine file hash and param hash
        cache_key = f"{file_hash}_{feature_type}_{morgan_bits}_{morgan_radius}_{combination_method}_{descriptor_count}_{param_hash}"
        
        return cache_key

    def _feature_schema(self, feature_type: str, descriptor_count: int) -> Dict[str, Any]:
        """Return dependency-aware schema data for cache invalidation."""
        schema: Dict[str, Any] = {}
        if feature_type not in {"morgan", "descriptors", "combined"}:
            return schema

        try:
            from rdkit import rdBase
            schema["rdkit_version"] = getattr(rdBase, "rdkitVersion", "unknown")

            if feature_type in {"descriptors", "combined"}:
                from rdkit.Chem import Descriptors
                descriptor_names = [x[0] for x in Descriptors._descList[:descriptor_count]]
                descriptor_blob = "|".join(descriptor_names)
                schema["descriptor_schema_hash"] = hashlib.md5(
                    descriptor_blob.encode()
                ).hexdigest()
        except Exception as exc:
            schema["rdkit_unavailable"] = exc.__class__.__name__

        return schema
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get path for cache file"""
        # Use first 8 chars of key as subdirectory for better organization
        subdir = self.by_file_dir / cache_key[:8]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{cache_key}.npz"
    
    def load_features(self, 
                     file_path: str,
                     feature_type: str,
                     morgan_bits: int = 1024,
                     morgan_radius: int = 2,
                     smiles_columns: list = None,
                     combination_method: str = "mean",
                     descriptor_count: int = 85) -> Optional[np.ndarray]:
        """
        Load cached features for a file
        
        Args:
            file_path: Input CSV file path
            feature_type: Type of features
            morgan_bits: Number of bits for Morgan fingerprint
            morgan_radius: Radius for Morgan fingerprint
            smiles_columns: List of SMILES column names
            combination_method: Method to combine multiple ligands
            
        Returns:
            Feature matrix if cached, None otherwise
        """
        try:
            cache_key = self.get_cache_key(
                file_path, feature_type, morgan_bits, 
                morgan_radius, smiles_columns, combination_method, descriptor_count
            )
            cache_path = self.get_cache_path(cache_key)
            
            if not cache_path.exists():
                return None
            
            # Load from npz file
            with np.load(cache_path) as data:
                features = data['features']
            
            # Update metadata
            if cache_key in self.metadata["cache_entries"]:
                self.metadata["cache_entries"][cache_key]["access_count"] += 1
                self._save_metadata()
            
            logger.info(f"Loaded features from cache: {cache_key}")
            return features
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def save_features(self,
                     features: np.ndarray,
                     file_path: str,
                     feature_type: str,
                     morgan_bits: int = 1024,
                     morgan_radius: int = 2,
                     smiles_columns: list = None,
                     combination_method: str = "mean",
                     descriptor_count: int = 85,
                     row_count: int = None,
                     failed_indices: list = None):
        """
        Save features to cache
        
        Args:
            features: Feature matrix to cache
            file_path: Input CSV file path
            feature_type: Type of features
            morgan_bits: Number of bits for Morgan fingerprint
            morgan_radius: Radius for Morgan fingerprint
            smiles_columns: List of SMILES column names
            combination_method: Method to combine multiple ligands
            row_count: Original number of rows in file
            failed_indices: List of indices that failed feature extraction
        """
        try:
            cache_key = self.get_cache_key(
                file_path, feature_type, morgan_bits,
                morgan_radius, smiles_columns, combination_method, descriptor_count
            )
            cache_path = self.get_cache_path(cache_key)
            
            # Save as compressed npz
            np.savez_compressed(
                cache_path,
                features=features,
                failed_indices=failed_indices if failed_indices else []
            )
            
            # Update metadata
            file_size = cache_path.stat().st_size
            self.metadata["cache_entries"][cache_key] = {
                "file_path": str(file_path),
                "feature_type": feature_type,
                "morgan_bits": morgan_bits,
                "morgan_radius": morgan_radius,
                "smiles_columns": smiles_columns,
                "combination_method": combination_method,
                "descriptor_count": descriptor_count,
                "feature_schema": self._feature_schema(feature_type, descriptor_count),
                "shape": features.shape,
                "row_count": row_count or features.shape[0],
                "failed_count": len(failed_indices) if failed_indices else 0,
                "cache_size": file_size,
                "access_count": 0
            }
            self._save_metadata()
            
            logger.info(f"Saved features to cache: {cache_key} ({file_size / 1024 / 1024:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache files
        
        Args:
            older_than_days: Only clear files older than this many days
        """
        cleared_count = 0
        cleared_size = 0
        
        if older_than_days is None:
            # Clear all cache
            for cache_file in self.by_file_dir.rglob("*.npz"):
                size = cache_file.stat().st_size
                cache_file.unlink()
                cleared_count += 1
                cleared_size += size
            
            # Clear metadata
            self.metadata["cache_entries"] = {}
            self._save_metadata()
            
        else:
            # Clear old cache files
            cutoff_age_seconds = older_than_days * 86400
            now = time.time()
            
            for cache_key, entry in list(self.metadata["cache_entries"].items()):
                cache_path = self.get_cache_path(cache_key)
                if cache_path.exists() and now - cache_path.stat().st_mtime > cutoff_age_seconds:
                    size = cache_path.stat().st_size
                    cache_path.unlink()
                    cleared_count += 1
                    cleared_size += size
                    del self.metadata["cache_entries"][cache_key]
            
            self._save_metadata()
        
        logger.info(f"Cleared {cleared_count} cache files ({cleared_size / 1024 / 1024:.2f} MB)")
        return cleared_count, cleared_size
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_files = len(self.metadata["cache_entries"])
        total_size = 0
        total_accesses = 0
        
        for entry in self.metadata["cache_entries"].values():
            total_size += entry.get("cache_size", 0)
            total_accesses += entry.get("access_count", 0)
        
        # Get actual disk usage
        actual_files = list(self.by_file_dir.rglob("*.npz"))
        actual_size = sum(f.stat().st_size for f in actual_files)
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size / 1024 / 1024,
            "actual_files": len(actual_files),
            "actual_size_mb": actual_size / 1024 / 1024,
            "total_accesses": total_accesses,
            "cache_dir": str(self.cache_dir),
            "most_accessed": self._get_most_accessed(5),
            "largest_files": self._get_largest_files(5)
        }
    
    def _get_most_accessed(self, n: int = 5) -> list:
        """Get most accessed cache entries"""
        entries = sorted(
            self.metadata["cache_entries"].items(),
            key=lambda x: x[1].get("access_count", 0),
            reverse=True
        )
        return [
            {
                "file": Path(e[1]["file_path"]).name,
                "feature_type": e[1]["feature_type"],
                "accesses": e[1].get("access_count", 0)
            }
            for e in entries[:n]
        ]
    
    def _get_largest_files(self, n: int = 5) -> list:
        """Get largest cache files"""
        entries = sorted(
            self.metadata["cache_entries"].items(),
            key=lambda x: x[1].get("cache_size", 0),
            reverse=True
        )
        return [
            {
                "file": Path(e[1]["file_path"]).name,
                "feature_type": e[1]["feature_type"],
                "size_mb": e[1].get("cache_size", 0) / 1024 / 1024
            }
            for e in entries[:n]
        ]
    
    def verify_cache(self) -> Tuple[int, int]:
        """
        Verify cache integrity
        
        Returns:
            (valid_count, invalid_count)
        """
        valid = 0
        invalid = 0
        
        for cache_key in list(self.metadata["cache_entries"].keys()):
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    # Try to load the file
                    with np.load(cache_path) as data:
                        _ = data['features']
                    valid += 1
                except Exception:
                    invalid += 1
                    # Remove invalid entry
                    del self.metadata["cache_entries"][cache_key]
                    cache_path.unlink()
            else:
                # File missing, remove from metadata
                invalid += 1
                del self.metadata["cache_entries"][cache_key]
        
        if invalid > 0:
            self._save_metadata()
            logger.info(f"Removed {invalid} invalid cache entries")
        
        return valid, invalid
