#!/usr/bin/env python3
"""
Universal Feature Extractor for AutoML System
Handles molecular (SMILES), tabular, and image data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional, Any
import hashlib
import pickle
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ========================================
#     Universal Feature Extractor
# ========================================

class FeatureExtractor:
    """
    Universal feature extractor that handles all data types:
    - Molecular data (SMILES strings)
    - Tabular data (numerical/categorical)
    - Image data (as flattened features)
    - Mixed data (combination of above)
    """
    
    def __init__(self, 
                 feature_type: str = "auto",
                 use_cache: bool = True,
                 cache_dir: str = "feature_cache",
                 morgan_bits: Optional[int] = None,
                 morgan_radius: Optional[int] = None,
                 descriptor_count: Optional[int] = None):
        """
        Initialize feature extractor
        
        Args:
            feature_type: Type of features ('auto', 'tabular', 'morgan', 'descriptors', 'combined')
            use_cache: Whether to cache extracted features
            cache_dir: Directory for cache files
        """
        self.feature_type = feature_type
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        
        if use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Molecular feature settings
        self.morgan_bits = 1024 if morgan_bits is None else int(morgan_bits)
        self.morgan_radius = 2 if morgan_radius is None else int(morgan_radius)
        self.descriptor_count = 85 if descriptor_count is None else int(descriptor_count)
        
        # RDKit modules (lazy loading)
        self._rdkit_imported = False
        self._rdkit_modules = {}
        
        # For backward compatibility
        self.feature_names_ = None
    
    # ========================================
    #     Data Type Detection
    # ========================================
    
    def detect_data_type(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        Automatically detect data type
        
        Returns:
            'molecular' if SMILES columns found, 'tabular' otherwise
        """
        if isinstance(data, pd.DataFrame):
            # Check for SMILES columns
            smiles_indicators = ['smiles', 'mol', 'molecule', 'ligand', 'l1', 'l2', 'l3']
            for col in data.columns:
                if any(indicator in col.lower() for indicator in smiles_indicators):
                    # Verify it contains SMILES-like strings
                    sample = data[col].dropna().head(5)
                    if len(sample) > 0 and self._looks_like_smiles(sample.iloc[0]):
                        return 'molecular'
        
        return 'tabular'
    
    def _looks_like_smiles(self, s: Any) -> bool:
        """Check if string looks like SMILES"""
        if pd.isna(s) or not isinstance(s, str):
            return False
        # Simple heuristic: contains typical SMILES characters
        smiles_chars = set('CNOPSFClBrI()[]=#@+\\/-123456789cnops')
        return len(s) > 1 and len(set(str(s)) - smiles_chars) < len(str(s)) * 0.2
    
    # ========================================
    #     Molecular Features (RDKit)
    # ========================================
    
    def _import_rdkit(self):
        """Lazy import RDKit modules"""
        if self._rdkit_imported:
            return
        
        try:
            from rdkit import Chem, DataStructs, rdBase
            from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator
            from rdkit.ML.Descriptors import MoleculeDescriptors
            
            self._rdkit_modules = {
                'rdBase': rdBase,
                'Chem': Chem,
                'AllChem': AllChem,
                'Descriptors': Descriptors,
                'MoleculeDescriptors': MoleculeDescriptors,
                'FingerprintGenerator': rdFingerprintGenerator,
                'DataStructs': DataStructs
            }
            self._rdkit_imported = True
        except ImportError:
            raise ImportError(
                "RDKit is required for molecular features. "
                "Install with: conda install -c conda-forge rdkit"
            )
    
    def extract_morgan_fingerprint(self, smiles: str) -> np.ndarray:
        """Extract Morgan fingerprint from SMILES"""
        self._import_rdkit()
        Chem = self._rdkit_modules['Chem']
        AllChem = self._rdkit_modules['AllChem']
        FingerprintGenerator = self._rdkit_modules.get('FingerprintGenerator')
        DataStructs = self._rdkit_modules.get('DataStructs')
        
        if pd.isna(smiles) or smiles == '':
            return np.zeros(self.morgan_bits)
        
        # Try cache first
        if self.use_cache:
            # Use MD5 hash for cache key to avoid special characters in filename
            cache_key = self._get_cache_key(
                f"morgan_{smiles}_{self.morgan_bits}_{self.morgan_radius}_{self._rdkit_version_token()}"
            )
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Compute fingerprint
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.morgan_bits)
        
        arr = None
        if FingerprintGenerator is not None and DataStructs is not None:
            gen = FingerprintGenerator.GetMorganGenerator(radius=self.morgan_radius, fpSize=self.morgan_bits)
            fp = gen.GetFingerprint(mol)
            arr = np.zeros(self.morgan_bits, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=self.morgan_bits)
            arr = np.array(fp)
        
        # Save to cache
        if self.use_cache:
            self._save_to_cache(cache_key, arr)
        
        return arr
    
    def extract_molecular_descriptors(self, smiles: str) -> np.ndarray:
        """Extract molecular descriptors from SMILES"""
        self._import_rdkit()
        Chem = self._rdkit_modules['Chem']
        Descriptors = self._rdkit_modules['Descriptors']
        MoleculeDescriptors = self._rdkit_modules['MoleculeDescriptors']
        
        if pd.isna(smiles) or smiles == '':
            return np.zeros(self.descriptor_count)
        
        # Try cache first
        if self.use_cache:
            # Use MD5 hash for cache key to avoid special characters in filename
            cache_key = self._get_cache_key(
                f"desc_{smiles}_{self.descriptor_count}_{self._descriptor_schema_token()}"
            )
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Compute descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.descriptor_count)
        
        descriptor_names = [x[0] for x in Descriptors._descList[:self.descriptor_count]]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        desc = calculator.CalcDescriptors(mol)
        arr = np.array(desc)
        
        # Handle NaN, Inf, and extremely large values
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extremely large values to reasonable range
        # Most molecular descriptors should be in reasonable range
        arr = np.clip(arr, -1e6, 1e6)
        
        # Save to cache
        if self.use_cache:
            self._save_to_cache(cache_key, arr)
        
        return arr
    
    def extract_from_smiles(self, smiles: Union[str, List[str]], 
                           feature_type: str = None) -> np.ndarray:
        """
        Extract features from SMILES string(s)
        
        Args:
            smiles: Single SMILES or list of SMILES
            feature_type: Override default feature type
        
        Returns:
            Feature array
        """
        if feature_type is None:
            feature_type = self.feature_type
            
        # Handle single SMILES
        if isinstance(smiles, str):
            smiles = [smiles]
        
        features = []
        for smi in smiles:
            if feature_type == "morgan":
                feat = self.extract_morgan_fingerprint(smi)
            elif feature_type == "descriptors":
                feat = self.extract_molecular_descriptors(smi)
            elif feature_type == "combined":
                morgan = self.extract_morgan_fingerprint(smi)
                desc = self.extract_molecular_descriptors(smi)
                feat = np.concatenate([morgan, desc])
            else:
                feat = self.extract_morgan_fingerprint(smi)
            features.append(feat)
        
        return np.array(features).squeeze()
    
    def extract_combination(self, smiles_list: List[str],
                           feature_type: str = None,
                           combination_method: str = "mean") -> np.ndarray:
        """
        Extract and combine features from multiple SMILES
        (Backward compatibility method)
        
        Args:
            smiles_list: List of SMILES strings
            feature_type: Type of features
            combination_method: How to combine ('mean', 'sum', 'concat')
        
        Returns:
            Combined feature array
        """
        if feature_type is None:
            feature_type = self.feature_type
        
        # Filter valid SMILES
        valid_smiles = [s for s in smiles_list if s and not pd.isna(s)]
        
        # For concat, always preserve L1/L2/L3 positions with zero-padding
        if combination_method == "concat":
            per_ligand_features = []
            for smi in smiles_list:
                if smi and not pd.isna(smi):
                    per_ligand_features.append(self.extract_from_smiles(smi, feature_type))
                else:
                    size = self.get_feature_size(feature_type)
                    per_ligand_features.append(np.zeros(size))
            return np.concatenate(per_ligand_features)
        
        if not valid_smiles:
            # Return zeros based on feature type
            if feature_type == "morgan":
                return np.zeros(self.morgan_bits)
            elif feature_type == "descriptors":
                return np.zeros(self.descriptor_count)
            else:  # combined
                return np.zeros(self.morgan_bits + self.descriptor_count)
        
        # Extract features for each valid SMILES
        features = []
        for smi in valid_smiles:
            feat = self.extract_from_smiles(smi, feature_type)
            features.append(feat)
        
        features = np.array(features)
        
        # Combine
        if combination_method == "mean":
            return np.mean(features, axis=0)
        elif combination_method == "sum":
            return np.sum(features, axis=0)
        else:
            return np.mean(features, axis=0)
    
    # ========================================
    #     Tabular Features
    # ========================================
    
    def extract_tabular_features(self, df: pd.DataFrame,
                                target_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract features from tabular data
        
        Args:
            df: Input DataFrame
            target_columns: Columns to exclude (targets)
        
        Returns:
            Feature array
        """
        target_columns = target_columns or []
        
        # Select feature columns (exclude targets)
        feature_cols = [col for col in df.columns if col not in target_columns]
        
        # Store feature names
        self.feature_names_ = feature_cols
        
        # Handle categorical columns
        df_encoded = df[feature_cols].copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                # Simple label encoding
                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        
        # Convert to array
        features = df_encoded.values.astype(np.float32)
        
        # Handle NaN
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    # ========================================
    #     Unified Interface
    # ========================================
    
    def extract_from_dataframe(self, df: pd.DataFrame,
                              smiles_columns: Optional[List[str]] = None,
                              target_columns: Optional[List[str]] = None,
                              feature_type: Optional[str] = None) -> np.ndarray:
        """
        Extract features from DataFrame (unified interface)
        
        Args:
            df: Input DataFrame
            smiles_columns: Columns containing SMILES (if molecular)
            target_columns: Target columns to exclude (if tabular)
            feature_type: Override feature type
        
        Returns:
            Feature array
        """
        if feature_type is None:
            feature_type = self.feature_type
        
        # Auto-detect if needed
        if feature_type == "auto":
            feature_type = self.detect_data_type(df)
            if feature_type == "molecular" and not smiles_columns:
                # Try to find SMILES columns
                smiles_columns = [col for col in df.columns 
                                 if any(ind in col.lower() for ind in ['smiles', 'l1', 'l2', 'l3'])]
        
        # Extract based on type
        if feature_type in ["morgan", "descriptors", "combined"]:
            # Molecular features
            if not smiles_columns:
                raise ValueError("No SMILES columns specified for molecular features")
            
            features = []
            for _, row in df.iterrows():
                smiles_list = [row[col] if col in row and pd.notna(row[col]) else None 
                              for col in smiles_columns]
                feat = self.extract_combination(smiles_list, feature_type)
                features.append(feat)
            
            return np.array(features)
        
        else:
            # Tabular features
            return self.extract_tabular_features(df, target_columns)
    
    # ========================================
    #     Caching
    # ========================================
    
    def _get_cache_key(self, data: str) -> str:
        """Generate cache key"""
        return hashlib.md5(data.encode()).hexdigest()

    def _rdkit_version_token(self) -> str:
        """Return the RDKit version for molecular-feature cache invalidation."""
        self._import_rdkit()
        rdBase = self._rdkit_modules.get('rdBase')
        return getattr(rdBase, 'rdkitVersion', 'unknown')

    def _descriptor_schema_token(self) -> str:
        """Return a stable token for the active RDKit descriptor schema."""
        self._import_rdkit()
        Descriptors = self._rdkit_modules['Descriptors']
        descriptor_names = [x[0] for x in Descriptors._descList[:self.descriptor_count]]
        descriptor_blob = "|".join(descriptor_names)
        descriptor_hash = hashlib.md5(descriptor_blob.encode()).hexdigest()
        return f"{self._rdkit_version_token()}_{descriptor_hash}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load from cache with robustness against partial writes"""
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            return None

        # Try multiple reads to mitigate transient empty reads during concurrent writes
        import time
        last_err = None
        for _ in range(3):
            try:
                if cache_file.stat().st_size == 0:
                    # Empty file (possibly concurrently created but not fully written), wait and retry
                    time.sleep(0.02)
                    continue
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                last_err = e
                time.sleep(0.02)
        # After multiple failures, treat cache as unavailable
        return None
    
    def _save_to_cache(self, cache_key: str, data: np.ndarray):
        """Save to cache using atomic write to avoid partial files"""
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        tmp_file = self.cache_dir / f"{cache_key}.pkl.tmp"

        try:
            with open(tmp_file, 'wb') as f:
                pickle.dump(data, f)
            # Atomic replace: rename is typically atomic on the same filesystem
            tmp_file.replace(cache_file)
        except Exception:
            # On write failure, try to clean temp file to avoid pollution
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                pass
    
    # ========================================
    #     Utility Methods
    # ========================================
    
    def get_feature_size(self, feature_type: Optional[str] = None, combination_method: Optional[str] = None) -> int:
        """Get the size of feature vector"""
        if feature_type is None:
            feature_type = self.feature_type
            
        if feature_type == "morgan":
            base = self.morgan_bits
        elif feature_type == "descriptors":
            base = self.descriptor_count
        elif feature_type == "combined":
            base = self.morgan_bits + self.descriptor_count
        else:
            return 0  # Tabular size varies

        if combination_method == "concat":
            return base * 3
        return base
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (for tabular data)"""
        return self.feature_names_ if self.feature_names_ else []


# ========================================
#     Backward Compatibility
# ========================================

# Constants for backward compatibility
MORGAN_BITS = 1024
DEFAULT_DESCRIPTOR_COUNT = 85
DESCRIPTOR_NAMES = []

# Load descriptor names if RDKit available
try:
    from rdkit.Chem import Descriptors
    DESCRIPTOR_NAMES = [x[0] for x in Descriptors._descList[:DEFAULT_DESCRIPTOR_COUNT]]
except ImportError:
    pass

# Convenience functions for backward compatibility
def smiles_to_fp(smiles: str, radius: int = 2, bits: int = MORGAN_BITS) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint"""
    extractor = FeatureExtractor(feature_type="morgan")
    extractor.morgan_bits = bits
    extractor.morgan_radius = radius
    return extractor.extract_from_smiles(smiles)

def smiles_to_descriptors(smiles: str) -> np.ndarray:
    """Convert SMILES to molecular descriptors"""
    extractor = FeatureExtractor(feature_type="descriptors")
    return extractor.extract_from_smiles(smiles)
