"""
Smart Data Loader for Enhanced Multi-Target SciBERT
Automatically detects and loads actual dataset or falls back to synthetic data
"""

import pandas as pd
import os
import warnings
from typing import Tuple, Optional
from pathlib import Path

class SmartDataLoader:
    """
    Smart data loader that tries actual dataset first, falls back to synthetic
    """
    
    def __init__(self):
        # Possible locations for the actual dataset
        self.actual_dataset_paths = [
            'rosssea_research_dataset.xlsx',           # Root directory
            'data/rosssea_research_dataset.xlsx',      # Data directory
            '../rosssea_research_dataset.xlsx',        # Parent directory
            './rosssea_research_dataset.xlsx'          # Current directory
        ]
        
        # Synthetic dataset location
        self.synthetic_dataset_path = 'data/synthetic_dataset.csv'
        
        # Alternative synthetic dataset paths
        self.synthetic_fallback_paths = [
            'data/synthetic_dataset.csv',
            'synthetic_dataset.csv',
            'examples/synthetic_dataset.csv'
        ]
        
    def load_dataset(self) -> Tuple[pd.DataFrame, bool]:
        """
        Load dataset with fallback mechanism
        
        Returns:
            tuple: (dataframe, is_actual_dataset)
                - dataframe: The loaded dataset
                - is_actual_dataset: True if actual dataset, False if synthetic
        """
        
        print("Enhanced Multi-Target SciBERT Data Loader")
        print("=" * 50)
        
        # Try to load actual dataset first
        df_actual, loaded_actual = self._try_load_actual_dataset()
        if loaded_actual:
            return df_actual, True
            
        # Fall back to synthetic dataset
        df_synthetic, loaded_synthetic = self._try_load_synthetic_dataset()
        if loaded_synthetic:
            return df_synthetic, False
            
        # If neither works, raise error
        raise FileNotFoundError(
            "Could not load any dataset. Please ensure either:\n"
            "1. Actual dataset 'rosssea_research_dataset.xlsx' is available, or\n"
            "2. Synthetic dataset 'data/synthetic_dataset.csv' exists"
        )
    
    def _try_load_actual_dataset(self) -> Tuple[Optional[pd.DataFrame], bool]:
        """Try to load the actual Ross Sea research dataset"""
        
        print("Attempting to load actual Ross Sea research dataset...")
        
        for path in self.actual_dataset_paths:
            if os.path.exists(path):
                try:
                    print(f"   Found dataset at: {path}")
                    
                    # Load Excel file
                    df = pd.read_excel(path)
                    
                    # Validate dataset structure
                    if self._validate_actual_dataset(df):
                        print(f"   Successfully loaded {len(df)} papers from actual dataset")
                        print(f"   Dataset spans {df['Year'].min()}-{df['Year'].max()}" if 'Year' in df.columns else "")
                        print(f"   Ready for enhanced multi-target classification")
                        return df, True
                    else:
                        print(f"   Dataset structure validation failed for {path}")
                        continue
                        
                except Exception as e:
                    print(f"   Error loading {path}: {str(e)}")
                    continue
        
        print("   Actual dataset not found in any expected location")
        return None, False
    
    def _try_load_synthetic_dataset(self) -> Tuple[Optional[pd.DataFrame], bool]:
        """Try to load synthetic dataset as fallback"""
        
        print("\nFalling back to synthetic dataset...")
        print("   Note: To use actual dataset:")
        print("   1. Obtain 'rosssea_research_dataset.xlsx' from Brooks & Ainley (2022)")
        print("   2. Place in root directory or data/ folder")
        print("   3. Re-run - system will automatically detect and use actual data")
        
        for path in self.synthetic_fallback_paths:
            if os.path.exists(path):
                try:
                    print(f"   Found synthetic dataset at: {path}")
                    
                    # Load CSV file
                    df = pd.read_csv(path)
                    
                    # Validate synthetic dataset structure
                    if self._validate_synthetic_dataset(df):
                        print(f"   Successfully loaded {len(df)} synthetic papers")
                        print(f"   Synthetic dataset ready for methodology validation")
                        print(f"   Note: Performance metrics will differ from actual dataset")
                        return df, True
                    else:
                        print(f"   Synthetic dataset structure validation failed for {path}")
                        continue
                        
                except Exception as e:
                    print(f"   Error loading synthetic dataset {path}: {str(e)}")
                    continue
        
        print("   No synthetic dataset found")
        return None, False
    
    def _validate_actual_dataset(self, df: pd.DataFrame) -> bool:
        """Validate that actual dataset has expected structure"""
        
        required_columns = [
            'Title', 'Abstract', 'Thematic_Tags',
            'CCAMLR_Objectives_Text', 'Management_zones', 'Monitoring_areas'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"   Missing required columns: {missing_columns}")
            return False
            
        # Check for reasonable data
        if len(df) < 10:
            print(f"   Dataset too small: {len(df)} papers")
            return False
            
        # Check for non-empty essential columns
        if df['Title'].isna().all() or df['Abstract'].isna().all():
            print(f"   Essential columns (Title/Abstract) are empty")
            return False
            
        return True
    
    def _validate_synthetic_dataset(self, df: pd.DataFrame) -> bool:
        """Validate that synthetic dataset has expected structure"""
        
        required_columns = [
            'Title', 'Abstract', 'Thematic_Tags',
            'CCAMLR_Objectives_Text', 'Management_zones', 'Monitoring_areas'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"   Missing required columns in synthetic dataset: {missing_columns}")
            return False
            
        # Check for reasonable synthetic data
        if len(df) < 5:
            print(f"   Synthetic dataset too small: {len(df)} papers")
            return False
            
        return True
    
    def get_dataset_info(self, df: pd.DataFrame, is_actual: bool) -> dict:
        """Get comprehensive dataset information"""
        
        info = {
            'dataset_type': 'actual' if is_actual else 'synthetic',
            'total_papers': len(df),
            'columns': list(df.columns),
            'date_range': None,
            'completeness': {}
        }
        
        # Date range analysis
        date_columns = ['Year', 'Publication_Year', 'research_year']
        for col in date_columns:
            if col in df.columns and not df[col].isna().all():
                try:
                    years = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(years) > 0:
                        info['date_range'] = f"{int(years.min())}-{int(years.max())}"
                        break
                except:
                    continue
        
        # Completeness analysis
        key_columns = ['Title', 'Abstract', 'Thematic_Tags', 'CCAMLR_Objectives_Text']
        for col in key_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                info['completeness'][col] = f"{non_null_count}/{len(df)} ({non_null_count/len(df)*100:.1f}%)"
        
        return info


def load_data_with_fallback() -> Tuple[pd.DataFrame, bool]:
    """
    Convenience function to load data with automatic fallback
    
    Returns:
        tuple: (dataframe, is_actual_dataset)
    """
    loader = SmartDataLoader()
    return loader.load_dataset()


def demonstrate_data_loading():
    """Demonstrate the data loading functionality"""
    
    print("Enhanced Multi-Target SciBERT Data Loading Demo")
    print("=" * 60)
    
    try:
        # Load data
        df, is_actual = load_data_with_fallback()
        
        # Get dataset info
        loader = SmartDataLoader()
        info = loader.get_dataset_info(df, is_actual)
        
        print(f"\nDataset Information:")
        print(f"   Type: {info['dataset_type'].upper()}")
        print(f"   Papers: {info['total_papers']}")
        print(f"   Date range: {info['date_range'] or 'Not available'}")
        print(f"   Columns: {len(info['columns'])}")
        
        print(f"\nData Completeness:")
        for col, completeness in info['completeness'].items():
            print(f"   {col}: {completeness}")
        
        print(f"\nData loading successful!")
        print(f"   Ready for enhanced multi-target classification")
        
        return df, is_actual
        
    except Exception as e:
        print(f"\nData loading failed: {str(e)}")
        return None, False


if __name__ == "__main__":
    # Run demonstration
    demonstrate_data_loading()
