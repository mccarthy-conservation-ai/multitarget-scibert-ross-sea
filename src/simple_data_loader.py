# simple_data_loader.py
"""
Simple Data Loader for BERT vs SciBERT Comparison
Loads Ross Sea research dataset with validation and fallback
"""

import pandas as pd
import os
from typing import Tuple, Optional, Dict
from pathlib import Path

class RossSeaDataLoader:
    """Clean data loader for Ross Sea research dataset"""
    
    def __init__(self):
        """Initialize with possible dataset locations"""
        
        # Possible locations for the actual dataset
        self.actual_dataset_paths = [
            'rosssea_research_dataset.xlsx',
            'data/rosssea_research_dataset.xlsx',
            '../rosssea_research_dataset.xlsx',
            './rosssea_research_dataset.xlsx',
            'datasets/rosssea_research_dataset.xlsx',
            '../data/rosssea_research_dataset.xlsx'
        ]
        
        # Synthetic dataset locations
        self.synthetic_dataset_paths = [
            'data/synthetic_dataset.csv',
            'synthetic_dataset.csv',
            'examples/synthetic_dataset.csv',
            '../data/synthetic_dataset.csv',
            './synthetic_dataset.csv'
        ]
        
        # Required columns for validation
        self.required_columns = [
            'Title',
            'Abstract',
            'Thematic_Tags',
            'CCAMLR_Objectives_Text',
            'Management_zones',
            'Monitoring_areas'
        ]
        
        # Optional but useful columns
        self.optional_columns = [
            'Keywords',
            'Year',
            'Authors',
            'DOI',
            'Journal'
        ]
    
    def load_dataset(self, prefer_actual: bool = True) -> Tuple[pd.DataFrame, bool, Dict]:
        """
        Load dataset with validation and comprehensive info
        
        Args:
            prefer_actual: If True, try actual dataset first
            
        Returns:
            tuple: (dataframe, is_actual_dataset, dataset_info)
        """
        
        print("Ross Sea Dataset Loader")
        print("=" * 50)
        
        if prefer_actual:
            # Try actual dataset first
            df, loaded, info = self._try_load_actual_dataset()
            if loaded:
                return df, True, info
            
            # Fall back to synthetic
            df, loaded, info = self._try_load_synthetic_dataset()
            if loaded:
                return df, False, info
        else:
            # Try synthetic first (for testing)
            df, loaded, info = self._try_load_synthetic_dataset()
            if loaded:
                return df, False, info
            
            # Fall back to actual
            df, loaded, info = self._try_load_actual_dataset()
            if loaded:
                return df, True, info
        
        # Neither worked
        raise FileNotFoundError(
            "Could not load any dataset. Please ensure either:\n"
            "1. Actual dataset 'rosssea_research_dataset.xlsx' is available\n"
            "2. Synthetic dataset 'synthetic_dataset.csv' exists\n"
            f"Searched in: {self.actual_dataset_paths + self.synthetic_dataset_paths}"
        )
    
    def _try_load_actual_dataset(self) -> Tuple[Optional[pd.DataFrame], bool, Dict]:
        """Try to load the actual Ross Sea research dataset"""
        
        print("\n📁 Attempting to load actual Ross Sea dataset...")
        
        for path in self.actual_dataset_paths:
            if os.path.exists(path):
                try:
                    print(f"   Found file at: {path}")
                    
                    # Load Excel file
                    df = pd.read_excel(path)
                    
                    # Validate structure
                    is_valid, validation_msg = self._validate_dataset(df)
                    if not is_valid:
                        print(f"   ❌ Validation failed: {validation_msg}")
                        continue
                    
                    # Get dataset info
                    info = self._get_dataset_info(df, path, 'actual')
                    
                    print(f"   ✅ Successfully loaded {len(df)} papers")
                    print(f"   ✅ All required columns present")
                    if 'Year' in df.columns:
                        print(f"   📅 Date range: {df['Year'].min()}-{df['Year'].max()}")
                    
                    return df, True, info
                    
                except Exception as e:
                    print(f"   ❌ Error loading {path}: {str(e)}")
                    continue
        
        print("   ❌ No valid actual dataset found")
        return None, False, {}
    
    def _try_load_synthetic_dataset(self) -> Tuple[Optional[pd.DataFrame], bool, Dict]:
        """Try to load synthetic dataset as fallback"""
        
        print("\n🔄 Attempting to load synthetic dataset...")
        
        for path in self.synthetic_dataset_paths:
            if os.path.exists(path):
                try:
                    print(f"   Found file at: {path}")
                    
                    # Load CSV file
                    df = pd.read_csv(path)
                    
                    # Validate structure
                    is_valid, validation_msg = self._validate_dataset(df)
                    if not is_valid:
                        print(f"   ❌ Validation failed: {validation_msg}")
                        continue
                    
                    # Get dataset info
                    info = self._get_dataset_info(df, path, 'synthetic')
                    
                    print(f"   ✅ Successfully loaded {len(df)} synthetic papers")
                    print(f"   ⚠️  Note: Using synthetic data for testing")
                    print(f"   ℹ️  For actual results, provide 'rosssea_research_dataset.xlsx'")
                    
                    return df, True, info
                    
                except Exception as e:
                    print(f"   ❌ Error loading {path}: {str(e)}")
                    continue
        
        print("   ❌ No valid synthetic dataset found")
        return None, False, {}
    
    def _validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate dataset has required structure
        
        Returns:
            tuple: (is_valid, validation_message)
        """
        
        # Check if empty
        if len(df) == 0:
            return False, "Dataset is empty"
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for reasonable number of papers
        if len(df) < 10:
            return False, f"Dataset too small: only {len(df)} papers"
        
        # Check that essential columns have data
        if df['Title'].isna().all():
            return False, "Title column is completely empty"
        
        if df['Abstract'].isna().all():
            return False, "Abstract column is completely empty"
        
        if df['Thematic_Tags'].isna().all():
            return False, "Thematic_Tags column is completely empty"
        
        # Check data quality
        non_empty_titles = df['Title'].notna().sum()
        non_empty_abstracts = df['Abstract'].notna().sum()
        
        if non_empty_titles < 0.8 * len(df):
            return False, f"Too many missing titles: {len(df) - non_empty_titles} missing"
        
        if non_empty_abstracts < 0.8 * len(df):
            return False, f"Too many missing abstracts: {len(df) - non_empty_abstracts} missing"
        
        return True, "Validation passed"
    
    def _get_dataset_info(self, df: pd.DataFrame, path: str, dataset_type: str) -> Dict:
        """Get comprehensive information about the dataset"""
        
        info = {
            'dataset_type': dataset_type,
            'file_path': path,
            'total_papers': len(df),
            'columns': {
                'all': list(df.columns),
                'required': [col for col in self.required_columns if col in df.columns],
                'optional': [col for col in self.optional_columns if col in df.columns],
                'extra': [col for col in df.columns if col not in self.required_columns + self.optional_columns]
            },
            'completeness': {},
            'date_range': None,
            'label_stats': {}
        }
        
        # Completeness analysis for key columns
        for col in self.required_columns + self.optional_columns:
            if col in df.columns:
                non_null = df[col].notna().sum()
                info['completeness'][col] = {
                    'count': non_null,
                    'percentage': (non_null / len(df)) * 100
                }
        
        # Date range if available
        date_columns = ['Year', 'Publication_Year', 'year']
        for col in date_columns:
            if col in df.columns and not df[col].isna().all():
                try:
                    years = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(years) > 0:
                        info['date_range'] = {
                            'min': int(years.min()),
                            'max': int(years.max()),
                            'span': int(years.max() - years.min() + 1)
                        }
                        break
                except:
                    continue
        
        # Basic label statistics
        if 'Thematic_Tags' in df.columns:
            theme_counts = df['Thematic_Tags'].notna().sum()
            info['label_stats']['themes'] = {
                'papers_with_themes': theme_counts,
                'percentage': (theme_counts / len(df)) * 100
            }
        
        if 'CCAMLR_Objectives_Text' in df.columns:
            obj_counts = df['CCAMLR_Objectives_Text'].notna().sum()
            info['label_stats']['objectives'] = {
                'papers_with_objectives': obj_counts,
                'percentage': (obj_counts / len(df)) * 100
            }
        
        if 'Management_zones' in df.columns:
            zone_counts = df['Management_zones'].notna().sum()
            info['label_stats']['zones'] = {
                'papers_with_zones': zone_counts,
                'percentage': (zone_counts / len(df)) * 100
            }
        
        if 'Monitoring_areas' in df.columns:
            area_counts = df['Monitoring_areas'].notna().sum()
            info['label_stats']['areas'] = {
                'papers_with_areas': area_counts,
                'percentage': (area_counts / len(df)) * 100
            }
        
        return info
    
    def print_dataset_summary(self, df: pd.DataFrame, is_actual: bool, info: Dict):
        """Print a nice summary of the loaded dataset"""
        
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 Dataset Type: {'ACTUAL' if is_actual else 'SYNTHETIC'}")
        print(f"📁 Source: {info['file_path']}")
        print(f"📝 Total Papers: {info['total_papers']}")
        
        if info['date_range']:
            print(f"📅 Date Range: {info['date_range']['min']}-{info['date_range']['max']} "
                  f"({info['date_range']['span']} years)")
        
        print(f"\n📋 Columns:")
        print(f"   Required ({len(info['columns']['required'])}): All present ✅")
        print(f"   Optional ({len(info['columns']['optional'])}): {', '.join(info['columns']['optional'])}")
        if info['columns']['extra']:
            print(f"   Extra ({len(info['columns']['extra'])}): {', '.join(info['columns']['extra'][:5])}...")
        
        print(f"\n🏷️ Label Coverage:")
        for label_type, stats in info['label_stats'].items():
            # The key is different based on label type
            if label_type == 'themes':
                count_key = 'papers_with_themes'
            elif label_type == 'objectives':
                count_key = 'papers_with_objectives'
            elif label_type == 'zones':
                count_key = 'papers_with_zones'
            elif label_type == 'areas':
                count_key = 'papers_with_areas'
            else:
                count_key = 'papers_with_labels'
            
            print(f"   {label_type.capitalize()}: {stats.get(count_key, 'N/A')} papers "
                  f"({stats.get('percentage', 0):.1f}%)")
        
        print(f"\n✅ Data Completeness:")
        for col in ['Title', 'Abstract', 'Keywords']:
            if col in info['completeness']:
                comp = info['completeness'][col]
                print(f"   {col}: {comp['count']}/{info['total_papers']} "
                      f"({comp['percentage']:.1f}%)")
        
        print("\n" + "=" * 70)


def load_ross_sea_dataset(prefer_actual: bool = True,
                         print_summary: bool = True) -> Tuple[pd.DataFrame, bool, Dict]:
    """
    Convenience function to load Ross Sea dataset
    
    Args:
        prefer_actual: Try actual dataset first if True
        print_summary: Print dataset summary if True
        
    Returns:
        tuple: (dataframe, is_actual_dataset, dataset_info)
    """
    
    loader = RossSeaDataLoader()
    df, is_actual, info = loader.load_dataset(prefer_actual=prefer_actual)
    
    if print_summary:
        loader.print_dataset_summary(df, is_actual, info)
    
    return df, is_actual, info


def validate_data_pipeline():
    """Validate that data can be loaded and has expected structure"""
    
    print("🔍 Validating Data Pipeline")
    print("=" * 70)
    
    try:
        # Load data
        df, is_actual, info = load_ross_sea_dataset(print_summary=False)
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Type: {'Actual' if is_actual else 'Synthetic'}")
        print(f"   Papers: {len(df)}")
        
        # Check critical columns
        print(f"\n🔍 Checking critical columns:")
        critical = ['Title', 'Abstract', 'Thematic_Tags', 'CCAMLR_Objectives_Text']
        for col in critical:
            if col in df.columns:
                non_empty = df[col].notna().sum()
                print(f"   ✅ {col}: {non_empty}/{len(df)} non-empty")
            else:
                print(f"   ❌ {col}: MISSING")
        
        # Sample data
        print(f"\n📝 Sample data (first paper):")
        if not df.empty:
            row = df.iloc[0]
            print(f"   Title: {str(row.get('Title', 'N/A'))[:60]}...")
            print(f"   Themes: {str(row.get('Thematic_Tags', 'N/A'))[:60]}...")
            print(f"   Zones: {str(row.get('Management_zones', 'N/A'))}")
        
        print(f"\n✅ Data pipeline validation complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    """Test the data loader"""
    
    print("Testing Ross Sea Data Loader")
    print("=" * 70)
    
    # Validate pipeline
    if validate_data_pipeline():
        print("\n" + "=" * 70)
        print("Loading with full summary:")
        print("=" * 70)
        
        # Load with summary
        df, is_actual, info = load_ross_sea_dataset()
        
        print(f"\n✅ Data loader test complete!")
