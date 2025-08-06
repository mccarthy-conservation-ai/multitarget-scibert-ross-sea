"""
Enhanced Data Preprocessing for Multi-Target SciBERT
Comprehensive preprocessing pipeline for Ross Sea research classification
UPDATED with Species_List and additional metadata features

This preprocessing methodology supports multi-target classification across:
- Research themes
- CCAMLR objectives  
- Management zones
- Monitoring areas
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataPreprocessor:
    """
    Enhanced data preprocessor for multi-target classification
    Now includes Species_List and other critical metadata
    """
    
    def __init__(self):
        """Initialize with comprehensive mapping dictionaries"""
        
        # COMPLETE THEME MAPPING (1-38)
        self.theme_mapping = {
            '1': 'Bioregionalisation & biodiversity mapping',
            '2': 'Physical & biological habitat changes',
            '3': 'Functional ecology processes',
            '4': 'Evolutionary biology processes',
            '5': 'Toothfish stock assessment methods',
            '6': 'Krill & silverfish population trends',
            '7': 'Prey availability effects on predators',
            '8': 'Toothfish & predator distributions',
            '9': 'Toothfish vertical & seasonal distribution',
            '10': 'Dependence on coastal habitats',
            '11': 'Toothfish population structure & roles',
            '12': 'Silverfish spawning & nursery factors',
            '13': 'Trophic productivity at ice edges',
            '14': 'Toothfish recruitment variation',
            '15': 'Under-ice & ice shelf habitat importance',
            '16': 'Krill swarming & top-down effects',
            '17': 'Seal & penguin foraging movements',
            '18': 'Demersal fish community ecology',
            '19': 'Exploitation effects on toothfish',
            '20': 'Toothfish spawning on Ross slope',
            '21': 'Benthic community structure & function',
            '22': 'Krill role in demersal slope environment',
            '23': 'Balleny Islands biota & habitats',
            '24': 'Balleny Islands prey & cetaceans',
            '25': 'Balleny Islands chinstrap penguins',
            '26': 'Balleny Islands endemic benthos',
            '27': 'Balleny Islands as nursery areas',
            '28': 'Toothfish spawning migrations N. Ross',
            '29': 'Toothfish spawning behaviour/dynamics',
            '30': 'Toothfish recruitment factors',
            '31': 'Seamount benthic communities',
            '32': 'Benthic habitats importance to toothfish',
            '33': 'Endemic seamount communities',
            '34': 'Toothfish stocks on seamounts & islands',
            '35': 'Krill hotspots & predator dependence',
            '36': 'Toothfish distribution & biomass NW Ross',
            '37': 'Benthic habitats NW Ross Sea',
            '38': 'Krill spatial dynamics & demography'
        }

        # CCAMLR OBJECTIVES MAPPING
        self.ccamlr_objectives_mapping = {
            'i': 'conserve natural ecological structure',
            'ii': 'scientific reference areas',
            'iii': 'promote research',
            'iv': 'representativeness of benthic and pelagic environments',
            'v': 'large-scale ecosystem processes/areas',
            'vi': 'trophically dominant pelagic prey species',
            'vii': 'key top predator foraging distributions',
            'viii': 'coastal/localized areas of particular ecosystem importance',
            'ix': 'D. mawsoni life cycle areas',
            'x': 'rare or vulnerable benthic habitats',
            'xi': 'promote research of Antarctic krill'
        }

        # COMPREHENSIVE MONITORING AREAS STANDARDIZATION
        self.monitoring_areas_mapping = {
            # Ross Sea Polynya variations
            'ross sea polynya': 'Ross Sea Polynya',
            'Ross Sea Polynya': 'Ross Sea Polynya',
            'ross sea polynia': 'Ross Sea Polynya',
            'Ross Sea polynya': 'Ross Sea Polynya',

            # McMurdo Sound variations
            'mcmurdo sound': 'McMurdo Sound',
            'McMurdo Sound': 'McMurdo Sound',
            'mcmurdo': 'McMurdo Sound',
            'mcmurdo sound polynya': 'McMurdo Sound Polynya',
            'McMurdo Sound Polynya': 'McMurdo Sound Polynya',

            # Erebus Bay variations
            'erebus bay': 'Erebus Bay',
            'Erebus Bay': 'Erebus Bay',
            'ErebusBay': 'Erebus Bay',
            'erebusbay': 'Erebus Bay',

            # Ross Sea regional variations
            'southwestern ross sea': 'Southwestern Ross Sea',
            'Southwestern Ross Sea': 'Southwestern Ross Sea',
            'southwest ross sea': 'Southwestern Ross Sea',
            'Southwest Ross Sea': 'Southwestern Ross Sea',
            'southwesteren ross sea': 'Southwestern Ross Sea',

            'western ross sea': 'Western Ross Sea',
            'Western Ross Sea': 'Western Ross Sea',

            'eastern ross sea': 'Eastern Ross Sea',
            'Eastern Ross Sea': 'Eastern Ross Sea',

            'southern ross sea': 'Southern Ross Sea',
            'Southern Ross Sea': 'Southern Ross Sea',

            'northern ross sea': 'Northern Ross Sea',
            'Northern Ross Sea': 'Northern Ross Sea',

            'central ross sea': 'Central Ross Sea',
            'Central Ross Sea': 'Central Ross Sea',

            'southeastern ross sea': 'Southeastern Ross Sea',
            'Southeastern Ross Sea': 'Southeastern Ross Sea',

            'northwest ross sea': 'Northwest Ross Sea',
            'Northwest Ross Sea': 'Northwest Ross Sea',

            # Ross Ice Shelf variations
            'ross ice shelf': 'Ross Ice Shelf',
            'Ross Ice Shelf': 'Ross Ice Shelf',
            'ross sea ice shelf': 'Ross Ice Shelf',
            'Ross Sea Ice Shelf': 'Ross Ice Shelf',
            'mcmurdo ice shelf': 'McMurdo Ice Shelf',
            'McMurdo Ice Shelf': 'McMurdo Ice Shelf',

            # Continental Shelf/Slope variations
            'continental shelf': 'Ross Sea Continental Shelf',
            'Continental Shelf': 'Ross Sea Continental Shelf',
            'ross sea continental shelf': 'Ross Sea Continental Shelf',
            'Ross Sea Continental Shelf': 'Ross Sea Continental Shelf',
            'ross sea continental slope': 'Ross Sea Continental Slope',
            'Ross Sea Continental Slope': 'Ross Sea Continental Slope',
            'ross sea continental shefl': 'Ross Sea Continental Shelf',  # Typo in data

            # Terra Nova Bay variations
            'terra nova bay': 'Terra Nova Bay',
            'Terra Nova Bay': 'Terra Nova Bay',
            'terra nova bay polynya': 'Terra Nova Bay Polynya',
            'Terra Nova Bay Polynya': 'Terra Nova Bay Polynya',

            # Victoria Land variations
            'victoria land': 'Victoria Land',
            'Victoria Land': 'Victoria Land',
            'victoria coast': 'Victoria Land',

            # Ross Island and Capes
            'ross island': 'Ross Island',
            'Ross Island': 'Ross Island',
            'cape crozier': 'Cape Crozier',
            'Cape Crozier': 'Cape Crozier',
            'cape royds': 'Cape Royds',
            'Cape Royds': 'Cape Royds',
            'cape bird': 'Cape Bird',
            'Cape Bird': 'Cape Bird',
            'cape hallett': 'Cape Hallett',
            'Cape Hallett': 'Cape Hallett',
            'cape washington': 'Cape Washington',
            'Cape Washington': 'Cape Washington',
            'cape colbeck': 'Cape Colbeck',
            'Cape Colbeck': 'Cape Colbeck',
            'cape adare': 'Cape Adare',
            'Cape Adare': 'Cape Adare',

            # Balleny Islands
            'balleny islands': 'Balleny Islands',
            'Balleny Islands': 'Balleny Islands',
            'balleny islands and vicinity': 'Balleny Islands',
            'Balleny Islands and Vicinity': 'Balleny Islands',
            'balleny': 'Balleny Islands',

            # Seamounts
            'admiralty seamount': 'Admiralty Seamount',
            'Admiralty Seamount': 'Admiralty Seamount',
            'admiralty & ross seamounts': 'Admiralty & Ross Seamounts',
            'Admiralty & Ross Seamounts': 'Admiralty & Ross Seamounts',
            'scott seamount': 'Scott Seamount',
            'Scott Seamount': 'Scott Seamount',

            # Ice-related features
            'marginal ice zone': 'Marginal Ice Zone',
            'Marginal Ice Zone': 'Marginal Ice Zone',
            'ross sea pack ice': 'Ross Sea Pack Ice',
            'Ross Sea Pack Ice': 'Ross Sea Pack Ice',
            'drygalski ice tongue': 'Drygalski Ice Tongue',
            'Drygalski Ice Tongue': 'Drygalski Ice Tongue',

            # Banks
            'pennell bank': 'Pennell Bank',
            'Pennell Bank': 'Pennell Bank',

            # Islands
            'coulman island': 'Coulman Island',
            'Coulman Island': 'Coulman Island',

            # General areas
            'all': 'All RSRMPA',
            'All': 'All RSRMPA',
            'ross sea': 'Ross Sea (General)',
            'Ross Sea': 'Ross Sea (General)'
        }

        # Zone mapping for text conversion
        self.zone_mapping = {
            'GPZ': 'General Protection Zone',
            'SRZ': 'Special Research Zone',
            'KRZ': 'Krill Research Zone'
        }

        print("Enhanced Data Preprocessor initialized with Species_List support")
        print(f"   Theme mapping: {len(self.theme_mapping)} themes")
        print(f"   CCAMLR objectives: {len(self.ccamlr_objectives_mapping)} objectives")
        print(f"   Monitoring areas: {len(self.monitoring_areas_mapping)} standardizations")
        print(f"   Management zones: {len(self.zone_mapping)} zones")

    def create_enhanced_combined_text(self, row: pd.Series) -> str:
        """
        Enhanced text combination with spatial, contextual, and SPECIES features
        UPDATED to include Species_List and additional metadata
        """
        # Basic text fields
        title = str(row['Title']) if pd.notna(row['Title']) else ''
        abstract = str(row['Abstract']) if pd.notna(row['Abstract']) else ''
        keywords = str(row['Keywords']) if pd.notna(row['Keywords']) else ''

        # CRITICAL ADDITION: Species List
        # This is essential for themes like "Exploitation effects on toothfish"
        species_list = str(row['Species_List']) if pd.notna(row['Species_List']) else ''
        
        # CRITICAL ADDITION: Method Tags
        # Essential for themes like "Ocean acidification" that require specific methods
        method_tags = str(row['Method_Tags']) if pd.notna(row['Method_Tags']) else ''
        
        # Additional metadata for improved classification
        named_features = str(row['Named_Features']) if pd.notna(row['Named_Features']) else ''
        temporal_scale = str(row['Temporal_Scale']) if pd.notna(row['Temporal_Scale']) else ''
        spatial_scale = str(row['Spatial_Scale']) if pd.notna(row['Spatial_Scale']) else ''
        data_availability = str(row['Data_Availability']) if pd.notna(row['Data_Availability']) else ''
        
        # Extract temporal information
        research_period = ''
        if pd.notna(row.get('Research_start_date')) or pd.notna(row.get('Research_end_date')):
            start_date = str(row.get('Research_start_date', '')).strip()
            end_date = str(row.get('Research_end_date', '')).strip()
            if start_date or end_date:
                research_period = f"{start_date}-{end_date}".strip('-')

        # Enhanced spatial information (existing)
        spatial_themes = str(row['RMP_Spatial_Themes']) if pd.notna(row['RMP_Spatial_Themes']) else ''

        # Add management zones as text features
        management_zones = ''
        if pd.notna(row['Management_zones']):
            zones = [zone.strip() for zone in str(row['Management_zones']).split(',')]
            zone_descriptions = []
            for zone in zones:
                if zone in self.zone_mapping:
                    zone_descriptions.append(self.zone_mapping[zone])
                else:
                    zone_descriptions.append(zone)
            management_zones = ' '.join(zone_descriptions)

        # Add standardized monitoring areas
        monitoring_areas = ''
        if pd.notna(row['Monitoring_areas']):
            areas = [area.strip() for area in str(row['Monitoring_areas']).split(',')]
            standardized_areas = []
            for area in areas:
                # Apply comprehensive standardization mapping
                area_lower = area.lower()
                standardized = self.monitoring_areas_mapping.get(area_lower, area)
                standardized_areas.append(standardized)
            monitoring_areas = ' '.join(standardized_areas)

        # Add CCAMLR objectives as text features
        objectives_text = ''
        if pd.notna(row.get('CCAMLR_Objectives_Text')):
            objectives_text = str(row['CCAMLR_Objectives_Text'])
            # Clean up the text by removing brackets and pipes
            objectives_text = objectives_text.replace('|', ' ').replace('(', '').replace(')', '')

        # Clean up all text fields
        title = ' '.join(str(title).split())
        abstract = ' '.join(str(abstract).split())
        keywords = ' '.join(str(keywords).split())
        species_list = ' '.join(str(species_list).split())
        method_tags = ' '.join(str(method_tags).split())
        named_features = ' '.join(str(named_features).split())
        spatial_themes = ' '.join(str(spatial_themes).split())
        management_zones = ' '.join(str(management_zones).split())
        monitoring_areas = ' '.join(str(monitoring_areas).split())
        objectives_text = ' '.join(str(objectives_text).split())
        temporal_scale = ' '.join(str(temporal_scale).split())
        spatial_scale = ' '.join(str(spatial_scale).split())

        # Combine with enhanced special tokens including SPECIES and METHODS
        # The order matters - put most important information first
        combined = (f"[TITLE] {title} "
                   f"[ABSTRACT] {abstract} "
                   f"[KEYWORDS] {keywords} "
                   f"[SPECIES] {species_list} "  # CRITICAL for species-specific themes
                   f"[METHODS] {method_tags} "   # CRITICAL for method-specific themes
                   f"[LOCATIONS] {named_features} "  # Geographic context
                   f"[TEMPORAL] {temporal_scale} {research_period} "  # Time context
                   f"[SPATIAL] {spatial_themes} {spatial_scale} "
                   f"[ZONES] {management_zones} "
                   f"[AREAS] {monitoring_areas} "
                   f"[OBJECTIVES] {objectives_text}")

        return combined

    def parse_themes(self, theme_str: str) -> List[str]:
        """Convert theme string to list"""
        if pd.isna(theme_str) or theme_str == '':
            return []
        try:
            # Handle both comma and pipe separators
            themes = str(theme_str).replace('|', ',').split(',')
            themes = [t.strip() for t in themes if t.strip()]
            return themes
        except:
            return []

    def parse_ccamlr_objectives(self, row: pd.Series) -> List[str]:
        """Parse CCAMLR objectives from text or binary columns"""
        objectives = []

        # Try text-based first (preferred)
        if pd.notna(row.get('CCAMLR_Objectives_Text')):
            text = str(row['CCAMLR_Objectives_Text'])
            # Extract roman numerals like (i), (ii), etc.
            roman_matches = re.findall(r'\(([ivx]+)\)', text.lower())
            objectives.extend(roman_matches)

        # Fallback to binary columns
        else:
            objectives_cols = ['Objective_I', 'Objective_II', 'Objective_III', 'Objective_IV',
                              'Objective_V', 'Objective_VI', 'Objective_VII', 'Objective_VIII',
                              'Objective_IX', 'Objective_X', 'Objective_XI']

            roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi']

            for i, col in enumerate(objectives_cols):
                if col in row and row[col] == 1:
                    objectives.append(roman_numerals[i])

        return objectives

    def parse_management_zones(self, zones_str: str) -> List[str]:
        """Parse management zones"""
        if pd.isna(zones_str) or zones_str == '':
            return []
        
        zones = [zone.strip() for zone in str(zones_str).split(',')]
        return [zone for zone in zones if zone in ['GPZ', 'SRZ', 'KRZ']]

    def parse_monitoring_areas(self, areas_str: str) -> List[str]:
        """Parse and standardize monitoring areas"""
        if pd.isna(areas_str) or areas_str == '':
            return []

        areas = [area.strip() for area in str(areas_str).split(',')]
        standardized_areas = []

        for area in areas:
            area_lower = area.lower()
            standardized = self.monitoring_areas_mapping.get(area_lower, area)
            standardized_areas.append(standardized)

        return standardized_areas

    def prepare_multi_target_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare multi-target data for classification
        Now includes metadata feature extraction
        """
        print("\nPreparing Multi-Target Data with Enhanced Metadata...")
        print("=" * 50)

        # Check for metadata columns and report availability
        metadata_cols = ['Species_List', 'Method_Tags', 'Named_Features',
                        'Temporal_Scale', 'Spatial_Scale', 'Research_start_date',
                        'Research_end_date', 'Data_Availability']
        
        print("Checking metadata availability:")
        for col in metadata_cols:
            if col in df.columns:
                non_empty = df[col].notna().sum()
                print(f"   ✓ {col}: {non_empty}/{len(df)} papers ({non_empty/len(df)*100:.1f}%)")
            else:
                print(f"   ✗ {col}: NOT FOUND in dataset")

        # Create enhanced combined text for all papers
        print("\n   Creating enhanced combined text with metadata...")
        df['enhanced_combined_text'] = df.apply(self.create_enhanced_combined_text, axis=1)

        # Sample text to verify metadata inclusion
        if len(df) > 0:
            sample_text = df['enhanced_combined_text'].iloc[0]
            print(f"\n   Sample enhanced text includes:")
            for token in ['[SPECIES]', '[METHODS]', '[LOCATIONS]', '[TEMPORAL]']:
                if token in sample_text:
                    print(f"     ✓ {token} section found")

        # Parse all targets
        print("\n   Parsing classification targets...")
        df['theme_list'] = df['Thematic_Tags'].apply(self.parse_themes)

        print("   Parsing CCAMLR objectives...")
        df['objectives_list'] = df.apply(self.parse_ccamlr_objectives, axis=1)

        print("   Parsing management zones...")
        df['zones_list'] = df['Management_zones'].apply(self.parse_management_zones)

        print("   Parsing monitoring areas...")
        df['monitoring_areas_list'] = df['Monitoring_areas'].apply(self.parse_monitoring_areas)

        # Filter monitoring areas to top areas (≥5 papers)
        print("   Filtering monitoring areas...")
        all_areas = []
        for areas in df['monitoring_areas_list']:
            all_areas.extend(areas)
        
        area_counts = Counter(all_areas)
        top_areas = [area for area, count in area_counts.items() if count >= 5]
        
        print(f"   Filtering monitoring areas: {len(area_counts)} → {len(top_areas)} (≥5 papers)")
        
        # Filter to top areas only
        df['top_monitoring_areas_list'] = df['monitoring_areas_list'].apply(
            lambda areas_list: [area for area in areas_list if area in top_areas]
        )
        
        # Create binary label matrices using MultiLabelBinarizer
        print("\n   Creating binary label matrices...")
        
        # Themes
        mlb_themes = MultiLabelBinarizer()
        y_themes = mlb_themes.fit_transform(df['theme_list'])
        theme_classes = mlb_themes.classes_

        # Objectives
        mlb_objectives = MultiLabelBinarizer()
        y_objectives = mlb_objectives.fit_transform(df['objectives_list'])
        objective_classes = mlb_objectives.classes_

        # Zones
        mlb_zones = MultiLabelBinarizer()
        y_zones = mlb_zones.fit_transform(df['zones_list'])
        zone_classes = mlb_zones.classes_

        # Areas
        mlb_areas = MultiLabelBinarizer()
        y_areas = mlb_areas.fit_transform(df['top_monitoring_areas_list'])
        area_classes = mlb_areas.classes_

        # Create label matrices dictionary
        label_matrices = {
            'themes': y_themes,
            'objectives': y_objectives,
            'zones': y_zones,
            'areas': y_areas
        }

        # Create class information
        class_info = {
            'theme_classes': theme_classes,
            'objective_classes': objective_classes,
            'zone_classes': zone_classes,
            'area_classes': area_classes,
            'mlb_themes': mlb_themes,
            'mlb_objectives': mlb_objectives,
            'mlb_zones': mlb_zones,
            'mlb_areas': mlb_areas
        }

        # Print summary statistics
        print(f"\nMulti-Target Data Summary:")
        print(f"   Total papers: {len(df)}")
        print(f"   Themes: {len(theme_classes)} classes")
        print(f"   CCAMLR objectives: {len(objective_classes)} classes")
        print(f"   Management zones: {len(zone_classes)} classes")
        print(f"   Monitoring areas: {len(area_classes)} classes (top areas)")
        print(f"   Enhanced text created with species and method information")

        # Label distribution statistics
        print(f"\nLabel Distribution:")
        print(f"   Themes per paper: {df['theme_list'].apply(len).mean():.2f} avg")
        print(f"   Objectives per paper: {df['objectives_list'].apply(len).mean():.2f} avg")
        print(f"   Zones per paper: {df['zones_list'].apply(len).mean():.2f} avg")
        print(f"   Areas per paper: {df['top_monitoring_areas_list'].apply(len).mean():.2f} avg")

        # Species coverage analysis (if Species_List exists)
        if 'Species_List' in df.columns:
            species_coverage = df['Species_List'].notna().sum()
            print(f"\nSpecies Information:")
            print(f"   Papers with species data: {species_coverage}/{len(df)} ({species_coverage/len(df)*100:.1f}%)")
            
            # Check for toothfish papers
            if species_coverage > 0:
                toothfish_papers = df[df['Species_List'].str.contains('Dissostichus', case=False, na=False)]
                print(f"   Papers mentioning Dissostichus (toothfish): {len(toothfish_papers)}")

        return df, {
            'label_matrices': label_matrices,
            'class_info': class_info,
            'mappings': {
                'theme_mapping': self.theme_mapping,
                'ccamlr_objectives_mapping': self.ccamlr_objectives_mapping,
                'monitoring_areas_mapping': self.monitoring_areas_mapping,
                'zone_mapping': self.zone_mapping
            }
        }

    def create_train_test_splits(self, df: pd.DataFrame, multi_target_data: Dict) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/validation/test splits for multi-target data
        """
        print("\nCreating train/validation/test splits...")
        
        # Get label matrices
        label_matrices = multi_target_data['label_matrices']
        
        # Create combined target matrix for stratification
        y_combined = np.hstack([
            label_matrices['themes'] * 3,  # Weight themes more heavily
            label_matrices['objectives'],
            label_matrices['zones'] * 2,   # Weight zones moderately
        ])
        
        X_indices = np.arange(len(df)).reshape(-1, 1)
        
        # First split: 85% train+val, 15% test
        X_temp_idx, y_temp_combined, X_test_idx, y_test_combined = iterative_train_test_split(
            X_indices, y_combined, test_size=0.15
        )
        
        # Second split: from 85%, split into train (70%) and val (15%)
        X_train_idx, y_train_combined, X_val_idx, y_val_combined = iterative_train_test_split(
            X_temp_idx, y_temp_combined, test_size=0.176  # 15%/85% = 0.176
        )
        
        train_indices = X_train_idx.flatten()
        val_indices = X_val_idx.flatten()
        test_indices = X_test_idx.flatten()
        
        # Create split data dictionary
        texts = df['enhanced_combined_text'].values
        
        # Text variables
        splits = {
            'X_train': texts[train_indices],
            'X_val': texts[val_indices],
            'X_test': texts[test_indices]
        }
        
        # Label variables for each target
        for target, matrix in label_matrices.items():
            splits[f'y_train_{target}'] = matrix[train_indices]
            splits[f'y_val_{target}'] = matrix[val_indices]
            splits[f'y_test_{target}'] = matrix[test_indices]
        
        print(f"Train/validation/test splits created:")
        print(f"   Train: {len(train_indices)} papers ({len(train_indices)/len(df)*100:.1f}%)")
        print(f"   Val: {len(val_indices)} papers ({len(val_indices)/len(df)*100:.1f}%)")
        print(f"   Test: {len(test_indices)} papers ({len(test_indices)/len(df)*100:.1f}%)")
        
        return splits, train_indices, val_indices, test_indices


def preprocess_dataset_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """
    Complete preprocessing pipeline for multi-target SciBERT training
    Now includes metadata features for improved classification
    
    Returns processed dataframe, multi-target data, train/val/test splits, and analysis results
    """
    
    print("Enhanced Multi-Target Data Preprocessing Pipeline with Metadata")
    print("======================================================================")
    
    # Initialize preprocessor
    preprocessor = EnhancedDataPreprocessor()
    
    # Prepare multi-target data
    processed_df, multi_target_data = preprocessor.prepare_multi_target_data(df)
    
    # Create train/validation/test splits
    splits, train_idx, val_idx, test_idx = preprocessor.create_train_test_splits(
        processed_df, multi_target_data
    )
    
    # Analysis results
    analysis_results = {
        'text_statistics': {
            'avg_length': processed_df['enhanced_combined_text'].str.len().mean(),
            'max_length': processed_df['enhanced_combined_text'].str.len().max(),
        },
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_indices': test_idx,
        'preprocessing_version': 'enhanced_multi_target_with_metadata_v2'
    }
    
    print(f"\nPreprocessing Pipeline Complete!")
    print(f"   {len(processed_df)} papers processed")
    print(f"   {len(multi_target_data['label_matrices'])} classification targets prepared")
    print(f"   Enhanced text with species and method features created")
    print(f"   Multi-target labels ready for training")
    print(f"   Train/validation/test splits created")
    
    # Verify variables are created correctly
    print(f"\nVerification of Variables:")
    for var_name in ['X_train', 'X_val', 'X_test']:
        print(f"   {var_name}: {len(splits[var_name])} samples")
    
    for target in ['themes', 'objectives', 'zones', 'areas']:
        for split in ['train', 'val', 'test']:
            var_name = f'y_{split}_{target}'
            if var_name in splits:
                shape = splits[var_name].shape
                print(f"   {var_name}: {shape}")
    
    return processed_df, multi_target_data, splits, analysis_results


def demonstrate_preprocessing():
    """Demonstrate preprocessing functionality with metadata"""
    
    print("Enhanced Data Preprocessing Demo with Species and Methods")
    print("=" * 70)
    
    try:
        # Import data loader
        from data_loader import load_data_with_fallback
        
        # Load dataset
        print("Loading dataset...")
        df, is_actual = load_data_with_fallback()
        
        # Run preprocessing
        processed_df, multi_target_data, splits, analysis = preprocess_dataset_for_training(df)
        
        # Show sample enhanced text
        print(f"\nSample Enhanced Text:")
        print("-" * 40)
        sample_text = processed_df['enhanced_combined_text'].iloc[0]
        print(f"Length: {len(sample_text)} characters")
        
        # Show what sections are included
        sections = ['[TITLE]', '[ABSTRACT]', '[SPECIES]', '[METHODS]', '[LOCATIONS]',
                   '[TEMPORAL]', '[SPATIAL]', '[ZONES]', '[AREAS]', '[OBJECTIVES]']
        print("\nSections included:")
        for section in sections:
            if section in sample_text:
                start = sample_text.find(section)
                end = sample_text.find('[', start + 1)
                if end == -1:
                    end = len(sample_text)
                content = sample_text[start:end].strip()
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"{section}: {content}")
        
        # Show variables created
        print(f"\nVariables Created:")
        print("-" * 50)
        
        # Text variables
        print("Text Variables:")
        for var_name in ['X_train', 'X_val', 'X_test']:
            if var_name in splits:
                print(f"   {var_name}: {len(splits[var_name])} samples")
        
        # Label variables
        print("\nLabel Variables:")
        for target in ['themes', 'objectives', 'zones', 'areas']:
            for split in ['train', 'val', 'test']:
                var_name = f'y_{split}_{target}'
                if var_name in splits:
                    shape = splits[var_name].shape
                    print(f"   {var_name}: {shape} (classes: {shape[1]})")
        
        # Show class counts
        print(f"\nClass Counts:")
        print("-" * 30)
        class_info = multi_target_data['class_info']
        print(f"   Themes: {len(class_info['theme_classes'])} classes")
        print(f"   Objectives: {len(class_info['objective_classes'])} classes")
        print(f"   Zones: {len(class_info['zone_classes'])} classes")
        print(f"   Areas: {len(class_info['area_classes'])} classes")
        
        # Verify enhanced text format
        print(f"\nEnhanced Text Format Verification:")
        print("-" * 35)
        sample = processed_df['enhanced_combined_text'].iloc[0]
        required_tokens = ['[TITLE]', '[ABSTRACT]', '[KEYWORDS]', '[SPECIES]', '[METHODS]',
                          '[LOCATIONS]', '[TEMPORAL]', '[SPATIAL]', '[ZONES]', '[AREAS]', '[OBJECTIVES]']
        for token in required_tokens:
            present = token in sample
            print(f"   {token}: {'Present' if present else 'Missing'}")
        
        # Show species coverage for toothfish themes
        if 'Species_List' in df.columns:
            print(f"\nSpecies Coverage Analysis:")
            print("-" * 30)
            toothfish_themes = ['5', '8', '9', '11', '14', '19', '20', '28', '29', '30', '32', '34', '36']
            toothfish_papers = processed_df[processed_df['theme_list'].apply(
                lambda themes: any(t in toothfish_themes for t in themes)
            )]
            
            if len(toothfish_papers) > 0:
                species_coverage = toothfish_papers['Species_List'].notna().sum()
                print(f"   Papers with toothfish themes: {len(toothfish_papers)}")
                print(f"   Of those, papers with species data: {species_coverage} ({species_coverage/len(toothfish_papers)*100:.1f}%)")
                
                # Check if Dissostichus is mentioned
                dissostichus_papers = toothfish_papers[
                    toothfish_papers['Species_List'].str.contains('Dissostichus', case=False, na=False)
                ]
                print(f"   Papers explicitly mentioning Dissostichus: {len(dissostichus_papers)}")
        
        print(f"\nPreprocessing Complete!")
        print(f"   Ready for model training with enhanced metadata")
        print(f"   Species and method information now included in text")
        
        return processed_df, multi_target_data, splits, analysis
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# Compatibility function for existing code
def preprocess_dataset_original_style(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """
    Compatibility wrapper for existing training code
    """
    return preprocess_dataset_for_training(df)


if __name__ == "__main__":
    # Run demonstration of preprocessing with metadata
    demonstrate_preprocessing()
