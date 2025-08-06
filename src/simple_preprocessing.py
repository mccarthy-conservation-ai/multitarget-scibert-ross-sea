# simple_preprocessing.py
"""
Simple Preprocessing for BERT vs SciBERT Comparison
Preserves all important mappings from the original preprocessing
ENHANCED with Species_List and other metadata features
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, List
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter

class SimplePreprocessor:
    """Simple preprocessor for multi-target classification with complete mappings"""
    
    def __init__(self):
        self.mlb_dict = {}
        
        # COMPLETE THEME MAPPING (1-38) - CRITICAL FOR PAPER
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

        # CCAMLR OBJECTIVES MAPPING - CRITICAL FOR POLICY ALIGNMENT
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

        # COMPREHENSIVE MONITORING AREAS STANDARDIZATION - CRITICAL FOR SPATIAL ANALYSIS
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

        # Zone mapping
        self.zone_mapping = {
            'GPZ': 'General Protection Zone',
            'SRZ': 'Special Research Zone',
            'KRZ': 'Krill Research Zone'
        }
    
    def create_simple_text(self, row: pd.Series) -> str:
        """
        ENHANCED: Combine title, abstract, keywords, AND METADATA into comprehensive text
        This enhanced version includes species, methods, locations, and temporal information
        """
        # Basic text fields
        title = str(row.get('Title', '')).strip() if pd.notna(row.get('Title')) else ''
        abstract = str(row.get('Abstract', '')).strip() if pd.notna(row.get('Abstract')) else ''
        keywords = str(row.get('Keywords', '')).strip() if pd.notna(row.get('Keywords')) else ''
        
        # ENHANCED METADATA FIELDS - CRITICAL FOR IMPROVED CLASSIFICATION
        species_list = str(row.get('Species_List', '')).strip() if pd.notna(row.get('Species_List')) else ''
        method_tags = str(row.get('Method_Tags', '')).strip() if pd.notna(row.get('Method_Tags')) else ''
        named_features = str(row.get('Named_Features', '')).strip() if pd.notna(row.get('Named_Features')) else ''
        spatial_scale = str(row.get('Spatial_Scale', '')).strip() if pd.notna(row.get('Spatial_Scale')) else ''
        temporal_scale = str(row.get('Temporal_Scale', '')).strip() if pd.notna(row.get('Temporal_Scale')) else ''
        
        # Additional potentially useful metadata
        data_availability = str(row.get('Data_Availability', '')).strip() if pd.notna(row.get('Data_Availability')) else ''
        publication_type = str(row.get('Publication_Type', '')).strip() if pd.notna(row.get('Publication_Type')) else ''
        
        # Extract temporal information
        research_dates = ''
        if pd.notna(row.get('Research_start_date')) or pd.notna(row.get('Research_end_date')):
            start_date = str(row.get('Research_start_date', '')).strip()
            end_date = str(row.get('Research_end_date', '')).strip()
            if start_date or end_date:
                research_dates = f"{start_date}-{end_date}".strip('-')
        
        # Clean up text fields
        title = ' '.join(title.split())
        abstract = ' '.join(abstract.split())
        keywords = ' '.join(keywords.split())
        species_list = ' '.join(species_list.split())
        method_tags = ' '.join(method_tags.split())
        named_features = ' '.join(named_features.split())
        spatial_scale = ' '.join(spatial_scale.split())
        temporal_scale = ' '.join(temporal_scale.split())
        
        # Enhanced concatenation with clear markers for each metadata type
        text_parts = []
        
        # Core content
        if title:
            text_parts.append(title + '.')
        if abstract:
            text_parts.append(abstract)
        
        # Enhanced metadata with clear markers
        if keywords:
            text_parts.append(f"Keywords: {keywords}")
        
        # CRITICAL ADDITIONS FOR SPECIES-RELATED THEMES
        if species_list:
            # This is crucial for themes like "Exploitation effects on toothfish"
            text_parts.append(f"Species studied: {species_list}")
        
        # CRITICAL ADDITIONS FOR METHOD-RELATED THEMES
        if method_tags:
            # This helps with themes like "Ocean acidification" that require specific methods
            text_parts.append(f"Methods used: {method_tags}")
        
        # SPATIAL CONTEXT
        if named_features:
            # Important for geographic-specific themes
            text_parts.append(f"Study locations: {named_features}")
        
        if spatial_scale:
            text_parts.append(f"Spatial scale: {spatial_scale}")
        
        # TEMPORAL CONTEXT
        if temporal_scale:
            # Critical for themes like "Climate-driven ocean circulation changes"
            text_parts.append(f"Temporal scale: {temporal_scale}")
        
        if research_dates:
            text_parts.append(f"Study period: {research_dates}")
        
        # Additional context
        if publication_type and publication_type.lower() != 'journal article':
            text_parts.append(f"Publication type: {publication_type}")
        
        return ' '.join(text_parts)
    
    def parse_themes(self, theme_str: str) -> List[str]:
        """Parse themes - handles both comma and pipe separators"""
        if pd.isna(theme_str) or theme_str == '':
            return []
        try:
            # Handle both comma and pipe separators
            themes = str(theme_str).replace('|', ',').split(',')
            themes = [t.strip() for t in themes if t.strip()]
            return themes
        except:
            return []
    
    def parse_objectives(self, row: pd.Series) -> List[str]:
        """Parse CCAMLR objectives from text or binary columns"""
        objectives = []
        
        # Try text-based first (preferred)
        if pd.notna(row.get('CCAMLR_Objectives_Text')):
            text = str(row['CCAMLR_Objectives_Text']).lower()
            # Extract roman numerals like (i), (ii), etc.
            matches = re.findall(r'\(([ivx]+)\)', text)
            objectives.extend(matches)
        else:
            # Fallback to binary columns if they exist
            objectives_cols = ['Objective_I', 'Objective_II', 'Objective_III', 'Objective_IV',
                              'Objective_V', 'Objective_VI', 'Objective_VII', 'Objective_VIII',
                              'Objective_IX', 'Objective_X', 'Objective_XI']
            
            roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi']
            
            for i, col in enumerate(objectives_cols):
                if col in row and row[col] == 1:
                    objectives.append(roman_numerals[i])
        
        return objectives
    
    def parse_zones(self, zones_str: str) -> List[str]:
        """Parse management zones"""
        if pd.isna(zones_str) or zones_str == '':
            return []
        
        zones = [zone.strip() for zone in str(zones_str).split(',')]
        # Only keep valid zones
        return [zone for zone in zones if zone in ['GPZ', 'SRZ', 'KRZ']]
    
    def parse_areas(self, areas_str: str) -> List[str]:
        """Parse and standardize monitoring areas"""
        if pd.isna(areas_str) or areas_str == '':
            return []
        
        areas = [area.strip() for area in str(areas_str).split(',')]
        standardized_areas = []
        
        for area in areas:
            # Apply standardization mapping
            area_lower = area.lower()
            standardized = self.monitoring_areas_mapping.get(area_lower, area)
            standardized_areas.append(standardized)
        
        return standardized_areas
    
    def parse_labels(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, List]]:
        """Parse all multi-label targets and return both matrices and class names"""
        
        # Parse all label types
        print("Parsing labels...")
        df['theme_list'] = df['Thematic_Tags'].apply(self.parse_themes)
        df['objective_list'] = df.apply(self.parse_objectives, axis=1)
        df['zone_list'] = df['Management_zones'].apply(self.parse_zones)
        df['area_list'] = df['Monitoring_areas'].apply(self.parse_areas)
        
        # Filter areas to only those with ≥5 papers (as in original)
        all_areas = []
        for areas in df['area_list']:
            all_areas.extend(areas)
        
        area_counts = Counter(all_areas)
        top_areas = [area for area, count in area_counts.items() if count >= 5]
        
        print(f"Filtering areas: {len(area_counts)} → {len(top_areas)} (≥5 papers)")
        
        # Filter to top areas only
        df['filtered_area_list'] = df['area_list'].apply(
            lambda areas: [area for area in areas if area in top_areas]
        )
        
        # Create binary matrices
        label_matrices = {}
        class_info = {}
        
        # Themes
        mlb_themes = MultiLabelBinarizer()
        label_matrices['themes'] = mlb_themes.fit_transform(df['theme_list'])
        self.mlb_dict['themes'] = mlb_themes
        class_info['themes'] = list(mlb_themes.classes_)
        
        # Objectives
        mlb_objectives = MultiLabelBinarizer()
        label_matrices['objectives'] = mlb_objectives.fit_transform(df['objective_list'])
        self.mlb_dict['objectives'] = mlb_objectives
        class_info['objectives'] = list(mlb_objectives.classes_)
        
        # Zones
        mlb_zones = MultiLabelBinarizer()
        label_matrices['zones'] = mlb_zones.fit_transform(df['zone_list'])
        self.mlb_dict['zones'] = mlb_zones
        class_info['zones'] = list(mlb_zones.classes_)
        
        # Areas (filtered)
        mlb_areas = MultiLabelBinarizer()
        label_matrices['areas'] = mlb_areas.fit_transform(df['filtered_area_list'])
        self.mlb_dict['areas'] = mlb_areas
        class_info['areas'] = list(mlb_areas.classes_)
        
        # Print summary
        print(f"\nLabel encoding complete:")
        for target, matrix in label_matrices.items():
            print(f"  {target}: {matrix.shape[1]} classes, "
                  f"{matrix.sum(axis=1).mean():.2f} avg labels per paper")
        
        # Print metadata usage summary
        print(f"\nMetadata features included in text:")
        metadata_cols = ['Species_List', 'Method_Tags', 'Named_Features', 'Spatial_Scale', 'Temporal_Scale']
        for col in metadata_cols:
            if col in df.columns:
                non_empty = df[col].notna().sum()
                print(f"  {col}: {non_empty}/{len(df)} papers ({non_empty/len(df)*100:.1f}%)")
        
        return label_matrices, class_info


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Prepare data for BERT vs SciBERT comparison
    ENHANCED VERSION with metadata features
    
    Returns:
        texts: Array of text strings (now includes metadata)
        labels: Dictionary of label matrices by target
        info: Additional information (classes, mappings, etc.)
    """
    preprocessor = SimplePreprocessor()
    
    # Create enhanced text with metadata
    print("Creating enhanced text representations with metadata...")
    df['text'] = df.apply(preprocessor.create_simple_text, axis=1)
    texts = df['text'].values
    
    # Parse labels
    labels, class_info = preprocessor.parse_labels(df)
    
    # Create comprehensive info dictionary
    info = {
        'num_papers': len(df),
        'class_counts': {k: v.shape[1] for k, v in labels.items()},
        'class_names': class_info,
        'mappings': {
            'theme_mapping': preprocessor.theme_mapping,
            'objectives_mapping': preprocessor.ccamlr_objectives_mapping,
            'areas_mapping': preprocessor.monitoring_areas_mapping,
            'zone_mapping': preprocessor.zone_mapping
        },
        'preprocessor': preprocessor,
        'text_stats': {
            'avg_length': np.mean([len(t) for t in texts]),
            'max_length': max([len(t) for t in texts]),
            'min_length': min([len(t) for t in texts])
        }
    }
    
    print(f"\n✓ Preprocessed {len(texts)} papers with enhanced metadata")
    print(f"  Average text length: {info['text_stats']['avg_length']:.0f} chars")
    print(f"  Text now includes: Species, Methods, Locations, Temporal/Spatial scales")
    for target, matrix in labels.items():
        print(f"  {target}: {matrix.shape[1]} classes")
    
    return texts, labels, info


def create_splits(texts: np.ndarray, labels: Dict[str, np.ndarray],
                  test_size: float = 0.15, val_size: float = 0.15,
                  random_state: int = 42) -> Dict:
    """
    Create train/val/test splits maintaining label distribution
    
    Args:
        texts: Array of text strings
        labels: Dictionary of label matrices
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing all splits
    """
    
    # Create indices
    indices = np.arange(len(texts))
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True
    )
    
    # Create splits dictionary
    splits = {
        'X_train': texts[train_idx],
        'X_val': texts[val_idx],
        'X_test': texts[test_idx],
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    
    # Add label splits for each target
    for target, label_matrix in labels.items():
        splits[f'y_train_{target}'] = label_matrix[train_idx]
        splits[f'y_val_{target}'] = label_matrix[val_idx]
        splits[f'y_test_{target}'] = label_matrix[test_idx]
    
    print(f"\n✓ Created splits:")
    print(f"  Train: {len(train_idx)} papers ({len(train_idx)/len(texts)*100:.1f}%)")
    print(f"  Val: {len(val_idx)} papers ({len(val_idx)/len(texts)*100:.1f}%)")
    print(f"  Test: {len(test_idx)} papers ({len(test_idx)/len(texts)*100:.1f}%)")
    
    # Verify label distribution
    print(f"\nLabel distribution check:")
    for target in labels.keys():
        train_avg = splits[f'y_train_{target}'].sum(axis=1).mean()
        val_avg = splits[f'y_val_{target}'].sum(axis=1).mean()
        test_avg = splits[f'y_test_{target}'].sum(axis=1).mean()
        print(f"  {target} avg labels - train: {train_avg:.2f}, val: {val_avg:.2f}, test: {test_avg:.2f}")
    
    return splits


def get_class_descriptions(info: Dict) -> Dict[str, Dict]:
    """
    Get human-readable descriptions for all classes
    
    Args:
        info: Info dictionary from prepare_data
        
    Returns:
        Dictionary mapping target -> class_index -> description
    """
    descriptions = {}
    
    # Themes - map from numbers to descriptions
    theme_classes = info['class_names']['themes']
    theme_mapping = info['mappings']['theme_mapping']
    descriptions['themes'] = {}
    for i, theme_num in enumerate(theme_classes):
        if str(theme_num) in theme_mapping:
            descriptions['themes'][i] = theme_mapping[str(theme_num)]
        else:
            descriptions['themes'][i] = f"Theme {theme_num}"
    
    # Objectives - map from roman numerals to descriptions
    obj_classes = info['class_names']['objectives']
    obj_mapping = info['mappings']['objectives_mapping']
    descriptions['objectives'] = {}
    for i, obj_code in enumerate(obj_classes):
        if obj_code in obj_mapping:
            descriptions['objectives'][i] = obj_mapping[obj_code]
        else:
            descriptions['objectives'][i] = f"Objective {obj_code}"
    
    # Zones - map from codes to full names
    zone_classes = info['class_names']['zones']
    zone_mapping = info['mappings']['zone_mapping']
    descriptions['zones'] = {}
    for i, zone_code in enumerate(zone_classes):
        if zone_code in zone_mapping:
            descriptions['zones'][i] = zone_mapping[zone_code]
        else:
            descriptions['zones'][i] = zone_code
    
    # Areas - already have full names
    area_classes = info['class_names']['areas']
    descriptions['areas'] = {i: area for i, area in enumerate(area_classes)}
    
    return descriptions


if __name__ == "__main__":
    """Test the preprocessing pipeline with enhanced features"""
    print("Testing Enhanced Simple Preprocessing Pipeline")
    print("=" * 70)
    
    # Test with synthetic data including metadata
    test_df = pd.DataFrame({
        'Title': ['Antarctic Krill Study', 'Ross Sea Toothfish Research'],
        'Abstract': ['Study of krill populations...', 'Analysis of toothfish distribution...'],
        'Keywords': ['krill, antarctica', 'toothfish, ross sea'],
        'Thematic_Tags': ['6,7', '8,11,19'],  # Note: 19 is "Exploitation effects on toothfish"
        'CCAMLR_Objectives_Text': ['(iii) research (vi) prey', '(i) ecological (vii) predators'],
        'Management_zones': ['GPZ,SRZ', 'GPZ'],
        'Monitoring_areas': ['Ross Sea Polynya,McMurdo Sound', 'Western Ross Sea'],
        # ENHANCED METADATA
        'Species_List': ['Euphausia superba', 'Dissostichus mawsoni; Dissostichus eleginoides'],
        'Method_Tags': ['Acoustic Methods; Field Sampling', 'Tagging/Tracking; Genomic/Molecular'],
        'Named_Features': ['ross sea polynya, mcmurdo sound', 'western ross sea, ross sea continental slope'],
        'Spatial_Scale': ['Regional (beyond RSRMPA)', 'RSRMPA-wide'],
        'Temporal_Scale': ['Multi-year', 'Multi-year'],
        'Research_start_date': ['2010', '2015'],
        'Research_end_date': ['2012', '2020']
    })
    
    # Test preprocessing
    texts, labels, info = prepare_data(test_df)
    print(f"\nProcessed {len(texts)} test papers")
    print(f"\nSample enhanced text (Paper 1):")
    print("-" * 50)
    print(texts[0][:500] + "..." if len(texts[0]) > 500 else texts[0])
    
    print(f"\nSample enhanced text (Paper 2 - Toothfish):")
    print("-" * 50)
    print(texts[1][:500] + "..." if len(texts[1]) > 500 else texts[1])
    
    # Show that the toothfish paper includes species info
    print(f"\n✓ Note: Paper 2 now includes 'Dissostichus mawsoni' in text")
    print(f"  This helps the model learn theme 19: 'Exploitation effects on toothfish'")
    
    # Test splits
    splits = create_splits(texts, labels)
    
    # Test descriptions
    descriptions = get_class_descriptions(info)
    print(f"\nSample theme descriptions:")
    for idx, desc in list(descriptions['themes'].items())[:3]:
        print(f"  Index {idx}: {desc}")
