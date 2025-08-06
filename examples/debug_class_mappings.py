"""
Debug script to extract actual class mappings from training data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_loader import load_data_with_fallback
    from data_preprocessing import preprocess_dataset_original_style
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def debug_class_mappings():
    """Debug what the actual class mappings are"""
    
    print("DEBUGGING ACTUAL CLASS MAPPINGS")
    print("=" * 50)
    
    # Load and preprocess data exactly like training
    df, is_actual = load_data_with_fallback()
    processed_df, multi_target_data, original_splits, analysis_results = preprocess_dataset_original_style(df)
    
    print(f"\nMulti-target data keys: {multi_target_data.keys()}")
    print(f"Class info keys: {multi_target_data.get('class_info', {}).keys()}")
    
    # Extract class info
    class_info = multi_target_data.get('class_info', {})
    
    print(f"\nACTUAL CLASS MAPPINGS:")
    print("-" * 30)
    
    # Themes
    theme_classes = class_info.get('theme_classes', [])
    print(f"THEMES ({len(theme_classes)} classes):")
    for i, theme in enumerate(theme_classes):
        print(f"  {i}: {theme}")
    
    # Objectives
    objective_classes = class_info.get('objective_classes', [])
    print(f"\nOBJECTIVES ({len(objective_classes)} classes):")
    for i, obj in enumerate(objective_classes):
        print(f"  {i}: {obj}")
    
    # Zones
    zone_classes = class_info.get('zone_classes', [])
    print(f"\nZONES ({len(zone_classes)} classes):")
    for i, zone in enumerate(zone_classes):
        print(f"  {i}: {zone}")
    
    # Areas
    area_classes = class_info.get('area_classes', [])
    print(f"\nAREAS ({len(area_classes)} classes):")
    for i, area in enumerate(area_classes):
        print(f"  {i}: {area}")
    
    # Check if preprocessor mappings are available
    print(f"\nPREPROCESSOR MAPPINGS:")
    print("-" * 25)
    
    from data_preprocessing import EnhancedDataPreprocessor
    preprocessor = EnhancedDataPreprocessor()
    
    print(f"Theme mapping sample: {list(preprocessor.theme_mapping.items())[:3]}")
    print(f"Objective mapping sample: {list(preprocessor.ccamlr_objectives_mapping.items())[:3]}")
    print(f"Zone mapping: {preprocessor.zone_mapping}")
    
    # Show the mapping from indices to descriptions
    print(f"\nINDEX TO DESCRIPTION MAPPING:")
    print("-" * 35)
    
    print("THEMES:")
    for i, theme_num in enumerate(theme_classes):
        if str(theme_num) in preprocessor.theme_mapping:
            desc = preprocessor.theme_mapping[str(theme_num)]
            print(f"  Index {i} (Theme {theme_num}): {desc}")
        else:
            print(f"  Index {i} (Theme {theme_num}): NO MAPPING FOUND")
    
    print("\nOBJECTIVES:")
    for i, obj_code in enumerate(objective_classes):
        if obj_code in preprocessor.ccamlr_objectives_mapping:
            desc = preprocessor.ccamlr_objectives_mapping[obj_code]
            print(f"  Index {i} (Objective {obj_code}): {desc}")
        else:
            print(f"  Index {i} (Objective {obj_code}): NO MAPPING FOUND")
    
    print("\nZONES:")
    for i, zone_code in enumerate(zone_classes):
        if zone_code in preprocessor.zone_mapping:
            desc = preprocessor.zone_mapping[zone_code]
            print(f"  Index {i} (Zone {zone_code}): {desc}")
        else:
            print(f"  Index {i} (Zone {zone_code}): NO MAPPING FOUND")
    
    print("\nAREAS:")
    for i, area_name in enumerate(area_classes):
        print(f"  Index {i}: {area_name}")

if __name__ == "__main__":
    debug_class_mappings()
