# experiment_config.py
"""
Simple configuration to switch between semantic-only and full model
"""

# CHANGE THIS LINE TO SWITCH MODES
MODE = "full"  # Options: "semantic" or "full"

def get_active_targets():
    """Returns which targets to use based on MODE"""
    if MODE == "semantic":
        return ["themes", "objectives"]
    else:
        return ["themes", "objectives", "zones", "areas"]

def should_use_target(target_name):
    """Check if we should use this target"""
    return target_name in get_active_targets()
