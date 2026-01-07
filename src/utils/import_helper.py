"""Helper for robust imports."""

import sys
from pathlib import Path

def get_config():
    """Get config module, handling both relative and absolute imports."""
    try:
        from . import config
        return config
    except ImportError:
        # Fallback: add parent to path and import
        current_dir = Path(__file__).parent
        if str(current_dir.parent) not in sys.path:
            sys.path.insert(0, str(current_dir.parent))
        from utils import config
        return config

def get_helpers():
    """Get helpers module."""
    try:
        from . import helpers
        return helpers
    except ImportError:
        current_dir = Path(__file__).parent
        if str(current_dir.parent) not in sys.path:
            sys.path.insert(0, str(current_dir.parent))
        from utils import helpers
        return helpers



