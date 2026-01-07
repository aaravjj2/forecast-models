"""Script to create a Colab cell that contains all source files as strings.
This allows building the entire project in Colab without manual uploads."""

import base64
import json
from pathlib import Path

def create_colab_setup_cell():
    """Create a cell that sets up all source files in Colab."""
    
    project_root = Path(__file__).parent
    src_dir = project_root / 'src'
    
    # Files to create (in order of dependencies)
    files_to_create = {
        'src/utils/config.py': project_root / 'src/utils/config.py',
        'src/utils/helpers.py': project_root / 'src/utils/helpers.py',
        'src/data/price_fetcher.py': project_root / 'src/data/price_fetcher.py',
        'src/data/news_fetcher.py': project_root / 'src/data/news_fetcher.py',
        'src/features/feature_builder.py': project_root / 'src/features/feature_builder.py',
        'src/models/base_model.py': project_root / 'src/models/base_model.py',
        'src/models/xgboost_model.py': project_root / 'src/models/xgboost_model.py',
        'src/models/lightgbm_model.py': project_root / 'src/models/lightgbm_model.py',
        'src/models/sentiment_model.py': project_root / 'src/models/sentiment_model.py',
        'src/models/rule_based_model.py': project_root / 'src/models/rule_based_model.py',
        'src/ensemble/meta_ensemble.py': project_root / 'src/ensemble/meta_ensemble.py',
        'src/backtest/walkforward_backtest.py': project_root / 'src/backtest/walkforward_backtest.py',
    }
    
    # Create __init__ files content
    init_files = {
        'src/data/__init__.py': 'from .price_fetcher import PriceFetcher\nfrom .news_fetcher import NewsFetcher\n__all__ = ["PriceFetcher", "NewsFetcher"]',
        'src/features/__init__.py': 'from .feature_builder import FeatureBuilder\n__all__ = ["FeatureBuilder"]',
        'src/models/__init__.py': 'from .base_model import BaseModel, ModelSignal\nfrom .xgboost_model import XGBoostModel\nfrom .lightgbm_model import LightGBMModel\nfrom .sentiment_model import SentimentModel\nfrom .rule_based_model import RuleBasedModel\n__all__ = ["BaseModel", "ModelSignal", "XGBoostModel", "LightGBMModel", "SentimentModel", "RuleBasedModel"]',
        'src/ensemble/__init__.py': 'from .meta_ensemble import MetaEnsemble\n__all__ = ["MetaEnsemble"]',
        'src/backtest/__init__.py': 'from .walkforward_backtest import WalkForwardBacktest\n__all__ = ["WalkForwardBacktest"]',
    }
    
    cell_code = '''# AUTO-CREATE: Build all source files in Colab
# This creates the entire project structure without manual uploads

from pathlib import Path
import os

project_dir = Path('/content/ml_research_pipeline')
os.chdir(project_dir)

# Create all source files
files_content = {
'''
    
    # Add file contents
    for rel_path, file_path in files_to_create.items():
        if file_path.exists():
            content = file_path.read_text()
            # Escape for Python string
            content = content.replace('\\', '\\\\').replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")
            cell_code += f"    '{rel_path}': '''{content}''',\n"
    
    # Add init files
    for rel_path, content in init_files.items():
        cell_code += f"    '{rel_path}': '''{content}''',\n"
    
    cell_code += '''}

# Write all files
print("Creating source files...")
for file_path, content in files_content.items():
    full_path = project_dir / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    print(f"  ✓ Created {file_path}")

print(f"\\n✓ All source files created!")
print(f"   Project ready at: {project_dir}")
'''
    
    return cell_code

if __name__ == "__main__":
    code = create_colab_setup_cell()
    output_file = Path(__file__).parent / 'colab_auto_setup_source.py'
    output_file.write_text(code)
    print(f"✓ Created {output_file}")
    print(f"   This file contains code to create all source files in Colab")

