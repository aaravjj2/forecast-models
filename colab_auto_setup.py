"""Automated Colab setup script - run this in Colab to set up everything."""

# This script should be run in a Colab notebook cell
# It will help set up secrets and verify the setup

import os
from pathlib import Path

def setup_colab_environment():
    """Set up Colab environment with all necessary configurations."""
    
    print("="*60)
    print("COLAB AUTOMATED SETUP")
    print("="*60)
    
    # Check if running on Colab
    try:
        import google.colab
        print("✓ Running on Google Colab")
        ON_COLAB = True
    except ImportError:
        print("⚠ Not running on Colab - some features may not work")
        ON_COLAB = False
    
    # Step 1: Install dependencies
    print("\n[1/4] Installing dependencies...")
    import subprocess
    try:
        subprocess.run(['pip', 'install', '-q', 
                       'pandas', 'numpy', 'scikit-learn', 
                       'xgboost', 'lightgbm', 'yfinance', 
                       'requests', 'python-dotenv', 'tqdm', 
                       'joblib', 'transformers', 'torch', 
                       'sentencepiece'], 
                      check=True, capture_output=True)
        print("✓ Dependencies installed")
    except Exception as e:
        print(f"⚠ Installation error: {e}")
    
    # Step 2: Set up project structure
    print("\n[2/4] Setting up project structure...")
    project_dir = Path('/content/ml_research_pipeline')
    project_dir.mkdir(exist_ok=True)
    
    # Create necessary directories
    dirs = ['data/raw', 'data/processed', 'models/specialists', 
            'models/meta', 'results', 'artifacts']
    for dir_path in dirs:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Project structure created at {project_dir}")
    
    # Step 3: Set up secrets
    print("\n[3/4] Setting up API keys...")
    
    if ON_COLAB:
        from google.colab import userdata
        
        # Required keys for the pipeline
        required_keys = {
            'FINNHUB_API_KEY': 'd28ndhhr01qmp5u9g65gd28ndhhr01qmp5u9g660',
            'NEWS_API_KEY': '9ff201f1e68b4544ab5d358a261f1742',
            'TIINGO_API_KEY': 'b815ff7c64c1a7370b9ae8c0b8907673fdb5eb5f',
        }
        
        # Try to get existing secrets, or set them programmatically
        secrets_set = {}
        for key, default_value in required_keys.items():
            try:
                value = userdata.get(key)
                secrets_set[key] = True
                os.environ[key] = value
                print(f"  ✓ {key} found in Colab secrets")
            except:
                # Secret doesn't exist - we'll need to add it
                # For now, set as environment variable (temporary)
                os.environ[key] = default_value
                secrets_set[key] = False
                print(f"  ⚠ {key} not in secrets - using provided value")
                print(f"     → Add to Colab secrets (🔑 → Secrets) for persistence")
        
        # Also set other useful keys
        other_keys = {
            'FINNHUB2_API_KEY': 'd38b891r01qlbdj4nnlgd38b891r01qlbdj4nnm0',
            'POLYGON_API_KEY': 'xVilYBLLH5At9uE3r6CIMrusXxWwxp0G',
        }
        
        for key, value in other_keys.items():
            os.environ[key] = value
        
        print(f"\n✓ API keys configured")
    else:
        # Not on Colab - try loading from keys.env
        env_file = Path('keys.env')
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print("✓ Loaded keys from keys.env")
        else:
            print("⚠ No keys.env found - some features may not work")
    
    # Step 4: Verify setup
    print("\n[4/4] Verifying setup...")
    
    checks = {
        'Project directory exists': project_dir.exists(),
        'FINNHUB_API_KEY set': bool(os.getenv('FINNHUB_API_KEY')),
        'NEWS_API_KEY set': bool(os.getenv('NEWS_API_KEY')),
    }
    
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print(f"\nProject directory: {project_dir}")
    print(f"Working directory: {os.getcwd()}")
    print("\nNext steps:")
    print("1. Upload ml_research_pipeline folder to Colab")
    print("2. Run notebooks 00-05 in sequence")
    print("3. Or use the run_on_colab.ipynb notebook")
    
    return project_dir

if __name__ == "__main__":
    setup_colab_environment()



