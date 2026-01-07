"""Script to help set up Colab secrets from keys.env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

def get_keys_from_env():
    """Load all keys from keys.env file."""
    project_root = Path(__file__).parent.parent
    env_file = project_root / "keys.env"
    
    if not env_file.exists():
        print(f"Error: {env_file} not found")
        return {}
    
    load_dotenv(env_file)
    
    # Get all keys
    keys = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                keys[key] = value
    
    return keys

def generate_colab_secrets_code():
    """Generate Python code to add secrets to Colab."""
    keys = get_keys_from_env()
    
    code = """# Run this cell in Google Colab to set up all secrets
# This will add all keys from keys.env to Colab secrets

from google.colab import userdata
import json

# Keys to add (from keys.env)
secrets_to_add = {
"""
    
    for key, value in keys.items():
        # Skip empty values and variable references
        if value and not value.startswith('${'):
            # Escape quotes in values
            value_escaped = value.replace('"', '\\"').replace("'", "\\'")
            code += f'    "{key}": "{value_escaped}",\n'
    
    code += """}

# Add each secret
print("Adding secrets to Colab...")
for key, value in secrets_to_add.items():
    try:
        # Check if already exists
        try:
            existing = userdata.get(key)
            print(f"  ⚠ {key} already exists, skipping...")
        except:
            # Add new secret
            # Note: Colab secrets must be added via UI, but we can verify
            print(f"  ✓ {key}: {'*' * min(len(value), 20)}")
            print(f"    → Add this manually via Colab UI: 🔑 → Secrets → Add new secret")
            print(f"    → Key: {key}")
            print(f"    → Value: {value[:50]}...")
    except Exception as e:
        print(f"  ✗ Error with {key}: {e}")

print("\\n" + "="*60)
print("IMPORTANT: Colab secrets must be added via the UI")
print("="*60)
print("\\nSteps:")
print("1. Click the 🔑 icon in Colab's left sidebar")
print("2. Go to 'Secrets' tab")
print("3. Click 'Add new secret' for each key above")
print("4. Copy the key name and value from the output above")
print("\\nAlternatively, you can set them programmatically if you have")
print("the right permissions, but the UI method is recommended.")
"""
    
    return code

def create_colab_setup_notebook():
    """Create a notebook cell code for Colab setup."""
    code = generate_colab_secrets_code()
    
    # Also create a script that can be run
    script_path = Path(__file__).parent / "colab_secrets_setup.py"
    with open(script_path, 'w') as f:
        f.write(code)
    
    print("="*60)
    print("COLAB SECRETS SETUP")
    print("="*60)
    print("\nGenerated code to set up Colab secrets.")
    print(f"\nCode saved to: {script_path}")
    print("\nTo use:")
    print("1. Open Google Colab")
    print("2. Create a new notebook")
    print("3. Copy and paste the code from colab_secrets_setup.py")
    print("4. Run the cell - it will show you what to add")
    print("5. Add secrets via Colab UI (🔑 → Secrets)")
    print("\nOr run this script to see the code:")
    print(f"   python {script_path}")
    
    return code

if __name__ == "__main__":
    create_colab_setup_notebook()


