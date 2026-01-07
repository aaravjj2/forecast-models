"""Script to add all keys from keys.env to Colab secrets.
Run this in a Colab notebook cell to set up all secrets.
"""

# This code should be run in Google Colab
# It will help you add all keys from keys.env to Colab secrets

from google.colab import userdata
import os

# All keys from keys.env
KEYS_FROM_ENV = {
    "TIINGO_API_KEY": "b815ff7c64c1a7370b9ae8c0b8907673fdb5eb5f",
    "FINAGE_API_KEY": "API_KEY6aZPLW0IIOEOAZFW1IMW46CC8WIMRP23",
    "NEWS_API_KEY": "9ff201f1e68b4544ab5d358a261f1742",
    "FINNHUB_API_KEY": "d28ndhhr01qmp5u9g65gd28ndhhr01qmp5u9g660",
    "FINNHUB2_API_KEY": "d38b891r01qlbdj4nnlgd38b891r01qlbdj4nnm0",
    "POLYGON_API_KEY": "xVilYBLLH5At9uE3r6CIMrusXxWwxp0G",
    "TWELVEDATA_API_KEY": "77c34e29fa104ee9bd7834c3b476b824",
    "QUANDL_API_KEY": "fN3R5X9VPSaeqFC6R2hF",
    "GROQ_API_KEY": "<GROQ_API_KEY>",
    "FRED_API_KEY": "3c86f2f10c5e2b13454447d184ddb268",
    "SEC_API_KEY": "0cb9c45a821668958bab90d73e70bc26b28b68ffeb83065da0495d0b7db2c138",
    "OPENROUTER_KEY": "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",
}

print("="*60)
print("COLAB SECRETS SETUP FROM keys.env")
print("="*60)
print("\nNote: Colab secrets must be added via the UI.")
print("This script will show you which keys to add.\n")

# Check which keys already exist
existing_keys = []
missing_keys = []

for key, value in KEYS_FROM_ENV.items():
    try:
        existing = userdata.get(key)
        existing_keys.append(key)
        print(f"✓ {key} - Already exists in Colab secrets")
    except:
        missing_keys.append((key, value))
        print(f"✗ {key} - Needs to be added")

if missing_keys:
    print("\n" + "="*60)
    print("KEYS TO ADD VIA COLAB UI")
    print("="*60)
    print("\nSteps:")
    print("1. Click 🔑 icon in Colab's left sidebar")
    print("2. Go to 'Secrets' tab")
    print("3. Click 'Add new secret' for each key below\n")
    
    for key, value in missing_keys:
        print(f"Key Name: {key}")
        print(f"Value: {value}")
        print(f"  → Click 'Add new secret', paste above, then click 'Add secret'\n")
        print("-" * 60)
    
    print("\nAfter adding, run this cell again to verify all keys are set.")
else:
    print("\n✓ All keys are already in Colab secrets!")

# Set environment variables for keys that exist
print("\n" + "="*60)
print("SETTING ENVIRONMENT VARIABLES")
print("="*60)

for key, value in KEYS_FROM_ENV.items():
    try:
        secret_value = userdata.get(key)
        os.environ[key] = secret_value
        print(f"✓ {key} set from Colab secrets")
    except:
        # Use value from keys.env as fallback
        os.environ[key] = value
        print(f"⚠ {key} not in secrets, using keys.env value")

print("\n✓ All environment variables set!")
print("\nYou can now use these keys in your code via os.getenv()")


