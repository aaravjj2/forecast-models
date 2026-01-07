# Setting Up Colab Secrets from keys.env

## Quick Setup

All keys from `keys.env` can be added to Colab secrets. The code will automatically use the same key names.

## Method 1: Add Keys Manually (Recommended)

1. Open Google Colab
2. Click the 🔑 icon (left sidebar)
3. Go to "Secrets" tab
4. Click "Add new secret" for each key below

### Required Keys (for ML pipeline):
- **FINNHUB_API_KEY**: `d28ndhhr01qmp5u9g65gd28ndhhr01qmp5u9g660`
- **NEWS_API_KEY**: `9ff201f1e68b4544ab5d358a261f1742`
- **TIINGO_API_KEY**: `b815ff7c64c1a7370b9ae8c0b8907673fdb5eb5f`

### Optional Keys (for extended features):
- **FINNHUB2_API_KEY**: `d38b891r01qlbdj4nnlgd38b891r01qlbdj4nnm0`
- **POLYGON_API_KEY**: `xVilYBLLH5At9uE3r6CIMrusXxWwxp0G`
- **TWELVEDATA_API_KEY**: `77c34e29fa104ee9bd7834c3b476b824`
- **QUANDL_API_KEY**: `fN3R5X9VPSaeqFC6R2hF`
- **GROQ_API_KEY**: `<GROQ_API_KEY>`
- **FRED_API_KEY**: `3c86f2f10c5e2b13454447d184ddb268`
- **SEC_API_KEY**: `0cb9c45a821668958bab90d73e70bc26b28b68ffeb83065da0495d0b7db2c138`
- **OPENROUTER_KEY**: `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`

## Method 2: Use Setup Script

Run this in a Colab cell to see which keys need to be added:

```python
# Copy and paste the content from add_all_keys_to_colab.py
# Or run: exec(open('add_all_keys_to_colab.py').read())
```

## Key Name Matching

The code uses the **exact same key names** as in `keys.env`:
- `FINNHUB_API_KEY` in keys.env → `FINNHUB_API_KEY` in Colab secrets
- `NEWS_API_KEY` in keys.env → `NEWS_API_KEY` in Colab secrets
- etc.

This ensures consistency between local and Colab environments.

## Verification

After adding keys, run this in Colab to verify:

```python
from google.colab import userdata
import os

keys_to_check = ['FINNHUB_API_KEY', 'NEWS_API_KEY', 'TIINGO_API_KEY']
for key in keys_to_check:
    try:
        value = userdata.get(key)
        os.environ[key] = value
        print(f"✓ {key} loaded")
    except:
        print(f"✗ {key} not found")
```

## Notes

- Keys added to Colab secrets are persistent across sessions
- Keys are encrypted and secure
- The master notebook will automatically load keys from secrets
- If a key is missing from secrets, it will use the value from keys.env as fallback


