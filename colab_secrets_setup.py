# Run this cell in Google Colab to set up all secrets
# This will add all keys from keys.env to Colab secrets

from google.colab import userdata
import json

# Keys to add (from keys.env)
secrets_to_add = {
    "TIINGO_API_KEY": "b815ff7c64c1a7370b9ae8c0b8907673fdb5eb5f",
    "FINAGE_API_KEY": "API_KEY6aZPLW0IIOEOAZFW1IMW46CC8WIMRP23",
    "NEWS_API_KEY": "9ff201f1e68b4544ab5d358a261f1742",
    "FINNHUB_API_KEY": "d28ndhhr01qmp5u9g65gd28ndhhr01qmp5u9g660",
    "FINNHUB2_API_KEY": "d38b891r01qlbdj4nnlgd38b891r01qlbdj4nnm0",
    "X_API_BEARER": "AAAAAAAAAAAAAAAAAAAAANec3QEAAAAAyosWBX%2FkYLjp7F5NwiEwlqTq2Qo%3DpLbS46M7Mbj8TBUi0GzK8qthzcVJUhqA6nPLIMilfK2UZAnUdU",
    "POLYGON_API_KEY": "xVilYBLLH5At9uE3r6CIMrusXxWwxp0G",
    "TWELVEDATA_API_KEY": "77c34e29fa104ee9bd7834c3b476b824",
    "QUANDL_API_KEY": "fN3R5X9VPSaeqFC6R2hF",
    "PRIXIE_API_KEY": "72b9a11d-721b-422b-a499-ab3174b4f6f8",
    "APCA_API_KEY_ID": "PKMZZAL28UP5G05AECSW",
    "APCA_API_SECRET_KEY": "QavdtLfphkusZaXaVgcL4xBULaXHcUIFagIrupnT",
    "APCA_ENDPOINT": "https://paper-api.alpaca.markets",
    "APCA_EMERGENCY_KEY": "b385d9ef-a399-42ce-aa63-937de3cdb34d",
    "OPTIONS_USE_ALPACA": "1",
    "OPTIONS_LAB_ALLOW_ALPACA": "1",
    "ALPACA2_KEY": "PKLYVWGCORNRTMRJLIYH7GFN6V",
    "ALPACA2_SECRET": "2zhuh4JAA8XW7Xk2U2HtJsD3LAHsMD5uysWAePJmgYkT",
    "REDDIT_SECRET": "ykp6PRyM3ahpTgd4SooD5p66JvCjkg",
    "REDDIT_CLIENT_ID": "x69iMELZkGKprVYoZv2ySw",
    "REDDIT_UID": "Brave-Vehicle3052",
    "REDDIT_PASSWORD": "stockforecastmodel",
    "FRED_API_KEY": "3c86f2f10c5e2b13454447d184ddb268",
    "SEC_API_KEY": "0cb9c45a821668958bab90d73e70bc26b28b68ffeb83065da0495d0b7db2c138",
    "POSTGRES_USER": "dashboard_user",
    "POSTGRES_PASSWORD": "newpassword",
    "POSTGRES_DB": "financial_dashboard",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "GROQ_API_KEY": "<GROQ_API_KEY>",
    "ALPACA3_KEY": "PK3OFL2DZZVBK75O3HON4URWAJ",
    "ALPACA3_SECRET": "76TzT5eFr5sn7NKKZ2visigC9LMZAs2usqcZuALjSKb5",
    "ALPACA3_ENDPOINT": "https://paper-api.alpaca.markets",
    "LLM_PROVIDER": "ollama",
    "OLLAMA_HOST": "http://localhost:11434",
    "OLLAMA_MODEL": "mistral:7b",
    "OPENROUTER_KEY": "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",
    "OLLAMA_KEY": "831c934cf0c24383af9a6cf7d283cb5d.sXbgSA3UVQbjgBJ5iddYL3pr",
}

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

print("\n" + "="*60)
print("IMPORTANT: Colab secrets must be added via the UI")
print("="*60)
print("\nSteps:")
print("1. Click the 🔑 icon in Colab's left sidebar")
print("2. Go to 'Secrets' tab")
print("3. Click 'Add new secret' for each key above")
print("4. Copy the key name and value from the output above")
print("\nAlternatively, you can set them programmatically if you have")
print("the right permissions, but the UI method is recommended.")
