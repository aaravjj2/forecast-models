# Running on Colab via Cursor Connection

## Step-by-Step Guide

### Step 1: Install Colab Extension in Cursor

1. **Open Extensions in Cursor:**
   - Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac)
   - Or click the Extensions icon in the left sidebar

2. **Search for Colab:**
   - Type "Colab" in the search box
   - Look for "Colab" by Google (official extension)

3. **Install:**
   - Click "Install" on the Colab extension
   - Wait for installation to complete

### Step 2: Connect to Colab

1. **Open Command Palette:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Or go to View → Command Palette

2. **Connect to Colab:**
   - Type: `Colab: Connect to Colab`
   - Select it from the dropdown
   - A browser window will open for authentication

3. **Authenticate:**
   - Sign in with your Google account
   - Grant permissions to the Colab extension
   - You'll see a success message

4. **Select Runtime:**
   - Choose a Colab runtime (CPU, GPU, or TPU)
   - Or create a new runtime
   - The connection will be established

### Step 3: Open the Master Notebook

1. **Open the notebook:**
   - Open `MASTER_RUNNER_COLAB.ipynb` in Cursor
   - You should see it as a Jupyter notebook

2. **Select Colab Kernel:**
   - Look at the top-right corner of the notebook
   - Click on the kernel selector (shows current kernel)
   - Select your connected Colab runtime
   - It should show something like "Colab Runtime" or "Google Colab"

### Step 4: Add API Keys to Colab Secrets

**Important:** Keys must be added via Colab's web interface.

1. **Open Google Colab in Browser:**
   - Go to https://colab.research.google.com
   - Sign in with the same Google account

2. **Access Secrets:**
   - Click the 🔑 (key) icon in the left sidebar
   - Go to the "Secrets" tab

3. **Add Keys:**
   - Click "Add new secret" for each key
   - Use these exact key names and values:

   **Required Keys:**
   - Key: `FINNHUB_API_KEY`, Value: `d28ndhhr01qmp5u9g65gd28ndhhr01qmp5u9g660`
   - Key: `NEWS_API_KEY`, Value: `9ff201f1e68b4544ab5d358a261f1742`
   - Key: `TIINGO_API_KEY`, Value: `b815ff7c64c1a7370b9ae8c0b8907673fdb5eb5f`

   **Optional Keys (for extended features):**
   - `POLYGON_API_KEY`: `xVilYBLLH5At9uE3r6CIMrusXxWwxp0G`
   - `TWELVEDATA_API_KEY`: `77c34e29fa104ee9bd7834c3b476b824`
   - `QUANDL_API_KEY`: `fN3R5X9VPSaeqFC6R2hF`
   - `GROQ_API_KEY`: `<GROQ_API_KEY>`
   - `FRED_API_KEY`: `3c86f2f10c5e2b13454447d184ddb268`
   - `SEC_API_KEY`: `0cb9c45a821668958bab90d73e70bc26b28b68ffeb83065da0495d0b7db2c138`
   - `OPENROUTER_KEY`: `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`

### Step 5: Upload Project to Colab

**Option A: Upload via Colab Web Interface**

1. In Colab web interface, click the folder icon (left sidebar)
2. Click "Upload" button
3. Upload the entire `ml_research_pipeline` folder
   - Or zip it first: `zip -r ml_research_pipeline.zip ml_research_pipeline/`
   - Then upload the zip and unzip: `!unzip ml_research_pipeline.zip`

**Option B: Upload via Notebook Cell**

Run this in a Colab cell (in Colab web interface):
```python
from google.colab import files
uploaded = files.upload()  # Select your zip file
!unzip ml_research_pipeline.zip
```

**Option C: From Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
# Copy from drive to /content
!cp -r /content/drive/MyDrive/ml_research_pipeline /content/
```

### Step 6: Run the Notebook in Cursor

1. **Verify Connection:**
   - Check that Colab kernel is selected (top-right)
   - You should see "Connected to Colab" or similar

2. **Run Cells:**
   - Click on the first cell
   - Press `Shift+Enter` to run it
   - Or click the "Run" button
   - Wait for execution to complete

3. **Run All Cells:**
   - Right-click in the notebook
   - Select "Run All" or "Run All Cells"
   - Or use Command Palette: `Notebook: Run All`

4. **Monitor Execution:**
   - Watch the output in each cell
   - Check for any errors
   - The notebook will execute on Colab's servers

### Step 7: Verify It's Running on Colab

Run this test cell to confirm:
```python
try:
    import google.colab
    print("✓ Running on Google Colab!")
    print(f"✓ Colab module: {google.colab}")
except ImportError:
    print("✗ Not running on Colab")
```

## Troubleshooting

### "Can't find Colab kernel"
- Make sure Colab extension is installed
- Reconnect: `Ctrl+Shift+P` → `Colab: Connect to Colab`
- Restart Cursor if needed

### "Module not found" errors
- Project files need to be uploaded to Colab
- Check that files are at `/content/ml_research_pipeline/`
- Verify `src/` directory exists

### "API key not found"
- Keys must be added via Colab web UI (🔑 → Secrets)
- Make sure you're signed into the same Google account
- Re-run the key setup cell after adding secrets

### "Connection lost"
- Reconnect: `Ctrl+Shift+P` → `Colab: Connect to Colab`
- Check your internet connection
- Colab sessions timeout after inactivity

### "Out of memory"
- Use a smaller dataset
- Reduce model complexity
- Use CPU runtime instead of GPU

## Quick Checklist

- [ ] Colab extension installed in Cursor
- [ ] Connected to Colab via Command Palette
- [ ] Colab kernel selected in notebook
- [ ] API keys added to Colab secrets (via web UI)
- [ ] Project files uploaded to Colab
- [ ] Notebook running on Colab server

## Expected Behavior

When running on Colab:
- ✅ Cells execute on Google's servers
- ✅ Output appears in Cursor
- ✅ Files are saved to `/content/ml_research_pipeline/`
- ✅ You can use Colab's free GPU/TPU resources
- ✅ All processing happens in the cloud

## Next Steps

Once connected and running:
1. Run the master notebook cell by cell
2. Monitor progress in each cell's output
3. Results will be saved to `results/` directory
4. Download results from Colab when complete


