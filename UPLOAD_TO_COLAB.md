# How to Upload Project to Colab and Run

## Method 1: Via Colab Web UI (Recommended - Easiest)

### Step 1: Prepare the Folder
1. **Zip the project folder** (optional but faster):
   ```bash
   cd "/home/aarav/Forecast models"
   zip -r ml_research_pipeline.zip ml_research_pipeline/
   ```

### Step 2: Upload to Colab
1. **Open Google Colab** in your browser:
   - Go to https://colab.research.google.com
   - Sign in with your Google account

2. **Upload the folder**:
   - Click the **folder icon (📁)** in the left sidebar
   - Click the **"Upload"** button (or drag and drop)
   - Select the `ml_research_pipeline` folder (or the zip file)
   - Wait for upload to complete

3. **If you uploaded a zip file**, extract it:
   - In a Colab cell, run:
   ```python
   !unzip ml_research_pipeline.zip -d /content/
   ```

### Step 3: Verify Upload
Run this in a Colab cell to verify:
```python
import os
from pathlib import Path

# Check if files are there
project_dir = Path('/content/ml_research_pipeline')
if (project_dir / 'src').exists():
    print("✓ Project files uploaded successfully!")
    print(f"  Location: {project_dir}")
    print(f"  Contents: {list((project_dir / 'src').iterdir())}")
else:
    print("✗ Project files not found")
    print(f"  Current contents: {list(Path('/content').iterdir())}")
```

### Step 4: Run the Notebook
1. **In Cursor** (with Colab connection):
   - Open `MASTER_RUNNER_COLAB.ipynb`
   - Make sure Colab kernel is selected
   - Run all cells

2. **Or in Colab web UI**:
   - Upload `MASTER_RUNNER_COLAB.ipynb` to Colab
   - Run all cells there

## Method 2: Via Cursor → Colab Connection

### Step 1: Upload Files
1. **Connect Cursor to Colab** (as before)
2. **In Colab web UI**, upload the folder:
   - Go to https://colab.research.google.com
   - Click folder icon → Upload
   - Upload `ml_research_pipeline` folder

### Step 2: Run in Cursor
1. **Open notebook in Cursor**
2. **Select Colab kernel** (top-right)
3. **Run all cells** - they'll execute on Colab servers

## Method 3: From Google Drive

### Step 1: Upload to Drive
1. Upload `ml_research_pipeline` folder to Google Drive
2. Note the path (e.g., `MyDrive/ml_research_pipeline`)

### Step 2: Mount Drive in Colab
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from drive
!cp -r /content/drive/MyDrive/ml_research_pipeline /content/
```

### Step 3: Verify and Run
```python
# Verify
from pathlib import Path
if (Path('/content/ml_research_pipeline') / 'src').exists():
    print("✓ Files ready!")
```

## Quick Checklist

- [ ] Project folder uploaded to `/content/ml_research_pipeline/`
- [ ] `src/` directory exists with all modules
- [ ] API keys added to Colab secrets (🔑 → Secrets)
- [ ] Notebook kernel set to Colab
- [ ] Ready to run!

## Troubleshooting

**"Project files not found"**
- Check upload completed successfully
- Verify folder is at `/content/ml_research_pipeline/`
- Check that `src/` subdirectory exists

**"Import errors"**
- Make sure entire `ml_research_pipeline` folder is uploaded
- Not just individual files - need the full folder structure
- Verify `src/data/`, `src/models/`, etc. all exist

**"Permission denied"**
- Colab files are read-only by default
- Use `/content/` for writable files
- The notebook handles this automatically

## File Structure After Upload

```
/content/
  └── ml_research_pipeline/
      ├── src/
      │   ├── data/
      │   ├── features/
      │   ├── models/
      │   ├── ensemble/
      │   ├── backtest/
      │   └── utils/
      ├── notebooks/
      ├── tests/
      └── MASTER_RUNNER_COLAB.ipynb
```

Once uploaded, the notebook will automatically find and use these files!

