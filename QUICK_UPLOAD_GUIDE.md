# Quick Upload Guide - 3 Simple Steps

## Step 1: Zip the Project (Optional but Faster)

The project is already zipped at:
```
/home/aarav/Forecast models/ml_research_pipeline.zip
```

Or create a fresh zip:
```bash
cd "/home/aarav/Forecast models"
zip -r ml_research_pipeline.zip ml_research_pipeline/
```

## Step 2: Upload to Colab

1. **Open Colab in browser**: https://colab.research.google.com
2. **Click folder icon** (📁) in left sidebar
3. **Click "Upload"** button
4. **Select**: `ml_research_pipeline.zip` (or the folder)
5. **Wait** for upload to complete

## Step 3: Extract (if you uploaded zip)

In a Colab cell, run:
```python
!unzip ml_research_pipeline.zip -d /content/
```

## Step 4: Verify

Run this in Colab to verify:
```python
from pathlib import Path
if (Path('/content/ml_research_pipeline') / 'src').exists():
    print("✓ Upload successful! Ready to run notebook.")
else:
    print("✗ Files not found - check upload")
```

## Step 5: Run Notebook

**In Cursor:**
- Open `MASTER_RUNNER_COLAB.ipynb`
- Select Colab kernel
- Run all cells

**Or in Colab:**
- Upload `MASTER_RUNNER_COLAB.ipynb`
- Run all cells

That's it! The notebook will find the files automatically.


