# Troubleshooting Colab Connection

## Issue: Kernel Selector Not Showing Colab Option

If Colab is enabled but you can't select the Colab kernel, try these solutions:

### Solution 1: Verify Connection Status

1. **Check if you're actually connected:**
   - Press `Ctrl+Shift+P`
   - Type: `Colab: Show Connection Status`
   - Or: `Colab: List Runtimes`
   - This will show if the connection is active

2. **Reconnect if needed:**
   - `Ctrl+Shift+P` → `Colab: Disconnect from Colab`
   - Then: `Ctrl+Shift+P` → `Colab: Connect to Colab`
   - Authenticate again

### Solution 2: Alternative Kernel Selection Methods

**Method A: Via Command Palette**
1. Press `Ctrl+Shift+P`
2. Type: `Notebook: Select Notebook Kernel`
3. Look for "Colab" or "Google Colab" in the list
4. Select it

**Method B: Via Status Bar**
1. Look at the bottom-right status bar in Cursor
2. Click on the kernel indicator (might show "Python 3" or similar)
3. Select "Select Another Kernel"
4. Choose Colab runtime

**Method C: Right-click Method**
1. Right-click in the notebook
2. Select "Select Kernel" or "Change Kernel"
3. Choose Colab

### Solution 3: Check Extension Settings

1. **Verify extension is active:**
   - `Ctrl+Shift+X` → Search "Colab"
   - Make sure it shows "Installed" and "Enabled"
   - If disabled, click "Enable"

2. **Check extension output:**
   - `Ctrl+Shift+P` → `View: Show Output`
   - Select "Colab" from the dropdown
   - Look for error messages

### Solution 4: Manual Kernel Selection

If the UI doesn't work, you can manually set the kernel:

1. **Check available kernels:**
   ```python
   # Run this in a notebook cell
   import sys
   print(sys.executable)
   ```

2. **Verify Colab connection:**
   ```python
   # Run this to check if Colab is accessible
   try:
       import google.colab
       print("✓ Colab is accessible!")
   except ImportError:
       print("✗ Colab not accessible - need to connect")
   ```

### Solution 5: Use Colab Web Interface Instead

If Cursor connection isn't working, you can:

1. **Upload notebook to Colab directly:**
   - Go to https://colab.research.google.com
   - Upload `MASTER_RUNNER_COLAB.ipynb`
   - Run it there
   - Results will be the same

2. **Or use Colab's file upload:**
   - In Colab, click folder icon
   - Upload entire `ml_research_pipeline` folder
   - Run the notebook

### Solution 6: Restart and Reconnect

1. **Restart Cursor:**
   - Close Cursor completely
   - Reopen it
   - Try connecting again

2. **Clear extension cache:**
   - `Ctrl+Shift+P` → `Developer: Reload Window`
   - Then reconnect to Colab

### Solution 7: Check for Updates

1. **Update Colab extension:**
   - `Ctrl+Shift+X` → Search "Colab"
   - Check if there's an "Update" button
   - Update if available

2. **Update Cursor:**
   - Check for Cursor updates
   - Some kernel selection issues are fixed in newer versions

## Quick Diagnostic Commands

Run these in Cursor's terminal or Command Palette:

```bash
# Check if Colab extension is installed
code --list-extensions | grep colab

# Check Cursor version
cursor --version
```

## Alternative: Run Directly in Colab Web

If the Cursor connection continues to have issues:

1. **Open Colab in browser:**
   - Go to https://colab.research.google.com
   - Sign in

2. **Upload notebook:**
   - File → Upload notebook
   - Select `MASTER_RUNNER_COLAB.ipynb`

3. **Upload project:**
   - Click folder icon (left sidebar)
   - Upload `ml_research_pipeline` folder
   - Or use: `!git clone <your-repo>` if you have it in a repo

4. **Add secrets:**
   - Click 🔑 icon → Secrets
   - Add API keys

5. **Run:**
   - Run all cells
   - Everything works the same, just in browser instead of Cursor

## Still Not Working?

If none of these work, the issue might be:
- Extension compatibility with your Cursor version
- Network/firewall blocking Colab connection
- Authentication token expired

**Best workaround:** Use Colab web interface directly - it's more reliable and has the same functionality.



