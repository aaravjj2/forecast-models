"""Monitor notebook execution progress by reading outputs."""

import json
from pathlib import Path
from datetime import datetime

def check_notebook_progress(notebook_path):
    """Check which cells have been executed and show progress."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    total_cells = len([c for c in cells if c.get('cell_type') == 'code'])
    executed_cells = 0
    completed_steps = []
    
    print("="*60)
    print("NOTEBOOK EXECUTION PROGRESS")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    step_names = {
        2: "Diagnostic: Colab Connection",
        3: "Install Dependencies",
        4: "Setup Environment & Keys",
        7: "Setup Project Structure",
        9: "Data Fetching (Prices + News)",
        11: "Feature Engineering",
        13: "Train Price Models (XGBoost + LightGBM)",
        15: "Train Sentiment Models",
        17: "Train Meta-Ensemble",
        19: "Walk-Forward Backtest"
    }
    
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            has_output = 'outputs' in cell and len(cell['outputs']) > 0
            execution_count = cell.get('execution_count')
            
            if execution_count is not None or has_output:
                executed_cells += 1
                step_name = step_names.get(i, f"Cell {i}")
                completed_steps.append(step_name)
                
                # Check for errors
                if has_output:
                    for output in cell['outputs']:
                        if output.get('output_type') == 'error':
                            print(f"⚠ {step_name} - ERROR DETECTED")
                            print(f"   {output.get('ename', 'Unknown error')}")
                            return
                
                print(f"✓ {step_name}")
    
    print(f"\nProgress: {executed_cells}/{total_cells} code cells executed")
    print(f"Completion: {executed_cells/total_cells*100:.1f}%")
    
    # Show next steps
    remaining = []
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            has_output = 'outputs' in cell and len(cell['outputs']) > 0
            execution_count = cell.get('execution_count')
            
            if execution_count is None and not has_output:
                step_name = step_names.get(i, f"Cell {i}")
                remaining.append((i, step_name))
    
    if remaining:
        print(f"\nNext steps:")
        for idx, step in remaining[:3]:  # Show next 3
            print(f"  → Run Cell {idx}: {step}")
    
    return {
        'executed': executed_cells,
        'total': total_cells,
        'percentage': executed_cells/total_cells*100 if total_cells > 0 else 0,
        'completed_steps': completed_steps
    }

if __name__ == "__main__":
    notebook_path = Path(__file__).parent / "MASTER_RUNNER_COLAB.ipynb"
    if notebook_path.exists():
        check_notebook_progress(notebook_path)
    else:
        print(f"Notebook not found: {notebook_path}")


