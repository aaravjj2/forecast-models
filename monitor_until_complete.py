"""Continuously monitor notebook execution until all cells complete."""

import json
import time
from pathlib import Path
from datetime import datetime

def check_notebook_progress(notebook_path):
    """Check which cells have been executed and return status."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    total_cells = len(code_cells)
    executed_cells = 0
    completed_steps = []
    errors = []
    
    step_names = {
        2: "Diagnostic: Colab Connection",
        3: "Install Dependencies",
        4: "Setup Environment & Keys",
        6: "Check Secrets Status",
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
                            error_info = {
                                'cell': i,
                                'step': step_name,
                                'error': output.get('ename', 'Unknown'),
                                'message': output.get('evalue', '')
                            }
                            errors.append(error_info)
    
    return {
        'executed': executed_cells,
        'total': total_cells,
        'percentage': executed_cells/total_cells*100 if total_cells > 0 else 0,
        'completed_steps': completed_steps,
        'errors': errors,
        'is_complete': executed_cells == total_cells
    }

def monitor_continuously(notebook_path, check_interval=15):
    """Monitor notebook execution continuously until complete."""
    print("="*70)
    print("CONTINUOUS NOTEBOOK MONITORING")
    print("="*70)
    print(f"Monitoring: {notebook_path}")
    print(f"Check interval: {check_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    previous_progress = 0
    start_time = time.time()
    last_update_time = start_time
    
    try:
        while True:
            status = check_notebook_progress(notebook_path)
            current_time = time.time()
            elapsed = current_time - start_time
            time_since_update = current_time - last_update_time
            
            # Clear screen for better readability (optional)
            # print("\033[2J\033[H")  # Uncomment for screen clearing
            
            print("="*70)
            print(f"PROGRESS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
            print("="*70)
            print(f"Elapsed time: {elapsed/60:.1f} minutes")
            print(f"Progress: {status['executed']}/{status['total']} cells ({status['percentage']:.1f}%)")
            
            if status['completed_steps']:
                print(f"\n✓ Completed steps ({len(status['completed_steps'])}):")
                for step in status['completed_steps']:
                    print(f"  • {step}")
            
            # Show progress change
            if status['percentage'] > previous_progress:
                progress_delta = status['percentage'] - previous_progress
                print(f"\n📈 Progress increased: +{progress_delta:.1f}%")
                last_update_time = current_time
            
            # Check for errors
            if status['errors']:
                print(f"\n⚠ ERRORS DETECTED ({len(status['errors'])}):")
                for error in status['errors']:
                    print(f"  ✗ {error['step']} (Cell {error['cell']})")
                    print(f"    Error: {error['error']}")
                    print(f"    Message: {error['message'][:100]}")
                print("\n⚠ Monitoring will continue, but please check the notebook for details.")
            
            # Check if complete
            if status['is_complete']:
                print("\n" + "="*70)
                print("🎉 ALL CELLS COMPLETED!")
                print("="*70)
                print(f"Total time: {elapsed/60:.1f} minutes")
                print(f"All {status['total']} cells executed successfully!")
                print("\nFinal completed steps:")
                for step in status['completed_steps']:
                    print(f"  ✓ {step}")
                
                if status['errors']:
                    print(f"\n⚠ Note: {len(status['errors'])} errors were detected during execution")
                    print("   Please review the notebook for error details.")
                else:
                    print("\n✅ No errors detected - pipeline completed successfully!")
                
                break
            
            # Show what's likely running next
            remaining = status['total'] - status['executed']
            if remaining > 0:
                print(f"\n⏳ {remaining} cells remaining...")
                if status['executed'] >= 4 and status['executed'] < 7:
                    print("   → Currently: Setting up project structure")
                elif status['executed'] >= 7 and status['executed'] < 9:
                    print("   → Currently: Fetching data (this takes 2-5 minutes)")
                elif status['executed'] >= 9 and status['executed'] < 11:
                    print("   → Currently: Building features")
                elif status['executed'] >= 11 and status['executed'] < 13:
                    print("   → Currently: Training price models (3-5 minutes)")
                elif status['executed'] >= 13 and status['executed'] < 15:
                    print("   → Currently: Training sentiment models")
                elif status['executed'] >= 15 and status['executed'] < 17:
                    print("   → Currently: Training meta-ensemble (2-3 minutes)")
                elif status['executed'] >= 17:
                    print("   → Currently: Running backtest (5-10 minutes)")
            
            previous_progress = status['percentage']
            
            print(f"\n⏱ Next check in {check_interval} seconds...")
            print("="*70 + "\n")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠ Monitoring stopped by user")
        print(f"Final progress: {status['percentage']:.1f}%")
    except Exception as e:
        print(f"\n\n⚠ Monitoring error: {e}")
        print("Continuing to monitor...")
        time.sleep(check_interval)

if __name__ == "__main__":
    notebook_path = Path(__file__).parent / "MASTER_RUNNER_COLAB.ipynb"
    if notebook_path.exists():
        monitor_continuously(notebook_path, check_interval=15)
    else:
        print(f"Error: Notebook not found at {notebook_path}")


