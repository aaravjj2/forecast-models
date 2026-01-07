#!/bin/bash
# Continuous monitoring script for notebook execution

echo "Starting continuous monitoring..."
echo "Press Ctrl+C to stop"
echo ""

NOTEBOOK="MASTER_RUNNER_COLAB.ipynb"
PREVIOUS_PROGRESS=0

while true; do
    clear
    echo "============================================================"
    echo "LIVE PROGRESS MONITOR - $(date '+%H:%M:%S')"
    echo "============================================================"
    echo ""
    
    python3 monitor_progress.py
    
    # Check for errors
    if grep -q '"output_type": "error"' "$NOTEBOOK" 2>/dev/null; then
        echo ""
        echo "⚠ ERROR DETECTED IN NOTEBOOK!"
        echo "Check the notebook for error details"
    fi
    
    sleep 10  # Check every 10 seconds
done

