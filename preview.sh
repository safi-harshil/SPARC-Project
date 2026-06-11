#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate realtime_v2

# ------------------------------------------------------------------------------------
# Preview script to show live camera feeds from all connected cameras.
# ------------------------------------------------------------------------------------

echo "[INFO] Starting live camera previews..."
python3 ~/Desktop/SPARC-Project/scripts/preview_all_cams.py