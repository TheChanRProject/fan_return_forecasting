#!/bin/bash

# Check if adviser binary exists
if ! command -v adviser &> /dev/null
then
    echo "adviser binary could not be found. Please ensure it is in your PATH or current directory."
    echo "Checking local directory..."
    if [ -f "./adviser" ]; then
        adviser="./adviser"
    else
        echo "Please locate the adviser binary and update this script or add it to PATH."
        exit 1
    fi
else
    adviser="adviser"
fi

# Submit job
# Assuming CLI syntax: adviser job submit --name <name> --command <cmd>
echo "Submitting job to adviser..."
$adviser job submit \
    --name "fan-stock-forecast" \
    --image "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime" \
    --command "pip install -r requirements.txt && python train.py" \
    --resources "gpu=1"

echo "Job submitted."
