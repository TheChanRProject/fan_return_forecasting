#!/bin/bash

# Check if advisor binary exists
if ! command -v advisor &> /dev/null
then
    echo "advisor binary could not be found. Please ensure it is in your PATH or current directory."
    echo "Checking local directory..."
    if [ -f "./advisor" ]; then
        ADVISOR="./advisor"
    else
        echo "Please locate the advisor binary and update this script or add it to PATH."
        exit 1
    fi
else
    ADVISOR="advisor"
fi

# Submit job
# Assuming CLI syntax: advisor job submit --name <name> --command <cmd>
echo "Submitting job to Advisor..."
$ADVISOR job submit \
    --name "fan-stock-forecast" \
    --image "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime" \
    --command "pip install -r requirements.txt && python train.py" \
    --resources "gpu=1"

echo "Job submitted."
