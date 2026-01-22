#!/bin/bash

# Setup script for active learning benchmark

echo "Setting up active learning benchmark environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install pytorch_mask_rcnn from source
git clone https://github.com/yhenon/pytorch-mask-rcnn.git
cd pytorch-mask-rcnn
pip install -r requirements.txt
python setup.py install
cd ..

# Create directory structure
mkdir -p config src experiments notebooks scripts results
mkdir -p results/figures results/checkpoints results/logs
mkdir -p experiments/configs

echo "Environment setup complete!"
echo "To activate: source venv/bin/activate"