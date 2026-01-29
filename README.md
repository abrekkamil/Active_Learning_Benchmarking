# Active Learning Benchmark for Mask R-CNN

A comprehensive benchmark for evaluating active learning strategies with Mask R-CNN for object detection and segmentation.

## Features

- Multiple cold-start initialization strategies
- Various active learning query strategies
- Support for multiple datasets (COCO, custom)
- Extensive logging with WandB integration
- Modular and extensible architecture

## Installation

```bash
git clone https://github.com/abrekkamil/Active_Learning_Benchmarking.git
cd active_learning_benchmark
pip install -r requirements.txt
# Test everything before starting experiments
python test_all_modules.py
```
You can start with notebooks to donwloand and test the datasets