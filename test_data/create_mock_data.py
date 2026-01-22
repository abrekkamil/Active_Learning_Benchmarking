#!/usr/bin/env python3
"""
Create mock data for testing.
"""

import numpy as np
import torch
import json
from pathlib import Path
import shutil

def create_mock_coco_dataset(output_dir: str = "test_data/coco"):
    """Create a mock COCO dataset for testing."""
    output_dir = Path(output_dir)
    
    # Create directories
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    
    for dir_path in [images_dir, annotations_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create mock annotations
    annotations = {
        "info": {
            "description": "Mock COCO Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "Test Suite"
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
            {"id": 3, "name": "car", "supercategory": "vehicle"}
        ],
        "images": [],
        "annotations": []
    }
    
    # Create 10 mock images
    for i in range(10):
        image_id = i + 1
        
        # Add image info
        annotations["images"].append({
            "id": image_id,
            "file_name": f"image_{image_id:06d}.jpg",
            "width": 640,
            "height": 480,
            "license": 0,
            "coco_url": "",
            "date_captured": ""
        })
        
        # Add 1-3 annotations per image
        for j in range(np.random.randint(1, 4)):
            annotation_id = len(annotations["annotations"]) + 1
            
            # Random bounding box
            x = np.random.randint(0, 500)
            y = np.random.randint(0, 350)
            width = np.random.randint(50, 150)
            height = np.random.randint(50, 150)
            
            annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": np.random.randint(1, 4),
                "bbox": [x, y, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            })
    
    # Save annotations
    annotations_path = annotations_dir / "instances_train2017.json"
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created mock COCO dataset at {output_dir}")
    print(f"  - Images: {len(annotations['images'])}")
    print(f"  - Annotations: {len(annotations['annotations'])}")
    print(f"  - Categories: {len(annotations['categories'])}")
    
    return output_dir

def create_mock_torch_dataset(output_dir: str = "test_data/torch"):
    """Create a mock torch dataset for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 100 random images and masks
    images = []
    masks = []
    
    for i in range(100):
        # Random RGB image
        image = torch.randn(3, 64, 64)
        images.append(image)
        
        # Random mask with 3 classes
        mask = torch.randint(0, 4, (64, 64))  # 0=background, 1-3=objects
        masks.append(mask)
    
    # Save as torch tensors
    torch.save({
        'images': images,
        'masks': masks,
        'classes': [1, 2, 3]
    }, output_dir / "mock_dataset.pth")
    
    print(f"Created mock torch dataset at {output_dir}")
    print(f"  - Samples: {len(images)}")
    print(f"  - Image shape: {images[0].shape}")
    print(f"  - Classes: {[1, 2, 3]}")
    
    return output_dir

def create_test_results(output_dir: str = "test_data/results"):
    """Create mock results for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple result files
    strategies = ["random", "diversity", "uncertainty", "hybrid"]
    
    for strategy in strategies:
        result = {
            "strategy": strategy,
            "final_ap": np.random.uniform(0.3, 0.7),
            "best_ap": np.random.uniform(0.35, 0.75),
            "final_labeled_count": np.random.randint(100, 200),
            "history": {
                "val_ap": list(np.random.uniform(0.1, 0.6, 10)),
                "labeled_count": list(range(100, 200, 10)),
                "train_loss": list(np.random.uniform(0.5, 2.0, 10))
            },
            "timing": {
                "total_time": np.random.uniform(600, 1800),  # 10-30 minutes
                "training_time": np.random.uniform(400, 1200),
                "query_time": np.random.uniform(50, 200)
            }
        }
        
        # Save as JSON
        result_path = output_dir / f"{strategy}_results.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"Created mock results at {output_dir}")
    print(f"  - Strategies: {strategies}")
    
    return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create mock data for testing")
    parser.add_argument("--all", action="store_true", help="Create all mock data")
    parser.add_argument("--coco", action="store_true", help="Create mock COCO dataset")
    parser.add_argument("--torch", action="store_true", help="Create mock torch dataset")
    parser.add_argument("--results", action="store_true", help="Create mock results")
    
    args = parser.parse_args()
    
    if args.all or not any([args.coco, args.torch, args.results]):
        args.coco = args.torch = args.results = True
    
    if args.coco:
        create_mock_coco_dataset()
    
    if args.torch:
        create_mock_torch_dataset()
    
    if args.results:
        create_test_results()
    
    print("\nMock data creation complete!")