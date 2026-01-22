#!/bin/bash

# Download and Prepare Datasets for Active Learning Benchmark

set -e  # Exit on error
set -o pipefail

# Configuration
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR="${PROJECT_DIR}/data"
COCO_DIR="${DATA_DIR}/coco"
VOC_DIR="${DATA_DIR}/voc"
CUSTOM_DIR="${DATA_DIR}/custom"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check for wget
    if ! command -v wget &> /dev/null; then
        log_error "wget is not installed. Please install it first."
        exit 1
    fi
    
    # Check for unzip
    if ! command -v unzip &> /dev/null; then
        log_error "unzip is not installed. Please install it first."
        exit 1
    fi
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed. Please install it first."
        exit 1
    fi
    
    log_success "All dependencies are installed"
}

# Function to create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    mkdir -p "$DATA_DIR"
    mkdir -p "$COCO_DIR"
    mkdir -p "$VOC_DIR"
    mkdir -p "$CUSTOM_DIR"
    mkdir -p "$DATA_DIR/processed"
    mkdir -p "$DATA_DIR/temp"
    
    log_success "Directory structure created"
}

# Function to download COCO dataset
download_coco() {
    local download_dir="$1"
    local year="$2"
    
    log_info "Downloading COCO $year dataset..."
    
    # Create directory
    mkdir -p "$download_dir"
    cd "$download_dir"
    
    # Define URLs
    if [ "$year" = "2017" ]; then
        # COCO 2017 dataset URLs
        images_url="http://images.cocodataset.org/zips/train2017.zip"
        val_images_url="http://images.cocodataset.org/zips/val2017.zip"
        test_images_url="http://images.cocodataset.org/zips/test2017.zip"
        annotations_url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    elif [ "$year" = "2014" ]; then
        # COCO 2014 dataset URLs
        images_url="http://images.cocodataset.org/zips/train2014.zip"
        val_images_url="http://images.cocodataset.org/zips/val2014.zip"
        test_images_url="http://images.cocodataset.org/zips/test2014.zip"
        annotations_url="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    else
        log_error "Unsupported COCO year: $year"
        return 1
    fi
    
    # Download files
    log_info "Downloading annotations..."
    wget -q --show-progress -c "$annotations_url" -O annotations.zip
    
    log_info "Downloading training images..."
    wget -q --show-progress -c "$images_url" -O train_images.zip
    
    log_info "Downloading validation images..."
    wget -q --show-progress -c "$val_images_url" -O val_images.zip
    
    # Optional: test images (large, skip by default)
    # log_info "Downloading test images..."
    # wget -q --show-progress -c "$test_images_url" -O test_images.zip
    
    # Extract files
    log_info "Extracting annotations..."
    unzip -q -o annotations.zip
    rm annotations.zip
    
    log_info "Extracting training images..."
    unzip -q -o train_images.zip -d images/
    rm train_images.zip
    
    log_info "Extracting validation images..."
    unzip -q -o val_images.zip -d images/
    rm val_images.zip
    
    # Optional: extract test images
    # log_info "Extracting test images..."
    # unzip -q -o test_images.zip -d images/
    # rm test_images.zip
    
    # Create symlinks for easier access
    ln -sf "$download_dir" "$COCO_DIR/$year"
    
    log_success "COCO $year dataset downloaded and extracted"
}

# Function to download Pascal VOC dataset
download_voc() {
    local download_dir="$1"
    local year="$2"
    
    log_info "Downloading Pascal VOC $year dataset..."
    
    # Create directory
    mkdir -p "$download_dir"
    cd "$download_dir"
    
    # VOC 2012 URLs
    if [ "$year" = "2012" ]; then
        voc_url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    elif [ "$year" = "2007" ]; then
        voc_url="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
        voc_test_url="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
    else
        log_error "Unsupported VOC year: $year"
        return 1
    fi
    
    # Download main dataset
    log_info "Downloading VOC $year dataset..."
    wget -q --show-progress -c "$voc_url" -O VOCtrainval.tar
    
    # Extract
    log_info "Extracting VOC $year dataset..."
    tar -xf VOCtrainval.tar
    rm VOCtrainval.tar
    
    # Download test set for VOC 2007
    if [ "$year" = "2007" ]; then
        log_info "Downloading VOC 2007 test set..."
        wget -q --show-progress -c "$voc_test_url" -O VOCtest.tar
        tar -xf VOCtest.tar
        rm VOCtest.tar
    fi
    
    # Create symlinks
    ln -sf "$download_dir" "$VOC_DIR/$year"
    
    log_success "Pascal VOC $year dataset downloaded and extracted"
}

# Function to download sample dataset for testing
download_sample_dataset() {
    log_info "Downloading sample dataset for testing..."
    
    # Create sample directory
    sample_dir="$DATA_DIR/sample"
    mkdir -p "$sample_dir"
    
    # Download sample images (using a small subset)
    cd "$sample_dir"
    
    # Create a simple structure
    mkdir -p images
    mkdir -p annotations
    
    # Download a few sample images
    log_info "Downloading sample images..."
    
    # Sample images from COCO (you can replace with your own URLs)
    sample_images=(
        "http://images.cocodataset.org/val2017/000000039769.jpg"
        "http://images.cocodataset.org/val2017/000000039770.jpg"
        "http://images.cocodataset.org/val2017/000000039771.jpg"
        "http://images.cocodataset.org/val2017/000000039772.jpg"
        "http://images.cocodataset.org/val2017/000000039773.jpg"
    )
    
    for i in "${!sample_images[@]}"; do
        img_url="${sample_images[$i]}"
        img_name=$(basename "$img_url")
        wget -q --show-progress -c "$img_url" -O "images/$img_name"
        
        # Create simple annotation
        cat > "annotations/${img_name%.*}.json" << EOF
{
    "image_id": $((i+1)),
    "file_name": "$img_name",
    "width": 640,
    "height": 480,
    "annotations": [
        {
            "bbox": [100, 100, 200, 200],
            "category_id": 1,
            "category_name": "object"
        }
    ]
}
EOF
    done
    
    # Create dataset info file
    cat > "dataset_info.json" << EOF
{
    "name": "sample_dataset",
    "description": "Sample dataset for testing active learning",
    "num_images": ${#sample_images[@]},
    "num_classes": 1,
    "classes": ["object"],
    "split": {
        "train": [1, 2, 3],
        "val": [4],
        "test": [5]
    }
}
EOF
    
    log_success "Sample dataset created in $sample_dir"
}

# Function to create synthetic dataset
create_synthetic_dataset() {
    log_info "Creating synthetic dataset..."
    
    synth_dir="$DATA_DIR/synthetic"
    mkdir -p "$synth_dir/images" "$synth_dir/annotations"
    
    cd "$synth_dir"
    
    # Create synthetic dataset using Python
    python3 -c "
import numpy as np
import json
import os
from PIL import Image, ImageDraw

# Create synthetic images
num_images = 100
image_size = 256

for i in range(num_images):
    # Create random image
    img = Image.new('RGB', (image_size, image_size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw random rectangles
    num_objects = np.random.randint(1, 5)
    annotations = []
    
    for j in range(num_objects):
        # Random position and size
        x1 = np.random.randint(0, image_size - 50)
        y1 = np.random.randint(0, image_size - 50)
        width = np.random.randint(20, 100)
        height = np.random.randint(20, 100)
        x2 = min(x1 + width, image_size - 1)
        y2 = min(y1 + height, image_size - 1)
        
        # Random color
        color = tuple(np.random.randint(0, 255, 3))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='black')
        
        # Annotation
        annotations.append({
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'category_id': np.random.randint(0, 3),
            'area': width * height
        })
    
    # Save image
    img.save(f'images/{i:06d}.jpg')
    
    # Save annotation
    with open(f'annotations/{i:06d}.json', 'w') as f:
        json.dump({
            'image_id': i,
            'file_name': f'{i:06d}.jpg',
            'width': image_size,
            'height': image_size,
            'annotations': annotations
        }, f, indent=2)

print(f'Created {num_images} synthetic images')
"
    
    # Create dataset config
    cat > "dataset_config.yaml" << EOF
name: "synthetic_dataset"
description: "Synthetic dataset for active learning experiments"
num_classes: 3
classes: ["class_0", "class_1", "class_2"]
image_size: 256
train_split: 0.7
val_split: 0.15
test_split: 0.15
EOF
    
    log_success "Synthetic dataset created in $synth_dir"
}

# Function to process dataset for active learning
process_dataset() {
    local dataset_type=$1
    local dataset_path=$2
    
    log_info "Processing $dataset_type dataset..."
    
    # Create processed directory
    processed_dir="$DATA_DIR/processed/$dataset_type"
    mkdir -p "$processed_dir"
    
    # Run processing script
    cd "$PROJECT_DIR"
    
    if [ -f "scripts/process_dataset.py" ]; then
        python3 scripts/process_dataset.py \
            --dataset-type "$dataset_type" \
            --input-dir "$dataset_path" \
            --output-dir "$processed_dir"
    else
        # Simple processing
        log_warning "No processing script found. Creating basic structure..."
        
        # Create train/val splits
        mkdir -p "$processed_dir/train"
        mkdir -p "$processed_dir/val"
        mkdir -p "$processed_dir/test"
        
        # Create info file
        cat > "$processed_dir/dataset_info.json" << EOF
{
    "dataset_type": "$dataset_type",
    "path": "$dataset_path",
    "processed_date": "$(date -I)",
    "splits": ["train", "val", "test"]
}
EOF
    fi
    
    log_success "Dataset processed and saved to $processed_dir"
}

# Function to verify dataset
verify_dataset() {
    local dataset_path=$1
    local dataset_type=$2
    
    log_info "Verifying $dataset_type dataset..."
    
    if [ ! -d "$dataset_path" ]; then
        log_error "Dataset directory not found: $dataset_path"
        return 1
    fi
    
    # Check for required files
    if [ "$dataset_type" = "coco" ]; then
        required_files=("annotations/instances_train2017.json" "images/train2017")
    elif [ "$dataset_type" = "voc" ]; then
        required_files=("Annotations" "JPEGImages")
    else
        required_files=("images" "annotations")
    fi
    
    for file in "${required_files[@]}"; do
        if [ ! -e "$dataset_path/$file" ]; then
            log_warning "Missing file/directory: $dataset_path/$file"
        fi
    done
    
    # Count files
    if [ -d "$dataset_path/images" ]; then
        num_images=$(find "$dataset_path/images" -name "*.jpg" -o -name "*.png" | wc -l)
        log_info "Found $num_images images"
    fi
    
    if [ -d "$dataset_path/annotations" ]; then
        num_annotations=$(find "$dataset_path/annotations" -name "*.json" -o -name "*.xml" | wc -l)
        log_info "Found $num_annotations annotation files"
    fi
    
    log_success "$dataset_type dataset verified"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all                     Download all datasets"
    echo "  --coco [YEAR]            Download COCO dataset (2017 or 2014)"
    echo "  --voc [YEAR]             Download Pascal VOC dataset (2012 or 2007)"
    echo "  --sample                 Download sample dataset"
    echo "  --synthetic              Create synthetic dataset"
    echo "  --process TYPE PATH      Process dataset for active learning"
    echo "  --verify TYPE PATH       Verify dataset integrity"
    echo "  --setup                  Setup directories only"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all                 Download all datasets"
    echo "  $0 --coco 2017           Download COCO 2017"
    echo "  $0 --sample              Download sample dataset for testing"
    echo "  $0 --process coco ./data/coco/2017"
}

# Main function
main() {
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    # Banner
    echo -e "${BLUE}"
    echo "=========================================="
    echo "  Dataset Download and Preparation"
    echo "=========================================="
    echo -e "${NC}"
    
    # Check dependencies
    check_dependencies
    
    # Create directories
    create_directories
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                # Download all datasets
                download_coco "$COCO_DIR/2017" "2017"
                download_voc "$VOC_DIR/2012" "2012"
                download_sample_dataset
                create_synthetic_dataset
                shift
                ;;
            --coco)
                if [ -n "$2" ] && [[ "$2" =~ ^(2017|2014)$ ]]; then
                    download_coco "$COCO_DIR/$2" "$2"
                    shift 2
                else
                    # Default to 2017
                    download_coco "$COCO_DIR/2017" "2017"
                    shift
                fi
                ;;
            --voc)
                if [ -n "$2" ] && [[ "$2" =~ ^(2012|2007)$ ]]; then
                    download_voc "$VOC_DIR/$2" "$2"
                    shift 2
                else
                    # Default to 2012
                    download_voc "$VOC_DIR/2012" "2012"
                    shift
                fi
                ;;
            --sample)
                download_sample_dataset
                shift
                ;;
            --synthetic)
                create_synthetic_dataset
                shift
                ;;
            --process)
                if [ -n "$2" ] && [ -n "$3" ]; then
                    process_dataset "$2" "$3"
                    shift 3
                else
                    log_error "Missing arguments for --process"
                    show_usage
                    exit 1
                fi
                ;;
            --verify)
                if [ -n "$2" ] && [ -n "$3" ]; then
                    verify_dataset "$3" "$2"
                    shift 3
                else
                    log_error "Missing arguments for --verify"
                    show_usage
                    exit 1
                fi
                ;;
            --setup)
                # Already handled above
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Final summary
    echo -e "${GREEN}"
    echo "=========================================="
    echo "  Dataset Preparation Completed"
    echo "=========================================="
    echo -e "${NC}"
    
    # Show downloaded datasets
    log_info "Downloaded datasets:"
    
    if [ -d "$COCO_DIR" ] && [ "$(ls -A $COCO_DIR)" ]; then
        echo "  - COCO:"
        for year_dir in "$COCO_DIR"/*; do
            if [ -d "$year_dir" ]; then
                year=$(basename "$year_dir")
                echo "    * $year"
            fi
        done
    fi
    
    if [ -d "$VOC_DIR" ] && [ "$(ls -A $VOC_DIR)" ]; then
        echo "  - Pascal VOC:"
        for year_dir in "$VOC_DIR"/*; do
            if [ -d "$year_dir" ]; then
                year=$(basename "$year_dir")
                echo "    * $year"
            fi
        done
    fi
    
    if [ -d "$DATA_DIR/sample" ]; then
        echo "  - Sample dataset"
    fi
    
    if [ -d "$DATA_DIR/synthetic" ]; then
        echo "  - Synthetic dataset"
    fi
    
    # Show disk usage
    echo ""
    log_info "Disk usage:"
    du -sh "$DATA_DIR"/* 2>/dev/null | sort -h
    
    # Show next steps
    echo ""
    log_info "Next steps:"
    log_info "1. Update config files to point to the correct data paths"
    log_info "2. Run experiments: ./scripts/run_all_experiments.sh --setup"
    log_info "3. Start with sample dataset for testing"
}

# Run main function with all arguments
main "$@"