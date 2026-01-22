#!/bin/bash

# Run All Active Learning Experiments
# This script runs a complete benchmark of active learning experiments

set -e  # Exit on error
set -o pipefail

# Configuration
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_DIR="$PROJECT_DIR/experiments/configs"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_DIR="$PROJECT_DIR/results/logs"
SCRIPT_DIR="$PROJECT_DIR/experiments"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR/checkpoints"
mkdir -p "$RESULTS_DIR/figures"

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

# Function to run a single experiment
run_experiment() {
    local config_file=$1
    local exp_name=$2
    local log_file="$LOG_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
    
    log_info "Starting experiment: $exp_name"
    log_info "Config: $config_file"
    log_info "Log file: $log_file"
    
    # Run the experiment
    cd "$PROJECT_DIR"
    python "$SCRIPT_DIR/run_experiment.py" \
        --config "$config_file" \
        --cold-start "$3" \
        --query-strategy "$4" 2>&1 | tee "$log_file"
    
    # Check if experiment succeeded
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Experiment $exp_name completed successfully"
        return 0
    else
        log_error "Experiment $exp_name failed"
        return 1
    fi
}

# Function to run benchmark
run_benchmark() {
    local benchmark_name=$1
    local config_file=$2
    
    log_info "Starting benchmark: $benchmark_name"
    log_info "Using config: $config_file"
    
    # Run benchmark script
    cd "$PROJECT_DIR"
    python "$SCRIPT_DIR/benchmark_strategies.py" 2>&1 | tee "$LOG_DIR/${benchmark_name}_benchmark.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Benchmark $benchmark_name completed successfully"
    else
        log_error "Benchmark $benchmark_name failed"
    fi
}

# Function to compare results
compare_results() {
    log_info "Comparing experiment results..."
    
    cd "$PROJECT_DIR"
    python "$SCRIPT_DIR/compare_results.py" \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$RESULTS_DIR/comparison" \
        --generate-report 2>&1 | tee "$LOG_DIR/comparison.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Results comparison completed"
    else
        log_error "Results comparison failed"
    fi
}

# Function to clean up old results
cleanup_old_results() {
    local days_old=$1
    
    log_info "Cleaning up results older than $days_old days..."
    
    # Find and remove old checkpoint files
    find "$RESULTS_DIR/checkpoints" -name "*.pth" -mtime +$days_old -delete 2>/dev/null || true
    
    # Find and remove old log files
    find "$LOG_DIR" -name "*.log" -mtime +$days_old -delete 2>/dev/null || true
    
    # Keep only recent result files
    find "$RESULTS_DIR" -name "*.json" -mtime +$days_old -exec mv {} "$RESULTS_DIR/archive/" \; 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Function to setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Check if virtual environment exists
    if [ ! -d "$PROJECT_DIR/venv" ]; then
        log_warning "Virtual environment not found. Creating..."
        python3 -m venv "$PROJECT_DIR/venv"
    fi
    
    # Activate virtual environment
    source "$PROJECT_DIR/venv/bin/activate"
    
    # Install requirements
    log_info "Installing requirements..."
    pip install --upgrade pip
    pip install -r "$PROJECT_DIR/requirements.txt"
    
    # Install pytorch_mask_rcnn if not installed
    if ! python -c "import pytorch_mask_rcnn" 2>/dev/null; then
        log_info "Installing pytorch_mask_rcnn..."
        cd "$PROJECT_DIR"
        if [ ! -d "pytorch-mask-rcnn" ]; then
            git clone https://github.com/yhenon/pytorch-mask-rcnn.git
        fi
        cd pytorch-mask-rcnn
        pip install -r requirements.txt
        python setup.py install
        cd "$PROJECT_DIR"
    fi
    
    log_success "Environment setup completed"
}

# Function to check system requirements
check_system() {
    log_info "Checking system requirements..."
    
    # Check Python version
    python_version=$(python3 --version | cut -d' ' -f2)
    required_version="3.7.0"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
        log_success "Python version OK: $python_version"
    else
        log_error "Python version $python_version is too old. Need at least $required_version"
        exit 1
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        gpu_name=$(echo "$gpu_info" | cut -d',' -f1)
        gpu_memory=$(echo "$gpu_info" | cut -d',' -f2)
        
        log_info "GPU detected: $gpu_name"
        log_info "GPU memory: $((gpu_memory / 1024)) GB"
        
        if [ $gpu_memory -lt 8000 ]; then
            log_warning "GPU memory is limited. Some experiments may be slow."
        fi
    else
        log_warning "No GPU detected. Experiments will run on CPU (slow)."
    fi
    
    # Check disk space
    disk_space=$(df "$PROJECT_DIR" | tail -1 | awk '{print $4}')
    if [ $disk_space -lt 10485760 ]; then  # Less than 10GB
        log_warning "Low disk space: $((disk_space / 1024 / 1024)) GB available"
    else
        log_success "Disk space OK: $((disk_space / 1024 / 1024)) GB available"
    fi
    
    # Check memory
    total_mem=$(free -g | grep Mem | awk '{print $2}')
    if [ $total_mem -lt 16 ]; then
        log_warning "Limited RAM: ${total_mem}GB available"
    else
        log_success "RAM OK: ${total_mem}GB available"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all                    Run all experiments"
    echo "  --cold-start             Run cold start benchmark"
    echo "  --query-strategy         Run query strategy benchmark"
    echo "  --single CONFIG          Run single experiment with config file"
    echo "  --compare                Compare results from all experiments"
    echo "  --cleanup DAYS           Clean up old results (older than DAYS days)"
    echo "  --setup                  Setup environment only"
    echo "  --check-system           Check system requirements only"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all                 Run complete benchmark"
    echo "  $0 --cold-start          Run cold start experiments"
    echo "  $0 --single config.yaml  Run single experiment"
    echo "  $0 --compare             Compare existing results"
}

# Main function
main() {
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    # Default values
    RUN_ALL=false
    RUN_COLD_START=false
    RUN_QUERY_STRATEGY=false
    RUN_SINGLE=false
    RUN_COMPARE=false
    RUN_CLEANUP=false
    RUN_SETUP=false
    RUN_CHECK=false
    
    CONFIG_FILE=""
    CLEANUP_DAYS=30
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                RUN_ALL=true
                shift
                ;;
            --cold-start)
                RUN_COLD_START=true
                shift
                ;;
            --query-strategy)
                RUN_QUERY_STRATEGY=true
                shift
                ;;
            --single)
                RUN_SINGLE=true
                CONFIG_FILE="$2"
                shift 2
                ;;
            --compare)
                RUN_COMPARE=true
                shift
                ;;
            --cleanup)
                RUN_CLEANUP=true
                CLEANUP_DAYS="$2"
                shift 2
                ;;
            --setup)
                RUN_SETUP=true
                shift
                ;;
            --check-system)
                RUN_CHECK=true
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
    
    # Banner
    echo -e "${BLUE}"
    echo "=========================================="
    echo "  Active Learning Benchmark Suite"
    echo "=========================================="
    echo -e "${NC}"
    
    # Check system requirements
    if $RUN_CHECK || $RUN_ALL; then
        check_system
    fi
    
    # Setup environment
    if $RUN_SETUP || $RUN_ALL; then
        setup_environment
    fi
    
    # Run experiments
    if $RUN_ALL; then
        log_info "Running complete benchmark suite..."
        
        # Run cold start benchmark
        run_benchmark "cold_start" "$CONFIG_DIR/cold_start_config.yaml"
        
        # Run query strategy benchmark
        run_benchmark "query_strategy" "$CONFIG_DIR/query_strategy_config.yaml"
        
        # Compare results
        compare_results
        
    elif $RUN_COLD_START; then
        run_benchmark "cold_start" "$CONFIG_DIR/cold_start_config.yaml"
        
    elif $RUN_QUERY_STRATEGY; then
        run_benchmark "query_strategy" "$CONFIG_DIR/query_strategy_config.yaml"
        
    elif $RUN_SINGLE; then
        if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
            log_error "Config file not found: $CONFIG_FILE"
            exit 1
        fi
        
        exp_name=$(basename "$CONFIG_FILE" .yaml)
        run_experiment "$CONFIG_FILE" "$exp_name"
        
    elif $RUN_COMPARE; then
        compare_results
        
    elif $RUN_CLEANUP; then
        cleanup_old_results "$CLEANUP_DAYS"
        
    elif $RUN_SETUP; then
        # Already handled above
        :
        
    elif $RUN_CHECK; then
        # Already handled above
        :
    fi
    
    # Final summary
    echo -e "${GREEN}"
    echo "=========================================="
    echo "  Benchmark Suite Completed"
    echo "=========================================="
    echo -e "${NC}"
    
    # Show where results are stored
    if [ -d "$RESULTS_DIR" ]; then
        log_info "Results stored in: $RESULTS_DIR"
        
        # Count files
        num_results=$(find "$RESULTS_DIR" -name "*.json" | wc -l)
        num_checkpoints=$(find "$RESULTS_DIR/checkpoints" -name "*.pth" 2>/dev/null | wc -l || echo 0)
        num_figures=$(find "$RESULTS_DIR/figures" -name "*.png" 2>/dev/null | wc -l || echo 0)
        
        log_info "  - Result files: $num_results"
        log_info "  - Checkpoints: $num_checkpoints"
        log_info "  - Figures: $num_figures"
    fi
    
    # Show next steps
    echo ""
    log_info "Next steps:"
    log_info "1. View comparison report: less $RESULTS_DIR/comparison/benchmark_report.md"
    log_info "2. Open analysis notebooks: jupyter notebook notebooks/"
    log_info "3. Check WandB dashboard for interactive results"
}

# Run main function with all arguments
main "$@"