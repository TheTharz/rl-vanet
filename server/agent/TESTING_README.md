# Model Testing Guide

This directory contains scripts for testing and evaluating trained RL models.

## Quick Start

### 1. List Available Models

```bash
# List all models (sorted by most recent)
python list_models.py

# List only PPO models
python list_models.py --type ppo

# List only final models (not checkpoints)
python list_models.py --final

# Show full paths
python list_models.py --path

# Get latest model only
python list_models.py --latest
```

### 2. Test a Specific Model

**Basic test:**
```bash
python test_model.py --model models/ppo_dual_final_20251112_031440.pth
```

**Test with WandB logging:**
```bash
python test_model.py --model models/ppo_dual_final_20251112_031440.pth --wandb
```

**Test with all features (save results, generate plots, WandB):**
```bash
python test_model.py \
    --model models/ppo_dual_final_20251112_031440.pth \
    --wandb \
    --save \
    --plot \
    --max_steps 1000
```

**Test and display plots:**
```bash
python test_model.py \
    --model models/ppo_dual_final_20251112_031440.pth \
    --plot \
    --show
```

### 3. Quick Test Script

Test the latest model automatically:
```bash
# Make script executable (first time only)
chmod +x quick_test.sh

# Test latest model with 500 steps
./quick_test.sh

# Test specific model with custom steps
./quick_test.sh models/ppo_dual_final_XXX.pth 1000
```

## Command-Line Options

### test_model.py

```
Required:
  --model PATH              Path to model file (.pth)

Optional:
  --model_type TYPE         Model type: auto, ppo_dual, dqn_dual, dqn_simple
                           (default: auto-detect from filename)
  --port PORT              ZMQ port for NS3 (default: 5555)
  --max_steps N            Maximum test steps (default: 500)
  --log_interval N         Print stats every N steps (default: 10)
  
Output:
  --save                   Save results to JSON and CSV
  --plot                   Generate visualization plots
  --show                   Display plots interactively
  --output_dir DIR         Output directory (default: test_results)
  
WandB:
  --wandb                  Enable WandB logging
  --wandb_project NAME     WandB project name
  --wandb_tags TAG [TAG]   Additional WandB tags
  
Other:
  --quiet                  Suppress verbose output
```

### list_models.py

```
  --dir DIR                Models directory (default: models)
  --type TYPE              Filter by type: ppo, dqn, dqn_dual, dqn_simple
  --sort SORT              Sort by: time, name (default: time)
  --path                   Show full paths
  --latest                 Show only latest model
  --final                  Show only final models (not checkpoints)
```

## Examples

### Example 1: Test PPO Model with WandB

```bash
python test_model.py \
    --model models/ppo_dual_final_20251112_031440.pth \
    --wandb \
    --wandb_project "vanet-evaluation" \
    --wandb_tags "final" "production" \
    --max_steps 1000 \
    --save \
    --plot
```

### Example 2: Test DQN Model

```bash
python test_model.py \
    --model models/dqn_dual_final_20251108_050303.pth \
    --max_steps 800 \
    --save \
    --plot
```

### Example 3: Quick Comparison

```bash
# Test multiple models
for model in models/ppo_dual_final_*.pth; do
    echo "Testing $model..."
    python test_model.py --model "$model" --max_steps 500 --save --quiet
done

# Compare results
python compare_models.py --plot
```

### Example 4: Find and Test Latest Model

```bash
# Find latest model
LATEST=$(python list_models.py --latest --final | grep "models/" | head -1)

# Test it
python test_model.py --model "$LATEST" --wandb --save --plot
```

## Output Files

When using `--save`, the following files are created in `test_results/`:

1. **summary_[model]_[timestamp].json** - Complete test results and statistics
2. **details_[model]_[timestamp].csv** - Step-by-step metrics data

When using `--plot`, an additional file is created:

3. **plots_[model]_[timestamp].png** - Comprehensive visualization

## Test Results Summary

The test script provides:

### Reward Statistics
- Total reward
- Average, std dev, min, max, median

### Network Performance
- PDR (Packet Delivery Ratio)
- Throughput
- CBR (Channel Busy Ratio)
- Average neighbors

### Action Analysis
- Action diversity (entropy)
- Most common actions
- BeaconHz distribution
- TxPower distribution

### Visualizations (when using --plot)
- PDR over time
- Throughput over time
- CBR over time
- Reward per step
- Cumulative reward
- Neighbor count
- BeaconHz actions
- TxPower actions
- Action distributions
- Histograms

## WandB Integration

When using `--wandb`, metrics are logged to Weights & Biases in real-time:

- All step-by-step metrics
- Windowed averages (50-step windows)
- Final summary statistics
- Visualization plots

**WandB Dashboard includes:**
- Real-time metric plots
- Action distribution analysis
- Performance comparisons
- Model metadata

## Tips

1. **Before Testing:**
   - Ensure NS3 simulation is ready to connect on the specified port
   - Check that the model file exists and is not corrupted

2. **For Best Results:**
   - Use at least 500-1000 steps for meaningful statistics
   - Save results for future comparison
   - Use WandB for organized experiment tracking

3. **Troubleshooting:**
   - If test hangs, check NS3 simulation is running
   - If model type detection fails, specify `--model_type` explicitly
   - Check ZMQ port conflicts with `--port`

4. **Performance Analysis:**
   - Compare baseline (no RL) vs trained models
   - Look for stability in metrics (less variance = better)
   - Check action diversity (too low = not learning, too high = unstable)

## Model Comparison

To compare multiple models:

1. Test each model with `--save`:
```bash
python test_model.py --model models/model1.pth --save
python test_model.py --model models/model2.pth --save
```

2. Compare results:
```bash
python compare_models.py --plot
```

This generates a comparison plot and summary table.

## Integration with Training

You can test models during training by:

1. Training with save intervals:
```bash
python ppo_dual_continuous.py --train --save_interval 100
```

2. Testing checkpoints:
```bash
python test_model.py --model models/ppo_dual_step500_XXX.pth --wandb
```

3. Monitoring progress in WandB dashboard

## Notes

- Testing runs in **evaluation mode** (no learning, greedy action selection)
- Models are loaded from saved checkpoints
- All metrics match training definitions for consistency
- CSV exports allow custom analysis in Excel, Python, R, etc.
