# Archive Directory

This directory contains scripts that are no longer actively used but kept for reference.

## 📁 Directory Structure

### `legacy_training/`
Scripts from earlier training approaches, superseded by `train_streaming.py`:
- **`train.py`** - Original training script with local files
- **`train_cached_data.py`** - Training with pre-cached data
- **`train_fast.py`** - Experimental fast training (moved to experimental/)

### `debug_tools/`
Debugging and monitoring utilities:
- **`debug_tokenizer.py`** - Debug tokenizer training progress
- **`monitor_training.py`** - Monitor training via log files
- **`simple_test.py`** - Simple RLHF comparison test

### `data_tools/`
Data analysis and exploration scripts:
- **`estimate_tokenization.py`** - Estimate tokenization timing
- **`explore_openwebtext.py`** - Explore OpenWebText samples
- **`generate_dataset.py`** - Legacy dataset generation

### `experimental/`
Experimental features and one-off scripts:
- **`train_fast.py`** - Fast training with existing cache

## 🔄 Accessing Archived Scripts

To use any archived script:
```bash
# From project root
python archive/debug_tools/monitor_training.py
python archive/data_tools/explore_openwebtext.py
```

## 📝 Notes

- These scripts may have dependencies on older code structures
- Some may need path adjustments to work with current codebase
- Keep for reference and potential future use
- Regularly review for permanent deletion candidates