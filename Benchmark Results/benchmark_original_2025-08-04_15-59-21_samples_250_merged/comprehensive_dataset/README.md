# Merged Dataset: ORIGINAL_1_CHANNEL

## Overview
This dataset combines 200 + 50 = 250 samples from multiple benchmark runs.

## Contents
- **Total samples:** 250
- **Feasible samples:** 190
- **Infeasible samples:** 60
- **With physical trajectories:** 250

## Files
- `complete_dataset.pkl` - Main merged dataset
- `dataset_metadata.json` - Dataset statistics and structure
- `README.md` - This file

## Usage
```python
import pickle
with open('complete_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
print(f"Loaded 250 samples")
```

## Merge Information
- **Merge timestamp:** 2025-08-04T15:59:23.450017
- **Description:** Combined 200 + 50 sample datasets
- **Model type:** ORIGINAL_1_CHANNEL
