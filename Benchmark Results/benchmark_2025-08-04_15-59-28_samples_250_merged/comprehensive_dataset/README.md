# Merged Dataset: 3_CHANNEL

## Overview
This dataset combines 200 + 50 = 250 samples from multiple benchmark runs.

## Contents
- **Total samples:** 250
- **Feasible samples:** 222
- **Infeasible samples:** 28
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
- **Merge timestamp:** 2025-08-04T15:59:31.684140
- **Description:** Combined 200 + 50 sample datasets
- **Model type:** 3_CHANNEL
