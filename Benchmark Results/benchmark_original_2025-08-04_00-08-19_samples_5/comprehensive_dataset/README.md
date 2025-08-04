
# GTO Halo Comprehensive Dataset Usage Guide (ORIGINAL 1-CHANNEL MODEL)

This directory contains a comprehensive dataset with all generated samples from the ORIGINAL 1-channel model,
SNOPT simulation results, converged trajectory data, and physical orbital trajectories.

## Model Type: ORIGINAL_1_CHANNEL
- Input: 67-dimensional vectors directly 
- Architecture: 1 channel, 67 sequence length
- No 3-channel conversion needed

## Files:

1. **complete_dataset.pkl**: Full dataset in Python pickle format
   - Load with: `data = pickle.load(open('complete_dataset.pkl', 'rb'))`
   
2. **complete_dataset.json**: Full dataset in JSON format (human-readable)

3. **feasible_trajectories.pkl**: Only the feasible/optimal trajectory data
   - Quick access to successful convergence results
   
4. **dataset_[status].pkl**: Data organized by convergence status
   - dataset_locally_optimal.pkl: SNOPT inform = 1 solutions
   - dataset_feasible.pkl: Feasible but not necessarily optimal
   - dataset_infeasible.pkl: Failed convergence
   - dataset_processing_error.pkl: Processing errors

5. **dataset_summary.json**: Overview of dataset contents

## Data Structure:

Each entry contains:
- **sample_idx**: Sample identifier
- **generated_sample**: Original diffusion model output
  - halo_energy: Physical halo energy parameter
  - trajectory_params: 66-dimensional trajectory parameters  
  - full_sample_vector: Complete input vector for SNOPT
  - model_type: 'ORIGINAL_1_CHANNEL'
- **simulation_config**: SNOPT simulation parameters used
- **snopt_results**: SNOPT convergence statistics
- **converged_trajectory**: Complete trajectory data (if feasible)
- **physical_trajectories**: Physical orbital state trajectories
- **processing_metadata**: Timestamps, convergence status, and model type

## Comparing with 3-Channel Model:

```python
import pickle

# Load original model results
with open('complete_dataset.pkl', 'rb') as f:
    original_data = pickle.load(f)

# Load 3-channel model results  
with open('../3channel_results/complete_dataset.pkl', 'rb') as f:
    channel3_data = pickle.load(f)

# Compare model performance
original_feasible = [entry for entry in original_data 
                    if entry['snopt_results']['feasibility']]
                    
channel3_feasible = [entry for entry in channel3_data 
                    if entry['snopt_results']['feasibility']]

print(f"Original 1-channel: {len(original_feasible)}/{len(original_data)} feasible")
print(f"3-channel: {len(channel3_feasible)}/{len(channel3_data)} feasible")
```
