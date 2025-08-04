
# GTO Halo Comprehensive Dataset Usage Guide

This directory contains a comprehensive dataset with all generated samples, 
SNOPT simulation results, and converged trajectory data.

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
- **simulation_config**: SNOPT simulation parameters used
- **snopt_results**: SNOPT convergence statistics
  - feasibility: Boolean success flag
  - snopt_inform: SNOPT convergence code
  - solving_time: Computation time in seconds
  - snopt_control_evaluations: Full SNOPT evaluation history (if available)
- **converged_trajectory**: Complete trajectory data (if feasible)
  - control_vector: Full SNOPT control solution
  - trajectory_parameters: Key trajectory parameters
  - control_segments: Thrust control for each segment
- **processing_metadata**: Timestamps and convergence status

## Example Usage:

```python
import pickle
import numpy as np

# Load complete dataset
with open('complete_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Filter for locally optimal solutions
optimal_solutions = [entry for entry in data 
                    if entry['processing_metadata']['convergence_status'] == 'LOCALLY_OPTIMAL']

# Extract feasible trajectories
feasible_controls = [entry['converged_trajectory']['control_vector'] 
                    for entry in data 
                    if entry['snopt_results']['feasibility']]

# Analyze solving times
solving_times = [entry['snopt_results']['solving_time'] 
                for entry in data 
                if entry['snopt_results']['solving_time'] is not None]
```
