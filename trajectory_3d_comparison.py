import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import random
import os

def load_dataset(dataset_path):
    """Load the comprehensive dataset."""
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")
    return data

def extract_trajectory_data(sample):
    """Extract trajectory data from a sample."""
    if 'physical_trajectories' not in sample:
        return None, None
    
    trajectory_data = sample['physical_trajectories']
    
    if 'converged_states' not in trajectory_data or 'predicted_states' not in trajectory_data:
        return None, None
    
    converged_states = np.array(trajectory_data['converged_states'])
    predicted_states = np.array(trajectory_data['predicted_states'])
    
    # Ensure both trajectories have the same length
    min_length = min(len(converged_states), len(predicted_states))
    converged_states = converged_states[:min_length]
    predicted_states = predicted_states[:min_length]
    
    return converged_states, predicted_states

def plot_3d_trajectory_comparison(converged_states, predicted_states, sample_idx, output_dir):
    """Create a 3D plot comparing converged vs predicted trajectories."""
    
    # Extract position data (first 3 columns)
    converged_pos = converged_states[:, :3]
    predicted_pos = predicted_states[:, :3]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(converged_pos[:, 0], converged_pos[:, 1], converged_pos[:, 2], 
            'b-', linewidth=2, label='Converged Trajectory', alpha=0.8)
    ax.plot(predicted_pos[:, 0], predicted_pos[:, 1], predicted_pos[:, 2], 
            'r--', linewidth=2, label='Predicted Trajectory', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(converged_pos[0, 0], converged_pos[0, 1], converged_pos[0, 2], 
               c='green', s=100, marker='o', label='Start Point')
    ax.scatter(converged_pos[-1, 0], converged_pos[-1, 1], converged_pos[-1, 2], 
               c='blue', s=100, marker='s', label='Converged End')
    ax.scatter(predicted_pos[-1, 0], predicted_pos[-1, 1], predicted_pos[-1, 2], 
               c='red', s=100, marker='^', label='Predicted End')
    
    # Calculate deviation metrics
    pos_deviation = np.linalg.norm(converged_pos - predicted_pos, axis=1)
    max_deviation = np.max(pos_deviation)
    mean_deviation = np.mean(pos_deviation)
    final_deviation = pos_deviation[-1]
    
    # Set labels and title
    ax.set_xlabel('X (DU)')
    ax.set_ylabel('Y (DU)')
    ax.set_zlabel('Z (DU)')
    ax.set_title(f'Trajectory Comparison - Sample {sample_idx}\n'
                f'Max Dev: {max_deviation:.4f} DU, Mean Dev: {mean_deviation:.4f} DU, Final Dev: {final_deviation:.4f} DU')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save plot
    output_path = os.path.join(output_dir, f'trajectory_comparison_sample_{sample_idx}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory comparison plot: {output_path}")
    print(f"  - Max deviation: {max_deviation:.4f} DU")
    print(f"  - Mean deviation: {mean_deviation:.4f} DU")
    print(f"  - Final deviation: {final_deviation:.4f} DU")
    
    return {
        'sample_idx': sample_idx,
        'max_deviation': max_deviation,
        'mean_deviation': mean_deviation,
        'final_deviation': final_deviation,
        'trajectory_length': len(converged_pos)
    }

def main():
    # Define dataset path for 3-channel model
    dataset_path = "Benchmark Results/benchmark_2025-08-04_02-14-56_samples_200/comprehensive_dataset/complete_dataset.pkl"
    
    # Create output directory
    output_dir = "trajectory_3d_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("3-CHANNEL MODEL TRAJECTORY COMPARISON")
    print("="*60)
    
    # Load dataset
    data = load_dataset(dataset_path)
    
    # Find feasible samples with trajectory data
    feasible_samples = []
    for i, sample in enumerate(data):
        if sample.get('snopt_results', {}).get('feasibility', False):
            converged_states, predicted_states = extract_trajectory_data(sample)
            if converged_states is not None and predicted_states is not None:
                feasible_samples.append((i, sample, converged_states, predicted_states))
    
    print(f"Found {len(feasible_samples)} feasible samples with trajectory data")
    
    if len(feasible_samples) < 3:
        print(f"Not enough feasible samples with trajectory data. Found {len(feasible_samples)} samples.")
        return
    
    # Select 3 random samples
    selected_samples = random.sample(feasible_samples, 3)
    
    # Generate plots for each selected sample
    results = []
    for sample_idx, sample, converged_states, predicted_states in selected_samples:
        print(f"\nProcessing sample {sample_idx}...")
        result = plot_3d_trajectory_comparison(converged_states, predicted_states, sample_idx, output_dir)
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("3-CHANNEL MODEL TRAJECTORY COMPARISON SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(f"Sample {result['sample_idx']}:")
        print(f"  - Trajectory length: {result['trajectory_length']} time steps")
        print(f"  - Max deviation: {result['max_deviation']:.4f} DU")
        print(f"  - Mean deviation: {result['mean_deviation']:.4f} DU")
        print(f"  - Final deviation: {result['final_deviation']:.4f} DU")
        print()
    
    # Calculate overall statistics
    max_deviations = [r['max_deviation'] for r in results]
    mean_deviations = [r['mean_deviation'] for r in results]
    final_deviations = [r['final_deviation'] for r in results]
    
    print(f"3-CHANNEL MODEL OVERALL STATISTICS (3 samples):")
    print(f"  - Average max deviation: {np.mean(max_deviations):.4f} ± {np.std(max_deviations):.4f} DU")
    print(f"  - Average mean deviation: {np.mean(mean_deviations):.4f} ± {np.std(mean_deviations):.4f} DU")
    print(f"  - Average final deviation: {np.mean(final_deviations):.4f} ± {np.std(final_deviations):.4f} DU")
    
    # Now run for original model
    print(f"\n{'='*60}")
    print("ORIGINAL MODEL TRAJECTORY COMPARISON")
    print(f"{'='*60}")
    
    # Define dataset path for original model
    original_dataset_path = "Benchmark Results/benchmark_original_2025-08-04_02-14-53_samples_200/comprehensive_dataset/complete_dataset.pkl"
    
    # Create output directory for original model
    original_output_dir = "trajectory_3d_plots_original"
    os.makedirs(original_output_dir, exist_ok=True)
    
    # Load original dataset
    original_data = load_dataset(original_dataset_path)
    
    # Find feasible samples with trajectory data for original model
    original_feasible_samples = []
    for i, sample in enumerate(original_data):
        if sample.get('snopt_results', {}).get('feasibility', False):
            converged_states, predicted_states = extract_trajectory_data(sample)
            if converged_states is not None and predicted_states is not None:
                original_feasible_samples.append((i, sample, converged_states, predicted_states))
    
    print(f"Found {len(original_feasible_samples)} feasible samples with trajectory data")
    
    if len(original_feasible_samples) < 3:
        print(f"Not enough feasible samples with trajectory data. Found {len(original_feasible_samples)} samples.")
        return
    
    # Select 3 random samples for original model
    original_selected_samples = random.sample(original_feasible_samples, 3)
    
    # Generate plots for each selected sample
    original_results = []
    for sample_idx, sample, converged_states, predicted_states in original_selected_samples:
        print(f"\nProcessing original model sample {sample_idx}...")
        result = plot_3d_trajectory_comparison(converged_states, predicted_states, sample_idx, original_output_dir)
        original_results.append(result)
    
    # Print summary for original model
    print(f"\n{'='*60}")
    print("ORIGINAL MODEL TRAJECTORY COMPARISON SUMMARY")
    print(f"{'='*60}")
    for result in original_results:
        print(f"Sample {result['sample_idx']}:")
        print(f"  - Trajectory length: {result['trajectory_length']} time steps")
        print(f"  - Max deviation: {result['max_deviation']:.4f} DU")
        print(f"  - Mean deviation: {result['mean_deviation']:.4f} DU")
        print(f"  - Final deviation: {result['final_deviation']:.4f} DU")
        print()
    
    # Calculate overall statistics for original model
    original_max_deviations = [r['max_deviation'] for r in original_results]
    original_mean_deviations = [r['mean_deviation'] for r in original_results]
    original_final_deviations = [r['final_deviation'] for r in original_results]
    
    print(f"ORIGINAL MODEL OVERALL STATISTICS (3 samples):")
    print(f"  - Average max deviation: {np.mean(original_max_deviations):.4f} ± {np.std(original_max_deviations):.4f} DU")
    print(f"  - Average mean deviation: {np.mean(original_mean_deviations):.4f} ± {np.std(original_mean_deviations):.4f} DU")
    print(f"  - Average final deviation: {np.mean(original_final_deviations):.4f} ± {np.std(original_final_deviations):.4f} DU")
    
    # Compare the two models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"3-Channel Model vs Original Model:")
    print(f"  - Max deviation: {np.mean(max_deviations):.4f} vs {np.mean(original_max_deviations):.4f} DU")
    print(f"  - Mean deviation: {np.mean(mean_deviations):.4f} vs {np.mean(original_mean_deviations):.4f} DU")
    print(f"  - Final deviation: {np.mean(final_deviations):.4f} vs {np.mean(original_final_deviations):.4f} DU")
    
    improvement_max = ((np.mean(original_max_deviations) - np.mean(max_deviations)) / np.mean(original_max_deviations)) * 100
    improvement_mean = ((np.mean(original_mean_deviations) - np.mean(mean_deviations)) / np.mean(original_mean_deviations)) * 100
    improvement_final = ((np.mean(original_final_deviations) - np.mean(final_deviations)) / np.mean(original_final_deviations)) * 100
    
    print(f"\nImprovements (3-Channel vs Original):")
    print(f"  - Max deviation: {improvement_max:+.1f}%")
    print(f"  - Mean deviation: {improvement_mean:+.1f}%")
    print(f"  - Final deviation: {improvement_final:+.1f}%")
    
    print(f"\nPlots saved to:")
    print(f"  - 3-Channel model: {output_dir}")
    print(f"  - Original model: {original_output_dir}")
    
    # Load dataset
    data = load_dataset(dataset_path)
    
    # Find feasible samples with trajectory data
    feasible_samples = []
    for i, sample in enumerate(data):
        if sample.get('snopt_results', {}).get('feasibility', False):
            converged_states, predicted_states = extract_trajectory_data(sample)
            if converged_states is not None and predicted_states is not None:
                feasible_samples.append((i, sample, converged_states, predicted_states))
    
    print(f"Found {len(feasible_samples)} feasible samples with trajectory data")
    
    if len(feasible_samples) < 3:
        print(f"Not enough feasible samples with trajectory data. Found {len(feasible_samples)} samples.")
        return
    
    # Select 3 random samples
    selected_samples = random.sample(feasible_samples, 3)
    
    # Generate plots for each selected sample
    results = []
    for sample_idx, sample, converged_states, predicted_states in selected_samples:
        print(f"\nProcessing sample {sample_idx}...")
        result = plot_3d_trajectory_comparison(converged_states, predicted_states, sample_idx, output_dir)
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAJECTORY COMPARISON SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(f"Sample {result['sample_idx']}:")
        print(f"  - Trajectory length: {result['trajectory_length']} time steps")
        print(f"  - Max deviation: {result['max_deviation']:.4f} DU")
        print(f"  - Mean deviation: {result['mean_deviation']:.4f} DU")
        print(f"  - Final deviation: {result['final_deviation']:.4f} DU")
        print()
    
    # Calculate overall statistics
    max_deviations = [r['max_deviation'] for r in results]
    mean_deviations = [r['mean_deviation'] for r in results]
    final_deviations = [r['final_deviation'] for r in results]
    
    print(f"OVERALL STATISTICS (3 samples):")
    print(f"  - Average max deviation: {np.mean(max_deviations):.4f} ± {np.std(max_deviations):.4f} DU")
    print(f"  - Average mean deviation: {np.mean(mean_deviations):.4f} ± {np.std(mean_deviations):.4f} DU")
    print(f"  - Average final deviation: {np.mean(final_deviations):.4f} ± {np.std(final_deviations):.4f} DU")
    print(f"\nPlots saved to: {output_dir}")

if __name__ == "__main__":
    main() 