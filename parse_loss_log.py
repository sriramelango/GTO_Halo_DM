#!/usr/bin/env python3
"""
Script to parse training log file and plot loss vs iteration/epoch.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def parse_log_file(log_file_path):
    """
    Parse the log file to extract loss values and iteration numbers.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        tuple: (iterations, losses) - lists of iteration numbers and loss values
    """
    iterations = []
    losses = []
    
    # Pattern to match lines like: "loss: 0.0849:   7%|▋         | 723/10000 [19:06<3:53:39,  1.51s/it]"
    pattern = r'loss:\s+([\d.]+):\s+\d+%\|.*?\|\s+(\d+)/\d+'
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                iteration = int(match.group(2))
                losses.append(loss)
                iterations.append(iteration)
    
    return iterations, losses

def plot_loss(iterations, losses, output_file=None, show_plot=True):
    """
    Plot loss vs iteration.
    
    Args:
        iterations (list): List of iteration numbers
        losses (list): List of loss values
        output_file (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create the main plot
    plt.subplot(2, 1, 1)
    plt.plot(iterations, losses, 'b-', alpha=0.7, linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Iteration')
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    if len(losses) > 10:
        window_size = min(50, len(losses) // 10)
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        moving_avg_iterations = iterations[window_size-1:]
        plt.plot(moving_avg_iterations, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.legend()
    
    # Create a zoomed-in plot for the last portion
    plt.subplot(2, 1, 2)
    if len(iterations) > 100:
        # Show last 20% of the data
        start_idx = int(len(iterations) * 0.8)
        plt.plot(iterations[start_idx:], losses[start_idx:], 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Iteration (Last 20%)')
        plt.grid(True, alpha=0.3)
        
        # Add moving average for zoomed plot
        if len(losses[start_idx:]) > window_size:
            moving_avg_zoom = np.convolve(losses[start_idx:], np.ones(window_size)/window_size, mode='valid')
            moving_avg_iterations_zoom = iterations[start_idx + window_size - 1:]
            plt.plot(moving_avg_iterations_zoom, moving_avg_zoom, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
            plt.legend()
    else:
        plt.plot(iterations, losses, 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Iteration')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    if show_plot:
        plt.show()

def calculate_statistics(losses):
    """
    Calculate basic statistics about the loss values.
    
    Args:
        losses (list): List of loss values
        
    Returns:
        dict: Dictionary containing statistics
    """
    if not losses:
        return {}
    
    losses_array = np.array(losses)
    stats = {
        'min_loss': np.min(losses_array),
        'max_loss': np.max(losses_array),
        'mean_loss': np.mean(losses_array),
        'std_loss': np.std(losses_array),
        'final_loss': losses[-1],
        'total_iterations': len(losses),
        'loss_decrease': losses[0] - losses[-1] if len(losses) > 1 else 0
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Parse training log and plot loss vs iteration')
    parser.add_argument('log_file', help='Path to the log file')
    parser.add_argument('--output', '-o', help='Output file path for the plot (e.g., loss_plot.png)')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')
    parser.add_argument('--stats-only', action='store_true', help='Only print statistics, do not plot')
    
    args = parser.parse_args()
    
    # Check if log file exists
    if not Path(args.log_file).exists():
        print(f"Error: Log file '{args.log_file}' not found.")
        return
    
    # Parse the log file
    print(f"Parsing log file: {args.log_file}")
    iterations, losses = parse_log_file(args.log_file)
    
    if not iterations or not losses:
        print("No loss data found in the log file.")
        return
    
    print(f"Found {len(iterations)} data points")
    print(f"Training ran for {iterations[-1]} iterations")
    
    # Calculate and print statistics
    stats = calculate_statistics(losses)
    if stats:
        print("\nLoss Statistics:")
        print(f"  Initial Loss: {losses[0]:.6f}")
        print(f"  Final Loss: {stats['final_loss']:.6f}")
        print(f"  Total Loss Decrease: {stats['loss_decrease']:.6f}")
        print(f"  Minimum Loss: {stats['min_loss']:.6f}")
        print(f"  Maximum Loss: {stats['max_loss']:.6f}")
        print(f"  Mean Loss: {stats['mean_loss']:.6f}")
        print(f"  Standard Deviation: {stats['std_loss']:.6f}")
    
    if args.stats_only:
        return
    
    # Plot the data
    plot_loss(iterations, losses, output_file=args.output, show_plot=not args.no_show)

if __name__ == "__main__":
    main() 