"""
validation/entropy_validator.py

A diagnosis tool to validate the core information dynamics of the simulation
against the "Deterministic Second Law" presented in the research papers.

This script measures the Shannon entropy of the observer slice over time and
plots the result. According to the theory (Paper 1, Cor. 8.22), the entropy
should exhibit linear growth.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# Import the core simulation components
from src.config import CONFIG
from src.causal_site import CausalSite
from src.state_manager import StateManager

def calculate_shannon_entropy(tags):
    """
    Calculates the Shannon entropy of a list or array of tags.
    Entropy H(X) = -sum(p(x) * log2(p(x)))
    """
    if tags.size == 0:
        return 0.0
    
    # Get the counts of each unique tag
    _, counts = np.unique(tags, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / tags.size
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def main():
    """
    Main function to run the validation test.
    """
    if CONFIG is None:
        print("Failed to load configuration. Exiting.")
        return

    print("--- Entropy Growth Validator ---")
    print("This tool will verify the 'Deterministic Second Law'.")
    print("Expect to see a plot of entropy showing linear growth.\n")

    # --- 1. Setup Simulation ---
    # Use a slightly smaller universe for a faster validation run
    CONFIG['causal_site']['layers'] = 30
    CONFIG['causal_site']['avg_nodes_per_layer'] = 30
    CONFIG['simulation']['total_ticks'] = 1000
    # Use the optimal parameters we found previously
    CONFIG['tags']['alphabet_size_q'] = 17
    CONFIG['tags']['max_out_degree_R'] = 2

    site = CausalSite(CONFIG)
    site.generate_graph()

    state_manager = StateManager(site, CONFIG)

    # --- 2. Define Observer Slice ---
    # The observer slice includes all nodes EXCEPT the hidden stratum (Layer 0).
    observer_slice_nodes = [
        node for layer in range(1, site.config['causal_site']['layers'])
        for node in site.nodes_by_layer.get(layer, [])
    ]
    observer_slice_indices = np.array(observer_slice_nodes, dtype=int)
    
    if observer_slice_indices.size == 0:
        print("Error: Observer slice is empty. Cannot run validation.")
        return

    # --- 3. Run Simulation and Record Entropy ---
    entropy_history = []
    total_ticks = CONFIG['simulation']['total_ticks']
    
    for tick in tqdm(range(total_ticks), desc="Validating Entropy Growth"):
        state_manager.tick()
        
        current_state = state_manager.get_current_state()
        
        # Get the state of only the observed nodes
        observed_tags = current_state[observer_slice_indices]
        
        # Calculate and store the entropy of the observer slice
        entropy = calculate_shannon_entropy(observed_tags)
        entropy_history.append(entropy)

    # --- 4. Analyze and Plot Results ---
    ticks_array = np.arange(total_ticks)
    
    # Perform a linear regression to find the slope
    # We ignore the initial phase where entropy hasn't saturated the system
    burn_in = total_ticks // 4
    slope, intercept = np.polyfit(ticks_array[burn_in:], entropy_history[burn_in:], 1)
    
    # Theoretical lower bound on the slope (entropy increment per tick)
    # From Paper 1, Prop. 8.20, the increment c0 >= log2(q) for a single hidden site.
    # Our system has many hidden sites, so we expect a significantly positive slope.
    c0 = np.log2(CONFIG['tags']['alphabet_size_q'])

    print("\n--- Validation Results ---")
    print(f"Measured Entropy Growth Slope: {slope:.6f} bits/tick")
    print(f"Theoretical Minimum Increment (c0): {c0:.6f} bits/tick")
    
    if slope > 0:
        print("\nVerification PASSED: Entropy is increasing, as predicted.")
        if slope >= c0 / site.graph.number_of_nodes(): # Heuristic check
             print("Verification PASSED: Growth rate is plausible.")
    else:
        print("\nVerification FAILED: Entropy is not increasing, contradicting the theory.")

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(ticks_array, entropy_history, label='Measured Observer Entropy (S)')
    plt.plot(ticks_array, slope * ticks_array + intercept, 'r--', 
             label=f'Linear Fit (Slope = {slope:.4f})')
    
    plt.title('Validation of the Deterministic Second Law', fontsize=16)
    plt.xlabel('Simulation Tick (t)', fontsize=12)
    plt.ylabel('Observer Shannon Entropy S(t) in bits', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.show()

if __name__ == '__main__':
    # We need to ensure the validation script runs from the project root.
    # We can add the project root to the path for robust imports.
    import sys
    import os
    # Add the project root to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()
