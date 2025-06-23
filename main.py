"""
main.py

This is the main entry point for the Emergent Universe simulation.
It orchestrates the initialization and execution of all core components.
This version is updated to correctly initialize the refactored ParticleDetector.
"""
import os
from tqdm import tqdm

# Import the core components of our simulation
from src.config import CONFIG
from src.causal_site import CausalSite
from src.state_manager import StateManager
from src.particle_detector import ParticleDetector
from visualization.visualizer import Visualizer

def main():
    """
    The main function to run the simulation.
    """
    if CONFIG is None:
        print("Failed to load configuration. Exiting.")
        return

    print("--- Emergent Universe Simulation ---")
    
    # --- Setup output directory for images if needed ---
    if CONFIG['visualization']['enabled'] and CONFIG['visualization']['save_frames']:
        os.makedirs("results/images", exist_ok=True)

    # === Phase 1: Initialization ===
    print("1. Initializing Universe Components...")
    
    causal_site = CausalSite(CONFIG)
    causal_site.generate_graph()
    causal_site.assign_grid_cells()

    state_manager = StateManager(causal_site, CONFIG)
    
    # --- AMENDED: Pass the state_manager to the ParticleDetector ---
    # The detector now needs access to the state_manager to know which nodes are hidden.
    particle_detector = ParticleDetector(causal_site, state_manager, CONFIG)
    
    visualizer = None
    if CONFIG['visualization']['enabled']:
        print("Initializing Visualizer...")
        visualizer = Visualizer(causal_site, CONFIG)
    
    print("\nInitialization Complete.\n")
    
    # === Phase 2: Main Simulation Loop with Logging ===
    print("2. Starting Simulation Loop...")
    
    total_ticks = CONFIG['simulation']['total_ticks']
    log_interval = CONFIG['simulation'].get('log_interval', 100)
    
    # Throttling the progress bar to prevent console display issues.
    progress_bar = tqdm(range(total_ticks), desc="Simulating", mininterval=0.5)
    
    for tick in progress_bar:
        state_manager.tick()
        current_state = state_manager.get_current_state()
        particles = particle_detector.detect(current_state, tick)

        if tick > 0 and tick % log_interval == 0:
            num_looping = len(particle_detector.looping_cells_last_tick)
            num_particles = len(particles)
            # Use tqdm.write, which is safe to use with the progress bar
            tqdm.write(f"Tick {tick}: {num_looping} looping cells, {num_particles} particles found.")

        if visualizer and tick % CONFIG['visualization']['update_interval'] == 0:
            visualizer.update_plot(current_state, particles, tick)
            if CONFIG['visualization']['save_frames']:
                visualizer.save_frame(tick)

    print("\nSimulation Loop Complete.\n")

    # === Phase 3: Final Report (Restored) ===
    print("3. Final Simulation Report...")
    
    final_particles = particle_detector.particles
    if not final_particles:
        print("No stable particles were detected at the end of the simulation.")
    else:
        print(f"Detected {len(final_particles)} stable particle(s) at the end of the simulation:")
        for particle_id, particle in final_particles.items():
            print(f"  - Particle ID: {particle.id}")
            print(f"    - Period: {particle.period} ticks")
            print(f"    - Size: {len(particle.cells)} cells")
            print(f"    - First Seen: Tick {particle.first_seen_tick}")
            print(f"    - Lifetime: {particle.lifetime} ticks")
    
    if visualizer:
        visualizer.close()
    
    print("\n--- Simulation Finished ---")


if __name__ == '__main__':
    main()
