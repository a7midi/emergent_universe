"""
particle_detector.py

VERSION 3: Implements observer blindness and other robustness fixes.
- Now requires the StateManager to be passed in during initialization.
- The detect() method filters out hidden nodes before hashing cell states,
  making the detector a true observer as required by the theory.
- Incorporates critical fixes from the code review checklist:
  - Uses config.get() for safe access to parameters.
  - Skips appending None hashes to the history.
  - Removes unused global CONFIG import.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

# Import our project modules
from src.causal_site import CausalSite
from src.state_manager import StateManager # Added for type hinting
from src.utils.hashing import hash_cell_state
from src.utils.graph_algorithms import find_connected_clusters

@dataclass
class Particle:
    """A simple data class to hold information about a detected particle."""
    id: int
    period: int
    cells: frozenset
    last_seen_tick: int
    first_seen_tick: int
    
    @property
    def lifetime(self):
        return self.last_seen_tick - self.first_seen_tick

class ParticleDetector:
    """
    Identifies and tracks emergent particles from the simulation's state evolution.
    """

    def __init__(self, causal_site: CausalSite, state_manager: StateManager, config: dict):
        """
        Initializes the ParticleDetector.
        
        Args:
            causal_site: The CausalSite object for the simulation.
            state_manager: The StateManager object to access its hidden_nodes set.
            config: The global configuration dictionary.
        """
        self.causal_site = causal_site
        self.state_manager = state_manager
        self.config = config

        # --- AMENDED: Use .get() for safe config access ---
        detector_config = self.config.get('detector', {})
        history_len = detector_config.get('max_history_length', 10000)
        self.min_loop_period = detector_config.get('min_loop_period', 5)
        self.min_particle_size = detector_config.get('min_particle_size', 2)

        self.hash_history = {
            cell_id: deque(maxlen=history_len)
            for cell_id in self.causal_site.nodes_by_cell.keys()
        }
        
        self.particles = {}
        self._next_particle_id = 0
        self._cluster_to_particle_id = {}
        
        self.looping_cells_last_tick = set()

    def detect(self, current_state: np.ndarray, current_tick: int) -> dict:
        """
        The main detection method, called at every simulation tick.
        """
        looping_cells_by_period = {}

        for cell_id, node_ids in self.causal_site.nodes_by_cell.items():
            if not node_ids:
                continue

            # --- AMENDED: Implement Observer Blindness ---
            # Filter out hidden nodes before computing the hash for a cell.
            # This ensures the detector only "sees" the observable universe.
            visible_node_ids = [nid for nid in node_ids if nid not in self.state_manager.hidden_nodes]

            if not visible_node_ids:
                continue # Skip cells that are fully hidden

            cell_node_indices = np.array(visible_node_ids, dtype=int)
            cell_tags = current_state[cell_node_indices]
            
            new_hash = hash_cell_state(cell_tags)
            
            # --- AMENDED: Do not add None hashes to the history ---
            if new_hash is None:
                continue
            
            history = self.hash_history[cell_id]
            if history:
                try:
                    # Search for the new hash in the cell's history to find loops.
                    # The reversed iterator is efficient and finds the most recent match first.
                    reversed_history = reversed(history)
                    index_from_end = next(i for i, h in enumerate(reversed_history) if h == new_hash)
                    
                    # Add 1 to convert the "distance from the end" to a period length.
                    period = index_from_end + 1
                    
                    if period >= self.min_loop_period:
                        if period not in looping_cells_by_period:
                            looping_cells_by_period[period] = set()
                        looping_cells_by_period[period].add(cell_id)

                except StopIteration:
                    # This is expected if no loop is found.
                    pass
            
            self.hash_history[cell_id].append(new_hash)

        all_looping_cells = set()
        for period_group in looping_cells_by_period.values():
            all_looping_cells.update(period_group)
        self.looping_cells_last_tick = all_looping_cells

        self._update_particles(looping_cells_by_period, current_tick)
        return self.particles
    
    def _update_particles(self, looping_cells_by_period: dict, current_tick: int):
        """Identifies and updates particle records based on detected clusters."""
        active_clusters = set()

        for period, cells in looping_cells_by_period.items():
            clusters = find_connected_clusters(cells, self.causal_site)
            
            for cluster in clusters:
                if len(cluster) < self.min_particle_size:
                    continue
                
                cluster_set = frozenset(cluster)
                active_clusters.add(cluster_set)

                if cluster_set in self._cluster_to_particle_id:
                    particle_id = self._cluster_to_particle_id[cluster_set]
                    self.particles[particle_id].last_seen_tick = current_tick
                else:
                    new_id = self._next_particle_id
                    new_particle = Particle(
                        id=new_id,
                        period=period,
                        cells=cluster_set,
                        first_seen_tick=current_tick,
                        last_seen_tick=current_tick
                    )
                    self.particles[new_id] = new_particle
                    self._cluster_to_particle_id[cluster_set] = new_id
                    self._next_particle_id += 1
        
        stale_clusters = set(self._cluster_to_particle_id.keys()) - active_clusters
        for cluster in stale_clusters:
            particle_id = self._cluster_to_particle_id.pop(cluster)
            if particle_id in self.particles:
                del self.particles[particle_id]
