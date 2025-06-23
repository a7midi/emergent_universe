"""
particle_detector.py

VERSION 2: Includes a new attribute to store the set of cells that were
found to be looping in the most recent tick. This provides a hook for
the diagnostic logging requested in the root-cause analysis.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Import our project modules
from src.causal_site import CausalSite
from src.config import CONFIG
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

    def __init__(self, causal_site, config):
        """
        Initializes the ParticleDetector.
        """
        self.causal_site = causal_site
        self.config = config

        history_len = self.config['detector']['max_history_length']
        self.hash_history = {
            cell_id: deque(maxlen=history_len)
            for cell_id in self.causal_site.nodes_by_cell.keys()
        }
        
        self.particles = {}
        self._next_particle_id = 0
        self._cluster_to_particle_id = {}
        
        # --- NEW: Logging Hook ---
        self.looping_cells_last_tick = set()
        # --- END NEW ---

    def detect(self, current_state, current_tick):
        """
        The main detection method, called at every simulation tick.
        """
        looping_cells_by_period = {}

        for cell_id, node_ids in self.causal_site.nodes_by_cell.items():
            if not node_ids:
                continue

            cell_node_indices = np.array(node_ids, dtype=int)
            cell_tags = current_state[cell_node_indices]
            
            new_hash = hash_cell_state(cell_tags)
            if new_hash is None:
                self.hash_history[cell_id].append(new_hash)
                continue
            
            history = self.hash_history[cell_id]
            if history:
                try:
                    reversed_history = reversed(history)
                    index_from_end = next(i for i, h in enumerate(reversed_history) if h == new_hash)
                    period = index_from_end + 1
                    
                    if period >= self.config['detector']['min_loop_period']:
                        if period not in looping_cells_by_period:
                            looping_cells_by_period[period] = set()
                        looping_cells_by_period[period].add(cell_id)

                except StopIteration:
                    pass
            
            self.hash_history[cell_id].append(new_hash)

        # --- NEW: Update the logging set ---
        all_looping_cells = set()
        for period_group in looping_cells_by_period.values():
            all_looping_cells.update(period_group)
        self.looping_cells_last_tick = all_looping_cells
        # --- END NEW ---

        self._update_particles(looping_cells_by_period, current_tick)
        return self.particles
    
    def _update_particles(self, looping_cells_by_period, current_tick):
        """Identifies and updates particle records based on detected clusters."""
        active_clusters = set()

        for period, cells in looping_cells_by_period.items():
            clusters = find_connected_clusters(cells, self.causal_site)
            
            for cluster in clusters:
                if len(cluster) < self.config['detector']['min_particle_size']:
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
