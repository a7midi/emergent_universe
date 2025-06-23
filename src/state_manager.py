"""
state_manager.py

VERSION 35 (Final, Corrected): Implements the "Time-as-Value" injective
fusion rule as specified in the final audit. This is the definitive version
designed to produce complex, non-global, and localized dynamics.
"""

import numpy as np
from math import gcd
from src.causal_site import CausalSite

class InjectiveFusionTable:
    """The 'injective' fusion mode: complex, history-dependent."""
    def __init__(self, q):
        self.q = q
        self.mapping = {}
        self.next_tag_internal = 0

    # --- AMENDED: Definitive "Time-as-Value" Fusion Rule ---
    def fuse(self, pred_tags_tuple: tuple, t: int) -> int:
        """
        Maps a set of predecessor tags to a stable internal value, but then
        perturbs that value with the current tick to prevent global synchrony.
        """
        key = tuple(sorted(pred_tags_tuple))
        if key not in self.mapping:
            # The mapping table only grows when a new *spatial* pattern is seen.
            self.mapping[key] = self.next_tag_internal
            self.next_tag_internal += 1
        
        # The output tag is a function of both the stable mapping and the global time.
        return (self.mapping[key] + t) % self.q


class StateManager:
    """
    Manages the state of all sites and evolves them according to a selected
    deterministic tag-fusion update rule.
    """

    def __init__(self, causal_site: CausalSite, config: dict):
        if not config:
            raise ValueError("Configuration could not be loaded. Aborting.")
        
        self.causal_site = causal_site
        self.config = config
        
        self.num_nodes = self.causal_site.graph.number_of_nodes()
        self.q = self.config['tags']['alphabet_size_q']
        self.state = np.zeros(self.num_nodes, dtype=int)
        
        self.fusion_mode = self.config['tags'].get('fusion_mode', 'injective')
        self.fusion_table = None
        if self.fusion_mode == 'injective':
            self.fusion_table = InjectiveFusionTable(self.q)
        
        self.tick_counter = 0
        self.hidden_nodes = set()
        
        sim_config = self.config.get('simulation', {})
        hide_layer_index = sim_config.get('hide_layer_index')

        if hide_layer_index is not None:
            self.hidden_nodes = set(self.causal_site.nodes_by_layer.get(hide_layer_index, []))
            if sim_config.get('verbose', True):
                print(f"Hiding layer {hide_layer_index} ({len(self.hidden_nodes)} nodes) from observer.")
        else:
            if sim_config.get('verbose', True):
                print("Universe is fully observable.")
        
        self.hidden_increments = {}
        if self.hidden_nodes:
            rng = np.random.default_rng(sim_config.get('seed', 42))
            for n in self.hidden_nodes:
                inc = rng.integers(1, self.q)
                while gcd(inc, self.q) != 1:
                    inc = rng.integers(1, self.q)
                self.hidden_increments[n] = int(inc)
        
        if sim_config.get('verbose', True):
            print(f"Using fusion mode: '{self.fusion_mode}'")
        
        self.initialize_state()

    def initialize_state(self):
        for node_id in self.causal_site.graph.nodes:
            self.state[node_id] = (node_id * 173) % self.q

    def _fusion(self, predecessor_tags: tuple):
        """Dispatches to the correct fusion rule based on config."""
        if not predecessor_tags:
            return None

        if self.fusion_mode == 'injective':
            if self.fusion_table:
                # Pass the tick counter for the time-as-value salt.
                return self.fusion_table.fuse(predecessor_tags, self.tick_counter)
            raise RuntimeError("Injective mode selected but fusion_table not initialized.")
        
        elif self.fusion_mode == 'sum_mod_q':
            return sum(predecessor_tags) % self.q
        elif self.fusion_mode == 'quadratic':
            s_mod = sum(predecessor_tags) % self.q
            return (s_mod * s_mod) % self.q
        
        raise ValueError(f"Unknown fusion_mode: '{self.fusion_mode}' in config.yaml")

    def tick(self):
        """Applies a single, causally consistent deterministic update to all nodes."""
        self.tick_counter += 1
        state_at_t = self.state
        next_state = state_at_t.copy()

        # Phase 1: Update hidden nodes.
        for node_id in self.hidden_nodes:
            increment = self.hidden_increments[node_id]
            next_state[node_id] = (state_at_t[node_id] + increment) % self.q

        # Phase 2: Update visible nodes.
        if self.causal_site.nodes_by_layer:
            max_layer = max(self.causal_site.nodes_by_layer.keys())
            for layer_index in range(1, max_layer + 1):
                for node_id in self.causal_site.nodes_by_layer.get(layer_index, []):
                    if node_id in self.hidden_nodes:
                        continue
                    
                    predecessors = list(self.causal_site.get_predecessors(node_id))
                    if not predecessors:
                        continue
                    
                    predecessor_tags = tuple(state_at_t[p_id] for p_id in predecessors)
                    
                    new_tag = self._fusion(predecessor_tags)
                    
                    if new_tag is not None:
                        next_state[node_id] = new_tag

        self.state = next_state

    def get_current_state(self):
        """Returns the current state array of the simulation."""
        return self.state
