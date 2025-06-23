"""
state_manager.py

VERSION 37 (Definitive): Implements the final audit's recommendations.
- Hidden clocks are now "drifting," with their coefficients perturbed
  periodically to prevent long-term synchronization.
- The injective fusion rule is now order-sensitive and site-specific,
  using the receiver's node_id to salt the key. This is the most
  faithful and complex implementation of the theory.
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

    # --- AMENDED: Key is now order-sensitive and site-specific ---
    def fuse(self, pred_tags_tuple: tuple, receiver_id: int) -> int:
        """
        Creates a unique mapping based on the ordered tags and the receiving node.
        """
        # The key now includes the ordered predecessors and the receiver's ID.
        key = pred_tags_tuple + (receiver_id,)

        if key not in self.mapping:
            self.mapping[key] = self.next_tag_internal
            self.next_tag_internal += 1
        return self.mapping[key] % self.q


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
        
        self.hidden_params = {}
        if self.hidden_nodes:
            rng = np.random.default_rng(sim_config.get("seed", 42))
            for n in self.hidden_nodes:
                mult = rng.integers(2, self.q)
                while gcd(mult, self.q) != 1:
                    mult = rng.integers(2, self.q)
                add = rng.integers(0, self.q)
                self.hidden_params[n] = (int(mult), int(add))
        
        if sim_config.get('verbose', True):
            print(f"Using fusion mode: '{self.fusion_mode}'")
        
        self.initialize_state()

    def initialize_state(self):
        for node_id in self.causal_site.graph.nodes:
            self.state[node_id] = (node_id * 173) % self.q

    def _fusion(self, predecessor_tags: tuple, node_id: int):
        """Dispatches to the correct fusion rule based on config."""
        if not predecessor_tags:
            return None

        if self.fusion_mode == 'injective':
            if self.fusion_table:
                return self.fusion_table.fuse(predecessor_tags, node_id)
            raise RuntimeError("Injective mode selected but fusion_table not initialized.")
        
        # Other modes are not site-specific
        elif self.fusion_mode == 'sum_mod_q':
            return sum(predecessor_tags) % self.q
        elif self.fusion_mode == 'quadratic':
            s_mod = sum(predecessor_tags) % self.q
            return (s_mod * s_mod) % self.q
        
        raise ValueError(f"Unknown fusion_mode: '{self.fusion_mode}' in config.yaml")

    def tick(self):
        """Applies a deterministic update with drifting hidden clocks."""
        self.tick_counter += 1
        state_at_t = self.state
        next_state = state_at_t.copy()

        # Phase 1: Update hidden layer with drifting multiplicative-affine clocks.
        for node_id in self.hidden_nodes:
            mult, add = self.hidden_params[node_id]
            # --- AMENDED: Drifting coefficients ---
            if self.tick_counter > 0 and self.tick_counter % 11 == 0:
                add = (add + node_id) % self.q
                new_mult = (mult + 1 + node_id) % self.q or 1
                while gcd(new_mult, self.q) != 1:
                    new_mult = (new_mult + 1) % self.q or 1
                mult = new_mult
                self.hidden_params[node_id] = (mult, add)
            
            next_state[node_id] = (mult * state_at_t[node_id] + add) % self.q

        # Phase 2: Update observable nodes.
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
                    
                    # --- AMENDED: Pass node_id for site-specific fusion key ---
                    new_tag = self._fusion(predecessor_tags, node_id)
                    
                    if new_tag is not None:
                        next_state[node_id] = new_tag

        self.state = next_state

    def get_current_state(self):
        """Returns the current state array of the simulation."""
        return self.state
