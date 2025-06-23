"""
state_manager.py

VERSION 12: Implements a "per-tick" probabilistic hidden noise model.
A new `hidden_noise_probability` setting in config.yaml controls the
rate at which new information is injected into the hidden layer as a whole. This
allows for fine-tuning the system's "temperature" to find the "habitable
zone" for particle emergence, as suggested by the user.
"""

import numpy as np
from src.causal_site import CausalSite
from src.config import CONFIG

class InjectiveFusionTable:
    """The 'injective' fusion mode: complex, history-dependent."""
    def __init__(self, q):
        self.q = q
        self.mapping = {}
        self.next_tag_internal = 0

    def fuse(self, pred_tags_tuple):
        pred_tags_tuple = tuple(sorted(pred_tags_tuple))
        if pred_tags_tuple not in self.mapping:
            self.mapping[pred_tags_tuple] = self.next_tag_internal
            self.next_tag_internal += 1
        return self.mapping[pred_tags_tuple] % self.q


class StateManager:
    """
    Manages the state of all sites and evolves them according to a selected
    deterministic tag-fusion update rule.
    """

    def __init__(self, causal_site, config):
        self.causal_site = causal_site
        self.config = config
        if not self.config:
            raise ValueError("Configuration could not be loaded. Aborting.")
        
        self.num_nodes = self.causal_site.graph.number_of_nodes()
        self.q = self.config['tags']['alphabet_size_q']
        self.rng = np.random.default_rng(self.config['simulation']['seed'])
        self.state = np.zeros(self.num_nodes, dtype=int)
        
        self.fusion_mode = self.config['tags'].get('fusion_mode', 'injective')
        self.fusion_table = None
        if self.fusion_mode == 'injective':
            self.fusion_table = InjectiveFusionTable(self.q)
        print(f"Using fusion mode: '{self.fusion_mode}'")
        
        self.tick_counter = 0
        self.layer_0_nodes = self.causal_site.nodes_by_layer.get(0, [])
        
        self.use_hidden_noise = self.config['simulation'].get('hidden_noise', True)
        self.noise_prob = self.config['simulation'].get('hidden_noise_probability', 1.0)
        print(f"Hidden layer noise enabled: {self.use_hidden_noise} (Global Probability: {self.noise_prob})")
        
        self.info_stream = np.array([], dtype=int)
        if self.use_hidden_noise:
            num_hidden_sites = len(self.layer_0_nodes)
            total_ticks = self.config['simulation']['total_ticks']
            if num_hidden_sites > 0 and total_ticks > 0:
                stream_size = num_hidden_sites * (total_ticks + 1)
                self.info_stream = self.rng.integers(0, self.q, size=stream_size)
        
        self.initialize_state()

    def initialize_state(self):
        if not self.use_hidden_noise:
            for node_id in self.layer_0_nodes:
                self.state[node_id] = node_id % self.q
        elif self.info_stream.size > 0:
            initial_tags = self.info_stream[:len(self.layer_0_nodes)]
            for i, node_id in enumerate(self.layer_0_nodes):
                self.state[node_id] = initial_tags[i]

    def _fusion(self, predecessor_tags):
        """Dispatches to the correct fusion rule based on config."""
        if self.fusion_mode == 'injective':
            if self.fusion_table:
                return self.fusion_table.fuse(predecessor_tags)
            raise RuntimeError("Injective mode selected but fusion_table not initialized.")
        elif self.fusion_mode == 'sum_mod_q':
            if not predecessor_tags: return 0
            return sum(predecessor_tags) % self.q
        raise ValueError(f"Unknown fusion_mode: '{self.fusion_mode}' in config.yaml")

    def tick(self):
        state_at_t = np.copy(self.state)
        state_at_t_plus_1 = np.copy(state_at_t)

        # --- MODIFIED NOISE INJECTION ---
        # Perform a single Bernoulli trial for the entire hidden layer.
        if self.use_hidden_noise and self.rng.random() < self.noise_prob:
            # If the trial succeeds, the entire hidden layer is refreshed.
            if self.info_stream.size > 0:
                num_hidden = len(self.layer_0_nodes)
                start_index = (self.tick_counter + 1) * num_hidden
                end_index = start_index + num_hidden

                if end_index <= len(self.info_stream):
                    next_hidden_tags = self.info_stream[start_index:end_index]
                    for i, node_id in enumerate(self.layer_0_nodes):
                        state_at_t_plus_1[node_id] = next_hidden_tags[i]
        # --- END MODIFICATION ---

        for layer_index in range(1, len(self.causal_site.nodes_by_layer)):
            for node_id in self.causal_site.nodes_by_layer[layer_index]:
                predecessors = list(self.causal_site.get_predecessors(node_id))
                predecessor_tags = tuple(state_at_t[p_id] for p_id in predecessors)
                state_at_t_plus_1[node_id] = self._fusion(predecessor_tags)
        
        self.state = state_at_t_plus_1
        self.tick_counter += 1

    def get_current_state(self):
        return self.state
