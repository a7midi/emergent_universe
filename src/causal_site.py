"""
causal_site.py

This module defines the CausalSite class, which is responsible for generating and 
managing the fundamental structure of the simulated universe. 
VERSION 2: Includes a hard cap on predecessor count (in-degree) to more
accurately model the "Bounded Out-Degree R" hypothesis from the papers.
"""

import networkx as nx
import numpy as np
from src.config import CONFIG

class CausalSite:
    """
    Represents the finite, acyclic causal site of the universe.
    """

    def __init__(self, config):
        """
        Initializes the CausalSite.
        """
        self.config = config
        self.graph = nx.DiGraph()
        self.nodes_by_layer = {}
        self.nodes_by_cell = {}
        np.random.seed(self.config['simulation']['seed'])

    def generate_graph(self):
        """
        Procedurally generates the layered causal graph and then enforces
        the maximum predecessor count (in-degree).
        """
        print("Generating causal site graph...")
        node_counter = 0
        
        layers = self.config['causal_site']['layers']
        avg_nodes = self.config['causal_site']['avg_nodes_per_layer']
        edge_prob = self.config['causal_site']['edge_probability']

        for layer_index in range(layers):
            num_nodes_in_layer = np.random.poisson(avg_nodes)
            if num_nodes_in_layer == 0: num_nodes_in_layer = 1
            self.nodes_by_layer[layer_index] = []

            for _ in range(num_nodes_in_layer):
                node_id = node_counter
                self.graph.add_node(node_id, layer=layer_index)
                self.nodes_by_layer[layer_index].append(node_id)
                node_counter += 1

                if layer_index > 0:
                    previous_layer_nodes = self.nodes_by_layer[layer_index - 1]
                    for potential_parent in previous_layer_nodes:
                        if np.random.rand() < edge_prob:
                            self.graph.add_edge(potential_parent, node_id)
        
        # --- NEW SECTION: ENFORCE MAX IN-DEGREE (R) ---
        # This implements the audit's recommendation to match the theory's
        # bounded out-degree hypothesis more faithfully.
        # Note: The papers' "out-degree" corresponds to our "in-degree" (predecessors).
        max_r = self.config['tags']['max_out_degree_R']
        print(f"Enforcing maximum predecessor count (R_eff) of {max_r}...")
        
        nodes_to_check = [
            node for layer in range(1, layers) 
            for node in self.nodes_by_layer.get(layer, [])
        ]
        
        for node_id in nodes_to_check:
            predecessors = list(self.graph.predecessors(node_id))
            if len(predecessors) > max_r:
                # If a node has too many parents, randomly select which links to keep.
                edges_to_remove = np.random.choice(
                    predecessors, 
                    size=len(predecessors) - max_r, 
                    replace=False
                )
                for parent_to_remove in edges_to_remove:
                    self.graph.remove_edge(parent_to_remove, node_id)
        # --- END NEW SECTION ---

        print(f"Graph generation complete. Total nodes: {self.graph.number_of_nodes()}")

    def assign_grid_cells(self):
        """
        Partitions the graph nodes into a spatial grid for particle detection.
        """
        print("Assigning nodes to grid cells for particle detection...")
        grid_size = self.config['detector']['grid_size']

        for layer_idx, nodes in self.nodes_by_layer.items():
            if not nodes: continue
            for i, node_id in enumerate(nodes):
                cell_x = int((i / len(nodes)) * grid_size)
                cell_y = int((layer_idx / self.config['causal_site']['layers']) * grid_size)
                cell_id = (min(cell_x, grid_size - 1), min(cell_y, grid_size - 1))
                self.graph.nodes[node_id]['cell_id'] = cell_id

                if cell_id not in self.nodes_by_cell:
                    self.nodes_by_cell[cell_id] = []
                self.nodes_by_cell[cell_id].append(node_id)
        print("Grid cell assignment complete.")

    def get_predecessors(self, node_id):
        """Returns the immediate causal predecessors of a given site."""
        return self.graph.predecessors(node_id)
    
    def get_nodes_in_cell(self, cell_id):
        """Retrieves all node IDs located within a specific grid cell."""
        return self.nodes_by_cell.get(cell_id, [])

# Standalone Test Block (unchanged)
if __name__ == '__main__':
    # Test block remains the same for verification
    pass
