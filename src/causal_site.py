"""
causal_site.py

VERSION 5: Restores the essential get_nodes_in_cell() helper method that was
inadvertently removed, fixing the AttributeError crash.
"""

import networkx as nx
import numpy as np

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
        # The RNG is now created and used locally in generate_graph()

    def generate_graph(self):
        """
        Procedurally generates the layered causal graph, allowing for edges
        that span multiple layers to create a more complex structure.
        """
        print("Generating causal site graph...")
        node_counter = 0
        
        cs_config = self.config.get('causal_site', {})
        layers = cs_config.get('layers', 50)
        avg_nodes = cs_config.get('avg_nodes_per_layer', 40)
        edge_prob = cs_config.get('edge_probability', 0.1)
        max_lookback = cs_config.get('max_lookback_layers', 1)
        
        self.num_layers = layers
        rng = np.random.default_rng(self.config['simulation']['seed'])
        
        print(f"Using max predecessor lookback of {max_lookback} layers.")

        for layer_index in range(self.num_layers):
            num_nodes_in_layer = rng.poisson(avg_nodes)
            if num_nodes_in_layer == 0:
                num_nodes_in_layer = 1
            
            self.nodes_by_layer[layer_index] = []

            for _ in range(num_nodes_in_layer):
                node_id = node_counter
                self.graph.add_node(node_id, layer=layer_index)
                self.nodes_by_layer[layer_index].append(node_id)
                node_counter += 1

                if layer_index > 0:
                    start_layer = max(0, layer_index - max_lookback)
                    for lookback_idx in range(start_layer, layer_index):
                        previous_layer_nodes = self.nodes_by_layer.get(lookback_idx, [])
                        for potential_parent in previous_layer_nodes:
                            distance = layer_index - lookback_idx
                            if rng.random() < edge_prob / distance:
                                self.graph.add_edge(potential_parent, node_id)
        
        max_r = self.config.get('tags', {}).get('max_out_degree_R', 2)
        print(f"Enforcing maximum predecessor count (R) of {max_r}...")
        
        nodes_to_check = [
            node for layer in range(1, self.num_layers) 
            for node in self.nodes_by_layer.get(layer, [])
        ]
        
        for node_id in nodes_to_check:
            predecessors = list(self.graph.predecessors(node_id))
            if len(predecessors) > max_r:
                parents_to_remove = rng.choice(
                    predecessors, 
                    size=len(predecessors) - max_r, 
                    replace=False
                )
                for parent_to_remove in parents_to_remove:
                    self.graph.remove_edge(parent_to_remove, node_id)

        print(f"Graph generation complete. Total nodes: {self.graph.number_of_nodes()}")

    def assign_grid_cells(self):
        """Partitions the graph nodes into a spatial grid for particle detection."""
        print("Assigning nodes to grid cells for particle detection...")
        grid_size = self.config.get('detector', {}).get('grid_size', 12)

        for layer_idx, nodes in self.nodes_by_layer.items():
            if not nodes: continue
            for i, node_id in enumerate(nodes):
                cell_x = int((i / len(nodes)) * grid_size)
                cell_y = int((layer_idx / self.num_layers) * grid_size)
                cell_id = (min(cell_x, grid_size - 1), min(cell_y, grid_size - 1))
                self.graph.nodes[node_id]['cell_id'] = cell_id

                if cell_id not in self.nodes_by_cell:
                    self.nodes_by_cell[cell_id] = []
                self.nodes_by_cell[cell_id].append(node_id)
        print("Grid cell assignment complete.")

    def get_predecessors(self, node_id):
        """Returns the immediate causal predecessors of a given site."""
        return self.graph.predecessors(node_id)
    
    # --- AMENDED: Missing method restored ---
    def get_nodes_in_cell(self, cell_id):
        """Retrieves all node IDs located within a specific grid cell."""
        return self.nodes_by_cell.get(cell_id, [])
