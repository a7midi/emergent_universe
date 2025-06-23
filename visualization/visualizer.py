"""
visualization/visualizer.py

This module defines the Visualizer class, responsible for creating a real-time
graphical representation of the simulation. It uses Matplotlib to plot the
causal site nodes and dynamically updates their colors based on their tags,
and highlights discovered particles.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Visualizer:
    """Handles the real-time visualization of the simulation."""

    def __init__(self, causal_site, config):
        """
        Initializes the visualizer and sets up the plot.

        Args:
            causal_site (CausalSite): The universe's structure.
            config (dict): The global configuration dictionary.
        """
        self.causal_site = causal_site
        self.config = config
        self.node_positions = {}
        self.particle_patches = {} # To keep track of drawn patches

        # Set up interactive plotting
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))

        # Pre-calculate node positions for plotting
        self._calculate_node_positions()
        
        # Create a colormap for tags
        self.colormap = plt.cm.get_cmap(
            'viridis', self.config['tags']['alphabet_size_q']
        )
        
        # Initial drawing of the graph nodes
        self.scatter = self._draw_initial_nodes()
        self.ax.set_title("Emergent Universe - Tick: 0")
        self.fig.tight_layout()

    def _calculate_node_positions(self):
        """
        Calculates and stores the (x, y) coordinates for each node.
        
        The layout uses the layer as the y-coordinate and the node's
        position within the layer as the x-coordinate, creating an intuitive
        representation of the causal flow from top to bottom.
        """
        for layer_idx, nodes in self.causal_site.nodes_by_layer.items():
            num_nodes = len(nodes)
            for i, node_id in enumerate(nodes):
                # Spread nodes horizontally, centered at 0
                x = (i - num_nodes / 2)
                # Layer determines the vertical position
                y = -layer_idx 
                self.node_positions[node_id] = (x, y)
    
    def _draw_initial_nodes(self):
        """Draws the initial scatter plot of all nodes."""
        pos = np.array(list(self.node_positions.values()))
        
        self.ax.set_facecolor('black')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Draw edges first, in a faint color
        for edge in self.causal_site.graph.edges():
            pos_start = self.node_positions[edge[0]]
            pos_end = self.node_positions[edge[1]]
            self.ax.plot([pos_start[0], pos_end[0]], [pos_start[1], pos_end[1]], 
                         color='gray', alpha=0.1, zorder=1)

        return self.ax.scatter(pos[:, 0], pos[:, 1], s=10, zorder=2)

    def update_plot(self, state, particles, tick):
        """
        Updates the plot with the latest simulation state.

        Args:
            state (np.ndarray): The current tags of all nodes.
            particles (dict): A dictionary of currently detected Particle objects.
            tick (int): The current simulation tick.
        """
        # 1. Update node colors based on their tags
        node_colors = [self.colormap(state[node_id] / self.config['tags']['alphabet_size_q']) 
                       for node_id in self.node_positions.keys()]
        self.scatter.set_color(node_colors)
        
        # 2. Update particle highlights
        # Remove old patches
        for patch_id, patch_list in self.particle_patches.items():
            for p in patch_list:
                p.remove()
        self.particle_patches.clear()

        # Draw new patches for current particles
        for particle_id, particle in particles.items():
            min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
            
            # Find the bounding box of the particle's cells
            for cell_id in particle.cells:
                nodes_in_cell = self.causal_site.get_nodes_in_cell(cell_id)
                for node_id in nodes_in_cell:
                    x, y = self.node_positions[node_id]
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)

            if min_x != float('inf'):
                # Add a small margin to the box
                width = (max_x - min_x) + 2
                height = (max_y - min_y) + 2
                
                # Create a semi-transparent rectangle to highlight the particle
                rect = patches.Rectangle((min_x - 1, min_y - 1), width, height,
                                         linewidth=2, edgecolor='red',
                                         facecolor='red', alpha=0.2, zorder=3)
                
                label = self.ax.text(min_x, max_y + 1, f"P:{particle_id}",
                                     color='white', fontsize=8, zorder=4)
                
                self.ax.add_patch(rect)
                self.particle_patches[particle_id] = [rect, label]
        
        # 3. Update title and redraw canvas
        self.ax.set_title(f"Emergent Universe - Tick: {tick}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001) # Small pause to allow the GUI to update
    
    def save_frame(self, tick):
        """Saves the current plot frame as an image."""
        self.fig.savefig(f"results/images/frame_{tick:05d}.png", dpi=150, facecolor='black')

    def close(self):
        """Keeps the final plot window open until manually closed."""
        print("Simulation ended. Close the plot window to exit.")
        plt.ioff()
        plt.show()

