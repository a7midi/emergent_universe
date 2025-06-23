"""
utils/graph_algorithms.py

This utility module provides graph-based algorithms needed for particle
detection. Its primary function is to take a set of "looping" grid cells
and identify connected components among them, which represent the emergent
particles.
"""

from collections import deque

def find_connected_clusters(looping_cells, causal_site):
    """
    Finds connected components (clusters) within a set of looping grid cells.

    This function defines adjacency between two cells if there is at least one
    causal edge in the main graph connecting a node in one cell to a node in
    the other. It then uses Breadth-First Search (BFS) to find the clusters.

    Args:
        looping_cells (set): A set of cell_id tuples that have been detected
                             as being in a loop.
        causal_site (CausalSite): The main CausalSite object containing the
                                  graph and cell data.

    Returns:
        list: A list of clusters, where each cluster is a list of cell_ids.
    """
    if not looping_cells:
        return []

    # Build an adjacency list for the graph of looping cells.
    adj = {cell: [] for cell in looping_cells}
    
    node_to_cell = {
        node: data['cell_id'] 
        for node, data in causal_site.graph.nodes(data=True) 
        if 'cell_id' in data
    }
    
    cell_list = list(looping_cells)
    for i in range(len(cell_list)):
        for j in range(i + 1, len(cell_list)):
            c1 = cell_list[i]
            c2 = cell_list[j]

            # --- BUG FIX ---
            # The original logic only checked for edges from c1->c2.
            # We must check for edges in both directions (c1->c2 or c2->c1)
            # to correctly determine if two cells are part of the same cluster.
            is_connected = False
            
            # Check for edges from c1 to c2
            for node1 in causal_site.get_nodes_in_cell(c1):
                for neighbor in causal_site.graph.successors(node1):
                    if node_to_cell.get(neighbor) == c2:
                        is_connected = True
                        break
                if is_connected:
                    break
            
            # If no connection found, check for edges from c2 to c1
            if not is_connected:
                for node2 in causal_site.get_nodes_in_cell(c2):
                    for neighbor in causal_site.graph.successors(node2):
                        if node_to_cell.get(neighbor) == c1:
                            is_connected = True
                            break
                    if is_connected:
                        break
            
            # If a connection exists in either direction, add to adjacency list.
            if is_connected:
                adj[c1].append(c2)
                adj[c2].append(c1)
            # --- END FIX ---
    
    clusters = []
    visited = set()

    for cell in looping_cells:
        if cell not in visited:
            new_cluster = []
            q = deque([cell])
            visited.add(cell)

            while q:
                current_cell = q.popleft()
                new_cluster.append(current_cell)

                for neighbor in adj[current_cell]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            
            clusters.append(new_cluster)

    return clusters


# --- Standalone Test Block ---
if __name__ == '__main__':
    # We need to mock the CausalSite object for this standalone test.
    class MockCausalSite:
        def __init__(self):
            self.graph = None
            self.nodes_by_cell = {}
        
        def get_nodes_in_cell(self, cell_id):
            return self.nodes_by_cell.get(cell_id, [])

    print("--- Running Standalone Test for utils/graph_algorithms.py ---")

    # 1. Create a mock causal site
    mock_site = MockCausalSite()
    
    # Create a graph with nodes assigned to cells (0,0), (0,1), (1,1), (2,2), (2,3)
    import networkx as nx
    mock_site.graph = nx.DiGraph()
    mock_site.graph.add_nodes_from([
        (0, {'cell_id': (0,0)}), (1, {'cell_id': (0,0)}),
        (2, {'cell_id': (0,1)}), (3, {'cell_id': (0,1)}),
        (4, {'cell_id': (1,1)}),
        (5, {'cell_id': (2,2)}),
        (6, {'cell_id': (2,3)})
    ])
    # Connect cell (0,0) to (0,1)
    mock_site.graph.add_edge(1, 2)
    # Connect cell (0,1) to (1,1)
    mock_site.graph.add_edge(3, 4)
    # Create an isolated connection
    mock_site.graph.add_edge(5, 6)

    mock_site.nodes_by_cell = {
        (0,0): [0, 1], (0,1): [2, 3], (1,1): [4], (2,2): [5], (2,3): [6]
    }

    # 2. Define a set of looping cells
    # One cluster of three, and one isolated cluster of two
    looping_cells = {(0,0), (0,1), (1,1), (2,2), (2,3)}
    
    print(f"\nInput Looping Cells: {looping_cells}")

    # 3. Run the clustering algorithm
    clusters = find_connected_clusters(looping_cells, mock_site)

    print("\nDiscovered Clusters:")
    for i, cluster in enumerate(clusters):
        print(f"  - Cluster {i+1}: {sorted(cluster)}")

    # 4. Verification Checks
    print("\n--- Verification Checks ---")
    expected_num_clusters = 2
    if len(clusters) == expected_num_clusters:
        print(f"Verification PASSED: Correct number of clusters found ({expected_num_clusters}).")
    else:
        print(f"Verification FAILED: Expected {expected_num_clusters}, but found {len(clusters)}.")

    # Check the content of the clusters (order doesn't matter)
    cluster_sets = [set(c) for c in clusters]
    expected_set1 = {(0,0), (0,1), (1,1)}
    expected_set2 = {(2,2), (2,3)}
    
    if (expected_set1 in cluster_sets) and (expected_set2 in cluster_sets):
        print("Verification PASSED: Cluster contents are correct.")
    else:
        print("Verification FAILED: Cluster contents are incorrect.")
        print(f"  Expected: [{expected_set1}, {expected_set2}]")
        print(f"  Found:    {cluster_sets}")
        
    print("\n--- Test Complete ---")
