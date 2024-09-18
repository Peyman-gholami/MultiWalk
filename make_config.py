import networkx as nx
import json
from networkx.algorithms.approximation import maximum_independent_set


# The error you encountered indicates that the graph is not bipartite, which suggests that there might be a better approach to remove the minimum number of edges to make the graph bipartite. Here's an improved method to achieve this by attempting to find a maximum bipartite subgraph, using an approximate method:
#
# Create the Erdős-Rényi graph.
# Check if it is bipartite and, if not, remove edges to make it bipartite.
# Save both the original and bipartite graph configurations.
# Here’s the updated code to accomplish this:

def create_erdos_renyi_graph(num_nodes, probability):
    # Create the Erdős-Rényi graph
    G = nx.erdos_renyi_graph(num_nodes, probability)

    # Check if the graph is connected
    if not nx.is_connected(G):
        raise ValueError("The generated graph is not connected. Try increasing the probability.")

    # Prepare the configuration data
    config_data = {
        'num_nodes': num_nodes,
        'probability': probability,
        'edges': list(G.edges)
    }

    # Generate the output file name
    output_file = f'./configs/erdos_renyi_graph_{num_nodes}_nodes_{probability}_prob.json'

    # Save the configuration to a JSON file
    with open(output_file, 'w') as f:
        json.dump(config_data, f, indent=4)

    print(f"Graph with {num_nodes} nodes and probability {probability} saved to {output_file}")

    return G, output_file


def make_bipartite_and_save(G, num_nodes, probability):
    # Find a maximum independent set, which can be used to form a bipartite graph
    max_independent_set = maximum_independent_set(G)
    top_nodes = set(max_independent_set)
    bottom_nodes = set(G.nodes) - top_nodes

    # Create a copy of the graph to modify
    B = nx.Graph()
    B.add_nodes_from(G.nodes)

    # Add edges between nodes in different sets only
    for u, v in G.edges():
        if (u in top_nodes and v in bottom_nodes) or (u in bottom_nodes and v in top_nodes):
            B.add_edge(u, v)

    # Prepare the configuration data for the bipartite graph
    bipartite_config_data = {
        'num_nodes': num_nodes,
        'probability': probability,
        'edges': list(B.edges),
        'top_nodes': list(top_nodes),
        'bottom_nodes': list(bottom_nodes)
    }

    # Generate the output file name for the bipartite graph
    bipartite_output_file = f'./configs/bipartite_graph_{num_nodes}_nodes_{probability}_prob.json'

    # Save the bipartite configuration to a JSON file
    with open(bipartite_output_file, 'w') as f:
        json.dump(bipartite_config_data, f, indent=4)

    print(f"Bipartite graph with {num_nodes} nodes and probability {probability} saved to {bipartite_output_file}")

    return bipartite_output_file


def load_graph_as_dict(config_file):
    # Load the configuration from the JSON file
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    # Create an empty adjacency dictionary
    adjacency_dict = {i: [] for i in range(config_data['num_nodes'])}

    # Populate the adjacency dictionary
    for edge in config_data['edges']:
        node1, node2 = edge
        adjacency_dict[node1].append(node2)
        adjacency_dict[node2].append(node1)

    return adjacency_dict, config_data.get('top_nodes', []), config_data.get('bottom_nodes', [])


# Example usage
num_nodes = 5  # Replace with the desired number of nodes
probability = 0.3  # Replace with the desired probability of connection

try:
    # Create and save the graph
    G, config_file = create_erdos_renyi_graph(num_nodes, probability)

    # Save the bipartite version of the graph
    bipartite_config_file = make_bipartite_and_save(G, num_nodes, probability)

    # Load the original graph as a dictionary
    graph_dict, _, _ = load_graph_as_dict(config_file)
    print("Original graph as dictionary:", graph_dict)

    # Load the bipartite graph as a dictionary
    bipartite_graph_dict, top_nodes, bottom_nodes = load_graph_as_dict(bipartite_config_file)
    print("Bipartite graph as dictionary:", bipartite_graph_dict)
    print("Top nodes:", top_nodes)
    print("Bottom nodes:", bottom_nodes)
except ValueError as e:
    print(e)
