import networkx as nx
import json
from networkx.algorithms.approximation import maximum_independent_set


def create_erdos_renyi_graph(num_nodes, probability):
    G = nx.erdos_renyi_graph(num_nodes, probability)

    if not nx.is_connected(G):
        raise ValueError("The generated graph is not connected. Try increasing the probability.")

    config_data = {
        'num_nodes': num_nodes,
        'probability': probability,
        'edges': list(G.edges)
    }

    output_file = f'./configs/erdos_renyi_graph_{num_nodes}_nodes.json'
    with open(output_file, 'w') as f:
        json.dump(config_data, f, indent=4)

    print(f"Graph with {num_nodes} nodes and probability {probability} saved to {output_file}")
    return G, output_file


def make_bipartite_and_save(G, num_nodes, probability):
    max_independent_set = maximum_independent_set(G)
    top_nodes = set(max_independent_set)
    bottom_nodes = set(G.nodes) - top_nodes

    B = nx.Graph()
    B.add_nodes_from(G.nodes)
    for u, v in G.edges():
        if (u in top_nodes and v in bottom_nodes) or (u in bottom_nodes and v in top_nodes):
            B.add_edge(u, v)

    bipartite_config_data = {
        'num_nodes': num_nodes,
        'probability': probability,
        'edges': list(B.edges),
        'top_nodes': list(top_nodes),
        'bottom_nodes': list(bottom_nodes)
    }

    bipartite_output_file = f'./configs/bipartite_erdos_renyi_graph_{num_nodes}_nodes.json'
    with open(bipartite_output_file, 'w') as f:
        json.dump(bipartite_config_data, f, indent=4)

    print(f"Bipartite erdos_renyi graph with {num_nodes} nodes and probability {probability} saved to {bipartite_output_file}")
    return bipartite_output_file


def create_cycle_graph(num_nodes):
    G = nx.cycle_graph(num_nodes)
    top_nodes, bottom_nodes = set(), set()

    if num_nodes % 2 == 0:
        top_nodes = {i for i in range(0, num_nodes, 2)}
        bottom_nodes = {i for i in range(1, num_nodes, 2)}
    else:
        raise ValueError("Cycle graph is only bipartite for an even number of nodes.")

    cycle_config_data = {
        'num_nodes': num_nodes,
        'edges': list(G.edges),
        'top_nodes': list(top_nodes),
        'bottom_nodes': list(bottom_nodes)
    }

    output_file = f'./configs/bipartite_cycle_graph_{num_nodes}_nodes.json'
    with open(output_file, 'w') as f:
        json.dump(cycle_config_data, f, indent=4)

    output_file = f'./configs/cycle_graph_{num_nodes}_nodes.json'
    with open(output_file, 'w') as f:
        json.dump(cycle_config_data, f, indent=4)


    print(f"Cycle graph with {num_nodes} nodes saved to {output_file}")
    return G, output_file


def create_complete_graph(num_nodes):
    G = nx.complete_graph(num_nodes)

    # Save the complete graph configuration
    complete_config_data = {
        'num_nodes': num_nodes,
        'edges': list(G.edges)
    }
    complete_output_file = f'./configs/complete_graph_{num_nodes}_nodes.json'
    with open(complete_output_file, 'w') as f:
        json.dump(complete_config_data, f, indent=4)
    print(f"Complete graph with {num_nodes} nodes saved to {complete_output_file}")

    # Create bipartite version of the complete graph
    top_nodes = set(range(num_nodes // 2))
    bottom_nodes = set(range(num_nodes // 2, num_nodes))

    B = nx.Graph()
    B.add_nodes_from(G.nodes)
    for u in top_nodes:
        for v in bottom_nodes:
            B.add_edge(u, v)

    complete_bipartite_config_data = {
        'num_nodes': num_nodes,
        'edges': list(B.edges),
        'top_nodes': list(top_nodes),
        'bottom_nodes': list(bottom_nodes)
    }

    complete_bipartite_output_file = f'./configs/bipartite_complete_graph_{num_nodes}_nodes.json'
    with open(complete_bipartite_output_file, 'w') as f:
        json.dump(complete_bipartite_config_data, f, indent=4)

    print(f"Complete bipartite graph with {num_nodes} nodes saved to {complete_bipartite_output_file}")
    return G, complete_output_file, complete_bipartite_output_file


def load_graph_as_dict(config_file):
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    adjacency_dict = {i: [] for i in range(config_data['num_nodes'])}
    for edge in config_data['edges']:
        node1, node2 = edge
        adjacency_dict[node1].append(node2)
        adjacency_dict[node2].append(node1)

    return adjacency_dict, config_data.get('top_nodes', []), config_data.get('bottom_nodes', [])


# Example usage
num_nodes = 20
probability = 0.3

try:
    # Erdős-Rényi
    # G, config_file = create_erdos_renyi_graph(num_nodes, probability)
    # bipartite_config_file = make_bipartite_and_save(G, num_nodes, probability)

    # Cycle Graph
    cycle_G, cycle_file = create_cycle_graph(num_nodes)

    # Complete Graph
    complete_G, complete_file, complete_bipartite_file = create_complete_graph(num_nodes)

    # Loading as dictionaries
    # graph_dict, _, _ = load_graph_as_dict(config_file)
    # print("Original Erdős-Rényi graph as dictionary:", graph_dict)
    #
    # bipartite_graph_dict, top_nodes, bottom_nodes = load_graph_as_dict(bipartite_config_file)
    # print("Bipartite Erdős-Rényi graph as dictionary:", bipartite_graph_dict)

    cycle_graph_dict, cycle_top, cycle_bottom = load_graph_as_dict(cycle_file)
    print("Cycle graph as dictionary:", cycle_graph_dict)

    complete_graph_dict, _, _ = load_graph_as_dict(complete_file)
    print("Complete graph as dictionary:", complete_graph_dict)

    complete_bipartite_graph_dict, complete_top, complete_bottom = load_graph_as_dict(complete_bipartite_file)
    print("Complete bipartite graph as dictionary:", complete_bipartite_graph_dict)
except ValueError as e:
    print(e)
