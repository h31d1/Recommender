import networkx as nx
import numpy as np

def net_prop_dict(G):
    prop_dict = {}

    prop_dict['no_of_nodes'] = nx.number_of_nodes(G)
    prop_dict['no_of_edges'] = nx.number_of_edges(G)
    if nx.is_connected(G):
        prop_dict['average_shortest_path_length'] = nx.average_shortest_path_length(G)
        prop_dict['diameter'] = nx.diameter(G)
    prop_dict['transitivity'] = nx.transitivity(G)
    prop_dict['average_clustering'] = nx.average_clustering(G)   
    prop_dict['edge_density'] = nx.classes.function.density(G)
    prop_dict['average_degree'] = np.array([d for n, d in G.degree()]).sum()/nx.number_of_nodes(G)
    prop_dict['total_triangles'] = np.array(list(nx.triangles(G).values())).sum()
    prop_dict['number_connected_components'] = nx.algorithms.components.number_connected_components(G)
    return prop_dict

def net_prop_dict_whole(G, k=2):
    prop_dict = {}

    prop_dict['no_of_nodes'] = nx.number_of_nodes(G)
    prop_dict['no_of_edges'] = nx.number_of_edges(G)
    if nx.is_connected(G):
        prop_dict['average_shortest_path_length'] = nx.average_shortest_path_length(G)
        prop_dict['diameter'] = nx.diameter(G)
    prop_dict['transitivity'] = nx.transitivity(G)
    prop_dict['average_clustering'] = nx.average_clustering(G)   
    prop_dict['edge_density'] = nx.classes.function.density(G)
    prop_dict['average_degree'] = np.array([d for n, d in G.degree()]).sum()/nx.number_of_nodes(G)
    prop_dict['total_triangles'] = np.array(list(nx.triangles(G).values())).sum()
    prop_dict['number_connected_components'] = nx.algorithms.components.number_connected_components(G)
    prop_dict['giant_component_prop'] = net_prop_dict(G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0]))
    prop_dict['k_core_prop'] = net_prop_dict(nx.k_core(G))
    return prop_dict

def proximity_prestige(graph):
    prox = {}
    for node_i in graph.nodes():
        d = nx.shortest_path(graph,target=node_i)
        I = len(d.keys())-1
        dist_sum = 0
        for node_j in d.keys():
            dist = len(d[node_j])-1
            if dist > 0: dist_sum += dist
        prox[node_i] = dist_sum/I
    return prox

def color_generator(n):
    colors = np.random.choice(range(256),n)/256
    return colors