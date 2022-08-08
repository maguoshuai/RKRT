# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 下午4:06
# @Author  : nevermore.huachi
# @File    : delete_check.py
# @Software: PyCharm
# 确认节点是否删除


import networkx as nx
from tqdm import tqdm


def local_impact(graph, impact_list):
    local_impact_list = {}
    for node in graph.nodes:
        total_impact = 0
        for no in nx.neighbors(graph, node):
            total_impact += impact_list[no]
        local_impact_list[node] = total_impact
    return local_impact_list


def distinguish(graph, local_impact_list, impact_list, node, threshold, mode):
    if mode == 1:
        no_impact = impact_list[node] / local_impact_list[node]
        no_neighbor_impact = [impact_list[neighbor] / local_impact_list[neighbor] for neighbor in
                              nx.neighbors(graph, node)]
        checking = [(no_impact / nei) for nei in no_neighbor_impact]

    elif mode == 2:
        checking = [
            (impact_list[node] / (local_impact_list[neighbor] + impact_list[neighbor])) / (
                    impact_list[neighbor] / (local_impact_list[node] + impact_list[node])) for neighbor in
            nx.neighbors(graph, node)]

    elif mode == 3:
        node_ego = nx.ego_graph(graph, node, radius=1)
        node_neighbor = nx.neighbors(graph, node)

        checking = []
        for neigh in node_neighbor:
            neigh_ego = nx.ego_graph(graph, neigh, radius=1)

            neigh_factor = len(list(set(neigh_ego.edges).intersection(set(node_ego.edges)))) / len(node_ego.edges)
            node_factor = len(list(set(node_ego.edges).intersection(set(neigh_ego.edges)))) / len(neigh_ego.edges)
            print(neigh_factor, node_factor)
            neigh_impact = neigh_factor * impact_list[neigh] / (local_impact_list[node] + impact_list[node])
            node_impact = node_factor * impact_list[node] / (local_impact_list[neigh] + impact_list[neigh])
            checking.append(node_impact / neigh_impact)

    elif mode == 4:

        node_neighbor = [n for n in nx.neighbors(graph, node)]
        checking = []
        for neigh in node_neighbor:
            neigh_neighbor = [n for n in nx.neighbors(graph, neigh)]
            nei_factor = len(list(set(neigh_neighbor).intersection(set(nx.ego_graph(graph, node, 1).nodes)))) / len(
                neigh_neighbor)
            node_factor = len(list(set(node_neighbor).intersection(set(nx.ego_graph(graph, neigh, 1).nodes)))) / len(
                node_neighbor)
            neigh_impact = nei_factor * impact_list[neigh] / local_impact_list[node]
            node_impact = node_factor * impact_list[node] / local_impact_list[neigh]
            checking.append(node_impact / neigh_impact)

    checking = [value >= threshold for value in checking]

    if True not in checking:
        return True
    else:
        return False


def filtering_node(process_graph, impact_list, threshold, mode):
    """
    :param process_graph: 要处理的网络
    :param impact_list: 节点的影响力的列表 type:dict
    :param threshold:
    :param mode: 删除节点的模式
    :return: 待删除的节点列表  type:list
    """
    remove_node = []

    local_impact_list = local_impact(process_graph, impact_list)

    for node in tqdm(process_graph.nodes):
        check = distinguish(process_graph, local_impact_list, impact_list, node, threshold, mode)
        if check:
            remove_node.append(node)
    return remove_node
