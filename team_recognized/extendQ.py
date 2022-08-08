# -*- coding: utf-8 -*-
# @Time    : 2021/8/17 下午5:09
# @Author  : nevermore.huachi
# @File    : extendQ.py
# @Software: PyCharm

import networkx as nx


def ExtendQ(g, coms, edge_num, node_k, node_coms_num):
    """ 重叠模块度 EQ

    Args:
        A (np.array): 邻接矩阵
        coms (dict): 社区划分结果
        edge_num (int): 边数量
        node_k (np.array): 每个节点对应的度 dim: (node_num, 2)
        node_coms_num (np.array): 每个节点所属社团数量 dim: (1, node_num)

    Returns:
        float: 重叠模块度值
    """
    factor = 2.0 * edge_num
    # node_k = sorted(node_k, key=lambda x: x[0], reverse=False)
    first_item = 0.0
    second_item = 0.0
    EQ = 0.0
    for eachc in coms:
        for eachp in coms[eachc]:
            for eachq in coms[eachc]:
                if g.has_edge(eachp, eachq):
                    first_item += g[eachp][eachq]['weight'] / float(node_coms_num[eachp] * node_coms_num[eachq])
                    second_item += node_k[eachp] * node_k[eachq] / float(node_coms_num[eachp] * node_coms_num[eachq])
    EQ = first_item - second_item / factor
    return EQ / factor
