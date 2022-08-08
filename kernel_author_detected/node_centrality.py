# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 上午11:02
# @Author  : nevermore.huachi
# @File    : node_centrality.py
# @Software: PyCharm
import networkx as nx
from tqdm import tqdm

# 　Harmonic 贡献分配
def Harmonic(authorCnt, order):
    total = sum([1 / (1 + i) for i in range(authorCnt)])
    credit = 1 / order
    return credit / total


# 　Axiomatic 贡献分配
def Axiomatic(authorCnt, order):
    credit = sum([1 / i for i in range(order, authorCnt + 1)]) / authorCnt
    return credit


def Hindex(indexList):
    indexSet = sorted(list(set(indexList)), reverse=True)
    for index in indexSet:
        # clist为大于等于指定引用次数index的文章列表
        clist = [i for i in indexList if i >= index]
        # 由于引用次数index逆序排列，当index<=文章数量len(clist)时，得到H指数
        if index <= len(clist):
            break
    return index


# 引用量统计
def get_impact(graph, node, mode='am', db_col=None):
    try:
        id = int(graph.nodes[node]['id'])
    except:
        id = int(node)

    pub_papers = db_col.find_one({'id': id})['pub_papers']

    impact = 0
    if mode == 'am':
        for paper in pub_papers:
            impact += (Axiomatic(paper['authorCnt'], paper['au_order']))  # paper['citationCnt'] + 1) *
    elif mode == 'cita':
        for paper in pub_papers:
            impact += paper['citationCnt']
        impact += 1
    elif mode == 'paper':
        impact = len(pub_papers)+1
    elif mode == 'Hindex':
        citationlist = [paper['authorCnt'] for paper in pub_papers]
        impact = Hindex(citationlist)
        impact += 1
    return impact


# 计算每个几点的贡献总值
def node_centrality(graph, pattern, db_col):
    """
    :param graph:
    :param pattern:
                    'cc':closeness_centrality
                    'bc': betweenness_centrality
                    'ec':eigenvector_centrality
                    'ecc': eccentricity
                    'sc': subgraph_centrality
                    'inc': information_centrality
                    'cfbc': current_flow_betweenness_centrality
                    'dc': degree_centrality
                    'cita': cita
                    'am': # A-index
                    'paper': papers
                    'Hindex': H-index
    :param db_col: 数据库
    :return: 中心性元组list  [(node, centrality)]
    """
    if pattern == 'cc':
        node_importance = nx.closeness_centrality(graph)  # 接近中心性
    elif pattern == 'bc':
        node_importance = nx.betweenness_centrality(graph)  # 介数中心性
    elif pattern == 'ec':
        node_importance = nx.eigenvector_centrality(graph)  # 特征向量中心性
    elif pattern == 'ecc':
        node_importance = nx.eccentricity(graph)  # 离心中心性
    elif pattern == 'sc':
        node_importance = nx.subgraph_centrality(graph)  # 子图中心性
    elif pattern == 'inc':
        node_importance = nx.information_centrality(graph)  # 信息指标
    elif pattern == 'cfbc':
        node_importance = nx.current_flow_betweenness_centrality(graph)  # 随机游走中心性
    elif pattern == 'dc':
        node_importance = nx.degree_centrality(graph)  # 度中心性
    elif pattern == 'cita':  # 引用量
        node_importance = [(no, get_impact(graph, no, pattern, db_col)+1) for no in graph.nodes]
    elif pattern == 'am':  # A-index
        node_importance = [(no, get_impact(graph, no, pattern, db_col)) for no in graph.nodes]
    elif pattern == 'paper':  # 发文量
        node_importance = [(no, get_impact(graph, no, pattern, db_col)) for no in graph.nodes]
    elif pattern == 'Hindex':
        node_importance = [(no, get_impact(graph, no, pattern, db_col)) for no in graph.nodes]
    elif pattern == 'compound':
        node_importance = nx.closeness_centrality(graph)
        node_importance = [(node, centrality * get_impact(graph, node, db_col))
                           for (node, centrality) in node_importance.items()]
    if pattern in ['cc', 'bc', 'ec', 'ecc', 'sc', 'inc', 'cfbc', 'dc']:
        node_importance = [(node, centrality)
                           for (node, centrality) in node_importance.items()]
    return sorted(node_importance, key=lambda x: x[1], reverse=True)
