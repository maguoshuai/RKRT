# -*- coding: utf-8 -*-
# @Time    : 2020/7/27 上午10:50
# @Author  : nevermore.huachi
# @File    : evaluating_metrics.py
# @Software: PyCharm
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from node_centrality import node_centrality


def analysis_network(original_graph, max_congra, peeled_graph, db_col):
    """
    :param original_graph:
    :param max_congra:
    :param peeled_graph:
    :param db_col:
    :returns
            nodes
            egdes
            connection   ---> no radius
            density
            papers
            citations
            paper_gini
            cita_gini
            aindex_gini
    """
    # 综合来看
    aindex_list = dict(node_centrality(original_graph, 'am', db_col))
    # clique_number = nx.graph_number_of_cliques(graph)#clique_numbernx.average_shortest_path_length(graph),
    # 原始网络
    total_ori_papers, total_ori_citations, paper_gini, cita_gini, aindex_gini = graph_publications(original_graph,
                                                                                                   db_col, aindex_list)
    original_graph_detail = [len(original_graph.nodes), len(original_graph.edges), nx.is_connected(original_graph),
                             nx.density(original_graph), total_ori_papers, total_ori_citations, paper_gini, cita_gini,
                             aindex_gini]

    print('original net ' + str(original_graph_detail))

    # 最大联通片
    total_maxcongra_papers, total_maxcongra_citations, paper_gini, cita_gini, aindex_gini = graph_publications(
        max_congra, db_col, aindex_list)
    max_congra_detail = [len(max_congra.nodes), len(max_congra.edges), nx.is_connected(max_congra),
                         nx.density(max_congra), total_maxcongra_papers, total_maxcongra_citations, paper_gini,
                         cita_gini, aindex_gini]
    print('max con graph ' + str(max_congra_detail))

    # 删除后的网络
    total_peeledgra_papers, total_peeledgra_citations, paper_gini, cita_gini, aindex_gini = graph_publications(
        peeled_graph, db_col, aindex_list)
    peeled_graph_detail = [len(peeled_graph.nodes), len(peeled_graph.edges), nx.is_connected(peeled_graph),
                           nx.density(peeled_graph), total_peeledgra_papers, total_peeledgra_citations, paper_gini,
                           cita_gini, aindex_gini]
    print('peeled graph ' + str(peeled_graph_detail))

    # # 中心性节点组成的网络
    # # 'cc':接近中心性
    # cc_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'cc', aindex_list)
    # print('cc centrality graph ' + str(cc_cen_graph_detail))
    #
    # #  'bc':介数中心性
    # bc_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'bc', aindex_list)
    # print('bc centrality graph ' + str(bc_cen_graph_detail))
    #
    # #  'ec':特征向量中心性
    # # ec_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'ec', aindex_list)
    # # print('ec centrality graph ' + str(ec_cen_graph_detail))
    #
    # #  'ecc': 离心中心性
    # # ecc_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'ecc', aindex_list)
    # # print('ecc centrality graphh ' + str(ecc_cen_graph_detail))
    #
    # #  'sc': 子图中心性
    # # sc_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'sc', aindex_list)
    # # print('sc centrality graphh ' + str(sc_cen_graph_detail))
    #
    # #  'inc': 信息指标
    # # inc_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'inc', aindex_list)
    # # print('inc centrality graphh ' + str(inc_cen_graph_detail))
    #
    # #  'cfbc': 随机游走中心性
    # # cfbd_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'cfbc', aindex_list)
    # # print('cfbc centrality graphh ' + str(cfbd_cen_graph_detail))
    #
    # #  'dc': 度中心性
    # dc_cen_graph_detail = centrality_graph(original_graph, peeled_graph, db_col, 'dc', aindex_list)
    # print('dc centrality graphh ' + str(dc_cen_graph_detail))
    #
    # #  'cita': 引用量
    # cita_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'cita', aindex_list)
    # print('cita centrality graphh ' + str(cita_cen_graph_detail))
    #
    # #  'am': A-index
    # am_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'am', aindex_list)
    # print('am centrality graphh ' + str(am_cen_graph_detail))
    #
    # #  'paper': 发文量
    # paper_cen_graph_detail = centrality_graph(max_congra, peeled_graph, db_col, 'paper', aindex_list)
    # print('paper centrality graphh ' + str(paper_cen_graph_detail))


def graph_publications(graph, db_col, aindex_list):
    papers = []
    node_papers = []
    node_citations = []
    aindex = []
    for i in graph.nodes():
        aindex.append(aindex_list[i])
        id = int(graph.nodes[i]['id'])
        pub_papers = db_col.find_one({'id': id})['pub_papers']
        node_papers.append(len(pub_papers))
        cita = 0
        for p in pub_papers:
            papers.append((p['id'], p['citationCnt']))
            cita += p['citationCnt']
        node_citations.append(cita)
    papers = list(set(papers))

    # papers = []
    # for i in graph.nodes():
    #     id = int(graph.nodes[i]['id'])
    #     pub_papers = db_col.find_one({'id': id})['pub_papers']
    #     for p in pub_papers:
    #         papers.append((p['id'], p['citationCnt']))
    # papers = list(set(papers))

    # 基尼系数
    paper_gini = gini_coefficient(node_papers)
    cita_gini = gini_coefficient(node_citations)
    aindex_gini = gini_coefficient(aindex)

    # 平均值 方差
    paper_mean = np.mean(node_papers)
    paper_var = np.var(node_citations)

    cita_mean = np.mean(node_citations)
    cita_var = np.var(node_citations)

    aindex_mean = np.mean(aindex)
    aindex_var = np.var(aindex)

    print(paper_mean, paper_var, cita_mean, cita_var, aindex_mean, aindex_var)
    return len(papers), np.sum(np.array([p[1] for p in papers])), paper_gini, cita_gini, aindex_gini


def centrality_graph(original_graph, peeled_graph, db_col, cen_mode, aindex_list):
    """

    :param original_graph:
    :param peeled_graph:
    :param db_col:
    :param cen_mode:
                    'cc':接近中心性
                    'bc':介数中心性
                    'ec':特征向量中心性
                    'ecc': 离心中心性
                    'sc': 子图中心性
                    'inc': 信息指标
                    'cfbc': 随机游走中心性
                    'dc': 度中心性
                    'cita': 引用量
                    'am': A-index
                    'paper': 发文量
    :return:
    """
    print('-------------------------construct net-----------------------------------')
    retain_nodes = node_centrality(original_graph, cen_mode, db_col)[:len(peeled_graph.nodes)]
    retain_nodes = [nodes[0] for nodes in retain_nodes]
    cen_graph = nx.induced_subgraph(original_graph, retain_nodes)
    print('done')

    connection = nx.is_connected(cen_graph)
    # if nx.is_connected(cen_graph):
    #     cen_net_radius = nx.radius(cen_graph)
    # else:
    #     cen_net_radius = 'infinite'

    cen_papers = []
    node_papers = []
    node_citations = []
    aindex = []
    for i in cen_graph.nodes():
        aindex.append(aindex_list[i])
        id = int(cen_graph.nodes[i]['id'])
        pub_papers = db_col.find_one({'id': id})['pub_papers']
        node_papers.append(len(pub_papers))
        cita = 0
        for p in pub_papers:
            cen_papers.append((p['id'], p['citationCnt']))
            cita += p['citationCnt']
        node_citations.append(cita)
    cen_papers = list(set(cen_papers))

    total_cen_papers = len(cen_papers)
    total_cen_citations = np.sum(np.array([p[1] for p in cen_papers]))

    paper_gini = gini_coefficient(node_papers)
    cita_gini = gini_coefficient(node_citations)
    aindex_gini = gini_coefficient(aindex)

    # 平均值/方差
    paper_mean = np.mean(node_papers)
    paper_var = np.var(node_citations)

    cita_mean = np.mean(node_citations)
    cita_var = np.var(node_citations)

    aindex_mean = np.mean(aindex)
    aindex_var = np.var(aindex)

    print(paper_mean, paper_var, cita_mean, cita_var, aindex_mean, aindex_var)
    return [len(cen_graph.nodes), len(cen_graph.edges), connection,
            nx.density(cen_graph), total_cen_papers, total_cen_citations, paper_gini, cita_gini, aindex_gini]


def gini_coefficient(wealths):
    # 计算数组累计值,从 0 开始

    # 一共是100个数字
    #    wealths = [1.5, 2, 3.5, 10, 4.2, 2.1, 1.1, 2.2, 3.1, 5.1, 9.5, 9.7, 1.7, 2.3, 3.8, 1.7, 2.3, 5, 4.7, 2.3, 4.3, 12]
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    # 加上0，再排序，再计算cumsum
    # 取最后一个，也就是原数组的和
    sum_wealths = cum_wealths[-1]
    # 倒数第一个
    # 人数的累积占比
    # 就是每个点都会产生一个横坐标
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)
    # 均衡收入曲线
    # 就是45度曲线
    upper = xarray
    # 收入累积占比
    yarray = cum_wealths / sum_wealths
    # cumsum的占比
    # 绘制基尼系数对应的洛伦兹曲线
    # plt.plot(xarray, yarray)
    # plt.plot(xarray, upper)
    # 上面画的是45度线
    # ax.plot(xarray, yarray)
    # ax.plot(xarray, upper)
    # ax.set_xlabel(u'人数累积占比')
    # ax.set_ylabel(u'收入累积占比')
    # plt.show()
    # 计算曲线下面积的通用方法

    B = np.trapz(yarray, x=xarray)
    # 总面积 0.5
    A = 0.5 - B
    G = A / (A + B)
    return G