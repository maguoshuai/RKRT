# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 上午10:42
# @Author  : nevermore.huachi
# @File    : detect_core_net.py
# @Software: PyCharm
# 提取核心网络
from pymongo import MongoClient
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from node_centrality import node_centrality
from delete_check import filtering_node
from creat_coauthor_net import GetAllFiledLinks, GetLinksByFields


# 统计初始网络的数据 --- 节点数量、边的数量、网络密度、论文数量、引用量
def statgraph(graph, db_col):
    papers = []
    for i in graph.nodes():
        id = int(graph.nodes[i]['id'])
        pub_papers = db_col.find_one({'id': id})['pub_papers']
        for p in pub_papers:
            papers.append((p['id'], p['citationCnt']))
    papers = list(set(papers))
    return (len(graph.nodes), len(graph.edges), nx.density(graph),
            len(papers), np.sum(np.array([p[1] for p in papers])))


# 统计网络信息
def net_details(graph, db_col):
    papers = []
    for i in graph.nodes():
        id = int(graph.nodes[i]['id'])
        pub_papers = db_col.find_one({'id': id})['pub_papers']
        for p in pub_papers:
            papers.append((p['id'], p['citationCnt']))
    papers = list(set(papers))
    # clique_number = nx.graph_number_of_cliques(graph)#clique_numbernx.average_shortest_path_length(graph),
    return (len(graph.nodes), len(graph.edges), nx.radius(graph),
            len(papers), np.sum(np.array([p[1] for p in papers])))


# 提取科研核心团队成员网络
def extract_net(field, graph, max_iterations, mode, db_col, threshold, check_mode):
    """
    :param field: 领域
    :param graph:
    :param max_iterations:
    :param mode: 节点中心性计算模式
    :param db_col: 数据库
    :param threshold:
    :param check_mode: 删除节点的方法
    :return:
    """
    # 使用影响力进行删除节点，筛除边界节点

    original_graph = graph.copy()
    # 保存每次迭代过程中的网络的信息
    # net_info = []
    # net_info.append(net_details(graph, db_col))
    # print(net_info[0])
    print('初始网络中包含的节点数为:{}'.format(len(graph.nodes)))

    # proportion = []

    for t in range(max_iterations):
        print('--------------------peeling {} --------------------'.format(str(t)))

        impact_list = dict(node_centrality(graph, mode, db_col))
        process_graph = graph
        # 删除节点
        remove_nodes = filtering_node(process_graph, impact_list, threshold, check_mode)
        # 标记删除时间
        for d_node in remove_nodes:
            original_graph.nodes[d_node]['peeledt'] = t

        print('第' + str(t) + '次迭代删除节点的个数为' + str(len(remove_nodes)))

        if not remove_nodes:
            print('未有满足条件的节点，迭代终止。。。。。。。。。。。。。。。。。。。。')
            break

        try:
            process_graph.remove_nodes_from(remove_nodes)
            mgraph = process_graph.subgraph(max(nx.connected_components(process_graph), key=len)).copy()

            # 将不在最大联通子图的节点进行标记
            peeled_nodes = set(graph.nodes).difference(mgraph.nodes)
            print("构成完全图删除的节点数为{}".format(len(peeled_nodes)))
            for pn in peeled_nodes:
                original_graph.nodes[pn]['peeledt'] = t

            print('第{}次迭代网络中保留的节点数为:{}'.format(t, len(graph.nodes)))

            # 统计网络中的论文数量
            # sta_papers = net_details(mgraph, db_col)
            # print(sta_papers)
            # net_info.append(sta_papers)
            graph = mgraph

        except:
            pass
    # print(net_details(graph, db_col))
    # plt.figure()
    # x = range(np.array(proportion).shape[0])
    # plt.plot(x, np.array(proportion), marker='x')
    # plt.xticks(x)
    # plt.savefig('../figure/bpeeling' + mode + '.png')
    # # plt.show()
    # for n in original_graph.nodes:
    #     try:
    #         t = original_graph.nodes[n]['peeledt']
    #     except:
    #         original_graph.nodes[n]['peeledt'] = 100

    savepath = '../core_author_net/' + field + '.gml'
    nx.write_gml(original_graph, savepath)
    return graph


# 对比与评价
# def run_eval(original_graph, max_congra, peeled_graph, db_col):
#     return analysis_network(original_graph, max_congra, peeled_graph, db_col)


def get_core_net_by_field(client):
    # fields = os.listdir('./journalinfo/')
    fields = [
        'information security',
        'Artificial intelligence',
        'computer network',
        'Software engineering',
        'data mining',
        'Computer graphics and multimedia',
        'Human computer interaction',
        'Computer architecture',
        'comprehensive',
        'Theory of Computer Science'
    ]
    db_field_au = client.Fields_author

    for fie in fields:
        print('*****************loading  ' + str(fie) + '  ************************')
        au_col = db_field_au[fie]
        # 建立索引
        # au_col.create_index('id', unique=True)
        co_net_path = '../Cooperation_net/' + fie + '.txt'
        co_net = nx.read_weighted_edgelist(co_net_path, delimiter=',')
        # 原始网络
        # for node in tqdm.tqdm(co_net.nodes):
        #     co_net.nodes[node]['name'] = au_col.find_one({"id": int(node)})['displayName']

        print('----------------------------original networks----------------------------')
        # print(statgraph(co_netn, au_col))
        # 最大联通片 max connected components networks
        print('-------------------- max connected components networks--------------------')
        max_congra = co_net.subgraph(max(nx.connected_components(co_net), key=len)).copy()
        # print(net_details(max_congra, au_col))
        graph = max_congra
        # extract core research network
        peeled_graph = extract_net(fie, max_congra, max_iterations=20, mode='am', db_col=au_col, threshold=0.9,
                                   check_mode=4)
        # print('聯通巨片shell')
        # print(dict2seqlist(dict(Counter(list(nx.core_number(max_congra).values())))))
        # print('peeled net shell')
        # print(dict2seqlist(dict(Counter(list(nx.core_number(peeled_graph).values())))))
        grapic_path = '../core_author_net/' + fie + 'mode_{}_{}hold_{}.gpickle'.format('am', 4, 0.95)
        nx.write_gpickle(peeled_graph, grapic_path)
        # gml_path = '../core_author_net/' + fie + 'mode_{}_{}hold_{}.gml'.format('am', 4, 0.95)
        # nx.write_gml(peeled_graph, gml_path)


def get_core_net(client):
    db_CCF = client.CCFgraph
    au_col = db_CCF.authors

    print('*****************loading  all field  ************************')
    # 建立索引
    # au_col.create_index('id', unique=True)
    co_net_path = '../Cooperation_net/all_field_net.txt'
    co_net = nx.read_weighted_edgelist(co_net_path, delimiter=',')
    # 原始网络
    # for node in tqdm.tqdm(co_net.nodes):
    #     co_net.nodes[node]['name'] = au_col.find_one({"id": int(node)})['displayName']

    print('----------------------------original networks----------------------------')
    # print(statgraph(co_netn, au_col))
    # 最大联通片 max connected components networks
    print('-------------------- max connected components networks --------------------')
    max_congra = co_net.subgraph(max(nx.connected_components(co_net), key=len)).copy()
    # print(net_details(max_congra, au_col))
    graph = max_congra

    # extract core research network
    def extract_allnet(graph, max_iterations, mode, db_col, threshold, check_mode):
        """
        :param field: 领域
        :param graph:
        :param max_iterations:
        :param mode: 节点中心性计算模式
        :param db_col: 数据库
        :param threshold:
        :param check_mode: 删除节点的方法
        :return:
        """
        # 使用影响力进行删除节点，筛除边界节点

        original_graph = graph.copy()
        # 保存每次迭代过程中的网络的信息
        # net_info = []
        # net_info.append(net_details(graph, db_col))
        # print(net_info[0])
        print('初始网络中包含的节点数为:{}'.format(len(graph.nodes)))

        # impact_list = dict(node_centrality(graph, mode, db_col))
        for t in range(max_iterations):
            print('--------------------peeling {} --------------------'.format(str(t)))

            impact_list = dict(node_centrality(graph, mode, db_col))
            process_graph = graph
            # 删除节点
            remove_nodes = filtering_node(process_graph, impact_list, threshold, check_mode)
            # 标记删除时间
            for d_node in remove_nodes:
                original_graph.nodes[d_node]['peeledt'] = t

            print('第' + str(t) + '次迭代删除节点的个数为' + str(len(remove_nodes)))
            if not remove_nodes:
                print('未有满足条件的节点，迭代终止。。。。。。。。。。。。。。。。。。。。')
                break
            try:
                process_graph.remove_nodes_from(remove_nodes)
                mgraph = process_graph.subgraph(max(nx.connected_components(process_graph), key=len)).copy()
                # 将不在最大联通子图的节点进行标记
                peeled_nodes = set(graph.nodes).difference(mgraph.nodes)
                print("构成完全图删除的节点数为{}".format(len(peeled_nodes)))
                for pn in peeled_nodes:
                    original_graph.nodes[pn]['peeledt'] = t
                print('第{}次迭代网络中保留的节点数为:{}'.format(t, len(graph.nodes)))
                graph = mgraph

            except:
                pass
        # print(net_details(graph, db_col))
        return graph

    centrality_mode = 'am'
    threshold = 0.9
    check_mode = 4
    print(centrality_mode, threshold, check_mode)
    peeled_graph = extract_allnet(max_congra, max_iterations=20, mode=centrality_mode, db_col=au_col,
                                  threshold=threshold,
                                  check_mode=check_mode)
    grapic_path = '../core_author_net/allfield_mode_{}_{}hold_{}.gml'.format(centrality_mode, check_mode, threshold)
    nx.write_gml(peeled_graph, grapic_path)


if __name__ == '__main__':
    client = MongoClient(host='172.21.201.187', port=27017)
    GetAllFiledLinks(client)  # create coauthor network
    get_core_net(client)
    # GetLinksByFields(client)
    # get_core_net_by_field(client)
