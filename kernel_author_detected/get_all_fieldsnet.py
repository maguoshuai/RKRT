# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 下午6:01
# @Author  : nevermore.huachi
# @File    : get_all_fieldsnet.py
# @Software: PyCharm
# 获取所有作者的网络

# all fields co author net


import os
import json
from pymongo import MongoClient
from itertools import combinations
import networkx as nx
import tqdm
from collections import Counter

# 建立索引
# au_col.create_index('id', unique=True)

# 构建全作者联通网络
def getall_links():
    # 创建mongodb 链接
    client = MongoClient()
    db_MAG = client.CCFgraph
    nccf_col = db_MAG.CCFpapers
    au_col = db_MAG.CCFauthors

    # fields = os.listdir('../journalinfo/')

    co_aulinks = []
    G = nx.Graph()

    # 获取领域所有论文
    papers = []

    results_papers = nccf_col.find()
    for re in results_papers:
        re = {key: val for key, val in re.items() if key != '_id'}
        papers.append(re)

    # 　存储领域所有作者以及合作关系
    for paper_info in tqdm.tqdm(papers):
        aulist = [a['id'] for a in paper_info['authors']]
        # 提取合作链接
        for li in combinations(aulist, 2):
            co_aulinks.append(li)

        for au in paper_info['authors']:
            if au['id'] not in G.nodes:
                G.add_node(au['id'], name=au['displayName'])

    print('===============================generating network===============================')
    # create coauthor net 
    co_aulinks = [tuple(sorted(li)) for li in co_aulinks]
    # coauthor times

    swl = [(au1, au2, papers) for (au1, au2), papers in Counter(co_aulinks).items()]
    
    print('================================saving network=================================')
    
    G.add_weighted_edges_from(swl)
    
    # 添加节点的名字
    for no in tqdm.tqdm(G.nodes):
        G.nodes[no]['name'] = au_col.find_one({'id': no})['displayName']
    
    nx_path = './Cooperation_net/all_field_net.txt'
    nx.write_weighted_edgelist(G, nx_path, delimiter=',')
    
    # 最大联通子图
    max_congra = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    congra_path = '../Cooperation_net/all_field_congra_net.txt'
    nx.write_weighted_edgelist(max_congra, congra_path, delimiter=',')
    
    savepath = '../peeled_net/allfield.gml'
    nx.write_gml(max_congra, savepath)




def ext_pa_info(au_order, paper_detaril):
    p_info = {}
    p_info['id'] = paper_detaril['id']
    p_info['displayName'] = paper_detaril['displayName']
    p_info['description'] = paper_detaril['description']
    p_info['citationCnt'] = paper_detaril['citationCnt']
    p_info['au_order'] = au_order + 1
    p_info['authorCnt'] = len(paper_detaril['authors'])
    p_info['estimatedCitationCnt'] = paper_detaril['estimatedCitationCnt']
    p_info['publishedDate'] = paper_detaril['venueInfo']['publishedDate']
    return p_info


# 转存数据库 +  生成个新的网络
def combine_db():
    # 连接MAG旧数据库
    client = MongoClient()
    db_MAG = client.MAGgraph
    paper_col = db_MAG.NCCFpapers
    # 新建CCF数据库
    db_ccf = client.CCFgraph
    ccf_author_col = db_ccf.iauthors
    # ccf_paper_col = db_ccf.papers

    # 合作关系
    # co_aulinks = []
    papers_detail = paper_col.find(no_cursor_timeout=True).batch_size(1000)
    for paper_info in tqdm.tqdm(papers_detail):
        # 将paper数据转存
        paper_info = {key: val for key, val in paper_info.items() if key != '_id'}
        # ccf_paper_col.insert_one(paper_info)

        authors = paper_info['authors']
        aulist = [a['id'] for a in authors]
        # 提取合作链接
        # for li in combinations(aulist, 2):
        #     co_aulinks.append(li)
        # 提取作者信息
        for index, au in enumerate(authors):
            p_info = ext_pa_info(index, paper_info)
            check_au = ccf_author_col.find_one({'id': au['id']})
            if check_au:
                # if check_au['pub_papers']:
                check_au['pub_papers'].append(p_info)
                ccf_author_col.delete_one({'id': au['id']})
                ccf_author_col.insert_one(check_au)
            else:
                au_info = {}
                au_info['id'] = au['id']
                au_info['displayName'] = au['displayName']
                au_info['institutions'] = au['institutions']
                au_info['pub_papers'] = []
                au_info['pub_papers'].append(p_info)
                # print(au_info)
                ccf_author_col.insert_one(au_info)

    # print('===============================generating network===============================')
    # # 构建合作网络
    # # 将每个合作关系都从小到大排序，去除由于大小排序问题产生的合作次数不准确的问题
    # co_aulinks = [tuple(sorted(li)) for li in co_aulinks]
    # #
    # # 统计合作的次数Z
    # from collections import Counter
    # swl = [(au1, au2, papers) for (au1, au2), papers in Counter(co_aulinks).items()]
    #
    # print('================================saving network=================================')
    # G = nx.Graph()
    # G.add_weighted_edges_from(swl)
    #
    # # 添加节点的名字
    # # for no in tqdm.tqdm(G.nodes):
    # #     G.nodes[no]['name'] = au_col.find_one({'id': no})['displayName']
    # #     # print(nccau_col.find_one({"id": no})['displayName'])
    #
    # nx_path = '../Cooperation_net/all_field_net.txt'
    # nx.write_weighted_edgelist(G, nx_path, delimiter=',')


def check_data(author_id, graph):
    client = MongoClient()
    db_ccf = client.CCFgraph

    ccf_author_col = db_ccf.authors
    # 建立作者库索引
    # ccf_author_col.ensure_index("id", unique=True)

    # ccf_paper_col = db_ccf.papers
    # 建立论文库索引
    # ccf_paper_col.ensure_index("id", unique=True)

    # author_list = ['2582651959', '2581475345', '2286237009']
    # for id in author_list:
    #     print(ccf_author_col.find_one({'': id}))

    # checking for author name
    # au_name = "Christos Faloutsos"
    # for au in ccf_author_col.find({'displayName': au_name}):
    #     print(au['id'])
    #     for paper in au["pub_papers"]:
    #         print(ccf_paper_col.find_one({"id": paper["id"]})["displayName"])

    # checking for author id
    # 判断节点是否被删除
    # max_connected_net = nx.read_gml('../ccf_graph/max_connected_components.gml')
    # max_connected_net = nx.read_gml('../ccf_graph/peeled_ccf_graph_4.gml')

    # author_id = '2286237009'
    au_impact = get_impact(author_id, mode='am', db_col=ccf_author_col)
    au_local_impact = sum([get_impact(neighbor, mode='am', db_col=ccf_author_col) for neighbor in
                           nx.neighbors(graph, author_id)])
    au_munal_impact = au_impact / au_local_impact

    # 計算局部影響力的方法
    mode = 1

    if mode == 1:
        Tnei_impact_list = [get_impact(nei, mode='am', db_col=ccf_author_col) / sum(
            [get_impact(nei_nei, mode='am', db_col=ccf_author_col) for nei_nei in
             nx.neighbors(graph, nei)]) for nei in nx.neighbors(graph, author_id)]
        checking = [(au_munal_impact, nei) for nei in Tnei_impact_list]

    elif mode == 2:
        checking = [((au_impact / (au_local_impact + au_impact)),
                     (get_impact(nei, mode='am', db_col=ccf_author_col) / (
                             get_impact(nei, mode='am', db_col=ccf_author_col) + sum(
                         [get_impact(nei_nei, mode='am', db_col=ccf_author_col)
                          for nei_nei in nx.neighbors(graph, nei)])) for nei in
                      nx.neighbors(graph, author_id)))]

    elif mode == 4:

        node_neighbor = [n for n in nx.neighbors(graph, author_id)]
        checking = []
        for neigh in node_neighbor:
            neigh_neighbor = [n for n in nx.neighbors(graph, neigh)]

            nei_factor = len(
                list(set(neigh_neighbor).intersection(set(nx.ego_graph(graph, author_id, 1).nodes)))) / len(
                neigh_neighbor)
            node_factor = len(
                list(set(node_neighbor).intersection(set(nx.ego_graph(graph, neigh, 1).nodes)))) / len(
                node_neighbor)

            neigh_impact = nei_factor * get_impact(neigh, mode='am', db_col=ccf_author_col) / sum(
                [get_impact(neighbor, mode='am', db_col=ccf_author_col) for neighbor in
                 nx.neighbors(graph, author_id)])

            au_impact = node_factor * get_impact(author_id, mode='am', db_col=ccf_author_col) / sum(
                [get_impact(neighbor, mode='am', db_col=ccf_author_col) for neighbor in
                 nx.neighbors(graph, neigh)])
            checking.append((au_impact, neigh_impact))

    return checking


def average_muim(checking):
    return sum([au_impact / neigh_impact for (au_impact, neigh_impact) in checking]) / len(checking)


def supply_name():
    id_net_path = '../Cooperation_net/all_field_net.txt'
    id_graph = nx.read_weighted_edgelist(id_net_path, delimiter=',')
    print(len(id_graph.nodes), len(id_graph.edges))
    client = MongoClient()
    db_ccf = client.CCFgraph
    ccf_author_col = db_ccf.authors

    # 测试数据库查找name
    # nn = 2654498356
    # print(type(nn))
    # name = ccf_author_col.find_one({"id": nn})['displayName']
    # print(name)

    # 添加节点的名字
    for node in tqdm.tqdm(id_graph.nodes):
        id_graph.nodes[node]['name'] = ccf_author_col.find_one({"id": int(node)})['displayName']

    nx_path = '../ccf_graph/all_field_net.txt'
    nx.write_weighted_edgelist(id_graph, nx_path, delimiter=',')

    all_field_path = '../ccf_graph/all_field_net.gml'
    nx.write_gml(id_graph, all_field_path)

    # 最大联通子图
    max_congra = id_graph.subgraph(max(nx.connected_components(id_graph), key=len)).copy()
    congra_path = '../ccf_graph/max_connected_components.txt'
    nx.write_weighted_edgelist(max_congra, congra_path, delimiter=',')

    savepath = '../ccf_graph/max_connected_components.gml'
    nx.write_gml(max_congra, savepath)


def get_author_net(graph, author, db_col):
    ego_nets_nodes = []
    for au in db_col.find({'displayName': author}):
        try:
            ego_nodes = nx.ego_graph(graph, str(au["id"]), radius=2).nodes
            ego_nets_nodes.extend(ego_nodes)
        except:
            print('cant find ' + str(au["id"]))

    author_net = nx.induced_subgraph(graph, list(set(ego_nets_nodes)))
    return author_net


# def extract_author_ego_by_name(au_name):
#     # 连接数据库
#     client = MongoClient()
#     db_ccf = client.CCFgraph
#     ccf_author_col = db_ccf.authors

#     # load net
#     coop_net_ori = nx.read_gml('../ccf_graph/all_field_net.gml')
#     author_net_ori = get_author_net(coop_net_ori, au_name, ccf_author_col)
#     savepath = '../author_net/' + au_name + '.gml'
#     nx.write_gml(author_net_ori, savepath)

#     # load KRT_net
#     peeled_net = nx.read_gml('../ccf_graph/peeled_ccf_graph_1.gml')
#     author_net_peeled = get_author_net(peeled_net, au_name, ccf_author_col)
#     peeled_author_net_path = '../author_net/' + au_name + 'peeled.gml'
#     nx.write_gml(author_net_peeled, peeled_author_net_path)


# computer A-index
def Axiomatic(authorCnt, order):
    credit = sum([1 / i for i in range(order, authorCnt + 1)]) / authorCnt
    return credit


# get author impact
def get_impact(node, mode, db_col):
    pub_papers = db_col.find_one({'id': int(node)})['pub_papers']
    impact = 0
    if mode == 'am':
        for paper in pub_papers:
            impact += (Axiomatic(paper['authorCnt'], paper['au_order']))  # paper['citationCnt'] + 1) *
    elif mode == 'cita':
        for paper in pub_papers:
            impact += paper['citationCnt']
    elif mode == 'paper':
        impact = len(pub_papers)
    return impact


def node_centrality(graph, pattern, db_col):
    node_importance = [(no, get_impact(no, pattern, db_col)) for no in graph.nodes]
    return sorted(node_importance, key=lambda x: x[1], reverse=True)


from team_recognized.peel_check import filtering_node


# extract krt net
def extract_net(graph, max_iterations, impact_mode, db_col, threshold, check_mode):
    """
    :param graph:
    :param max_iterations:
    :param mode: node_centrality mode
    :param db_col: 
    :param threshold:
    :param check_mode: remove mode
    :return:
    """
    original_graph = graph.copy()
    # net_info = []
    # net_info.append(net_details(graph, db_col))
    # print(net_info[0])
    print('node num in original net:{}'.format(len(graph.nodes)))

    # proportion = []

    impact_list = dict(node_centrality(graph, impact_mode, db_col))

    for t in range(max_iterations):
        print('--------------------peeling {} --------------------'.format(str(t)))

        process_graph = graph

        # 删除节点
        remove_nodes = filtering_node(process_graph, impact_list, threshold, check_mode)
        # 标记删除时间
        for d_node in remove_nodes:
            original_graph.nodes[d_node]['peeledt'] = t

        print('No.' + str(t) + 'remove node num is' + str(len(remove_nodes)))

        if not remove_nodes:
            print('ending.......................')
            break
        try:
            process_graph.remove_nodes_from(remove_nodes)
            mgraph = process_graph.subgraph(max(nx.connected_components(process_graph), key=len)).copy()

            # 将不在最大联通子图的节点进行标记
            peeled_nodes = set(graph.nodes).difference(mgraph.nodes)
            print("the node num in max congra net is {}".format(len(peeled_nodes)))
            for pn in peeled_nodes:
                original_graph.nodes[pn]['peeledt'] = t
            print('No. {} retained node :{}'.format(t, len(graph.nodes)))
            # 统计网络中的论文数量
            # sta_papers = net_details(mgraph, db_col)
            # print(sta_papers)
            # net_info.append(sta_papers)
            graph = mgraph
        except:
            pass

    savepath = '../ccf_graph/peeled_ccf_graph_im{}_ch{}.gml'.format(impact_mode, check_mode)
    nx.write_gml(graph, savepath)
    return graph


def peeled_net(max_iterations=20, impact_mode='am', threshold=0.9, check_mode=1):
    print('******************* Connect to database ***************************')
    client = MongoClient(host='172.21.201.187', port=27017)
    db_ccf = client.CCFgraph
    ccf_author_col = db_ccf.authors

    print('***************** Loading  all author net  ************************')
    max_connected_net = nx.read_gml('../ccf_graph/max_connected_components.gml')
    # max_connected_net = nx.convert_node_labels_to_integers(max_connected_net, first_label=0, ordering="default",
    #                                                        label_attribute='id')
    peeled_graph = extract_net(max_connected_net, max_iterations=max_iterations, impact_mode=impact_mode,
                               db_col=ccf_author_col, threshold=threshold, check_mode=check_mode)


def record_name():
    client = MongoClient()
    db_ccf = client.CCFgraph
    ccf_author_col = db_ccf.iauthors
    ccf_author_col.create_index('id', unique=True)
    with open('./Cooperation_net/author_name.txt', 'a') as file:
        for au in tqdm.tqdm(ccf_author_col.find(no_cursor_timeout=True).batch_size(1000)):
            try:
                file.write(str(au['id']) + ',' + au['displayName'] + ',' + au['institutions'][0]['displayName'] + '\n')
            except:
                file.write(str(au['id']) + ',' + au['displayName'] + '\n')


def get_faction():
    client = MongoClient()
    db_ccf = client.CCFgraph
    ccf_author_col = db_ccf.authors

    author1 = '2642061000'
    author2 = '2286237009'
    mode = 1
    graph = nx.read_gml('../ccf_graph/max_connected_components.gml')

    impact_a1 = get_impact(author1, mode='am', db_col=ccf_author_col)
    impact_a2 = get_impact(author2, mode='am', db_col=ccf_author_col)
    local_impact_a1 = sum(
        [get_impact(nei_au, mode='am', db_col=ccf_author_col) for nei_au in nx.neighbors(graph, author1)])
    local_impact_a2 = sum(
        [get_impact(nei_au, mode='am', db_col=ccf_author_col) for nei_au in nx.neighbors(graph, author2)])

    if mode == 1:
        checking = (impact_a1 / local_impact_a1) / (impact_a2 / local_impact_a2)

    elif mode == 2:
        checking = (impact_a1 / (impact_a2 + local_impact_a2)) / (impact_a2 / (impact_a1 + local_impact_a1))

    elif mode == 3:
        pass

    elif mode == 4:

        neighbor_a1 = [n for n in nx.neighbors(graph, author1)]
        neighbor_a2 = [n for n in nx.neighbors(graph, author2)]

        a1_factor = len(list(set(neighbor_a1).intersection(set(nx.ego_graph(graph, author2, 1).nodes)))) / len(
            neighbor_a1)
        a2_factor = len(list(set(neighbor_a2).intersection(set(nx.ego_graph(graph, author1, 1).nodes)))) / len(
            neighbor_a2)
        a1_impact = a1_factor * impact_a1 / local_impact_a2
        a2_impact = a2_factor * impact_a2 / local_impact_a1
        checking = a1_impact / a2_impact
    print(checking)


if __name__ == '__main__':
    getall_links()
    # ana_allaunet()
    # find_author(author_name='Olaf Hellwich')  # 检查某一作者的信息
    # search_author(author_id=329910383)
    # combine_db() # 重新合并数据库all
    # 周志华 2582651959 2581475345 2286237009
    # check_data()
    # supply_name() # 在原始的网络中给每个节点添加作者的名字
    # extract_author_ego(au_name='Fei-Fei Li')  # 提取某一特定作者的合作者ego net
    peeled_net(max_iterations=20, impact_mode='am', threshold=0.9, check_mode=1)  # peeling net
    # record_name()
    # get_faction() # 獲得兩個作者局部影響力的比值
