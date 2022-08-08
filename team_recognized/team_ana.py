# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 上午11:39
# @Author  : nevermore.huachi
# @File    : team_ana.py
# @Software: PyCharm

import re
from tools.check_authors import check_data, average_muim, get_impact
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import math


class static2roles():
    def __init__(self):
        self.author_id = []
        self.aai = []
        self.ari = []
        self.aari = []
        self.apl1 = []
        self.adc = []
        self.ardc2nei = []
        self.aadc = []


def extract_member_info(team_info, roles_mark, graph, label, db_col):
    print('-----------------------' + label + '-----------------------------')
    # record author id   save
    author_id = []

    author_a_index = []
    author_relative_impact = []
    author_ave_relative_impact = []
    author_proportion_less_1 = []
    author_degree_centrality = []
    author_ave_degree_centrality = []
    author_relative_dc2nei = []

    try:
        pattern = roles_mark + '(.*)\n'
        role_info = re.search(pattern, team_info).group().replace('\n', '').replace(roles_mark.replace('\\', ''),
                                                                                    '').split('\t')
        for author in role_info:
            print(author)
            if author:
                name = author.split(',')[0]
                id_list = author.split(',')[1:]
                if id_list:
                    au_id = []
                    for id in id_list:
                        if id:
                            # author id extend
                            au_id.append(id)

                            id_a_index = get_impact(id, 'am', db_col)

                            id_to_coauthor_impact = check_data(id, graph)

                            id_ave_relative_impact = average_muim(id_to_coauthor_impact)

                            id_proportion_less_1 = np.sum(
                                list(map(lambda x: x <= 1, [m / n for m, n in id_to_coauthor_impact]))) / len(
                                id_to_coauthor_impact)

                            id_degree_centrality = nx.degree(graph, id)

                            # leader to neighbor average dc
                            id_relative_dc2nei = [(id_degree_centrality, nx.degree(graph, nei)) for nei in
                                                  nx.neighbors(graph, id)]

                            id_ave_relative_dc = np.mean([tar_dc / nei_dc for tar_dc, nei_dc in id_relative_dc2nei])

                            author_a_index.append(id_a_index)
                            author_relative_impact.extend(id_to_coauthor_impact)
                            author_ave_relative_impact.append(id_ave_relative_impact)
                            author_proportion_less_1.append(id_proportion_less_1)
                            author_degree_centrality.append(id_degree_centrality)
                            author_relative_dc2nei.extend(id_relative_dc2nei)
                            author_ave_degree_centrality.append(id_ave_relative_dc)

                    author_id.extend(au_id)
    except Exception as e:
        print(e)
    finally:
        return author_id, author_a_index, author_relative_impact, author_ave_relative_impact, author_proportion_less_1, author_degree_centrality, author_relative_dc2nei, author_ave_degree_centrality


def team_member_ana():
    from pymongo import MongoClient
    client = MongoClient()
    db_ccf = client.CCFgraph

    ccf_author_col = db_ccf.authors
    team_info_path = '../team_ground_truth/team_info.txt'
    all_net = nx.read_gml('../ccf_graph/all_field_net.gml')
    # max_connected_net = nx.read_gml('../ccf_graph/max_connected_components.gml')

    # graph = nx.karate_club_graph()
    roles_mark = {
        'leader': '##',
        'member': '%%',
        'assistant': '@@',
        'postdoc': '&&',
        'doctor': '\^\^',
        'master': '\$\$',
        'Alumni': '\+\+',
    }

    with open(team_info_path, 'r') as file:
        team_list = file.read().split('---\n')

        whole_aai = []
        whole_ari = []
        whole_aari = []
        whole_apl1 = []
        whole_adc = []
        whole_ardc2nei = []
        whole_aadc = []

        statistics = {
            'leader': static2roles(),
            'member': static2roles(),
            'assistant': static2roles(),
            'postdoc': static2roles(),
            'doctor': static2roles(),
            'master': static2roles(),
            'Alumni': static2roles(),
        }

        for team_info in team_list:
            print(team_info.split('\n')[0])
            print(team_info.split('\n')[1])

            for role in roles_mark.keys():
                author_id, aai, ari, aari, apl1, adc, ardc2nei, aadc = extract_member_info(
                    team_info, roles_mark[role], graph=all_net, label=role, db_col=ccf_author_col)

                whole_aai.extend(aai)
                whole_ari.extend(ari)
                whole_aari.extend(aari)
                whole_apl1.extend(apl1)
                whole_adc.extend(adc)
                whole_ardc2nei.extend(ardc2nei)
                whole_aadc.extend(aadc)

                statistics[role].author_id.extend(author_id)
                statistics[role].aai.extend(aai)
                statistics[role].ari.extend(ari)
                statistics[role].aari.extend(aari)
                statistics[role].apl1.extend(apl1)
                statistics[role].adc.extend(adc)
                statistics[role].ardc2nei.extend(ardc2nei)
                statistics[role].aadc.extend(aadc)
    # # 分析数据
    # # test_distribution_mi(statistics)
    #
    # statistics # # plot
    plot_scatter(statistics['leader'].ari, 'leader', "Leaders",
                 fig_path='../statis_fig/leader_ari.svg')
    plot_scatter(statistics['member'].ari, 'member', "Members",
                 fig_path='../statis_fig/member_ari.svg')
    plot_scatter(statistics['assistant'].ari, 'assistant', "Assistants",
                 fig_path='../statis_fig/assistant_ari.svg')
    plot_scatter(statistics['postdoc'].ari, 'postdoc', "Postdocs",
                 fig_path='../statis_fig/postdoc_ari.svg')
    plot_scatter(statistics['doctor'].ari, 'doctor', "Doctors",
                 fig_path='../statis_fig/doctor_ari.svg')
    plot_scatter(statistics['master'].ari, 'master', "Masters",
                 fig_path='../statis_fig/master_ari.svg')
    plot_scatter(statistics['Alumni'].ari, 'Alumni', "Alumnis",
                 fig_path='../statis_fig/Alumni_ari.svg')
    # #
    # plot_scatter(statistics['leader'].ardc2nei, 'leaders', "leaders relative degree centrality",
    #              fig_path='../statis_fig/leader_ardc.png')
    # plot_scatter(statistics['member'].ardc2nei, 'members', "members relative degree centrality",
    #              fig_path='../statis_fig/member_ardc.png')
    # plot_scatter(statistics['assistant'].ardc2nei, 'assistants', "assistants relative degree centrality",
    #              fig_path='../statis_fig/assistant_ardc.png')
    # plot_scatter(statistics['postdoc'].ardc2nei, 'postdocs', "postdocs relative degree centrality",
    #              fig_path='../statis_fig/postdoc_ardc.png')
    # plot_scatter(statistics['doctor'].ardc2nei, 'doctors', "doctors relative degree centrality",
    #              fig_path='../statis_fig/doctor_ardc.png')
    # plot_scatter(statistics['master'].ardc2nei, 'master', "master relative degree centrality",
    #              fig_path='../statis_fig/master_ardc.png')
    # plot_scatter(statistics['Alumni'].ardc2nei, 'Alumni', "Alumni relative degree centrality",
    #              fig_path='../statis_fig/Alumni_ardc.png')
    # #
    # # # author a-index impact --> aai
    # print('plot a-index impact')
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(statistics['leader'].aai, bins=100, alpha=0.6, label="leaders")
    # plt.hist(statistics['member'].aai, bins=100, alpha=0.6, label="members")
    # plt.hist(statistics['assistant'].aai, bins=100, alpha=0.6, label="assistants")
    # plt.hist(statistics['postdoc'].aai, bins=100, alpha=0.6, label="postdocs")
    # plt.hist(statistics['doctor'].aai, bins=100, alpha=0.6, label="doctors")
    # plt.hist(statistics['master'].aai, bins=100, alpha=0.6, label="masters")
    # plt.hist(statistics['Alumni'].aai, bins=100, alpha=0.6, label="Alumnis")
    # plt.title("the distribution of author a-index impact")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/apart_roles_distrib_aai.png')
    # plt.close()
    #
    # # author average relative impact --> aari
    # print('plot author relative impact')
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(statistics['leader'].aari, bins=100, alpha=0.6, label="leaders")
    # plt.hist(statistics['member'].aari, bins=100, alpha=0.6, label="members")
    # plt.hist(statistics['assistant'].aari, bins=100, alpha=0.6, label="assistants")
    # plt.hist(statistics['postdoc'].aari, bins=100, alpha=0.6, label="postdocs")
    # plt.hist(statistics['doctor'].aari, bins=100, alpha=0.6, label="doctors")
    # plt.hist(statistics['master'].aari, bins=100, alpha=0.6, label="masters")
    # plt.hist(statistics['Alumni'].aari, bins=100, alpha=0.6, label="Alumnis")
    # plt.title("the distribution of author relative impact")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/apart_roles_distrib_aari.png')
    # plt.close()
    #
    # # author proportion less 1 --> apl1
    # print('plot the distribution of author proportion less 1')
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(statistics['leader'].apl1, bins=100, alpha=0.6, label="leaders")
    # plt.hist(statistics['member'].apl1, bins=100, alpha=0.6, label="members")
    # plt.hist(statistics['assistant'].apl1, bins=100, alpha=0.6, label="assistants")
    # plt.hist(statistics['postdoc'].apl1, bins=100, alpha=0.6, label="postdocs")
    # plt.hist(statistics['doctor'].apl1, bins=100, alpha=0.6, label="doctors")
    # plt.hist(statistics['master'].apl1, bins=100, alpha=0.6, label="masters")
    # plt.hist(statistics['Alumni'].apl1, bins=100, alpha=0.6, label="Alumnis")
    # plt.title("the distribution of author proportion less 1")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/apart_roles_distrib_apl1.png')
    # plt.close()
    #
    # # author degree centrality --> adc
    # print('plot author derge centrality')
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(statistics['leader'].adc, bins=100, alpha=0.6, label="leaders")
    # plt.hist(statistics['member'].adc, bins=100, alpha=0.6, label="members")
    # plt.hist(statistics['assistant'].adc, bins=100, alpha=0.6, label="assistants")
    # plt.hist(statistics['postdoc'].adc, bins=100, alpha=0.6, label="postdocs")
    # plt.hist(statistics['doctor'].adc, bins=100, alpha=0.6, label="doctors")
    # plt.hist(statistics['master'].adc, bins=100, alpha=0.6, label="masters")
    # plt.hist(statistics['Alumni'].adc, bins=100, alpha=0.6, label="Alumnis")
    # plt.title("the distribution of author degree centrality")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/apart_roles_distrib_adc.png')
    # plt.close()
    #
    # # author average relative degree centrality  ardc
    # print("plot author relative degree centrality")
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(statistics['leader'].aadc, bins=100, alpha=0.6, label="leaders")
    # plt.hist(statistics['member'].aadc, bins=100, alpha=0.6, label="members")
    # plt.hist(statistics['assistant'].aadc, bins=100, alpha=0.6, label="assistants")
    # plt.hist(statistics['postdoc'].aadc, bins=100, alpha=0.6, label="postdocs")
    # plt.hist(statistics['doctor'].aadc, bins=100, alpha=0.6, label="doctors")
    # plt.hist(statistics['master'].aadc, bins=100, alpha=0.6, label="masters")
    # plt.hist(statistics['Alumni'].aadc, bins=100, alpha=0.6, label="Alumnis")
    # plt.title("the distribution of author relative degree centrality")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/apart_roles_distrib_ardc.png')
    # plt.close()
    #
    # # plot whole data
    # print("plot whole data")
    # plot_scatter(whole_ari, 'all authors', "all authors A-index relative  impact",
    #              fig_path='../statis_fig/whole_ari.png')
    # plot_scatter(whole_ardc2nei, 'all authors', "all authors A-index relative  impact",
    #              fig_path='../statis_fig/whole_ardc.png')
    #
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(whole_aai, bins=100, alpha=0.6, label="all authors")
    # plt.title("author a-index impact")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/whole_distrib_aai.png')
    # plt.close()
    #
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(whole_aari, bins=100, alpha=0.6, label="all authors")
    # plt.title("author average a-index relative impact")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/whole_distrib_aari.png')
    # plt.close()
    #
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(whole_apl1, bins=100, alpha=0.6, label="all authors")
    # plt.title("the distribution of all author proportion less 1")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/whole_distrib_apl1.png')
    # plt.close()
    #
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(whole_adc, bins=100, alpha=0.6, label="all authors")
    # plt.title("author degree centrality")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/whole_distrib_adc.png')
    # plt.close()
    #
    # plt.figure(figsize=(12, 8), dpi=320)
    # plt.hist(whole_aadc, bins=100, alpha=0.6, label="all authors")
    # plt.title("author average relative degree centrality")
    # plt.legend()
    # # plt.show()
    # plt.savefig('../statis_fig/whole_distrib_aadc.png')
    # plt.close()

    peeled_graph = nx.read_gml('../core_author_net/allfield_mode_am_4hold_0.9.gml')
    for key in statistics.keys():
        print(key, hit_ratio(statistics[key].author_id, peeled_graph), len(statistics[key].author_id))


def test_distribution_mi(statistics):
    weights = [[0.32664756446991405, 0.673352435530086],
               [0.4213857428381079, 0.5786142571618921],
               [0, 1],
               [0.4298245614035088, 0.5701754385964912],
               [0.5169811320754717, 0.4830188679245283],
               [0.55, 0.44999999999999996],
               [0.5112529392005375, 0.4887470607994625]]
    cs = ['red', 'orange']
    fig = plt.figure(figsize=(12, 8), dpi=320)
    for index, role in enumerate(statistics.keys()):
        try:
            less_one = np.sum(list(map(lambda x: x <= 1, [m / n for m, n in statistics[role].ari]))) / len(
                statistics[role].ari)
            weight = [less_one, 1 - less_one]
            print(weight)
            plt.pie(weight, autopct='%3.1f%%',
                    radius=(index + 1) * 0.3,
                    pctdistance=0.6,
                    startangle=90,
                    counterclock=True,
                    colors=cs,
                    # 锲形块边界属性字典
                    wedgeprops={'edgecolor': 'white',
                                'linewidth': 1,
                                'linestyle': '-'
                                },
                    # 锲形块标签文本和数据标注文本的字体属性
                    textprops=dict(color='k',  # 字体颜色
                                   fontsize=8,
                                   family='Arial'
                                   )
                    )
        except:
            pass
    plt.pie(x=[1],
            radius=0.3,
            colors=[fig.get_facecolor()]
            )
    plt.legend(labels=statistics.keys(),
               title='111111111111',
               facecolor=fig.get_facecolor(),  # 图例框的填充颜色
               fontsize=12
               )
    plt.savefig('../statis_fig/whole_distrib_aadc.png')
    plt.close(fig)


def top_author_ana():
    from kernel_author_detected.node_centrality import node_centrality
    from pymongo import MongoClient
    client = MongoClient(host='172.21.201.187', port=27017)
    db_ccf = client.CCFgraph
    ccf_author_col = db_ccf.authors

    graph = nx.read_gml('../Cooperation_net/max_connected_components.gml')
    node_centrality = node_centrality(graph, pattern='dc', db_col=ccf_author_col)

    # top节点的A-index的比值
    # 节点的A-index
    cen_node_muim = []
    for cen_node, centrality in node_centrality[:100]:
        cen_node_muim.extend(check_data(cen_node, graph))

    # 散点图
    plot_scatter(cen_node_muim, 'top authors')

    from matplotlib import pyplot as plt
    import numpy as np
    top_author_mi = [au_impact / neigh_impact for (au_impact, neigh_impact) in cen_node_muim]
    print(np.sum(list(map(lambda x: x <= 1, top_author_mi))) / len(top_author_mi))

    fig, ax = plt.subplots()
    ax.hist(top_author_mi, bins=100, edgecolor="black", alpha=0.7, label="leaders")
    ax.legend()
    plt.show()
    plt.close()

    # 　top 节点与其合作者的中心性的比值
    cen_nei_muim = []
    centrality_dict = dict(node_centrality)
    cen_nei_frac = []
    for cen_node, centrality in node_centrality[:100]:
        for nei in nx.neighbors(graph, cen_node):
            cen_nei_muim.append((centrality, centrality_dict[nei]))
            cen_nei_frac.append(centrality / centrality_dict[nei])

    plot_scatter(cen_nei_muim, 'top authors cen nei frac')

    print(np.sum(list(map(lambda x: x <= 1, cen_nei_frac))) / len(cen_nei_frac))

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(cen_nei_frac, bins=200, edgecolor="black", alpha=0.7, label="cen_nei_frac")
    ax.legend()
    plt.show()
    plt.close()

    # top节点的egonetwork中除了top节点的中心性的比值分布
    edge_frac = []
    for cen_node, centrality in node_centrality[:100]:
        cennode_ego_subgraph = nx.induced_subgraph(graph, nx.neighbors(graph, cen_node))
        for targe, source in cennode_ego_subgraph.edges:
            edge_frac.append(centrality_dict[targe] / centrality_dict[source])

    print(np.sum(list(map(lambda x: x <= 1, edge_frac))) / len(edge_frac))

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(edge_frac, bins=200, edgecolor="black", alpha=0.7, label="edge_frac")
    ax.legend()
    plt.show()
    plt.close()

    # top_author_mi = [au_impact / neigh_impact for (au_impact, neigh_impact) in cen_node_muim]
    # print(np.sum(list(map(lambda x: x <= 1, top_author_mi))) / len(top_author_mi))


def plot_scatter(data, label, title, fig_path):
    x1 = [x for (x, y) in data]
    y1 = [y for (x, y) in data]
    if x1:
        linex = range(math.ceil(max(x1) + 1))
    else:
        linex = range(1)
    font = {'family': 'arial',
            'size': 22
            }
    liney = linex
    plt.figure(figsize=(8, 6))
    plt.plot(linex, liney, "-", label="y=x", linewidth=4)
    plt.scatter(x1, y1, marker='x', color='blue', s=50, label=label)
    #                   记号形状       颜色           点的大小    设置标签
    # plt.scatter(x2, y2, marker='+', color='blue', s=40, label='Second')
    # plt.scatter(x3, y3, marker='o', color='green', s=40, label='Third')
    plt.legend(loc='best', prop={'size': 20})  # 设置 图例所在的位置 使用推荐位置
    plt.xlim(0, )
    plt.ylim(0, )
    plt.tick_params(labelsize=18)
    ax = plt.gca()
    bwith = 3
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.xlabel("author1", fontdict=font)
    plt.ylabel("author2", fontdict=font)
    plt.title(title, fontdict=font)
    # plt.show()
    plt.savefig(fig_path, format='svg')
    plt.close()


def plot_hist(data_list, label, title):
    from matplotlib import pyplot as plt
    plt.hist(data_list, bins=100, alpha=0.7, label=label)
    plt.title(title)
    plt.legend()
    plt.show()


def top_author_muimpact():
    from kernel_author_detected.node_centrality import node_centrality
    from pymongo import MongoClient
    import numpy as np
    import matplotlib.pyplot as plt

    graph = nx.read_gml('../Cooperation_net/max_connected_components.gml')

    client = MongoClient(host='172.21.201.187', port=27017)
    db_ccf = client.CCFgraph

    ccf_author_col = db_ccf.authors
    node_centrality = node_centrality(graph, pattern='dc', db_col=ccf_author_col)
    top_node_centrality = node_centrality[:100]

    centrality_dict = dict(node_centrality)

    # 对于top节点来说,平均的度的对比值
    average_dcfrac = []
    for node, centrality in top_node_centrality:
        average_dcfrac.append(
            np.mean([centrality / centrality_dict[neighbor] for neighbor in nx.neighbors(graph, node)]))

    plt.hist(average_dcfrac, bins=200, edgecolor="black", alpha=0.7)
    plt.title("the distribution for node degree fraction")
    plt.show()

    # top节点,所有合作边的度的对比
    all_dcfrac = []
    for node, centrality in top_node_centrality:
        for neighbor in nx.neighbors(graph, node):
            all_dcfrac.append((centrality, centrality_dict[neighbor]))

    plot_scatter(all_dcfrac, "all degree fraction for top authors",
                 "the all degree fraction for top authors to his/her co-authors")

    # the distribution of degree
    degree_list = nx.degree(graph)
    plt.figure(figsize=(40, 6), dpi=300)
    plt.hist(dict(degree_list).values(), bins=400, edgecolor="black", alpha=0.7)
    plt.title("the distribution for node degree")
    plt.xlim(0, 100)
    plt.show()

    # top节点A-index
    # 对于top节点来说,平均的Aindex的对比值
    average_A_index = []
    for node, centrality in top_node_centrality:
        average_A_index.append(
            np.mean([get_impact(node, 'am', ccf_author_col) / get_impact(neighbor, 'am', ccf_author_col) for neighbor in
                     nx.neighbors(graph, node)]))

    plt.hist(average_A_index, bins=200, edgecolor="black", alpha=0.7)
    plt.title("the distribution for node A-index fraction")
    plt.show()

    # top节点,所有合作边的度的对比
    all_dcfrac = []
    for node, centrality in top_node_centrality:
        for neighbor in nx.neighbors(graph, node):
            all_dcfrac.append((get_impact(node, 'am', ccf_author_col), get_impact(neighbor, 'am', ccf_author_col)))

    plot_scatter(all_dcfrac, "all A-index fraction for top authors",
                 "the all A-index fraction for top authors to his/her co-authors")

    # the distribution of A index
    A_index_list = [get_impact(node, 'am', ccf_author_col) for node in graph.nodes]
    plt.figure(figsize=(40, 6), dpi=300)
    plt.hist(A_index_list, bins=400, edgecolor="black", alpha=0.7)
    plt.title("the distribution for A-index")
    plt.xlim(0, 100)
    plt.show()


def plot_multi_pie():
    weights = [[0.32664756446991405, 0.673352435530086],
               [0.4213857428381079, 0.5786142571618921],
               [0, 1],
               [0.4298245614035088, 0.5701754385964912],
               [0.5169811320754717, 0.4830188679245283],
               [0.55, 0.44999999999999996],
               [0.5112529392005375, 0.4887470607994625]]

    cs = ['red', 'orange']
    fig = plt.figure(figsize=(14, 12), dpi=320)

    wedges1, texts1, autotexts1 = plt.pie(weights[0], autopct='%3.1f%%',
                                          radius=1.6,
                                          pctdistance=0.93,
                                          startangle=90,
                                          counterclock=False,
                                          colors=cs,
                                          # 锲形块边界属性字典
                                          wedgeprops={'edgecolor': 'white',
                                                      'linewidth': 1,
                                                      'linestyle': '-'
                                                      },
                                          # 锲形块标签文本和数据标注文本的字体属性
                                          textprops=dict(color='k'  # 字体颜色
                                                         )
                                          )
    wedges2, texts2, autotexts2 = plt.pie(weights[1], autopct='%3.1f%%',
                                          radius=1.4,
                                          pctdistance=0.93,
                                          startangle=90,
                                          counterclock=False,
                                          colors=cs,
                                          # 锲形块边界属性字典
                                          wedgeprops={'edgecolor': 'white',
                                                      'linewidth': 1,
                                                      'linestyle': '-'
                                                      },
                                          # 锲形块标签文本和数据标注文本的字体属性
                                          textprops=dict(color='k'  # 字体颜色
                                                         )
                                          )
    wedges3, texts3, autotexts3 = plt.pie(weights[2], autopct='%3.1f%%',
                                          radius=1.2,
                                          pctdistance=0.9,
                                          startangle=90,
                                          counterclock=False,
                                          colors=cs,
                                          # 锲形块边界属性字典
                                          wedgeprops={'edgecolor': 'white',
                                                      'linewidth': 1,
                                                      'linestyle': '-'
                                                      },
                                          # 锲形块标签文本和数据标注文本的字体属性
                                          textprops=dict(color='k'  # 字体颜色
                                                         )
                                          )
    wedges4, texts4, autotexts4 = plt.pie(weights[3], autopct='%3.1f%%',
                                          radius=1.0,
                                          pctdistance=0.9,
                                          startangle=90,
                                          counterclock=False,
                                          colors=cs,
                                          # 锲形块边界属性字典
                                          wedgeprops={'edgecolor': 'white',
                                                      'linewidth': 1,
                                                      'linestyle': '-'
                                                      },
                                          # 锲形块标签文本和数据标注文本的字体属性
                                          textprops=dict(color='k'  # 字体颜色
                                                         )
                                          )
    wedges5, texts5, autotexts5 = plt.pie(weights[4], autopct='%3.1f%%',
                                          radius=0.8,
                                          pctdistance=0.88,
                                          startangle=90,
                                          counterclock=False,
                                          colors=cs,
                                          # 锲形块边界属性字典
                                          wedgeprops={'edgecolor': 'white',
                                                      'linewidth': 1,
                                                      'linestyle': '-'
                                                      },
                                          # 锲形块标签文本和数据标注文本的字体属性
                                          textprops=dict(color='k'  # 字体颜色
                                                         )
                                          )
    wedges6, texts6, autotexts6 = plt.pie(weights[5], autopct='%3.1f%%',
                                          radius=0.6,
                                          pctdistance=0.85,
                                          startangle=90,
                                          counterclock=False,
                                          colors=cs,
                                          # 锲形块边界属性字典
                                          wedgeprops={'edgecolor': 'white',
                                                      'linewidth': 1,
                                                      'linestyle': '-'
                                                      },
                                          # 锲形块标签文本和数据标注文本的字体属性
                                          textprops=dict(color='k'  # 字体颜色
                                                         )
                                          )
    wedges7, texts7, autotexts7 = plt.pie(weights[6], autopct='%3.1f%%',
                                          radius=0.4,
                                          pctdistance=0.75,
                                          startangle=90,
                                          counterclock=False,
                                          colors=cs,
                                          # 锲形块边界属性字典
                                          wedgeprops={'edgecolor': 'white',
                                                      'linewidth': 1,
                                                      'linestyle': '-'
                                                      },
                                          # 锲形块标签文本和数据标注文本的字体属性
                                          textprops=dict(color='k'  # 字体颜色
                                                         )
                                          )

    plt.setp(autotexts1, size=12, weight='bold')
    plt.setp(autotexts2, size=12, weight='bold')
    plt.setp(autotexts3, size=12, weight='bold')
    plt.setp(autotexts4, size=12, weight='bold')
    plt.setp(autotexts5, size=12, weight='bold')
    plt.setp(autotexts6, size=12, weight='bold')
    plt.setp(autotexts7, size=12, weight='bold')

    plt.pie(x=[1],
            radius=0.2,
            colors=[fig.get_facecolor()]
            )

    plt.legend(labels=['<=1', '>1'],
               loc='best',
               title='the proportion in all links',
               facecolor=fig.get_facecolor(),  # 图例框的填充颜色
               fontsize=12
               )
    plt.savefig('../statis_fig/multi_pie.svg', dpi=600, format='svg')
    # plt.show()
    # plt.close(fig)


def hit_ratio(author_list, graph):
    num = 0
    for au in author_list:
        if au in graph.nodes:
            num += 1
    return num


def check_team_au():
    leaders = ['1984838606', '2227253382', '2052472465', '2069524442', '2146258202', '2681328136', '38845618',
               '2101365550',
               '2249793398', '2163446446', '329910383', '2104653613', '2106222798', '2036349267', '2055799763',
               '2171310226',
               '2081353546']
    members = ['404565408', '2343632159', '2670944070', '2631790958', '2152507340', '2113246774', '2147185896',
               '1601293841',
               '2174985400', '2136556746', '2146115641', '29005305', '2051098434', '2893607809', '2126101459',
               '2561234346',
               '2122707581', '2682373867', '2667020052', '2152999232', '2893439602', '2932589500', '2683234669',
               '2163255317',
               '2466710729', '305618809', '2573551016', '2146115641', '2897769005', '2134014566', '2133983782',
               '2138640236',
               '2141739595', '2503224341', '2930518073', '2153421060', '2263964151', '2167032303', '2595570217',
               '2687156700',
               '2684724500', '2163368592', '2655439487', '2698120280', '2096543487', '2323386473', '2120169796',
               '2887548897',
               '2630277302', '2639120638', '2070575593', '2777743730', '2168613017', '2182886034', '57898110',
               '2330102650',
               '2196817303', '152887991', '1999508070', '132103621', '2066841051', '303613142', '2899091106',
               '2113477095',
               '2574630407', '2311769716', '2630370053', '2571009248', '2145160160', '1978803655', '1607253913',
               '2924136942',
               '1964156647', '2156010532', '2102478470', '2924935227', '2096095416', '218175561', '2365212314',
               '2119893020',
               '2107917670', '2006188770', '2206719268', '2128450755', '2198070710', '2808777917', '2103579519',
               '2681876608',
               '2010875763', '1971261289', '2100818546', '2046410517', '162253719', '2573645052', '2118029367',
               '2167096806',
               '2161983439', '2776195728', '2149341496', '2096627015', '2048157592', '229175834', '2111990228',
               '2122193271',
               '2683512592', '1988747110', '2096939186', '2642278050', '2129309438', '2636614612', '2303451173',
               '2635065094',
               '2304396750', '2666159266', '2166371674', '2099811583', '2107434590', '2149514797', '2680908996',
               '2225259531',
               '2168024214', '2157708780', '2798650010', '2198983026', '2226543658', '2101141533', '2100886215',
               '416104820',
               '1119718656', '2153496801', '2671665018', '2518530940', '2160150884', '2652217057', '2124091997',
               '2240690536',
               '2231782831', '1836339439', '1811224162', '2001793595']
    assistants = []
    postdocs = ['1975861307', '2026325584', '2228814312', '2924838121', '2095943436', '1128290216', '2161476513',
                '2558271738',
                '2168180808', '2307389109', '2303724852', '2803707736', '2917990815', '2424925839', '2134277328',
                '2219775466',
                '2463967093', '2167158302', '2128187218', '100375071', '2077180693', '2103351368', '2749903649',
                '2749976761',
                '2682727269', '2607834907', '2104945716']
    doctors = ['2147918188', '2345205178', '2118252886', '2141588545', '2128865582', '2425799484', '2752358457',
               '2608282721',
               '1266965569', '2560294467', '2042663857', '2738305129', '2097492187', '2933028805', '2625268406',
               '1983914940',
               '2214898855', '2443078652', '2149913060', '2700287723', '2022203600', '2782446841', '2781179173',
               '2255033872',
               '2205809501', '2596602368', '2398964917', '2097350334', '2485676506', '2694064971', '2667928597',
               '2604345531',
               '2788159515', '1978292022', '2139651645', '2139651645', '2121075360', '2122817130', '2304364831',
               '2104203371',
               '2160828216', '2343557966', '301914378', '2225682492', '2528370491', '917262090', '2763245920',
               '2000086078',
               '2708433550', '337396571', '2908307367', '2535016339', '2806089582', '2811340330', '2735618500']
    masters = ['2116456411', '2102002796', '2634174050', '2144965526', '1988254826', '102740216', '2129908635',
               '2568769829',
               '2133088636', '2580518003', '662858474', '2120864060', '2112232458', '2099118133', '2898141097',
               '2883385364',
               '2607216286', '2537250171', '2683972733', '2799967932', '2790068275', '2645108987', '2678184513',
               '2234747992',
               '2892137050', '2486156316', '2254229294', '2145091067', '2097449386', '2099560555', '2358414896',
               '2103181844',
               '2166225333', '2627438164', '2609556378', '2570704126', '2573643773', '2573643773', '2605989135',
               '2225682492',
               '2528370491', '917262090', '2763245920', '2000086078', '2706131936']
    alumnis = ['2120761920', '46106380', '2137039922', '283042521', '2478973563', '2608828817', '2118640023',
               '1923218030',
               '2108025540', '98894233', '2786135038', '2099301034', '1221626345', '2885856570', '38970831',
               '2628878818',
               '2161208487', '2892558305', '2088536091', '2656625994', '2163499725', '2234452589', '1603383808',
               '2102013811',
               '2607727389', '2303613998', '2276079446', '2681328136', '565878027', '2028266199', '373876587',
               '2528506410',
               '2650987650', '2628878818', '2885856570', '2474966271', '2160732037', '2097273326', '1973247951',
               '1999770807',
               '1354816936', '2762740081', '2798958267', '2014530249', '380014765', '2303718488', '2130665893',
               '2496684464',
               '2168841453', '2190065093', '2307618570', '2171612916', '2599631056', '2650943115', '2892798998',
               '2685828925',
               '2462341530', '2263778494', '2631946017', '2790179500', '2930008496', '2792601924', '2651058430',
               '2634792150',
               '2165482772', '2305543518', '2686615858', '2669810403', '2777588807', '2701453572', '2428770388',
               '2639347120',
               '2111360953', '2696592548', '2700043532', '2765096203', '2108928151', '2663032478', '2429888869',
               '2628571622',
               '2296433447', '2616561934', '2643659814', '2148573281', '2656066957', '2650938711', '2681987992',
               '2708982487',
               '2673596703', '2308764093', '2686615858', '2587431914', '2616561934', '2755397222', '2099118133',
               '2754904664',
               '2292540048', '2920464777', '2599411263', '2673828586', '2883385364', '2031838762', '1966707555',
               '2111218445',
               '2164965353', '1031839550', '2116956451', '2145095518', '380014765', '2303746224', '1944499404',
               '2135058405',
               '2761678767', '2358414896', '2922937925', '2618037599', '2114922421', '2110279758', '2708413883',
               '2488092641',
               '2702051385', '2234260472', '2192416783', '2687995922', '2235956210', '2184982114', '2021404014',
               '2311379535',
               '2043608195', '2092742661', '2601969657', '2917906060', '2590036756', '1843239838', '2093664028',
               '2153400383',
               '2694980694', '2431339895', '2300191317', '2024573934', '2159509338', '2164620433', '2098934872',
               '2147670869',
               '2306357273', '2745745895', '1988090614', '2095763615', '2575753754', '2615592390', '242385703',
               '2300552631',
               '2114746241', '2931302653', '2210743192', '2066208128', '2013206790', '1833090809', '2137432717',
               '2136004617',
               '2013771523', '2147859899', '221544955', '788080121', '2307251823', '2460225895', '2150786432',
               '2305122628',
               '2310236397', '2107761078', '2136593781', '2164499344', '2327759715', '2288795585', '2346787949',
               '2113813718',
               '2047561791', '2670485545', '2202853099', '2146993022', '1972901956', '2083817176', '2309866919',
               '2077766773',
               '2310568174', '2169729263', '2111786531', '2591060587', '2558271738', '2167546878', '2093607815',
               '2100191659',
               '2163028900', '1891944859', '2893140854', '2046107018', '1896847774', '2183833945', '2225734442',
               '2477261816',
               '2134765091', '2103579519', '2440012093', '2222803834', '2230062339', '2005485966', '2182041801',
               '2541203325',
               '2132615128', '2288913131', '2155451605', '2215949004', '297711364', '2147266155', '2102820239',
               '2076547627',
               '2129077364', '2423058546', '2306260897', '2145498347', '2067730168', '2633073089', '2112566403',
               '2302071190',
               '2736981059', '2405812664', '2655571811', '1245689326', '2233371350', '2107856776', '2221020175',
               '2690457040',
               '2642278050', '2277394921', '2303583372', '2636614612', '2635065094', '2701048582', '2097653865',
               '2312122676',
               '2420013409', '2495035196', '2694064971', '2303556036', '2493676941', '2637816835', '2111591513',
               '2629424063',
               '1988747110', '2636614612', '2129309438', '2303583372', '2103280417', '2635065094', '2666159266',
               '2304396750',
               '2212530998', '2226053727', '2616397971', '2145506593', '2128683105', '2691080598', '2172071796',
               '2144337244',
               '2121797826', '2226363402', '2166371674', '2107434590', '2804759263', '2804759263', '2416370509',
               '2708938150',
               '2442767405', '2324914564', '2694992530', '2666159266', '2310999502', '2168716058', '2172071796',
               '2227394013',
               '2302736171', '2805104779', '2550380291', '2120426649', '2614530281', '2485433341', '2283061575',
               '2305718927',
               '2323257001', '2680908996', '2225259531', '2780491112', '2660763088', '2529019787', '2919393546',
               '2634373288',
               '2594717567', '2708882055', '2660734614', '2691080598', '2303325710', '2892625957', '2159186115',
               '2485713002',
               '1422584695', '1787861840', '2126694903', '2151574249', '1982743242', '2150469586', '2150469586',
               '2129779547',
               '2735789512', '2161123335', '2152164708', '1980488834', '2066503697', '31390454', '1996512423',
               '2164968040',
               '2113165487', '2113165487', '2108641532', '2206899879', '1988556028', '2195679574', '2278294079',
               '2237328951',
               '1969377669', '2676701685', '2171689288', '2057065545', '2102100994', '2776420735', '2100143089',
               '2165004325',
               '223720365', '2800424027', '1878631932', '2228563131', '2682927300', '2168649741', '2738814177',
               '2176877039',
               '2112894965', '2220299576', '2160510735', '1834773024', '2246752162', '127401815', '2159823517',
               '1984456616',
               '2312146965', '1764326631', '678912235', '2213385919', '2417229859', '1489048449', '2099243581',
               '2034935985',
               '2109788593', '87460347', '1747408227', '2147247199', '2110385854', '2096651107', '2130754085',
               '2167258035',
               '2150833760', '2649179961', '2275864655', '2150663838', '2588683129', '2101048513', '325046544',
               '2205399161',
               '2150261957', '2127689019', '28143959', '2171668805', '2026751896', '2149435715', '2758949846',
               '2611554460',
               '2758032081', '2140436085', '2114757771', '2096598770', '2647128834', '2608432063', '2149435715',
               '2306754820',
               '2484057079', '2682687117', '80552714', '2892559261', '2628051876', '2124669283', '1966009211',
               '666632189',
               '2570135626', '2047336310', '2918103248', '1999552226', '2687065695', '2150130079', '1729528123',
               '2305656273',
               '2910042736', '2071071684', '1957339657', '1924262089', '2046687935', '2140907756', '2299514913',
               '2620260231',
               '1991606086', '1197965035', '2466749593', '2107736785', '2001727836', '2155642141', '2405433388',
               '70605372',
               '259655449', '2102902193', '2795037202', '2691929280', '2777528128', '2534132368', '2534132368',
               '2802020631',
               '2786175503', '2689289499', '2615792163', '2628813804', '2539067361', '2700361395', '2900474256',
               '2900847785',
               '2601127596', '2436792609', '2489420623', '1951375226', '2105273232', '2916217127', '2139755848',
               '2011793866',
               '2125565904', '2097440269', '2085857916', '2294752828', '2127585911', '2081353546', '2634960625',
               '2198295625',
               '2123919020', '2735618500', '2026534069', '2552041327', '1017840353', '1811224162', '2180628279',
               '2591786134']

    # from team_recognized.node_centrality import node_centrality
    from pymongo import MongoClient
    #
    # graph = nx.read_gml('../ccf_graph/max_connected_components.gml')
    # client = MongoClient(host='172.21.201.187', port=27017)
    client = MongoClient()
    db_ccf = client.CCFgraph
    ccf_author_col = db_ccf.authors

    peeled_graph = nx.read_gml('../core_author_net/peeled_ccf_graph_4.gml')
    peeled_nodes = list(peeled_graph.nodes)
    print(len(peeled_nodes))
    # orginal_graph = nx.read_gml('../ccf_graph/max_connected_components.gml')
    #
    # print('data load sucess')
    # mode = 'paper'
    # max_centrality_node = node_centrality(orginal_graph, mode, db_col=ccf_author_col)
    # top_nodes = [no for no, cen in max_centrality_node[:len(peeled_graph.nodes)]]
    #
    def check_num(role_list, top_nodes):
        num = 0
        for ro in role_list:
            if ro in top_nodes:
                num += 1
        return num

    #
    # print('top nodes:' + mode)
    # print(check_num(leaders, top_nodes))
    # print(check_num(members, top_nodes))
    # print(check_num(postdocs, top_nodes))
    # print(check_num(doctors, top_nodes))
    # print(check_num(masters, top_nodes))
    # print(check_num(alumnis, top_nodes))
    #
    # print('peeled nodes:' + mode)
    # print(check_num(leaders, peeled_nodes))
    # print(check_num(members, peeled_nodes))
    # print(check_num(postdocs, peeled_nodes))
    # print(check_num(doctors, peeled_nodes))
    # print(check_num(masters, peeled_nodes))
    # print(check_num(alumnis, peeled_nodes))

    net = nx.read_weighted_edgelist('./topk_graph.txt', delimiter=',')
    net = net.subgraph(max(nx.connected_components(net), key=len)).copy()
    net_nodes = list(net.nodes)
    print(len(net_nodes))
    print(check_num(leaders, net_nodes))
    print(check_num(members, net_nodes))
    print(check_num(postdocs, net_nodes))
    print(check_num(doctors, net_nodes))
    print(check_num(masters, net_nodes))
    print(check_num(alumnis, net_nodes))

    def stat_num(role_list):
        papers = []
        citations = []
        for au in role_list:
            pub_papers = ccf_author_col.find_one({'id': int(au)})['pub_papers']
            papers.append(len(pub_papers))
            for paper in pub_papers:
                citations.append(paper['citationCnt'])

        return sum(citations) / (len(role_list) * sum(papers))

    print(stat_num(leaders))
    print(stat_num(members))
    print(stat_num(postdocs))
    print(stat_num(doctors))
    print(stat_num(masters))
    print(stat_num(alumnis))


from itertools import combinations

# 0-10 44| 10-20 5 | 20-30 3 | 30 -40 2| 40 -50 0 | 50 -  5
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot


def plot_mempie():
    import pyecharts.options as opts
    from pyecharts.charts import Pie
    from pyecharts.globals import ThemeType

    x_data = ["0-10", "10-20", "20-30", "30-40", ">50"]
    y_data = [44, 5, 3, 2, 5]
    data_pair = [list(z) for z in zip(x_data, y_data)]
    data_pair.sort(key=lambda x: x[1])

    c = (
        # Pie(init_opts=opts.InitOpts(width="400px", height="300px",renderer='svg', theme=ThemeType.VINTAGE))
        Pie(init_opts=opts.InitOpts(width="630px",
                                    height="600px",
                                    bg_color="White",
                                    renderer='svg'
                                    ))
            .add(
            series_name="ratio of members",
            data_pair=data_pair,
            center=["50%", "50%"],
            label_opts=opts.LabelOpts(is_show=True, position="center", font_size=20),
        )
            .set_global_opts(
            # title_opts=opts.TitleOpts(
            #     title="Distributions of members",
            #     pos_left="center",
            #     pos_top="10",
            #     title_textstyle_opts=opts.TextStyleOpts(color="black",font_family='Arial'),
            # ),
            legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical", pos_bottom='10%',
                                        textstyle_opts=opts.TextStyleOpts(color="black", font_family='Arial',
                                                                          font_size=22))
        )
            .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{b}: {c}"
            ),
            label_opts=opts.LabelOpts(color="rgba(255, 255, 255, 0.3)",
                                      font_family='Arial',
                                      font_size=22,
                                      position='insideTop',
                                      formatter="{d}%",
                                      ),
        )
    )
    # from pyecharts import options as opts
    # from pyecharts.charts import Pie
    # from pyecharts.globals import ThemeType
    # from pyecharts.faker import Faker
    # x_data = ["0-10", "10-20", "20-30", "30 -40", "40 -50","40 -50","> 50"]
    # y_data = [44, 5, 3, 2, 5]
    #
    # c = (
    #     Pie(init_opts=opts.InitOpts(width="300px", height="300px", renderer='svg', theme=ThemeType.VINTAGE))
    #         .add(
    #         "",
    #         [
    #             list(z)
    #             for z in zip(
    #             x_data,
    #             y_data,
    #         )
    #         ],
    #         center=["40%", "50%"],
    #     )
    #         .set_global_opts(
    #         title_opts=opts.TitleOpts(title="Pie-Legend 滚动"),
    #         legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
    #     )
    #         .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    # )
    return c


def country_pie():
    import pyecharts.options as opts
    from pyecharts.charts import Pie
    from pyecharts.globals import ThemeType

    x_data = ["0-10", "10-20", "20-30", "30-40", ">50"]
    y_data = [44, 5, 3, 2, 5]
    data_pair = [list(z) for z in zip(x_data, y_data)]
    data_pair.sort(key=lambda x: x[1])

    c = (
        # Pie(init_opts=opts.InitOpts(width="400px", height="300px",renderer='svg', theme=ThemeType.VINTAGE))
        Pie(init_opts=opts.InitOpts(width="630px",
                                    height="600px",
                                    bg_color="Whifte",
                                    renderer='svg'
                                    ))
            .add(
            series_name="ratio of members",
            data_pair=data_pair,
            center=["50%", "50%"],
            label_opts=opts.LabelOpts(is_show=True, position="center", font_size=20),
        )
            .set_global_opts(
            # title_opts=opts.TitleOpts(
            #     title="Distributions of members",
            #     pos_left="center",
            #     pos_top="10",
            #     title_textstyle_opts=opts.TextStyleOpts(color="black",font_family='Arial'),
            # ),
            legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical", pos_bottom='10%',
                                        textstyle_opts=opts.TextStyleOpts(color="black", font_family='Arial',
                                                                          font_size=22))
        )
            .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{b}: {c}"
            ),
            label_opts=opts.LabelOpts(color="rgba(255, 255, 255, 0.3)",
                                      font_family='Arial',
                                      font_size=22,
                                      position='insideTop',
                                      formatter="{d}%",
                                      ),
        )
    )
    # from pyecharts import options as opts
    # from pyecharts.charts import Pie
    # from pyecharts.globals import ThemeType
    # from pyecharts.faker import Faker
    # x_data = ["0-10", "10-20", "20-30", "30 -40", "40 -50","40 -50","> 50"]
    # y_data = [44, 5, 3, 2, 5]
    #
    # c = (
    #     Pie(init_opts=opts.InitOpts(width="300px", height="300px", renderer='svg', theme=ThemeType.VINTAGE))
    #         .add(
    #         "",
    #         [
    #             list(z)
    #             for z in zip(
    #             x_data,
    #             y_data,
    #         )
    #         ],
    #         center=["40%", "50%"],
    #     )
    #         .set_global_opts(
    #         title_opts=opts.TitleOpts(title="Pie-Legend 滚动"),
    #         legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
    #     )
    #         .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    # )
    return c


def plot_doublebars():
    from pyecharts import options as opts
    from pyecharts.charts import Bar
    from pyecharts.faker import Faker
    from pyecharts.commons.utils import JsCode

    weights = [[0.32664756446991405, 0.673352435530086],
               [0.4213857428381079, 0.5786142571618921],
               [0, 1],
               [0.5169811320754717, 0.4830188679245283],
               [0.55, 0.44999999999999996],
               [0.5112529392005375, 0.4887470607994625]]

    data1 = [{"value": we[0], "percent": we[0] / 1} for we in weights]
    data2 = [{"value": we[1], "percent": we[1] / 1} for we in weights]

    c = (
        Bar(init_opts=opts.InitOpts(width="800px",
                                    height="550px",
                                    bg_color="White",
                                    renderer='svg'))
            .add_xaxis(["Leader", "Members", "Assistants", "Doctors", "Masters", "Alimni"],
                       )
            .add_yaxis("<=1", data1, stack="stack1", category_gap="50%",
                       label_opts=opts.LabelOpts(font_size=22))
            .add_yaxis(">1", data2, stack="stack1", category_gap="50%",
                       label_opts=opts.LabelOpts(font_size=22))
            .reversal_axis()
            .set_series_opts(
            label_opts=opts.LabelOpts(
                position="inside",
                formatter=JsCode(
                    "function(x){return Number(x.data.percent * 100).toFixed() + '%';}"
                ),
                font_size=16,
            ),
        ).set_global_opts(
            legend_opts=opts.LegendOpts(pos_bottom=True,
                                        textstyle_opts=opts.TextStyleOpts(color="black", font_family='Arial',
                                                                          font_size=22)),
            xaxis_opts=opts.AxisOpts(
                #                 坐标轴标签的配置
                axislabel_opts=opts.LabelOpts(
                    font_size=20,
                    font_family='Arial',
                ),
            ),
            yaxis_opts=opts.AxisOpts(
                #                 坐标轴标签的配置
                axislabel_opts=opts.LabelOpts(
                    font_size=20,
                    font_family='Arial',
                ),
            ),
        )
    )
    return c


def plot_coverage():
    from pyecharts.charts import Bar
    from pyecharts import options as opts
    from pyecharts.render import make_snapshot
    from snapshot_phantomjs import snapshot

    bar = (
        Bar(init_opts=opts.InitOpts(bg_color="White", renderer='svg'))
            .add_xaxis(["Leader", "Member", "Postdoc", "Doctor", "Master", "Alumni"])
            .add_yaxis("Top A-index", [15, 87, 5, 5, 7, 158], label_opts=opts.LabelOpts(font_size=18))
            .add_yaxis("MDD", [16, 86, 6, 13, 10, 139], label_opts=opts.LabelOpts(font_size=18))
            .add_yaxis("All team members", [17, 140, 27, 55, 45, 464], label_opts=opts.LabelOpts(font_size=18))
            .set_global_opts(legend_opts=opts.LegendOpts(
            textstyle_opts=opts.TextStyleOpts(color="black", font_family='Arial', font_size=22)),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=22, font_family='Arial')),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=22, font_family='Arial')),
        )
    )
    # bar.render("mode4.html")  # 在本地生成静态网页
    make_snapshot(snapshot, bar.render(), "coverage.svg")  # 在本地生成图表截图


def comp_net():
    from pymongo import MongoClient
    import tqdm
    client = MongoClient()
    db_mag = client.CCFgraph
    col_paper = db_mag.papers
    co_aulinks = []
    for paper in tqdm.tqdm(col_paper.find()):
        authors = paper['authors']
        aulist = [a['id'] for a in authors][:2]
        # 提取合作链接
        for li in combinations(aulist, 2):
            # if li not in co_aulinks:
            co_aulinks.append(li)

    co_aulinks = [tuple(li) for li in co_aulinks]

    # 统计合作的次数
    wl = []
    swl = []
    for ul in tqdm.tqdm(co_aulinks):
        if ul not in wl:
            wl.append(ul)
            swl.append((ul[0], ul[1], co_aulinks.count(ul)))
    G = nx.Graph()
    G.add_weighted_edges_from(swl)
    nx.write_weighted_edgelist(G, './topk_graph.txt', delimiter=',')


if __name__ == '__main__':
    # data=[(1,2), (2,4)]
    # plot_scatter(data, 'leader', "Leaders",
    #              fig_path='../statis_fig/leader_ari.svg')
    # team_member_ana()
    # top_author_ana()
    # plot_scatter()
    # top_author_muimpact()
    # plot_multi_pie()
    # 2560294467
    # peeled_graph = nx.read_gml('../ccf_graph/peeled_ccf_graph_4.gml')
    # print('1984838606' in peeled_graph.nodes)
    check_team_au()
    # plot_coverage()
    # from team_recognized.get_all_fieldsnet import peeled_net
    # members_ana()
    # peeled_net(max_iterations=20, impact_mode='am', threshold=0.9, check_mode=1)
    # make_snapshot(snapshot, plot_mempie().render(), "pie.svg", pixel_ratio=20)
    # make_snapshot(snapshot, plot_doublebars().render(), "doublebars.svg", pixel_ratio=20)
    # comp_net()
