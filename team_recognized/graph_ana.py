# -*- coding: utf-8 -*-
# @Time    : 2021/10/19 下午3:12
# @Author  : nevermore.huachi
# @File    : graph_ana.py
# @Software: PyCharm
# distribution of degree
import matplotlib.pyplot as plt
import os
from math import log
from pymongo import MongoClient
import networkx as nx
import random
from LPALS import LPALS
import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from kernel_author_detected.node_centrality import get_impact

# 连接数据库
client = MongoClient()
db_ccf = client.CCFgraph
ccf_author_col = db_ccf.authors


def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    print(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def calcShannonEnt(data):
    numEntries = len(data)  # 样本数
    labelCounts = {}  # 该数据集每个类别的频数
    for featVec in data:  # 对每一行样本
        currentLabel = featVec
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    print(labelCounts)
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算p(xi)
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def plot_degree(degree, color):
    x = range(len(degree))  # 生成X轴序列，从1到最大度
    y = [z / float(sum(degree)) for z in degree]  # 将频次转化为频率
    plt.figure(figsize=(5.8, 5.2), dpi=150)
    plt.xlabel("Degree", size=14)  # Degree
    plt.ylabel("Frequency", size=14)  # Frequency
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.loglog(x, y, '.', color=color)
    plt.show()


def plot_total_degree():
    net_path = '../Cooperation_net/computer network.txt'
    total_graph = nx.read_weighted_edgelist(net_path, delimiter=',')
    max_gra = total_graph.subgraph(max(nx.connected_components(total_graph), key=len)).copy()

    mg_degree = nx.degree_histogram(max_gra)
    # print(mg_degree)
    # print(calcShannonEnt([d for n, d in nx.degree(max_gra)]))
    # print(gini([d for n, d in nx.degree(max_gra)]))
    x1 = range(len(mg_degree))  # 生成X轴序列，从1到最大度
    y1 = [z / float(sum(mg_degree)) for z in mg_degree]  # 将频次转化为频率
    plt.figure(figsize=(8, 6), dpi=150)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    plt.xlabel("Degree", size=20)  # Degree
    plt.ylabel("Frequency", size=20)  # Frequency
    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.loglog(x1, y1, '.', color='#243542', label="Original network")
    # plt.scatter(x1, y1, alpha=0.5, label="original")

    # peeled_net = nx.read_gml('../ccf_graph/peeled_ccf_graph_4.gml')
    peeled_net = nx.read_gpickle('../peeled_net/computer networkmode_am_4hold_0.95.gpickle')
    pn_degree = nx.degree_histogram(peeled_net)
    # print(calcShannonEnt([d for n, d in nx.degree(peeled_net)]))
    # print(gini([d for n, d in nx.degree(peeled_net)]))
    x2 = range(len(pn_degree))  # 生成X轴序列，从1到最大度
    y2 = [z / float(sum(pn_degree)) for z in pn_degree]  # 将频次转化为频率
    plt.loglog(x2, y2, '.', color='#B32126', label="Core research network")
    # plt.scatter(x2, y2, alpha=0.5, marker='^', label="peeled")
    plt.legend(loc='best', fontsize=16)
    plt.savefig('degree_distribution.svg', format='svg')
    plt.show()


def plt_degree_hist(name, o_graph, p_graph):
    az_degree = nx.degree_histogram(o_graph)
    print(calcShannonEnt([d for n, d in nx.degree(o_graph)]))
    print(gini([d for n, d in nx.degree(o_graph)]))
    x3 = range(len(az_degree))  # 生成X轴序列，从1到最大度
    y3 = [z / float(sum(az_degree)) for z in az_degree]  # 将频次转化为频率
    print(y3)
    plt.figure(figsize=(12, 5), dpi=150)
    plt.xlabel("Degree", size=14)  # Degree
    plt.ylabel("Frequency", size=14)  # Frequency
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.loglog(x3, y3, '.', color='blue', label="original")
    # plt.scatter(x3, y3, alpha=0.5, label="original")

    # pld_zhou_net = nx.read_gml('../author_net/Zhi-Hua Zhoupeeled.gml')
    pz_degree = nx.degree_histogram(p_graph)
    print(calcShannonEnt([d for n, d in nx.degree(p_graph)]))
    print(gini([d for n, d in nx.degree(p_graph)]))
    x4 = range(len(pz_degree))  # 生成X轴序列，从1到最大度
    y4 = [z / float(sum(pz_degree)) for z in pz_degree]  # 将频次转化为频率
    plt.loglog(x4, y4, '.', color='#FF0000', label="peeled")
    # plt.scatter(x4, y4, marker='^', alpha=0.5, label="peeled")
    plt.title(name)
    plt.legend(loc='best')
    plt.savefig('./author_hist_dis/' + name + '.png')
    # plt.show()


def plt_degree_scatter(name, o_graph, p_graph):
    az_degree = nx.degree_histogram(o_graph)
    print(calcShannonEnt([d for n, d in nx.degree(o_graph)]))
    print(gini([d for n, d in nx.degree(o_graph)]))
    x3 = range(len(az_degree))
    y3 = [z / float(sum(az_degree)) for z in az_degree]
    print(y3)
    plt.figure(figsize=(12, 5), dpi=150)
    plt.xlabel("Degree", size=14)  # Degree
    plt.ylabel("Frequency", size=14)  # Frequency
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.scatter(x3, y3, alpha=0.5, label="original")

    pz_degree = nx.degree_histogram(p_graph)
    print(calcShannonEnt([d for n, d in nx.degree(p_graph)]))
    print(gini([d for n, d in nx.degree(p_graph)]))
    x4 = range(len(pz_degree))  # 生成X轴序列，从1到最大度
    y4 = [z / float(sum(pz_degree)) for z in pz_degree]  # 将频次转化为频率
    plt.scatter(x4, y4, marker='^', alpha=0.5, label="peeled")
    plt.title(name)
    plt.legend(loc='best')
    plt.savefig('./author_scatter_dis/' + name + '.png')
    # plt.show()


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


def plt_scatter(name, o_graph, p_graph):
    oa_degree = nx.degree(o_graph)
    x1 = [n for n, d in oa_degree]
    y1 = [d for n, d in oa_degree]

    plt.figure(figsize=(12, 5), dpi=150)
    plt.xlabel("Node", size=14)  # Degree
    plt.ylabel("Degree", size=14)  # Frequency
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)

    pa_degree = nx.degree(p_graph)
    x2 = [n for n, d in pa_degree]
    y2 = [d for n, d in pa_degree]

    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4 ** 2  # 点面积
    # 画散点图
    plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='original')
    plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='peeled', marker='^')
    # plt.plot([0, 9.5], [9.5, 0], linewidth='0.5', color='#000000')
    plt.title(name)
    plt.legend(loc='best')
    plt.savefig('./author_scatter/' + name + '.png')


def plot_author_degree(au_name, coop_net_ori, peeled_net):
    author_net_ori = get_author_net(coop_net_ori, au_name, ccf_author_col)
    author_net_peeled = get_author_net(peeled_net, au_name, ccf_author_col)
    plt_degree_hist(au_name, author_net_ori, author_net_peeled)
    plt_degree_scatter(au_name, author_net_ori, author_net_peeled)
    plt_scatter(au_name, author_net_ori, author_net_peeled)


def plot_team_gini_log(name, gini1, gini2):
    gini1_labelCounts = {}
    for feat in gini1:  # 对每一行样本
        if feat not in gini1_labelCounts.keys(): gini1_labelCounts[feat] = 0
        gini1_labelCounts[feat] += 1
    x1 = list(gini1_labelCounts.keys())
    y1 = [z / float(sum(list(gini1_labelCounts.values()))) for z in list(gini1_labelCounts.values())]

    gini2_labelCounts = {}
    for feat in gini2:  # 对每一行样本
        if feat not in gini2_labelCounts.keys(): gini2_labelCounts[feat] = 0
        gini2_labelCounts[feat] += 1
    x2 = list(gini2_labelCounts.keys())
    y2 = [z / float(sum(list(gini2_labelCounts.values()))) for z in list(gini2_labelCounts.values())]

    plt.figure(figsize=(12, 6), dpi=150)
    plt.xlabel("Gini", size=14)  # Degree
    plt.ylabel("Frequency", size=14)  # Frequency
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)

    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4 ** 2  # 点面积
    # 画散点图
    plt.hist(x1, y1, '.', color=colors1, alpha=0.7, label='original')
    plt.loglog(x2, y2, '.', color=colors2, alpha=0.7, label='core net')
    # plt.plot([0, 9.5], [9.5, 0], linewidth='0.5', color='#000000')
    plt.title(name)
    plt.legend(loc='best')
    plt.show()


def plot_team_gini_hist(name, gini1, gini2):
    plt.figure(figsize=(12, 5), dpi=150)
    plt.xlabel("Gini", size=14)  # Degree
    plt.ylabel("Frequency", size=14)  # Frequency
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)

    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'

    plt.hist(gini1, bins=40, color=colors1, alpha=0.5, label='original')
    plt.hist(gini2, bins=40, color=colors2, alpha=0.5, label='core net')
    # plt.plot([0, 9.5], [9.5, 0], linewidth='0.5', color='#000000')
    plt.title(name)
    plt.legend(loc='best')
    plt.show()


def detected_team(G, file_output, lda_vec, db_col, cen_mode):
    algorithm = LPALS(G, file_output=file_output, lda_vec=lda_vec, db_col=db_col)
    algorithm.detect()
    eq, community = algorithm.output()
    gini_co = []
    for com in community:
        subgraph = nx.induced_subgraph(G, list(com)).copy()
        sub_gini = gini([get_impact(subgraph, n, mode=cen_mode, db_col=db_col) for n in subgraph.nodes()])
        # sub_gini = gini([d for n, d in nx.degree(subgraph)])
        gini_co.append(sub_gini)
    return gini_co


def plot_field_gini():
    total_gini = []
    db_field_au = client.Fields_author
    fields = os.listdir('../journalinfo/')
    fields = [fields[2]]
    fields_label = ['Field 1']
    # fields_label = ['Field 1', 'Field 2', 'Field 3', 'Field 4', 'Field 5', 'Field 6', 'Field 7', 'Field 8', 'Field 9', 'Field 10']
    for index, fie in enumerate(fields):
        au_col = db_field_au[fie]
        print("******************    load {} lda model     ********************".format(fie))
        ldavec_path = '../lda_model/' + fie + 'lda.pickle'
        with open(ldavec_path, 'rb') as fp:
            ldavec = pickle.load(fp)
        print('load lda model successful')
        # 原始网络
        co_net_path = '../Cooperation_net/' + fie + '.txt'
        co_net = nx.read_weighted_edgelist(co_net_path, delimiter=',')
        co_netn = nx.convert_node_labels_to_integers(co_net, first_label=0, ordering="default",
                                                     label_attribute='id')
        max_congra = co_netn.subgraph(max(nx.connected_components(co_netn), key=len)).copy()

        ocommunity_path = '../community_results/coriginal' + fie + '.txt'
        ogini_co = detected_team(max_congra, file_output=ocommunity_path, lda_vec=ldavec, db_col=au_col, cen_mode='cita')
        extend_ogini_co = [(g, 0, index) for g in ogini_co]

        # load graph
        print("..................    load krt {} graph success   .....................".format(fie))
        grapic_path = '../core_author_net/' + fie + '.gpickle'
        krt_G = nx.read_gpickle(grapic_path)
        pcommunity_path = '../community_results/krt' + fie + '.txt'
        pgini_co = detected_team(krt_G, file_output=pcommunity_path, lda_vec=ldavec, db_col=au_col, cen_mode='cita')
        extend_pgini_co = [(g, 1, index) for g in pgini_co]
        print(extend_pgini_co)
        total_gini = total_gini + extend_pgini_co  + extend_ogini_co

    team_ginis = np.array(total_gini)
    tips = pd.DataFrame(team_ginis, columns=['gini', 'type', 'Fields'])
    fig = plt.figure(figsize=(12, 5))
    # violinplot = sns.violinplot(x="Fields", y="gini", hue='type', data=tips, order=fields, split=True,
    #                             linewidth=1, palette=["#F7746A", "#36ACAE"])
    ax = sns.violinplot(x='Fields', y='gini', hue='type', kind='box',
                        data=tips, split=True, linewidth=2, palette=["#FF9933", "#009966"])
    # ax = sns.catplot(x='Fields', y='gini', hue='type', data=tips, kind='box',
    #                     linewidth=2)
    ax.set_xticklabels(fields_label)
    plt.setp(ax.collections, alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    ax.legend(handles=handles[0:], labels=['Original teams', 'Kernel teams'])
    # plt.legend(['Original teams','Kernel teams'], title="", loc='best',frameon=False)
    # plt.xticks(rotation=45)
    plt.tick_params(width=3, labelsize=14)
    plt.xlabel('Fields', fontdict={'family': 'DejaVu Sans', 'size': 16})
    plt.ylabel('gini', fontdict={'family': 'DejaVu Sans', 'size': 16})
    plt.savefig('violinplot_split.pdf')
    plt.savefig('violingraph.svg', format='svg')
    plt.show()


def plot_violin():
    data = [i for i in range(1000)]
    data = [data, data]

    # ----------------------------------(d) 多数据系列的豆状图-----------------------------
    # flatui = ["#F7746A", "#36ACAE", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    # sns.set_palette(sns.hls_palette(2, s = 0.95, l = 0.5, h=0.0417))
    sns.set_context("notebook", font_scale=1.5,
                    rc={'font.size': 12, 'axes.labelsize': 20, 'legend.fontsize': 15,
                        'xtick.labelsize': 15, 'ytick.labelsize': 15})

    fig = plt.figure(figsize=(5, 5.5))
    violinplot = sns.violinplot(data=data[0], inner='box', linewidth=1, split=True, palette=["#F7746A", "#36ACAE"])
    violinplot = sns.violinplot(data=data[1], inner='box', linewidth=1, split=True, palette=["#F7746A", "#36ACAE"])
    plt.show()


def plot_team_graph():
    db_field_au = client.Fields_author
    fields = os.listdir('../journalinfo/')
    for index, fie in enumerate(fields):
        au_col = db_field_au[fie]
        print("******************    load {} lda model     ********************".format(fie))
        ldavec_path = '../lda_model/' + fie + 'lda.pickle'
        with open(ldavec_path, 'rb') as fp:
            ldavec = pickle.load(fp)

        # load graph
        print("..................    load peeled {} graph success   .....................".format(fie))
        grapic_path = '../peeled_net/' + fie + '.gpickle'
        peel_G = nx.read_gpickle(grapic_path)
        pcommunity_path = '../community_results/peeled' + fie + '.txt'
        algorithm = LPALS(peel_G, file_output=pcommunity_path, lda_vec=ldavec, db_col=au_col)
        algorithm.detect()
        eq, community = algorithm.output()
        if not os.path.exists('../community_graph/{}'.format(fie)):
            os.mkdir('../community_graph/{}'.format(fie))
        for index, com in enumerate(community):
            subgraph = nx.induced_subgraph(peel_G, list(com))
            plt.figure(figsize=(12, 12))
            community_path = '../community_graph/{}/{}.png'.format(fie, index)
            pos = nx.spring_layout(subgraph, iterations=200)
            nx.draw_networkx(subgraph, pos, width=3.0, with_labels=False, node_size=1000)
            plt.savefig(community_path)
            plt.close()


if __name__ == '__main__':
    # coop_net_ori = nx.read_gml('../ccf_graph/all_field_net.gml')
    # peeled_net = nx.read_gml('../ccf_graph/peeled_ccf_graph_4.gml')
    # plot_total_degree()
    # for i in range(100):
    #     nodes = list(peeled_net.nodes())
    #     au_id = random.choice(nodes)
    #     try:
    #         au = ccf_author_col.find_one({'id': int(au_id)})
    #         print(au)
    #         au_name = au['displayName']
    #         plot_author_degree(au_name, coop_net_ori, peeled_net)
    #     except:
    #         print('not find id name' + str(au_id))
    plot_field_gini()
    # plot_total_degree()
    # plot_team_graph()
