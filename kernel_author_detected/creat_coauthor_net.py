# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 下午6:01
# @Author  : nevermore.huachi
# @File    : get_all_fieldsnet.py
# @Software: PyCharm

from itertools import combinations
import networkx as nx
import tqdm
from collections import Counter
import os


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


def GetAllFiledLinks(client):
    db_MAG = client.CCFgraph
    nccf_col = db_MAG.CCFpapers
    au_col = db_MAG.authors

    co_aulinks = []
    G = nx.Graph()

    # 获取领域所有论文
    papers = []

    results_papers = nccf_col.find()
    for re in tqdm.tqdm(results_papers):
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
    congra_path = './Cooperation_net/all_field_congra_net.txt'
    nx.write_weighted_edgelist(max_congra, congra_path, delimiter=',')


def GetLinksByFields(client):
    fields = os.listdir('../journalinfo/')
    db_field_au = client.Fields_author

    db_CCF = client.CCFgraph
    paper_col = db_CCF.CCFpapers

    for fie in fields:
        print('storing' + str(fie) + '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        db_au_col = db_field_au[fie]

        jourlistpath = '../journalinfo/' + str(fie) + '/'
        jourlistfile = [jourlistpath + 'A.txt', jourlistpath + 'B.txt', jourlistpath + 'C.txt']
        # 获取领域所有杂志
        journal = []
        for journamef in jourlistfile:
            with open(journamef, 'r') as f:
                for j in f:
                    journal.append(j.replace('\n', ''))
        # print(fie, len(journal))

        papers = []
        for j in journal:
            results_papers = paper_col.find({"venueInfo.displayName": j})
            for re in results_papers:
                re = {key: val for key, val in re.items() if key != '_id'}
                papers.append(re)

        co_aulinks = []
        for paper_info in tqdm.tqdm(papers):
            authors = paper_info['authors']
            aulist = [a['id'] for a in authors]
            # 提取合作链接
            for li in combinations(aulist, 2):
                co_aulinks.append(li)

            # 提取作者信息
            for index, au in enumerate(authors):
                p_info = ext_pa_info(index, paper_info)

                check_au = db_au_col.find_one({'id': au['id']})
                if check_au:
                    # if check_au['pub_papers']:
                    check_au['pub_papers'].append(p_info)
                    db_au_col.delete_one({'id': au['id']})
                    db_au_col.insert_one(check_au)
                else:
                    au_info = {}
                    au_info['id'] = au['id']
                    au_info['displayName'] = au['displayName']
                    au_info['pub_papers'] = []
                    au_info['pub_papers'].append(p_info)
                    # print(au_info)
                    db_au_col.insert_one(au_info)

        print('===============================generating network===============================')
        # 构建合作网络
        # 将每个合作关系都从小到大排序，去除由于大小排序问题产生的合作次数不准确的问题
        co_aulinks = [tuple(sorted(li)) for li in co_aulinks]
        # 统计合作的次数
        from collections import Counter
        swl = [(au1, au2, papers) for (au1, au2), papers in Counter(co_aulinks).items()]
        print('================================saving network=================================')
        G = nx.Graph()
        G.clear()
        G.add_weighted_edges_from(swl)
        nx_path = '../Cooperation_net/' + str(fie) + '.txt'
        nx.write_weighted_edgelist(G, nx_path, delimiter=',')
