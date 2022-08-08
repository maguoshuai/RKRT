from pymongo import MongoClient
import networkx as nx


def find_author_by_name(author_name):
    client = MongoClient(host='172.21.201.187', port=27017)
    db_ccf = client.CCFgraph
    ccf_author_col = db_ccf.authors
    au = ccf_author_col.find({'displayName': author_name})
    for a in au:
        print('-----------------------------------')
        print(a['id'])
        for papers in a['pub_papers']:
            print(papers)


def find_author_by_id(author_id):
    client = MongoClient()
    db_ccf = client.CCFgraph
    au_col = db_ccf.authors
    au = au_col.find_one({'id': author_id})
    for paper in au['pub_papers']:
        print(paper)


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


def Axiomatic(authorCnt, order):
    credit = sum([1 / i for i in range(order, authorCnt + 1)]) / authorCnt
    return credit
