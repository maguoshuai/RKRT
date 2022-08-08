from pymongo import MongoClient
import networkx as nx
import tqdm
import os
import pickle
import numpy as np
from team_recognized import generator
from kernel_author_detected.node_centrality import get_impact
from extendQ import ExtendQ
from LDA_model import lda_vector


class LPALS(object):

    def __init__(self, graph=None, network_filename=None, file_output=None, lda_vec=None, db_col=None):
        """
        Constructor
        :@param network_filename: the networkx filename
        :@param epsilon: the tolerance required in order to merge communities
        :@param min_community_size:min nodes needed to form a community
        :@param file_output: True/False
        """
        if graph is None:
            self.g = nx.Graph()
            if network_filename is not None:
                self.__read_graph(network_filename)
            else:
                raise ImportError
        else:
            self.g = graph
        self.db = db_col
        self.file_output = file_output
        self.lda_vector = lda_vec
        self.load_author_pubs()

    def load_author_pubs(self):
        for i in self.g.nodes():
            id = int(self.g.nodes[i]['id'])
            pub_papers = self.db.find_one({'id': id})['pub_papers']
            self.g.nodes[i]['pub_papers'] = len(pub_papers)
            self.g.nodes[i]['aindex'] = get_impact(self.g, i, mode='am', db_col=self.db)
        return self.g

    def __read_graph(self, network_filename):
        """
        Read .ncol network file
        :param network_filename: complete path for the .ncol file
        :return: an undirected network
        """
        self.g = nx.read_edgelist(network_filename, nodetype=int)

    def detect(self):
        self.compute_NI()
        self.compute_sim()
        self.compute_NNI()

        nodes = list(self.g.nodes)
        v_queue = []
        for node in nodes:
            v_queue.append((node, self.g.nodes[node]['NI']))
            self.g.nodes[node]['L'] = {node: 1}
            self.g.nodes[node]['dominant'] = 1
            self.g.nodes[node]['label'] = node
        v_queue = sorted(v_queue, key=lambda v: v[1])
        nodes = [v[0] for v in v_queue]

        # 定义最大迭代次数
        T = 10
        t = 0
        while t < T:
            change = False
            for node in tqdm.tqdm(nodes, ncols=35, bar_format='Exec: {l_bar}{bar}'):
                L_Ng = {}
                # 计算邻居们的标签和权重
                for neighbor in list(self.g.adj[node].keys()):
                    c, b = self.g.nodes[neighbor]['label'], self.g.nodes[neighbor]['dominant'] * \
                           self.g.nodes[node]['NNI'][neighbor]

                    # 修改更新label的方法为 影响力的相对大小大于一定程度之后进行更新

                    if L_Ng.get(c, -1) == -1:
                        L_Ng[c] = b
                    else:
                        L_Ng[c] += b

                # 除去权重过小的标签
                avg = sum(L_Ng.values()) / len(L_Ng)
                max_dominant = 0
                label = -1
                self.g.nodes[node]['L'] = {}
                for c in L_Ng.keys():
                    if L_Ng[c] >= avg:
                        self.g.nodes[node]['L'][c] = L_Ng[c]
                        if L_Ng[c] > max_dominant:
                            max_dominant = L_Ng[c]
                            label = c
                sum_dominant = sum(self.g.nodes[node]['L'].values())

                for c in self.g.nodes[node]['L'].keys():
                    self.g.nodes[node]['L'][c] /= sum_dominant

                if not self.g.nodes[node]['label'] == label:
                    self.g.nodes[node]['label'] = label
                    change = True
                self.g.nodes[node]['dominant'] = max_dominant / sum_dominant
            if not change:
                break
            t += 1

    def output(self):
        try:
            os.remove(self.file_output)
        except:
            pass

        gen = generator.GraphGenerator(0.1, self.g)
        edge_num = sum([self.g[u][v]['weight'] for (u, v) in self.g.edges])
        node_k = {}
        for n in self.g.nodes:
            node_k[n] = sum([self.g[n][nn]['weight'] for nn in nx.neighbors(self.g, n)])
        communities, node_coms_num = gen.get_Overlapping_communities()
        # node_coms_num = [(node, coms_num) for (node, coms_num) in zip(node_coms_num.keys(), node_coms_num.values())]
        # adj = np.array(nx.adjacency_matrix(self.g).todense())
        EQ = ExtendQ(self.g, communities, edge_num, node_k, node_coms_num)
        print(len(list(gen.get_Overlapping_nodes())))
        communities = list(communities.values())
        with open(self.file_output, 'a') as f:
            for community in communities:
                for u in community:
                    f.write(str(self.get_author_id(u))+ ' ')
                f.write('\n')
        return EQ, communities

    def compute_NI(self):
        """
        计算网络中每个节点的ni，将结果存储在节点的NI属性中
        :param g: 网络
        :return:None
        """
        for node in self.g.nodes:
            # self.g.nodes[node]['NI'] = 0.5 + 0.5 * (self.g.nodes[node]['NI'] - min_ni) / (max_ni - min_ni)
            self.g.nodes[node]['NI'] = self.g.nodes[node]['aindex']

    def compute_sim(self):
        """
        计算网络中节点u,v的相似度，其中(u,v)∈E
        根据[1]文的考量，将α设置为3.

        (u,v)∈E是因为在LPA算法中只有相邻节点才会互相传播标签
        不存在边的那些节点对计算了也没用
        :param g:网络图
        :return:None
        """
        for edge in self.g.edges:
            u, v = edge[0], edge[1]
            for i in (u, v):
                if self.g.nodes[i].get('sim', -1) == -1:
                    self.g.nodes[i]['sim'] = {}
            sim = self.cos_sim(u, v)
            self.g.nodes[u]['sim'][v] = sim
            self.g.nodes[v]['sim'][u] = sim

    def compute_NNI(self):
        """
        计算节点v对节点u的影响力，其中(v,u)∈E
        u节点的属性'NNI'是一个字典，其中包含所有与u相邻的节点V
        对于每一个v∈V，g.nodes[u]['NNI'][v]表示v给u造成的影响力

        :param g:网络图
        :return:None
        """
        for u in self.g.nodes:
            self.g.nodes[u]['NNI'] = {}
            u_nei = nx.neighbors(self.g, u)
            for v in list(self.g.adj[u].keys()):
                v_nei = nx.neighbors(self.g, v)

                u_nei_nodes, v_nei_nodes = set(u_nei), set(v_nei)
                common_uv = u_nei_nodes.intersection(v_nei_nodes)
                u_diff_v = u_nei_nodes.difference(v_nei_nodes)
                if len(common_uv) == 0:
                    self.g.nodes[u]['NNI'][v] = self.g.nodes[u]['sim'][v] / sum(
                        [self.g.nodes[u]['sim'][nei] for nei in list(self.g.adj[u].keys())])
                elif len(u_diff_v) == 0:
                    self.g.nodes[u]['NNI'][v] = sum(
                        [self.g.nodes[u]['sim'][nei] for nei in list(self.g.adj[u].keys())]) / sum(
                        [self.g.nodes[v]['sim'][nei] for nei in list(self.g.adj[v].keys())])
                elif len(common_uv) and len(u_diff_v):
                    com_weight = sum([self.g.nodes[u]['sim'][cn] for cn in common_uv])
                    com_weight = com_weight + self.g.nodes[u]['sim'][v]
                    # diff_weight = sum([self.g.nodes[u]['sim'][dn] for dn in u_diff_v])
                    total_weight = sum([self.g.nodes[u]['sim'][nei] for nei in list(self.g.adj[u].keys())])
                    self.g.nodes[u]['NNI'][v] = com_weight / total_weight

        # for u in g.nodes:
        #     g.nodes[u]['NNI'] = {}
        #     sim_max = max(g.nodes[u]['sim'].values())
        #     for v in list(g.adj[u].keys()):
        #         g.nodes[u]['NNI'][v] = (g.nodes[v]['NI'] * g.nodes[u]['sim'][v] / sim_max) ** 0.5
        # return self.g

    def get_author_id(self, index):
        return int(self.g.nodes[index]['id'])

    def cos_sim(self, na, nb):
        """
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a
        :param vector_b: 向量 b
        :return: sim
        """
        # a_pubs = self.g.nodes[na]['pub_papers']
        # b_pubs = self.g.nodes[nb]['pub_papers']
        a_pubs = sum([self.g[na][nan]['weight'] for nan in nx.neighbors(self.g, na)])
        b_pubs = sum([self.g[nb][nbn]['weight'] for nbn in nx.neighbors(self.g, nb)])
        vector_a = np.mat(self.lda_vector[self.get_author_id(na)])
        vector_b = np.mat(self.lda_vector[self.get_author_id(nb)])
        num = self.g[na][nb]['weight'] * float(vector_a * vector_b.T)
        denom = np.linalg.norm(a_pubs * vector_a) * np.linalg.norm(b_pubs * vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim


def detected_team(client):
    db_field_au = client.Fields_author
    fields = os.listdir('../journalinfo/')
    for fie in fields:
        au_col = db_field_au[fie]
        lda_vector(client, au_col)
        # load graph
        print("..................    load {} graph success   .....................".format(fie))
        grapic_path = '../core_author_net/' + fie + '.gpickle'
        G = nx.read_gpickle(grapic_path)
        # Gn = nx.convert_node_labels_to_integers(G, first_label=0, ordering="default")
        # load lda model
        print("******************    load {} lda model     ********************".format(fie))
        ldavec_path = '../lda_model/' + fie + 'lda.pickle'
        with open(ldavec_path, 'rb') as fp:
            ldavec = pickle.load(fp)

        community_path = '../community_results/com' + fie + '.txt'
        Q = []
        for i in range(10):
            algorithm = LPALS(G, file_output=community_path, lda_vec=ldavec, db_col=au_col)
            communities = algorithm.detect()
            EQ = algorithm.output()
            Q.append(EQ)
        print(Q)


if __name__ == '__main__':
    client = MongoClient(host='172.21.201.187', port=27017)
    detected_team(client)
