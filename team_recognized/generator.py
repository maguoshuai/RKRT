"""
这个文件定义了如何从图中提出初重叠社区和重叠节点
"""
import networkx as nx


class GraphGenerator:
    """
    定义归属系数的阈值
    也就是说，只有一个节点对一个社区的归属系数大于这个阈值时，我们才考虑将这个节点加入这个社区中
    """
    b_threshold = 0.
    g = nx.Graph()

    def __init__(self, b, g):
        self.b_threshold = b
        self.g = g

    def get_Overlapping_communities(self):
        """
            从图中将所有的重叠社区返回
            :return: list
            """
        node_coms_num = {}
        d = {}
        for node in self.g.nodes:
            coms_num = 0
            L = self.g.nodes[node]['L']
            for label in L.keys():
                if L[label] > self.b_threshold:
                    # 这个节点属于label 社区
                    coms_num += 1
                    if d.get(label, -1) == -1:
                        d[label] = {node}
                    else:
                        d[label].add(node)
            node_coms_num[node] = coms_num
        # return list(d.values()), node_coms_num
        return d, node_coms_num

    def get_Overlapping_nodes(self) -> set:
        """
        从图中将所有的重叠节点返回
        :return: 所有的重叠节点
        """
        overlapping_nodes = set()
        for node in self.g.nodes:
            L = self.g.nodes[node]['L']
            count = 0
            for label in L.keys():
                if L[label] > self.b_threshold:
                    count += 1
                    if count >= 2:
                        overlapping_nodes.add(node)
                        break
        return overlapping_nodes

