import networkx as nx
from gensim.models import Word2Vec
import random


def random_walk(graph: nx.Graph, walk_length, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        current_node = walk[-1]
        current_node_neighbors = list(graph.neighbors(current_node))
        if len(current_node_neighbors) > 0:
            walk.append(random.choice(current_node_neighbors))
        else:
            break

    return walk


def get_walks(graph: nx.Graph, num_walks, walk_length):
    nodes = list(graph.nodes())
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(graph, walk_length=walk_length, start_node=node))

    return walks

class Deepwalk:
    def __init__(self, graph, walk_length, num_walks, ):
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.walks = get_walks(graph, walk_length=walk_length, num_walks=num_walks)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.walks
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


if __name__ == "__main__":
    G = nx.read_edgelist(
        'data/wiki/Wiki_edgelist.txt',
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[('weight', int)]
    )

    model = Deepwalk(G, walk_length=10, num_walks=80)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()
    print('success')
