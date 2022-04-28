import torch.nn as nn
from tqdm import tqdm
import pdb
import torch.nn.functional as F
import torch
from dgl.nn import SAGEConv
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import GATConv
from dgl.nn import GraphConv

class Proposed_model(nn.Module):
    def __init__(self, args, graph, item_graph):
        super().__init__()
        self.args = args
        self.hid_dim = args.embed_size
        self.layer_num = args.layers

        # self.user_embedding = torch.nn.Parameter(torch.randn(graph.nodes('user').shape[0], self.hid_dim))
        # self.item_embedding = torch.nn.Parameter(torch.randn(graph.nodes('game').shape[0], self.hid_dim))

        self.user_embedding = torch.nn.Parameter(torch.load('./baselines/user_embedding.pt'))
        self.item_embedding = torch.nn.Parameter(torch.load('./baselines/item_embedding.pt'))

        self.item_conv = SAGEConv(self.hid_dim, self.hid_dim, 'mean')
        self.social_GAT = GATConv(self.hid_dim, self.hid_dim, num_heads = 1, allow_zero_in_degree = True)
        self.social_conv = SAGEConv(self.hid_dim, self.hid_dim, 'mean')
        self.linear = torch.nn.Linear(3 * self.hid_dim, self.hid_dim)

        self.build_model(item_graph)

    def build_layer(self, idx, graph):
        if idx == 0:
            input_dim = graph.ndata['h'].shape[1]
        else:
            input_dim = self.hid_dim
        dic = {
            rel: GraphConv(input_dim, self.hid_dim, weight = True, bias = False)
            for rel in graph.etypes
        }
        return dglnn.HeteroGraphConv(dic, aggregate = 'mean')

    def build_model(self, graph):
        self.layers = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx, graph)
            self.layers.append(h2h)

    def forward(self, graph, item_graph, social_graph):

        h_game = item_graph.ndata['h']
        for layer in self.layers:
            h_game = layer(item_graph, {'game': h_game})['game']

        graph_game2user = dgl.edge_type_subgraph(graph, ['played by'])

        weight = graph.edata['weight'][('game', 'played by', 'user')]
        h_user_aggregate = self.item_conv(graph_game2user, (h_game, self.user_embedding), edge_weight = weight)

        _, social_weight = self.social_GAT(social_graph, h_user_aggregate, get_attention = True)
        social_weight = social_weight.sum(1)
        h_user_social = self.social_conv(social_graph, self.user_embedding, edge_weight = social_weight)

        user_embed = (1 - self.args.social_g - self.args.item_g) * self.user_embedding + self.args.item_g * h_user_aggregate + self.args.social_g * h_user_social

        return {"user": user_embed, "game": self.item_embedding}
