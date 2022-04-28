import os
import sys
from dgl.data.utils import save_graphs
from tqdm import tqdm
from scipy import stats
from .NegativeSampler import NegativeSampler
import pdb
import torch
import logging
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
import numpy as np
import dgl
from dgl.data import DGLDataset
import pandas as pd
from sklearn import preprocessing
from dgl.data import DGLDataset

class Dataloader_item_graph(DGLDataset):
    def __init__(self, graph, app_id_path, publisher_path, developer_path, genre_path):
        self.app_id_path = app_id_path
        self.publisher_path = publisher_path
        self.developer_path = developer_path
        self.genre_path = genre_path

        logging.info("reading item graph")
        self.app_id_mapping = self.read_id_mapping(self.app_id_path)
        self.publisher = self.read_mapping(self.publisher_path)
        self.developer = self.read_mapping(self.developer_path)
        self.genre = self.read_mapping(self.genre_path)

        graph_data = {
            ('game', 'co_publisher', 'game'): self.publisher,
            ('game', 'co_developer', 'game'): self.developer,
            ('game', 'co_genre', 'game'): self.genre
        }
        self.graph = dgl.heterograph(graph_data)
        self.graph.nodes['game'].data['h'] = graph.ndata['h']['game'].float()


    def read_mapping(self, path):
        mapping = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                if line[1] != '':
                    if line[0] not in mapping:
                        mapping[self.app_id_mapping[line[0]]] = [line[1]]
                    else:
                        mapping[self.app_id_mapping[line[0]]].append(line[1])
        for key in mapping:
            mapping[key] = set(mapping[key])
        src = []
        dst = []
        keys = list(mapping.keys())
        for i in range(len(keys) - 1):
            for j in range(i +1, len(keys)):
                game1 = keys[i]
                game2 = keys[j]
                if len(mapping[game1] & mapping[game2]) > 0:
                    src.extend([game1, game2])
                    dst.extend([game2, game1])
        return (torch.tensor(src), torch.tensor(dst))

    def read_id_mapping(self, path):
        mapping = {}
        count = 0
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line not in mapping:
                    mapping[line] = count
                    count += 1
        return mapping
