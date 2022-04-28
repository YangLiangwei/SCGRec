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

class Dataloader_steam_filtered(DGLDataset):
    def __init__(self, args, root_path, user_id_path, app_id_path, app_info_path, friends_path, developer_path, publisher_path, genres_path, device = 'cpu', name = 'steam'):
        logging.info("steam dataloader init")

        self.args = args
        self.root_path = root_path
        self.user_id_path = user_id_path
        self.app_id_path = app_id_path
        self.app_info_path = app_info_path
        self.friends_path = friends_path
        self.developer_path = developer_path
        self.publisher_path = publisher_path
        self.genres_path = genres_path
        self.device = device
        self.graph_path = self.root_path + '/graph.bin'
        self.game_path = self.root_path + '/train_game.txt'
        self.time_path = self.root_path + '/train_time.txt'
        self.valid_path = self.root_path + '/valid_game.txt'
        self.test_path = self.root_path + '/test_game.txt'

        logging.info("reading user id mapping from {}".format(self.user_id_path))
        self.user_id_mapping = self.read_id_mapping(self.user_id_path)
        logging.info("reading app id mapping from {}".format(self.app_id_path))
        self.app_id_mapping = self.read_id_mapping(self.app_id_path)

        logging.info("build valid data")
        self.valid_data = self.build_valid_data(self.valid_path)

        logging.info("build test data")
        self.test_data = self.build_valid_data(self.test_path)

        if os.path.exists(self.graph_path):
            logging.info("loading preprocessed data")
            self.graph = dgl.load_graphs(self.graph_path)
            self.graph = self.graph[0][0]
            logging.info("reading user game information")
            self.dic_user_game = self.read_dic_user_game(self.game_path)

        else:
            self.process()
            dgl.save_graphs(self.graph_path, self.graph)

        self.dataloader = self.build_dataloader(self.args, self.graph)

    def build_valid_data(self, path):
        users = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                user = self.user_id_mapping[line[0]]
                games = [self.app_id_mapping[game] for game in line[1:]]
                users[user] = games
        return users

    def build_dataloader(self, args, graph):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.layers, return_eids = False)
        train_id = torch.tensor([i for i in range(graph.edges(etype = 'play')[0].shape[0])], dtype = torch.long)
        dataloader = dgl.dataloading.EdgeDataLoader(
            graph, {('user', 'play', 'game'): train_id},
            sampler, negative_sampler = NegativeSampler(self.dic_user_game), batch_size = args.batch_size, shuffle = True, num_workers = 2
        )
        return dataloader


    def process(self):
        logging.info("reading app info from {}".format(self.app_info_path))
        self.app_info = self.read_app_info(self.app_info_path)

        logging.info("reading publisher from {}".format(self.publisher_path))
        self.publisher = self.read_mapping(self.publisher_path)

        logging.info("reading developer from {}".format(self.developer_path))
        self.developer = self.read_mapping(self.developer_path)

        logging.info("reading genre from {}".format(self.genres_path))
        self.genre = self.read_mapping(self.genres_path)

        logging.info("reading user item play time from {}".format(self.game_path))

        self.user_game, self.dic_user_game = self.read_play_time_rank(self.game_path, self.time_path)

        logging.info("reading friend list from {}".format(self.friends_path))
        self.friends = self.read_friends(self.friends_path)

        graph_data = {
            ('user', 'friend of', 'user'): (self.friends[:, 0], self.friends[:, 1]),

            ('game', 'developed by', 'developer'): (torch.tensor(list(self.developer.keys())), torch.tensor(list(self.developer.values()))),

            ('developer', 'develop', 'game'): (torch.tensor(list(self.developer.values())), torch.tensor(list(self.developer.keys()))),

            ('game', 'published by', 'publisher'): (torch.tensor(list(self.publisher.keys())), torch.tensor(list(self.publisher.values()))),

            ('publisher', 'publish', 'game'): (torch.tensor(list(self.publisher.values())), torch.tensor(list(self.publisher.keys()))),

            ('game', 'genre', 'type'): (torch.tensor(list(self.genre.keys())), torch.tensor(list(self.genre.values()))),

            ('type', 'genred', 'game'): (torch.tensor(list(self.genre.values())), torch.tensor(list(self.genre.keys()))),

            ('user', 'play', 'game'): (self.user_game[:, 0].long(), self.user_game[:, 1].long()),

            ('game', 'played by', 'user'): (self.user_game[:, 1].long(), self.user_game[:, 0].long())
        }
        graph = dgl.heterograph(graph_data)

        ls_feature = []

        for node in graph.nodes('game'):
            node = int(node)
            if node in self.app_info:
                ls_feature.append(self.app_info[node])

        ls_feature = np.vstack(ls_feature)
        feature_mean = ls_feature.mean(0)

        ls_feature = []

        count_total = 0
        count_without_feature = 0
        for node in graph.nodes('game'):
            count_total += 1
            node = int(node)
            if node in self.app_info:
                ls_feature.append(self.app_info[node])
            else:
                count_without_feature += 1
                ls_feature.append(feature_mean)
        logging.info("total game number is {}, games without features number is {}".format(count_total,count_without_feature ))

        graph.nodes['game'].data['h'] = torch.tensor(np.vstack(ls_feature))
        graph.edges['play'].data['time'] = self.user_game[:, 2]
        graph.edges['played by'].data['time'] = self.user_game[:, 2]
        graph.edges['play'].data['percentile'] = self.user_game[:, 3]
        graph.edges['played by'].data['percentile'] = self.user_game[:, 3]
        self.graph = graph

    def __getitem__(self, i):
        pass

    def __len__(self):
        pass

    def generate_percentile(self, ls):
        dic = {}
        for ls_i in ls:
            if ls_i[1] in dic:
                dic[ls_i[1]].append(ls_i[2])
            else:
                dic[ls_i[1]] = [ls_i[2]]
        for key in tqdm(dic):
            dic[key] = sorted(list(set(dic[key])))
        dic_percentile = {}

        for key in tqdm(dic):
            dic_percentile[key] = {}
            length = len(dic[key])
            for i in range(len(dic[key])):
                time = dic[key][i]
                dic_percentile[key][time] = (i + 1) / length


        for i in tqdm(range(len(ls))):
            ls[i].append(dic_percentile[ls[i][1]][ls[i][2]])
        return ls


    def read_dic_user_game(self, game_path):
        dic_game = {}
        with open(game_path, 'r') as f_game:
            lines_game = f_game.readlines()
            for i in tqdm(range(len(lines_game))):
                line_game = lines_game[i].strip().split(',')
                user = self.user_id_mapping[line_game[0]]

                dic_game[user] = []
                for j in range(1, len(line_game)):
                    game = self.app_id_mapping[line_game[j]]
                    dic_game[user].append(game)
        return dic_game


    def read_play_time_rank(self, game_path, time_path):
        ls = []
        dic_game = {}
        dic_time = {}
        with open(game_path, 'r') as f_game:
            with open(time_path, 'r') as f_time:
                lines_game = f_game.readlines()
                lines_time = f_time.readlines()
                for i in tqdm(range(len(lines_game))):
                    line_game = lines_game[i].strip().split(',')
                    line_time = lines_time[i].strip().split(',')
                    user = self.user_id_mapping[line_game[0]]
                    dic_game[user] = []

                    for j in range(1, len(line_game)):
                        game = self.app_id_mapping[line_game[j]]
                        dic_game[user].append(game)
                        time = line_time[j]
                        if time == r'\N':
                            ls.append([user, game, 0])
                        else:
                            ls.append([user, game, float(time)])
        logging.info('generate percentiles')
        ls = self.generate_percentile(ls)
        return torch.tensor(ls), dic_game

    def read_play_time(self, path):
        ls = []
        with open(path, 'r', encoding = 'utf8') as f:
            for line in f:
                line = line.strip().split(',')
                if line[-1] == r'\N':
                    ls.append([self.user_id_mapping[line[0]], self.app_id_mapping[line[1]], 0])
                else:
                    ls.append([self.user_id_mapping[line[0]], self.app_id_mapping[line[1]], int(line[2])])
        logging.info('generate percentiles')
        ls = self.generate_percentile(ls)
        return torch.tensor(ls)

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

    def read_app_info(self, path):
        dic = {}
        df = pd.read_csv(path, header = None)
        df = pd.get_dummies(df, columns = [2])
        df_time = pd.to_datetime(df.iloc[:, 3])
        date_end = pd.to_datetime('2013-06-25')
        time_sub = date_end - df_time
        time_sub = time_sub.dt.days
        df = pd.concat([df, time_sub], axis = 1)
        column_num = len(df.columns)
        column_index = [2]
        column_index.extend([i for i in range(4, column_num)])

        logging.info("begin feature engineering")
        df.iloc[:, 4].replace(to_replace = -1, value = np.nan, inplace = True)
        mean = df.iloc[:, 4].mean()
        df.iloc[:, 4].replace(to_replace = np.nan, value = mean, inplace = True)
        columns_norm = [2, 4, 5, 11]
        mean = df.iloc[:, columns_norm].mean()
        std = df.iloc[:, columns_norm].std()
        df.iloc[:, columns_norm] = (df.iloc[:, columns_norm] - mean) / std

        for i in range(len(df)):
            app_id = self.app_id_mapping[str(df.iloc[i, 0])]
            feature = df.iloc[i, column_index].to_numpy()
            feature = feature.astype(np.float64)
            dic[app_id] = feature
        dic['feature_num'] = len(feature)
        return dic

    def read_friends(self, path):
        ls = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                ls.append([self.user_id_mapping[line[0]], self.user_id_mapping[line[1]]])
        return torch.tensor(ls)

    def read_mapping(self, path):
        mapping = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                if line[0] not in mapping:
                    if line[1] != '':
                        mapping[self.app_id_mapping[line[0]]] = line[1]
        mapping_value2id = {}
        count = 0
        for value in mapping.values():
            if value not in mapping_value2id:
                mapping_value2id[value] = count
                count += 1
        for key in mapping:
            mapping[key] = mapping_value2id[mapping[key]]
        return mapping
