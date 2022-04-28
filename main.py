import sys
import dgl
import dgl.function as fn
sys.path.append('../')
import os
import multiprocessing as mp
# mp.set_start_method('spawn')
from tqdm import tqdm
import pdb
import random
import numpy as np
import torch
import torch.nn as nn
import logging
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
from utils.parser import parse_args
from utils.metrics import MAE, RMSE, ndcg_at_k, recall_at_k, hit_at_k, precision_at_k
from utils.dataloader_steam import Dataloader_steam
from utils.dataloader_item_graph import Dataloader_item_graph
from models.RGCNModel_steam_rank import RGCNModel_steam_rank
from models.Predictor import HeteroDotProductPredictor
from models.model import Proposed_model

def validate(train_mask, dic, h, ls_k):
    users = torch.tensor(list(dic.keys())).long()
    user_embedding = h['user'][users]
    game_embedding = h['game']
    rating = torch.mm(user_embedding, game_embedding.t())
    rating[train_mask] = -float('inf')

    valid_mask = torch.zeros_like(train_mask)
    for i in range(users.shape[0]):
        user = int(users[i])
        items = torch.tensor(dic[user])
        valid_mask[i, items] = 1

    _, indices = torch.sort(rating, descending = True)
    ls = [valid_mask[i,:][indices[i, :]] for i in range(valid_mask.shape[0])]
    result = torch.stack(ls).float()

    res = []
    for k in ls_k:
        discount = (torch.tensor([i for i in range(k)]) + 2).log2()
        ideal, _ = result.sort(descending = True)
        idcg = (ideal[:, :k] / discount).sum(dim = 1)
        dcg = (result[:, :k] / discount).sum(dim = 1)
        ndcg = torch.mean(dcg / idcg)

        recall = torch.mean(result[:, :k].sum(1) / result.sum(1))
        hit = torch.mean((result[:, :k].sum(1) > 0).float())
        precision = torch.mean(result[:, :k].mean(1))

        logging_result = "For k = {}, ndcg = {}, recall = {}, hit = {}, precision = {}".format(k, ndcg, recall, hit, precision)
        logging.info(logging_result)
        res.append(logging_result)
    return ndcg, str(res)


def construct_negative_graph(graph, etype):
    utype, _ , vtype = etype
    src, _ = graph.edges(etype = etype)
    dst = torch.randint(graph.num_nodes(vtype), size = src.shape)
    return dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parse_args()
    setup_seed(2020)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    path = '/home/yangliangwei/datasets/steam/'

    user_id_path = path + '/users.txt'
    app_id_path = path + '/app_id.txt'
    app_info_path = path + '/App_ID_Info.txt'
    friends_path = path + '/friends.txt'
    developer_path = path + '/Games_Developers.txt'
    publisher_path = path + '/Games_Publishers.txt'
    genres_path = path + '/Games_Genres.txt'

    DataLoader = Dataloader_steam(args, path, user_id_path, app_id_path, app_info_path, friends_path, developer_path, publisher_path, genres_path)

    graph = DataLoader.graph
    DataLoader_item = Dataloader_item_graph(graph, app_id_path, publisher_path, developer_path, genres_path)

    graph_item = DataLoader_item.graph

    graph_social = dgl.edge_type_subgraph(graph, [('user', 'friend of', 'user')])

    graph = dgl.edge_type_subgraph(graph, [('user', 'play', 'game'), ('game', 'played by', 'user')])
    graph.update_all(fn.copy_edge('percentile', 'm'), fn.sum('m', 'total'), etype = 'played by')
    graph.apply_edges(func = fn.e_div_v('percentile', 'total', 'weight'), etype = 'played by')

    valid_user = list(DataLoader.valid_data.keys())
    train_mask = torch.zeros(len(valid_user), graph.num_nodes('game'))
    for i in range(len(valid_user)):
        user = valid_user[i]
        item_train = torch.tensor(DataLoader.dic_user_game[user])
        train_mask[i, :][item_train] = 1
    train_mask = train_mask.bool()

    model = Proposed_model(args, graph, graph_item)

    predictor = HeteroDotProductPredictor()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr)

    stop_count = 0
    ndcg_val_best = 0
    ls_k = args.k

    total_epoch = 0
    for epoch in range(args.epoch):
        model.train()
        graph_neg = construct_negative_graph(graph, ('user', 'play', 'game'))
        h = model(graph, graph_item, graph_social)

        score = predictor(graph, h, ('user', 'play', 'game'))
        score_neg = predictor(graph_neg, h, ('user', 'play', 'game'))
        loss = -(score - score_neg).sigmoid().log().sum()
        logging.info("loss = {}".format(loss))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_epoch += 1

        # score, h = model.forward_all(graph, 'play')
        logging.info('Epoch {}'.format(epoch))
        if total_epoch > 1:
            model.eval()
            logging.info("begin validation")

            ndcg, _ = validate(train_mask, DataLoader.valid_data, h, ls_k)

            if ndcg > ndcg_val_best:
                ndcg_val_best = ndcg
                stop_count = 0
                logging.info("begin test")
                ndcg_test, test_result = validate(train_mask, DataLoader.test_data, h, ls_k)
            else:
                stop_count += 1
                if stop_count > args.early_stop:
                    logging.info('early stop')
                    break

    logging.info('Final ndcg {}'.format(ndcg_test))
    logging.info(test_result)
