import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = 'Ciao', type = str,
                        help = 'Dataset to use')
    parser.add_argument('--train_percent', default = 0.8, type = float,
                        help = 'training_percent')
    parser.add_argument('--embed_size', default = 32, type = int,
                        help = 'embedding size for all layer')
    parser.add_argument('--lr', default = 0.03, type = float,
                        help = 'learning rate')
    parser.add_argument('--model', default = 'RGCN', type = str,
                        help = 'model selection')
    parser.add_argument('--epoch', default = 1000, type = int,
                        help = 'epoch number')
    parser.add_argument('--early_stop', default = 10, type = int,
                        help = 'early_stop validation')
    parser.add_argument('--batch_size', default = 1024, type = int,
                        help = 'batch size')
    parser.add_argument('--layers', default = 1, type = int,
                        help = 'layer number')
    parser.add_argument('--gpu', default = -1, type = int,
                        help = '-1 for cpu, 0 for gpu:0')
    parser.add_argument('--k', default = [5, 10, 20], type = list,
                        help = 'negative sampler number for each node')
    parser.add_argument('--g', default = 0.1, type = float,
                        help = 'hyper-parameter for aggregation weight')
    parser.add_argument('--social_g', default = 0.1, type = float,
                        help = 'hyper-parameter for aggregation weight')
    parser.add_argument('--item_g', default = 0.1, type = float,
                        help = 'hyper-parameter for aggregation weight')

    args = parser.parse_args()
    return args

