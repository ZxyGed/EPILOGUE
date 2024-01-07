# export
import os
import argparse
import torch
import numpy as np

np.random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-d', '--dataset', default='yeast', type=str)
    parser.add_argument('--feature_type', default='seqFeature', type=str)

    parser.add_argument('--num_neighbors', default='[2, 2, 2]', type=str)
    parser.add_argument('-ep', '--num_epoch', default=1200, type=int)
    parser.add_argument('-bs', '--batch_size', default=6400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.005, type=float)
    parser.add_argument('-ns', '--neg_slop', default=0.2, type=float)
    parser.add_argument('-dr', '--dropout_rate', default=0.15, type=float)
    parser.add_argument('-w', '--weight', default=0.05, type=float)
    parser.add_argument('-pdr', '--prob_drop', default=0.6, type=float)
    parser.add_argument('-pad', '--prob_add', default=0.1, type=float)
    parser.add_argument('-t', '--tao', default=0.5, type=float)
    parser.add_argument('-hd', '--hidden_dim', default=512, type=int)
    parser.add_argument('-nl', '--num_layers', default=3, type=int)
    parser.add_argument('-nh', '--num_heads', default=1, type=int)
    parser.add_argument('-pa', '--patience', default=35, type=int)

    parser.add_argument('-df', '--dataFolder', default='autodl-tmp/data', type=str)
    parser.add_argument('-sf', '--saveFolder', default='autodl-tmp/data', type=str)
    parser.add_argument('-sbi', '--subsetInfo', default='level1', type=str)
    parser.add_argument('-mrk', '--marker', default='', type=str)
    parser.add_argument('-gn', '--gpu_num', type=int, default=0)

    return parser.parse_known_args()


def main():
    args, unknown = parse_args()
    args.num_neighbors = eval(args.num_neighbors)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

    from models import EPILOGUE
    from utils import evaluate_level1
    model = EPILOGUE(args)
    model.training_gae_cl()
    # embeddings, embedding = model.pretraining(return_embedding=True)
    # embedding = embedding.cpu().numpy()
    # scores = evaluate_level1(embedding)
    # with open(r'{args.level}_ret.txt', 'a') as f:
    #     f.write(f'{scores} -- hd: {args.hidden_dim}, lr: {args.learning_rate}, dr: {args.dropout_rate}\n')


if __name__ == '__main__':
    main()
