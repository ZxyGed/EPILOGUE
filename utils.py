import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse
import hashlib
from GPUtil import showUtilization as gpu_usage
from numba import cuda

from torch import Tensor
from typing import Optional, Tuple, Union
from torch_geometric.typing import OptTensor
from torch_geometric.utils import remove_self_loops, dropout_edge, add_random_edge


def get_useful_sample_index(mat):
    # filter the columns filled with zeros and return the bool idx
    return mat.sum(axis=0) != 0


def build_consistency_loss(attention_weights):
    num_views = len(attention_weights)
    num_heads = attention_weights.shape[0]
    loss = 0
    loss_func = torch.nn.MSELoss(reduction='sum')
    for i in range(num_views - 1):
        for j in range(i + 1, num_views):
            loss += loss_func(attention_weights[i],
                              attention_weights[j]) / num_heads
    return loss


def rwr(graph, restart_prob=0.5):
    n = graph.shape[0]
    graph = graph - np.diag(np.diag(graph))
    graph = graph + np.diag(np.sum(graph, axis=0) == 0)
    norm_graph = graph / np.sum(graph, axis=0)
    ret = np.matmul(np.linalg.inv(np.eye(n) - (1 - restart_prob)
                                  * norm_graph), restart_prob * np.eye(n))
    return ret


def count_occurence(labels, is_save=False, file_name=None):
    num_class = labels.shape[1]
    A = np.zeros((num_class, num_class))
    for i in range(num_class):
        for j in range(i, num_class):
            if i == j:
                continue
            else:
                A[j, i] = A[i, j] = np.sum(labels[:, i] & labels[:, j])
    count = np.sum(labels, axis=0)
    if is_save:
        torch.save((A, count), f'datasets/occurence_count/{file_name}')
    return A, count


def construct_graph(A, count, threshold=0.5, p=0.5):
    # A, count = torch.load(f'datasets/occurence_count/{file_name}')
    A = A / count
    A[A < threshold] = 0
    A[A >= threshold] = p
    A = A + np.eye(A.shape[0])
    return A


def sparse_adj(adj):
    adj = sparse.coo_matrix(adj)
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)
    edge_weight = torch.tensor(adj.data.reshape(-1, 1), dtype=torch.float)
    return edge_index, edge_weight


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


def gen_md5ID(embeddings_dir, head, selected_params_dict):
    # embeddings_dir:
    # autodl-nas/embeddings/without_reconstruction_err
    # autodl-nas/embeddings/with_reconstruction_err
    # autodl-nas/embeddings/with_pgnn
    # if args.name == 'yeast':
    #     head = f"{args.name}_{args.level}_{args.type}"
    # else:
    #     head = f"{args.name}_{args.domain}_{args.size}_{args.type}"
    # For selected_params_dict:
    # train_representation: ['hidden_dim', 'embedding_dim', 'dropout_rate', 'learning_rate']
    # train_representation_withReconsErr: ['hidden_dim', 'embedding_dim', 'dropout_rate', 'learning_rate', 'weight']
    # train_supervised_pgnn: ['hidden_dim', 'embedding_dim', 'dropout_rate', 'learning_rate','num_layers', 'hidden_ft', 'GNN', 'threshold']
    params_str = str(sorted(selected_params_dict.items()))
    tale = hashlib.md5(params_str.encode('utf-8')).hexdigest()
    return f"{embeddings_dir}/{head}_{tale}.npz"


def gen_msks(adjs):
    msks = np.array(np.concatenate([sum(adj) for adj in adjs]))
    msks[msks == 1] = 0
    # msks=np.multiply(msks,1/msks.sum(axis=0))
    # msks[np.isnan(msks)]=0
    return msks


# code was borrowed from pyg
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def gen_negative_samples(X):
    # seed is set as 42
    np.random.seed(42)
    num_samples = X.shape[0]
    idx = np.random.permutation(num_samples)
    return X[idx, :]


def disturb_graph(edge_index, edge_weight, p_drop=0.5, p_add=0.5):
    disturbed_edge_index, disturbed_edge_weight = edge_index, edge_weight
    if p_drop >0:
        try:
            disturbed_edge_index, edge_msks = dropout_edge(edge_index, p_drop)
        except:
            print(edge_index)
            raise Exception
        disturbed_edge_weight = edge_weight[edge_msks]
    if p_add >0:
        _, w = remove_self_loops(edge_index, edge_weight)
        avg_w = w.mean()
        
        disturbed_edge_index, added_edges = add_random_edge(disturbed_edge_index, p_add)
        disturbed_edge_weight = torch.concat((disturbed_edge_weight, avg_w * torch.ones(added_edges.size(1)).to(edge_index.device)))
    return disturbed_edge_index, disturbed_edge_weight


def build_contrastive_loss(orig_embeddings, aug_embeddings, loss_weights, masks, tao=0.5):
    num_views = len(orig_embeddings)
    # cos_sims = []
    loss = 0
    for i in range(num_views):
        orig_embeddings[i] = F.normalize(orig_embeddings[i])
        aug_embeddings[i] = F.normalize(aug_embeddings[i])
        # cos_sims.append(torch.matmul(orig_embeddings[i],aug_embeddings[i].T)/tao)
        cos_sim = torch.matmul(orig_embeddings[i], aug_embeddings[i].T) / tao
        # loss -= F.log_softmax(cos_sim, dim=1).sum(dim=1).mean()
        # loss -= loss_weights[i] * ((cos_sim.diag() - torch.log(torch.exp(1.06 * cos_sim).sum(1)))*masks[i]).sum()/sum(masks[i])
        loss -= loss_weights[i] * (cos_sim.diag() - torch.log(torch.exp(1.06 * cos_sim).sum(1))).mean()
    return loss


def build_contrastive_loss_ori_final(orig_embeddings, final_embedding, loss_weights, tao=0.5):
    num_views = len(orig_embeddings)
    # cos_sims = []
    loss = 0
    for i in range(num_views):
        orig_embeddings[i] = F.normalize(orig_embeddings[i])
        final_embedding = F.normalize(final_embedding)
        # cos_sims.append(torch.matmul(orig_embeddings[i],aug_embeddings[i].T)/tao)
        cos_sim = torch.matmul(orig_embeddings[i], final_embedding.T) / tao
        # loss -= F.log_softmax(cos_sim, dim=1).sum(dim=1).mean()
        # loss -= (cos_sim.diag()-torch.log(torch.exp(1.06 * cos_sim).sum(1))).mean()
        logits = cos_sim.diag() - torch.log(torch.exp(1.06 * cos_sim).sum(1))
        # loss -= (msks[i]*logits).sum()/sum(msks[i])
        loss -= loss_weights[i] * logits.mean()
    return loss


# def build_reg_loss(orig_embeddings, aug_embeddings, msks, tao):
#     num_views = len(orig_embeddings)
#     loss = 0
#     for i in range(num_views):
#         orig_embeddings[i] = F.normalize(orig_embeddings[i])
#         aug_embeddings[i] = F.normalize(aug_embeddings[i])
#         # cos_sims.append(torch.matmul(orig_embeddings[i],aug_embeddings[i].T)/tao)
#         cos_sim = torch.matmul(orig_embeddings[i], aug_embeddings[i].T) / tao
#         # loss -= F.log_softmax(cos_sim, dim=1).sum(dim=1).mean()
#         loss -= (cos_sim.diag() - torch.log(torch.exp(1.06 * cos_sim).sum(1))).mean()
#         # logits = cos_sim.diag()-torch.log(torch.exp(1.06 * cos_sim).sum(1))
#         # loss -= (msks[i]*logits).sum()/sum(msks[i])
#     return loss


def evaluate_level1(embedding):
    from function_prediction import evaluate_features
    idx = np.load('idx.npy')
    anno = np.load('autodl-tmp/data/labels/yeast_level1_label.npy').T
    config = {'gamma': {'minimum': -3, 'maximum': 0, 'samples': 4}, 'regularization': {'minimum': -2, 'maximum': 2, 'samples': 5}, 'trials': 1, 'test_size': 0.1, 'folds': 5}
    content = evaluate_features(embedding[idx], anno[idx], config)
    return content
