import os
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import numpy as np
from scipy import sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse.tensor import SparseTensor

from LARC import LARC
from embedder import embedder
from gnn_backbone import GNN4GAE, GNN4Contrastive
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected, subgraph, remove_self_loops, add_self_loops
from utils import disturb_graph, build_contrastive_loss, build_contrastive_loss_ori_final, evaluate_level1,free_gpu_cache
from geneFuncPrediction import _evaluate,evaluate_func

# multi_view GAE with incomplete graphs(different graph size)

class MGAE_CL(nn.Module):
    def __init__(self, args):
        super(MGAE_CL, self).__init__()
        self.encoders = nn.ModuleList([GNN4GAE(args.num_features, args.hidden_dim, dropout_rate=args.dropout_rate, neg_slop=args.neg_slop, num_layers=args.num_layers, num_heads=args.num_heads) for i in range(args.num_views)])
        atten = nn.Parameter(torch.empty((args.num_views, args.hidden_dim, 1)))
        self.attenWeights = xavier_uniform_(atten)
        self.args = args
        
        

    def forward(self, features, edge_index_li, edge_weights_li, batch_msks, n_id_li):
        orig_embeddings = []
        aug_embeddings = []
                
        for i, (encoder, feature, eidx, ew, n_id) in enumerate(zip(self.encoders, features, edge_index_li, edge_weights_li, n_id_li)):
            # numCenterNode = batch_msks[i].sum().to(torch.int32)
            # orig_embedding = torch.zeros((len(n_id), self.args.hidden_dim)).to(self.args.device)
            # orig_embedding[batch_msks[i].squeeze()]=encoder(feature, eidx, ew, numCenterNode)
            # print('feature:',feature.is_cuda)
            # print('edge_index:',eidx.is_cuda())
            # print('edge_weight:',ew.is_cuda)
            # print('n_id:',len(n_id).is_cuda())
            orig_embedding=encoder(feature, eidx, ew, len(n_id))
            orig_embeddings.append(orig_embedding)
            
            if self.args.is_aug:
                row, col, _  = eidx.coo()
                edge_index = torch.stack([col, row], dim=0)
                disturbed_idx, disturbed_w = disturb_graph(edge_index, ew, self.args.prob_drop, self.args.prob_add)
                disturbed_idx, disturbed_w = to_undirected(disturbed_idx, disturbed_w)
                disturbed_idx, disturbed_w = remove_self_loops(disturbed_idx, disturbed_w)
                disturbed_idx, disturbed_w = add_self_loops(edge_index, disturbed_w)
                sparse_edge_index = SparseTensor.from_edge_index(disturbed_idx,sparse_sizes=eidx.sparse_sizes())
                
                # aug_embedding = torch.zeros((len(n_id), self.args.hidden_dim)).to(self.args.device)
                # aug_embedding[batch_msks[i].squeeze()] = encoder(feature, sparse_edge_index, disturbed_w, numCenterNode)
                
                aug_embedding = encoder(feature, sparse_edge_index, disturbed_w, len(n_id))
                del disturbed_idx, sparse_edge_index
                
                # shuffle_idx = torch.randperm(len(feature))
                # aug_embedding = encoder(feature[shuffle_idx], eidx, ew, len(n_id))
                
                aug_embeddings.append(aug_embedding)
                
                
        
        # self.embeddings = orig_embeddings
        return orig_embeddings, aug_embeddings

    def embed(self, features, edge_index_li, edge_weights_li, batch_msks, n_id_li, get_all_emb=False):
        embeddingList = []
        for i, (encoder, feature, eidx, ew, n_id) in enumerate(zip(self.encoders, features, edge_index_li, edge_weights_li, n_id_li)):
            # numCenterNode = batch_msks[i].sum().to(torch.int32)
            # orig_embedding = torch.zeros((len(n_id), self.args.hidden_dim)).to(self.args.device)
            # orig_embedding[batch_msks[i].squeeze()]=encoder(feature, eidx, ew, numCenterNode)
            orig_embedding=encoder(feature, eidx, ew, len(n_id))
            embeddingList.append(orig_embedding)
        embeddingTensor = torch.stack(embeddingList)
        
        atten = torch.bmm(embeddingTensor, self.attenWeights).squeeze()
        batch_msks[batch_msks==0]=float('-inf')
        atten = torch.multiply(batch_msks, atten)
        atten[atten==float('inf')] = float('-inf')
        atten = F.softmax(atten, dim=0)
        embedding = torch.einsum('bn,bns->ns',atten,embeddingTensor)
        # atten_msks = self.atten * self.args.w_msks
        # embedding = sum(atten_msks[i].view(-1, 1) * embeddings[i] for i in range(self.args.num_views))  
        # embedding = self.mergeLayer(torch.concat(embeddings, dim=1))
        # embedding = F.softmax(embedding, dim=0)     
        if get_all_emb:
            return embedding, embeddingList
        return embedding
    
    
class EPILOGUE(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.is_eval = True
    
    def training_gae_cl(self):
        embeddings = self.gae_cl_learning(is_aug=True)
        import pickle
        with open('autodl-tmp/epilogue.pckl','wb') as f:
            pickle.dump(embeddings, f)
            print(1)
        # print(embedding.shape)
        # np.save('yeast_emb.npy', embedding)
        # np.save(f'autodl-tmp/embeddings/{self.args.dataset}_emb_{self.args.hidden_dim}_{self.args.learning_rate}_{self.args.dropout_rate}_{self.args.batch_size}_{self.args.prob_add}_{self.args.prob_drop}_{self.args.weight}_{self.args.num_layers}_{str(self.args.num_neighbors)}_{self.args.num_heads}_{self.args.feature_type}_{self.args.marker}.npy', embedding)
        # free_gpu_cache()

        # message = f'-hd {self.args.hidden_dim} -lr {self.args.learning_rate} -dr {self.args.dropout_rate} -bs {self.args.batch_size} -pad {self.args.prob_add} -pdr {self.args.prob_drop} -w {self.args.weight} --num_neighbors {str(self.args.num_neighbors)} -nl {self.args.num_layers} -nh {self.args.num_heads} -d {self.args.dataset} --feature_type {self.args.feature_type}'
        # config = {'gamma': {'minimum': -3, 'maximum': 0, 'samples': 4}, 'regularization': {'minimum': -2, 'maximum': 2, 'samples': 5}, 'trials': 1, 'test_size': 0.2, 'folds': 3}
        # evaluate_func(embedding, self.args.dataset, self.args.subsetInfo, message, config, self.args.gpu_num)
    
        # with open('log.txt', 'a') as f:
        #     f.write(f'{scores} -hd {self.args.hidden_dim} -lr {self.args.learning_rate} -dr {self.args.dropout_rate} -bs {self.args.batch_size} -pad {self.args.prob_add} -pdr {self.args.prob_drop} -w {self.args.weight} --num_neighbors {str(self.args.num_neighbors)} -nl {self.args.num_layers} -nh {self.args.num_heads} -d {self.args.dataset} --feature_type {self.args.feature_type}\n')

    def gae_cl_learning(self, is_aug=True):
        bad_counter = 0
        best_performance = float('inf')
        best_epoch = self.args.num_epoch + 1
        gae_cl = MGAE_CL(self.args).to(self.args.device)
        # optimiser_gae_cl = torch.optim.Adam(gae_cl.parameters(), lr=self.args.learning_rate)
        optimiser_gae_cl = torch.optim.Adam(gae_cl.parameters(), lr=self.args.learning_rate)
        optimiser_gae_cl = LARC(optimiser_gae_cl)

        t = time.time()

        for epoch in range(self.args.num_epoch):
            gae_cl.train()
            total_loss = total_contrastive_loss = total_recons_loss = 0
            for n_id in iter(torch.split(torch.randperm(self.args.num_nodes),self.args.batch_size)):
                cur_loss = 0
                optimiser_gae_cl.zero_grad()
                
                batch_msks = self.args.masks[:,n_id]
                # numNodes = [batch_msks[i].sum().to(torch.int32) for i in range(self.args.num_views)]
                    
                features_li = []
                edge_index_li = []
                edge_weights_li = []
                n_id_li = []
                for d in self.args.dataList:
                    sampled_d = next(iter(NeighborLoader(d,num_neighbors=self.args.num_neighbors,input_nodes=n_id,batch_size=len(n_id))))
                    edge_index_li.append(sampled_d.adj_t)
                    # print(sampled_d.adj_t.nnz())
                    edge_weights_li.append(sampled_d.edge_attr)
                    features_li.append(self.args.features[sampled_d.n_id])
                    n_id_li.append(n_id)
                    # del sampled_d

                graphs = []
                loss_weights = []
                weight_msks = []
                
                masks = self.args.masks[:,n_id]
                for i in range(self.args.num_views):
                    row, col, _ = self.args.dataList[i].adj_t.coo()
                    edge_index = torch.stack([col, row], dim=0)
                    
                    n_id = n_id.to(self.args.device)
                    # print(edge_index.max(),edge_index.min(), n_id.max(),n_id.min())
                    # print(n_id.get_device(), edge_index.get_device())
                    edge_index, _ = subgraph(n_id, edge_index, relabel_nodes=True)
                    mask = masks[i]
                    
                    graph = SparseTensor.from_edge_index(edge_index, sparse_sizes=(len(n_id), len(n_id))).to_dense()
                    num_nodes= sum(mask)
                    pos_weight = float(num_nodes * num_nodes - graph.sum()) / graph.sum() # 计算0/1比
                    
                    connec_idx = graph.view(-1)==1
                    weight_msk = torch.ones(connec_idx.size(0))
                    mask = mask.to(torch.float)
                    weight_msk[(mask.view(-1,1)@mask.view(1,-1)).view(-1).to(torch.bool)] = 0 # 将不存在的节点置零
                    weight_msk[connec_idx] = pos_weight # 0/1比越高，即越稀疏，重构时的权重越高
                    weight_msks.append(weight_msk.to(self.args.device))
                    
                    loss_weight = num_nodes * num_nodes / float((num_nodes * num_nodes - graph.sum()) * 2) # 图中的0约少，权重越高，即图越稠密，权重越高
                    loss_weights.append(loss_weight) 
                    graphs.append(graph.to(self.args.device))
                
                
                if self.args.weight > 0:
                    orig_embeddings, aug_embeddings = gae_cl(features_li, edge_index_li, edge_weights_li, batch_msks, n_id_li)

                    if is_aug:
                        contrastive_loss = build_contrastive_loss(orig_embeddings, aug_embeddings, loss_weights, masks, self.args.tao)
                    else:
                        embedding = gae_cl.embed(features_li, edge_index_li, edge_weights_li, n_id_li)
                        contrastive_loss = build_contrastive_loss_ori_final(orig_embeddings, embedding, loss_weights, self.args.tao)   

                    cur_loss += self.args.weight * contrastive_loss 
                    total_contrastive_loss += contrastive_loss.item()
            
                if self.args.weight < 1:
                    embedding = gae_cl.embed(features_li, edge_index_li, edge_weights_li, batch_msks, n_id_li)
                    # print(embedding[0])
                    recons_adj = F.sigmoid(torch.matmul(embedding, embedding.T))

                    recons_loss = 0
                    for i in range(self.args.num_views):
                        recons_loss += loss_weights[i] * F.binary_cross_entropy(recons_adj.view(-1), graphs[i].view(-1), weight=weight_msks[i])
                    cur_loss += (1 - self.args.weight) * recons_loss
                    total_recons_loss += recons_loss.item()

                total_loss += cur_loss.item()
                
                cur_loss.backward()
                optimiser_gae_cl.step()
                
            if not epoch % 25:
                if self.args.weight == 1:
                    print(f'epoch:{epoch}, contrastive_loss: {total_contrastive_loss}, loss:{total_loss}')
                elif self.args.weight == 0:
                    print(f'epoch:{epoch}, recons_loss: {total_recons_loss}, loss:{total_loss}')
                else:
                    print(f'epoch:{epoch}, contrastive_loss: {total_contrastive_loss}, recons_loss: {total_recons_loss}, loss:{total_loss}')
            
#             if not epoch % 100 and epoch:   
#                 with torch.no_grad():
#                     gae_cl.eval()
#                     embeddings = []
#                     for n_id in iter(torch.split(torch.arange(self.args.num_nodes), self.args.batch_size)):

#                         batch_msks = self.args.masks[:,n_id]
#                         # numNodes = [batch_msks[i].sum() for i in range(self.args.numViews)]

#                         features_li = []
#                         edge_index_li = []
#                         edge_weights_li = []
#                         n_id_li = []

#                         for d in self.args.dataList:           
#                             sampled_d = next(iter(NeighborLoader(d,num_neighbors=self.args.num_neighbors,input_nodes=n_id,batch_size=len(n_id))))
#                             edge_index_li.append(sampled_d.adj_t)
#                             edge_weights_li.append(sampled_d.edge_attr)
#                             features_li.append(self.args.features[sampled_d.n_id])
#                             n_id_li.append(n_id)
#                             del sampled_d      
#                         eb = gae_cl.embed(features_li, edge_index_li, edge_weights_li, batch_msks, n_id_li)
#                         embeddings.append(eb.detach().cpu().numpy())
#                         del eb
#                     embedding = np.concatenate(embeddings)
#                     message = f'-hd {self.args.hidden_dim} -lr {self.args.learning_rate} -dr {self.args.dropout_rate} -bs {self.args.batch_size} -pad {self.args.prob_add} -pdr {self.args.prob_drop} -w {self.args.weight} --num_neighbors {str(self.args.num_neighbors)} -nl {self.args.num_layers} -nh {self.args.num_heads} -d {self.args.dataset} --feature_type {self.args.feature_type} EPOCH: {epoch}'
#                     config = {'gamma': {'minimum': -3, 'maximum': 0, 'samples': 4}, 'regularization': {'minimum': -2, 'maximum': 2, 'samples': 5}, 'trials': 1, 'test_size': 0.2, 'folds': 3}
#                     evaluate_func(embedding, self.args.dataset, self.args.subsetInfo, message, config, self.args.gpu_num)
                    
#                     gae_cl.train()

            if total_loss - best_performance < 5*1e-3:
                best_performance = total_loss
                best_epoch = epoch
                bad_counter = 0
                torch.save(gae_cl.state_dict(), f'{self.args.model_dir}/{self.args.dataset}_gae_cl_{self.args.marker}.pkl')
            else:
                bad_counter += 1
            if bad_counter == self.args.patience:
                print('Early Stoping at epoch %d' % epoch)
                break

                

        del cur_loss
        print(f"Optimization for gae_cl learning Finished! The Best Epoch was {best_epoch}")
        print("Total time elapsed: %.4fs" % (time.time() - t))
        
        if best_epoch < 100:
            import sys
            sys.exit()
        
        gae_cl.load_state_dict(torch.load(f"{self.args.model_dir}/{self.args.dataset}_gae_cl_{self.args.marker}.pkl"))
        with torch.no_grad():
            gae_cl.eval()
            embeddings = [[] for _ in range(7)]
            for n_id in iter(torch.split(torch.arange(self.args.num_nodes), self.args.batch_size)):
                
                batch_msks = self.args.masks[:,n_id]
                # numNodes = [batch_msks[i].sum() for i in range(self.args.numViews)]
                
                features_li = []
                edge_index_li = []
                edge_weights_li = []
                n_id_li = []
                
                for d in self.args.dataList:           
                    sampled_d = next(iter(NeighborLoader(d,num_neighbors=self.args.num_neighbors,input_nodes=n_id,batch_size=len(n_id))))
                    edge_index_li.append(sampled_d.adj_t)
                    edge_weights_li.append(sampled_d.edge_attr)
                    features_li.append(self.args.features[sampled_d.n_id])
                    n_id_li.append(n_id)
                    del sampled_d      
                eb, ebList = gae_cl.embed(features_li, edge_index_li, edge_weights_li, batch_msks, n_id_li, True)
                embeddings[-1].append(eb.detach().cpu().numpy())
                for i in range(6):
                    embeddings[i].append(ebList[i].detach().cpu().numpy())
                del eb, ebList
        
        del gae_cl
        for i,eb in enumerate(embeddings):
            embeddings[i] = np.concatenate(embeddings[i])
        return embeddings

# train gae and evaluate
if __name__ == '__main__':
    from main import parse_args
    from embedder import embedder
    args, unknown = parse_args()
    # eb=embedder(args)
    model = EPILOGUE(args)
    embeddings, embedding = model.pretraining(return_embedding=True)
    # embeddings, embedding = model.load_pretrained_embeddings(return_embedding=True)
    # del model, embeddings
    embedding = embedding.cpu().numpy()
    idx = np.load('idx.npy')
    anno = np.load('autodl-tmp/data/labels/yeast_level1_label.npy').T
    config = {'gamma': {'minimum': -3, 'maximum': 0, 'samples': 4}, 'regularization': {'minimum': -2, 'maximum': 2, 'samples': 5}, 'trials': 1, 'test_size': 0.2, 'folds': 5}
    content = _evaluate_features(embedding[idx], anno[idx], config)
    scores = content[0]
    print(scores)
    with open(r'{args.level}_ret.txt', 'a') as f:
        f.write(f'{scores} -- hd: {args.hidden_dim}, ed: {args.embedding_dim}, lr: {args.learning_rate}, dr: {args.dropout_rate}, p: {args.prob}\n')
    # model.training(is_pretrain=False)
