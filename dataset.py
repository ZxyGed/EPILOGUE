import os
from typing import Callable, List, Optional, Union

import torch
import numpy as np
from scipy import sparse
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, remove_self_loops, add_self_loops


class Networks:
    
    viewsList = [
        'coexpression', 'cooccurence', 'database', 'experimental', 'fusion',
        'neighborhood'
    ]
    
    def __init__(self, dataFolder, saveFolder, dataset):
        assert dataset in ('yeast', 'human')
        
        self.dataFolder = dataFolder
        self.saveFolder = saveFolder
        self.dataset = dataset
        self.saveFile = f'{saveFolder}/{dataset}.pt'
        
        self.numNodes = 6400 if self.dataset == 'yeast' else 18356
        
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        if not os.path.exists(self.saveFile):
            self.process()
        
        self.dataList = torch.load(self.saveFile)
        
    def process(self):
        dataList = []
        adjs = []
        # seqContent = torch.load(f'{self.dataFolder}/seqAttrs/{self.dataset}.pt')
        # x, subsetIdx = seqContent['x'], seqContent['idx'].to(torch.bool)
        for view in self.viewsList:
            adj_sparse = sparse.load_npz(f'{self.dataFolder}/adjs/{self.dataset}_{view}.npz')
            edge_index, edge_weight = from_scipy_sparse_matrix(adj_sparse)
            edge_index, edge_weight = to_undirected(edge_index, edge_weight)
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            
            mask = torch.zeros(self.numNodes).to(torch.bool)
            mask[edge_index[0].unique()]=True
            
            # node_nodes 必须要加，否则只会添加index中已有节点的对角线元素
            # 然后在subgraph的时候由于缺少某个节点而报错
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=self.numNodes) 
            
            # ToSparseTensor会将edge_weight默认转换如sparsetensor，因此将weight保存为attr
            data = Data(edge_index=edge_index, edge_attr=edge_weight, mask=mask, num_nodes=self.numNodes)
            dataList.append(data)
        torch.save(dataList, self.saveFile)