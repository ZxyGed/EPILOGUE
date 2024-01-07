# export
import gc
import os
import time
import numpy as np
import torch
from dataset import Networks
# from utils import process
import torch.nn as nn
from torch_geometric.transforms import ToSparseTensor
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from layers import AvgReadout
from torch.nn import functional as F


class embedder:
    def __init__(self, args):
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        args.model_dir = 'autodl-tmp/model'
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        
        args.dataList = Networks(args.dataFolder, args.saveFolder, dataset=args.dataset).dataList
        args.num_views = len(args.dataList)

        if args.feature_type == 'rwr':
            args.features = [content.data[i].x.to(args.device) for i in range(args.num_views)]
        elif args.feature_type == 'identity':
            args.features = torch.eye(args.dataList[0].num_nodes).to(args.device)
        elif args.feature_type == 'seqFeature':
            # seqContent = torch.load(f'{args.dataFolder}/seqAttrs/{args.dataset}.pt')
            # x, subsetIdx = seqContent['x'], seqContent['idx'].to(torch.bool)
            # args.features = x[subsetIdx].to(args.device)
            # for d in args.dataList:
            #     d = d.subgraph(subsetIdx)
            args.features = torch.load(f'{args.dataFolder}/seqAttrs/{args.dataset}.pt').to(args.device)
            # args.features = F.layer_norm(args.features, (args.features.shape[-1]))
        else:
            raise ValueError(f'feature_type({args.feature_type}) found, but expected either rwr or identity')
            
        args.num_nodes, args.num_features = args.features.shape
        for d in args.dataList:
            d = ToSparseTensor()(d).to(args.device)
        # args.edge_indexes = [content.data[i].edge_index.to(args.device) for i in range(args.num_views)]
        # args.edge_weights = [content.data[i].edge_weight.to(args.device) for i in range(args.num_views)]

        args.is_aug = True
        
        maskList = []
        for d in args.dataList:
            maskList.append(d.mask)
            del d.mask
        args.masks = torch.stack(maskList).to(args.device)
        self.args = args
        # How to aggregate
        # args.readout_func = AvgReadout()

        # Summary aggregation
        # args.readout_act_func = nn.Sigmoid()
        
        # del content
        # gc.collect()
        # torch.cuda.empty_cache()

    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s
