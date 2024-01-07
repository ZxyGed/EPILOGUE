import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
# from attention import Attention
# from layers import WGATConv

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class GNN4GAE(torch.nn.Module):
    # default activation function is relu
    def __init__(self, in_ft, hidden_ft, dropout_rate=0.1, neg_slop=0.2, num_layers=3, num_heads=1):
        super().__init__()
        
        self.hidden_ft = hidden_ft
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        # self.convs.append(GATConv(in_ft, hidden_ft, add_self_loops=False, neg_slop=neg_slop, heads=num_heads, concat=False))
        # self.convs.extend([GATConv(hidden_ft, hidden_ft, add_self_loops=False, neg_slop=neg_slop, heads=num_heads, concat=False) for _ in range(num_layers-1)])
        
        self.lin = nn.Linear(in_ft, hidden_ft)
        self.convs.extend([GATConv(hidden_ft, hidden_ft, add_self_loops=False, neg_slop=neg_slop, heads=num_heads, concat=False) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout_rate)
        # self.acFunc = Swish()

    def forward(self, x, edge_index, edge_weight, numNode, update='max'):
        x = self.lin(x)
        
        layers_ret = []
        
        for i, conv in enumerate(self.convs):
            tmp = conv(x, edge_index, edge_weight)
            # tmp = self.acFunc(tmp)
            
            if i < self.num_layers-1:
                x = x + tmp
                layers_ret.append(x[:numNode])
                x = self.dropout(x)
            else:
                # for the output layer, we dont't add residual
                layers_ret.append(tmp[:numNode])

        if update == 'concat':
            ret = torch.cat(layers_ret, dim=1)
        elif update == 'max':
            ret = torch.max(torch.stack(layers_ret, dim=0), dim=0)[0]
        # elif update == 'sum':
        #     ret = torch.sum(torch.stack(layers_ret, dim=0), dim=0)[0]

        return ret
# class GNN4GAE(torch.nn.Module):
#     # default activation function is relu
#     def __init__(self, in_ft, hidden_ft, dropout_rate=0.1, neg_slop=0.2, num_layers=3, num_heads=1):
#         super().__init__()
        
#         self.hidden_ft = hidden_ft
#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()
#         # self.convs.append(GATConv(in_ft, hidden_ft, add_self_loops=False, neg_slop=neg_slop, heads=num_heads, concat=False))
#         # self.convs.extend([GATConv(hidden_ft, hidden_ft, add_self_loops=False, neg_slop=neg_slop, heads=num_heads, concat=False) for _ in range(num_layers-1)])
        
#         self.lin = nn.Linear(in_ft, hidden_ft)
#         self.convs.extend([GATConv(hidden_ft, hidden_ft, add_self_loops=False, neg_slop=neg_slop, heads=num_heads, concat=False) for _ in range(num_layers)])

#         self.dropout = nn.Dropout(dropout_rate)
        
#         self.acFunc1 = nn.ReLU()
#         # self.acFunc2 = nn.Tanh()
#         self.acFunc2 = Swish()

#         # self.pre = nn.LayerNorm(in_ft)
#         # self.bn0 = nn.LayerNorm(hidden_ft)
#         # self.bn1 = nn.LayerNorm(hidden_ft)
#         # self.bn2 = nn.LayerNorm(hidden_ft)

#         self.bn0 = nn.BatchNorm1d(hidden_ft)
#         self.bn1 = nn.BatchNorm1d(hidden_ft)
#         self.bn2 = nn.BatchNorm1d(hidden_ft)

#     def forward(self, x, edge_index, edge_weight, numNode, update='max'):
#         x = self.lin(x)
#         x = self.bn0(x)

#         tmp = self.convs[0](x, edge_index, edge_weight)
#         tmp = self.bn1(tmp)
#         tmp = self.acFunc2(tmp)
#         tmp = self.dropout(tmp)

#         tmp = self.convs[1](tmp, edge_index, edge_weight)
#         tmp = self.bn2(tmp)
#         tmp = self.dropout(tmp)
#         x = x + tmp # add residule
#         # x = self.acFunc1(x)
#         x = self.acFunc2(x)

#         return x[:numNode]


class GNN4Contrastive(torch.nn.Module):
    # default activation function is relu
    def __init__(self, in_ft, hidden_ft, dropout_rate=0.1, neg_slop=0.2):
        super().__init__()
        # self.conv1 = GCNConv(in_ft, hidden_ft)
        # self.conv2 = GCNConv(hidden_ft, hidden_ft)

        self.conv1 = GATConv(in_ft, hidden_ft)
        self.conv2 = GATConv(hidden_ft, hidden_ft)
        # self.conv3 = WGATConv(hidden_ft, hidden_ft)

        # self.conv1 = WGATConv(in_ft, hidden_ft)
        # self.conv2 = WGATConv(hidden_ft, hidden_ft)
        # self.ac_fc = nn.LeakyReLU(neg_slop)
        self.ac_fc = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_weight, update='max'):
        layers_ret = []
        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = self.ac_fc(x1)
        layers_ret.append(x1)

        x2 = self.dropout(x1)
        x2 = self.conv2(x2, edge_index, edge_weight)
        layers_ret.append(x2)

        # x3 = self.dropout(x2)
        # x3 = self.conv3(x3, edge_index, edge_weight)
        # layers_ret.append(x3)
        # return F.log_softmax(x, dim=1)

        # print(f"ret's shape is {str(ret.shape)}")
        if update == 'concat':
            ret = torch.cat(layers_ret, dim=1)
        elif update == 'max':
            ret = torch.max(torch.stack(layers_ret, dim=0), dim=0)[0]
        # print(ret.shape)
        return ret


class MultiHopGNN(torch.nn.Module):
    def __init__(self, in_ft, hidden_ft, embedding_ft, num_layers, GNN, device, dropout_rate=0.1, neg_slop=0.2):
        super().__init__()
        assert GNN in ["GCN", "GAT"]
        self.GNN = GNN
        if GNN == "GCN":
            self.layers = [
                GCNConv(in_ft, hidden_ft, add_self_loops=True, normalize=True).to(device)]
            for i in range(num_layers - 1):
                self.layers.append(GCNConv(hidden_ft, hidden_ft).to(device))
        elif GNN == "GAT":
            self.layers = [GATConv(
                in_ft, hidden_ft, add_self_loops=True, negative_slop=neg_slop, dropout=dropout_rate).to(device)]
            for i in range(num_layers - 1):
                self.layers.append(GATConv(hidden_ft, hidden_ft).to(device))
        else:
            raise ValueError(
                f'GNN {GNN} found, but expected either GCN or GAT'
            )

        self.num_layers = num_layers
        self.neg_slop = neg_slop
        self.dropout_rate = dropout_rate
        self.hypo = nn.Linear(self.num_layers * hidden_ft, embedding_ft)

    def forward(self, x, edge_index, edge_weight=None):
        layers = []
        if self.GNN == "GCN":
            for i in range(self.num_layers):
                x = self.layers[i](x, edge_index, edge_weight)
                x = F.leaky_relu(x, self.neg_slop)
                x = F.dropout(x, self.dropout_rate)
                layers.append(x)
        elif self.GNN == "GAT":
            for i in range(self.num_layers):
                # leakyrelu and drop are in GATConv settings
                # print(f"layer idx: {i}")
                # print(f"x's shape is {x.shape}")
                # print(f"edge_index's shape is {edge_index.shape}")
                x = self.layers[i](x, edge_index)
                layers.append(x)
        else:
            pass
        x = torch.cat(layers, -1)
        return self.hypo(x)

# if __name__=='__main__':
#
#     mhgnn=MultiHopGNN()
