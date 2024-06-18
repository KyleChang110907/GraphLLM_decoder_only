import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import init
from torch.nn.utils import weight_norm
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import math



class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act=False, dropout=False, p=0.5, **kwargs):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.act = act
        self.p = p
        concat_dim = [input_dim] + list(hidden_dim) + [output_dim]

        self.module_list = nn.ModuleList()
        for i in range(len(concat_dim)-1):
            self.module_list.append(nn.Linear(concat_dim[i], concat_dim[i+1]))
    
    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            if self.act and i != len(self.module_list)-1:
                x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=self.p, training=self.training)
        return x

        

class GraphNetwork_layer(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_attr_dim=15, message_dim=None, aggr='add', act=False):
        super(GraphNetwork_layer, self).__init__()
        self.aggr = aggr
        self.act = act
        message_dim = input_dim if message_dim is None else message_dim

        self.messageMLP = MLP(input_dim * 2 + edge_attr_dim, [], message_dim, act=self.act, dropout=False)
        self.outputMLP = MLP(input_dim + message_dim, [], output_dim, act=self.act, dropout=False)
        

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, messageMLP=self.messageMLP, outputMLP=self.outputMLP)

    def message(self, x_i, x_j, edge_attr, messageMLP):
        return messageMLP(torch.cat((x_i, x_j, edge_attr), dim=-1))

    def update(self, aggr_out, x, outputMLP):
        return outputMLP(torch.cat((x, aggr_out), dim=-1))




class SectionMessagePassingLayer(MessagePassing):
    def __init__(self, edge_attr_dim=15, gm_dim=10, aggr='add'):
        super(SectionMessagePassingLayer, self).__init__()
        '''
        Edge attr dim = 15
        Input  x: [node_num, feature_num]
        Output x: [node_num, feature_num + edge_attr]
        '''
        self.aggr = aggr
        self.edge_attr_dim = edge_attr_dim
        self.gm_dim = gm_dim

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return edge_attr

    def update(self, aggr_out, x):
        gm_space = torch.zeros((x.shape[0], self.gm_dim))
        return torch.cat([x[:, :-self.gm_dim], aggr_out, gm_space], dim=1)





class LayerNormLSTMCell(nn.Module):
    """
    - https://github.com/daehwannam/pytorch-rnn-util/blob/master/rnn_util/seq.py
    It's based on tf.contrib.rnn.LayerNormBasicLSTMCell
    Reference:
    - https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LayerNormBasicLSTMCell
    - https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1335
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fiou_linear = nn.Linear(input_size + hidden_size, hidden_size * 4)

        self.fiou_ln_layers = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(4))
        self.cell_ln = nn.LayerNorm(hidden_size)


    def forward(self, input, state):
        """
        :param input: a tensor of of shape (batch_size, input_size)
        :param state: a pair of a hidden tensor and a cell tensor whose shape is (batch_size, hidden_size).
                      ex. (h_0, c_0)
        :returns: hidden and cell
        """
        hidden_tensor, cell_tensor = state

        fiou_linear = self.fiou_linear(torch.cat([input, hidden_tensor], dim=1))
        fiou_linear_tensors = fiou_linear.split(self.hidden_size, dim=1)

        # if self.layer_norm_enabled:
        fiou_linear_tensors = tuple(ln(tensor) for ln, tensor in zip(self.fiou_ln_layers, fiou_linear_tensors))

        f, i, o = tuple(torch.sigmoid(tensor) for tensor in fiou_linear_tensors[:3])
        u = torch.tanh(fiou_linear_tensors[3])

        new_cell = self.cell_ln(i * u + (f * cell_tensor))
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell


















class Edge_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, bias=True, aggr="add"):
        super(Edge_GCNConv, self).__init__(aggr=aggr)

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        self.edge_dim = edge_dim
        self.edge_update = torch.nn.Parameter(torch.Tensor(out_channels + edge_dim, out_channels))  # new

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_update)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        x = torch.matmul(x, self.weight)
        edge_weight = torch.ones((edge_index.size(1),),
                                 dtype=x.dtype,
                                 device=edge_index.device)  
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, x.size(0)) 

        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5) 
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        return norm.view(-1, 1) * x_j 

    def update(self, aggr_out):
        aggr_out = torch.mm(aggr_out, self.edge_update)
        if self.bias is not None:
            return aggr_out + self.bias
        else:
            return aggr_out


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


