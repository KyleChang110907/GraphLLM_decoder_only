import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import global_mean_pool
import torch.utils.checkpoint as cp
from .layers import *
from LLM import load_LLM
import torch.nn.functional as F

import loralib as lora

class GraphLatentEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, gnn_num_layers, head_num, gnn_hidden_dim, latent_dim):
        super(GraphLatentEncoder, self).__init__()
        self.gnn_num_layers = gnn_num_layers
        self.conv_layers = nn.ModuleList()
        for i in range(gnn_num_layers):
            in_dim = node_dim if i == 0 else gnn_hidden_dim
            out_dim = latent_dim if i == (gnn_num_layers - 1) else gnn_hidden_dim
            self.conv_layers.append(tg.nn.GATv2Conv(in_dim, out_dim, heads=head_num, concat=False, edge_dim=edge_dim))

    def forward(self, x, edge_index, edge_attr, batch):

        for i in range(self.gnn_num_layers):
            if i == 0:
                x, (return_edge_index, attention_weights) = self.conv_layers[i](x, edge_index, edge_attr, return_attention_weights=True)
            else:
                x = self.conv_layers[i](x, edge_index, edge_attr)

        latent = global_mean_pool(x, batch)
        node_embeddings = x
        
        return latent, return_edge_index, attention_weights, node_embeddings





# revised for LLM
class GraphNodeTimeSeriesEncoder(nn.Module):
    def __init__(self, latent_dim,  node_dim,GraphLLM_hidden_dim,output_time_length,time_patch,device):
        super(GraphNodeTimeSeriesEncoder, self).__init__()

        self.device = device
        self.time_patch = time_patch
        self.input_dim = 2*latent_dim + 2*time_patch + 1
        self.node_dim = node_dim
        self.latent_dim = latent_dim
        self.output_time_length = output_time_length

        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.encoder = nn.Linear(self.input_dim, GraphLLM_hidden_dim)
        
    def create_ground_motion_graph(self, ground_motion_x,ground_motion_y, node_embeddings, latent, ptr):
        bs = len(ptr) - 1
        x = torch.zeros(ptr[bs],self.output_time_length , self.input_dim).to(self.device)
        
        for b in range(bs):

            for node in range(ptr[b],ptr[b+1]):

                for time in range(self.output_time_length):

                    # node embeddings and graph embeddings
                    x[node, time, 0:self.latent_dim] = node_embeddings[node].view(1,1,-1)
                    x[node, time, self.latent_dim : 2 *self.latent_dim ] = latent[b].view(1,1,-1)
            
                    # gronund motion in x and y direction
                    x[node, time, -(2*self.time_patch+1) : -(self.time_patch+1)] = ground_motion_x[b,time,:]
                    x[node, time, -(self.time_patch+1) : -1] = ground_motion_y[b,time,:]

                    # positional feature for time
                    x[node, time, -1 : ] = time/self.output_time_length

        
        x = self.layer_norm(x)
        x = self.encoder(x)
        x = F.relu(x)
        
        return x   

    def forward(self, ground_motion_x,ground_motion_y, node_embeddings, latent, ptr):
        x = self.create_ground_motion_graph(ground_motion_x,ground_motion_y, node_embeddings, latent, ptr)
        
        return x

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)
    


class NodeTimeSeriesDecoder(nn.Module):
    def __init__(self, node_dim, GraphLLM_hidden_dim, ground_motion_dim,
                 output_time_length, device):
        super(NodeTimeSeriesDecoder, self).__init__()

        self.node_dim = node_dim
        self.GraphLLM_hidden_dim = GraphLLM_hidden_dim
        self.ground_motion_dim = ground_motion_dim
        
        self.output_dim = output_time_length
        self.device = device

        self.response_decoder =nn.Linear(GraphLLM_hidden_dim, 1)
        # self.response_decoder =nn.Linear(GraphLLM_hidden_dim, output_time_length)
        

    def forward(self, LLM_output):
        

        node_out = self.response_decoder(LLM_output).to(self.device)

        return node_out



class GraphLLM(nn.Module):
    def __init__(self, node_dim, edge_dim, gnn_num_layers, head_num, gnn_hidden_dim,
                 latent_dim,GraphLLM_hidden_dim,LoRA_rank,LoRA_alpha,ground_motion_dim, output_dim,output_time_length,configs,model_name,time_patch,device):
        super(GraphLLM, self).__init__()

        self.device = device
        self.time_patch = time_patch
        self.output_time_length = output_time_length
        self.graphLatentEncoder = GraphLatentEncoder(node_dim, edge_dim, gnn_num_layers, head_num, gnn_hidden_dim, latent_dim)
        self.graphNodeTimeSeriesEncoder = GraphNodeTimeSeriesEncoder(latent_dim, node_dim,GraphLLM_hidden_dim,output_time_length,time_patch,device)

        input_dim = 2*time_patch + 2*latent_dim +1
        self.llM_QLoRA = load_LLM.LLM_PEFT(configs,input_dim,output_dim,LoRA_alpha,LoRA_rank,model_name,device)
        
        self.nodeTimeSeriesDecoder = NodeTimeSeriesDecoder(node_dim, GraphLLM_hidden_dim, ground_motion_dim,
                 output_time_length, device)



    def forward(self, x, edge_index, edge_attr, batch, ptr, ground_motions):
        
        # graph latent
        latent, _, _, node_embeddings = self.graphLatentEncoder(x, edge_index, edge_attr, batch)
   

        ground_motion_x = ground_motions[:,:,0:self.time_patch]
        ground_motion_y = ground_motions[:,:,self.time_patch:2*self.time_patch]
        
        # graph + 2 dir groundmotion + node as input of pretrained LLM
        inputs_embeds_1 = self.graphNodeTimeSeriesEncoder( ground_motion_x,ground_motion_y, node_embeddings, latent, ptr)

        inputs_embeds = inputs_embeds_1

        # finetune pretrained LLM model through LoRA
        LLM_output = self.llM_QLoRA (inputs_embeds)

        output = self.nodeTimeSeriesDecoder(LLM_output)
        
        output = output.reshape(-1,self.output_time_length).unsqueeze(2)
        
        return output

