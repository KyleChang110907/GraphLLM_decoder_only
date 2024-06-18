import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
import random
import sys
sys.path.insert(0, "E:/TimeHistoryAnalysis/Time-History-Analysis/")
sys.path.append("../")

from Utils import dataset
from Utils import normalization
from Models.layers import *

from captum.attr import IntegratedGradients




class LSTM_interpretability(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device, gm_1, gm_2, ptr, x):
        super(LSTM_interpretability, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = device

        self.gm_1 = gm_1
        self.gm_2 = gm_2
        self.ptr = ptr
        self.x = x

        self.graph_encoder = MLP(self.input_dim, [], self.hidden_dim, act=True, dropout=False)
        self.lstmCellList = nn.ModuleList()
        for i in range(num_layers):
            self.lstmCellList.append(nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim))
        self.response_decoder = MLP(self.hidden_dim*2, [64], self.output_dim, act=True, dropout=False)


    def create_ground_motion_graph(self, gm_1, gm_2, graph_node, ptr):
        bs = len(ptr)-1
        gm_graph = graph_node.clone()
        for b in range(bs):
            gm_graph[ptr[b]:ptr[b+1], -20:-10] = gm_1[b*10 : (b+1)*10]
            gm_graph[ptr[b]:ptr[b+1], -10:] = gm_2[b*10 : (b+1)*10]
        gm_graph = self.graph_encoder(gm_graph)
        return gm_graph

    def set_hidden_state(self, graph, H):
        if H is None:
            H = self.graph_encoder(graph)
        return H

    def set_cell_state(self, graph, C):
        if C is None:
            C = self.graph_encoder(graph)
        return C

    def next_cell_input(self, H, gm_1, gm_2, ptr):
        H_gm = H.clone()
        bs = len(ptr)-1
        for b in range(bs):
            H_gm[ptr[b]:ptr[b+1], -20:-10] = gm_1[b*10 : (b+1)*10]
            H_gm[ptr[b]:ptr[b+1], -10:] = gm_2[b*10 : (b+1)*10]
        return H_gm

    def create_response(self, H, C):
        state = torch.cat([H, C], dim=1)
        node_out = self.response_decoder(state)
        return node_out


    def forward(self, graph_node):
        '''
        gm    : ground motion at time t [10*batch_size]
        graph : x[node_num(batch), feature_num]
        H     : hidden state at time t [node_num(batch), hidden_feature]
        C     : cell state at time t [node_num(batch), hidden_feature]
        '''

        X = self.create_ground_motion_graph(self.gm_1, self.gm_2, graph_node, self.ptr)

        H_0 = self.graph_encoder(self.x)
        H_1 = self.graph_encoder(self.x)
        C_0 = self.graph_encoder(self.x)
        C_1 = self.graph_encoder(self.x)

        for i in range(self.num_layers):
            if i == 0:
                H_0, C_0 = self.lstmCellList[i](X, (H_0, C_0))
            else:
                H_1, C_1 = self.lstmCellList[i](self.next_cell_input(H_0, self.gm_1, self.gm_2, self.ptr),
                                                (H_1, C_1))

        y = self.create_response(H_1, C_1)

        return y








# %%
save_model_dir = "../Results/Nonlinear_Dynamic_Analysis_World_Taipei3_Design/2022_08_09__00_30_59/"

args_path = save_model_dir + "training_args.json"
args = json.load(open(args_path, 'r'))
args["data_num"] = 330
args["batch_size"] = 300

# hyperparameters
print("hyperparameters:")
print(args)

# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU_name = torch.cuda.get_device_name()



SEED = 731
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


norm_path = save_model_dir + "norm_dict.json"
norm_dict = json.load(open(norm_path, 'r'))

# Dataset object which contains args.data_num structure graphs
dataset = dataset.GroundMotionDataset(folder=args["dataset_name"],
                                      graph_type=args["whatAsNode"],
                                      data_num=args["data_num"],
                                      other_folders=args["other_datasets"])



dataset_norm, dataset_sampled, norm_dict = normalization.normalize_with_normDict(norm_dict, dataset, args["reduce_sample"], args["yield_factor"])
final_dataset = dataset_sampled if args["sample_node"] else dataset_norm
print("dataset length:", len(final_dataset))
dataloader = DataLoader(final_dataset, batch_size=args["batch_size"], shuffle=False)


graph = next(iter(dataloader))
print("graph.x.shape:", graph.x.shape)
graph = graph.to(device)
graph.ground_motion_1 = graph.ground_motion_1.permute(1, 0)
graph.ground_motion_2 = graph.ground_motion_2.permute(1, 0)

gm_1 = graph.ground_motion_1[500]
gm_2 = graph.ground_motion_2[500]



model_constructor_args = {
    'input_dim': 47, 'hidden_dim': args["hidden_dim"], 'output_dim': 8,
    'num_layers': args["num_layers"], 'device': device, 'gm_1': gm_1, 'gm_2': gm_2, 'ptr': graph.ptr, 'x': graph.x}

model = LSTM_interpretability(**model_constructor_args).to(device)
model.load_state_dict(torch.load(save_model_dir + 'model.pt'))
model.eval()



ig = IntegratedGradients(model)
ig_attr_test = ig.attribute(graph.x, n_steps=1, target=0)
ig_attr_test = ig_attr_test[:, :27]
print("ig_attr_test shape:", ig_attr_test.shape)

# y = model(graph.x)
# print("y shape:", y.shape)

ig_attr_test_sum = ig_attr_test.cpu().detach().numpy().sum(0)
ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)



# visualize
x_axis_data = np.arange(27)
x_axis_data_labels = ['x_grid_num', 'y_grid_num', 'z_grid_num', 'x_coord', 'y_coord', 'z_coord', 'period', 'if_bottom', 'if_top', 'if_side', 'beta_x', 'beta_z',
                      'Ux', 'Uz', 'Ry', 'x_n_len', 'x_n_My', 'x_p_len', 'x_p_My', 'y_n_len', 'y_n_My', 'y_p_len', 'y_p_My', 'z_n_len', 'z_n_My', 'z_p_len', 'z_p_My']


ig_attr_test_norm_sum_sort, feature_index_sort = torch.sort(torch.tensor(ig_attr_test_norm_sum).abs(), descending=True)
x_axis_data_labels_sort = [x_axis_data_labels[i] for i in feature_index_sort]

colors = ['b' if val < 0 else 'r' for val in ig_attr_test_norm_sum]
colors_sort = [colors[i] for i in feature_index_sort]

ax = plt.subplot()
plt.rcParams['font.size'] = '16'
ax.set_title(f'Top 10 Feature Importance of displacement_X prediction \n(from {args["batch_size"]} structures)')
ax.set_xlabel('Attributions', fontsize=17)
ax.set_ylabel('Importance', fontsize=17)
# ax.bar(x_axis_data[:10], ig_attr_test_norm_sum_sort[:10], align='center', alpha=0.8, color=colors_sort[:10])
ax.bar(x_axis_data[:10], ig_attr_test_norm_sum_sort[:10], align='center', alpha=0.8)

ax.set_xticks(x_axis_data[:10])
ax.set_xticklabels(x_axis_data_labels_sort[:10], rotation=45, fontsize=15)

# blue_patch = mpatches.Patch(color='blue', label='positive effect')
# red_patch = mpatches.Patch(color='red', label='negative effect')
# plt.legend(handles=[blue_patch, red_patch])

# plt.savefig("Interpretability/feature_imprtance.png")
plt.show()