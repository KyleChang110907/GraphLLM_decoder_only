from pathlib import Path
import json
from argparse import ArgumentParser
import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

from Models.GraphLLM import GraphLSTM
from Utils.dataset import GroundMotionDataset
from Utils.animation import make_displacement_animation
from DataGeneration.sections_tw import beam_sections, column_sections



device = "cuda" if torch.cuda.is_available() else "cpu"

# read in trained model and args
ckpt_name = "Nonlinear_Dynamic_Analysis_WorldGM_TaiwanSection_BSE-2/2023_03_26__10_35_37"
model_folder = Path(f"../Results/{ckpt_name}")
print("if model exist:", os.path.exists(model_folder))

model_path = model_folder / "model.pt"
args_path = model_folder / "training_args.json"
norm_path = model_folder / "norm_dict.json"

args_json = json.loads(open(args_path, 'r').read())
args = ArgumentParser().parse_args()
args_dict = vars(args)
args_dict.update(args_json)
print("--- Loading args from:", str(args_path))

norm_dict = json.loads(open(norm_path, 'r').read())
print("--- Loading norm_dict from:", str(norm_path))

model_kwargs = {
        'node_dim': 35, 'edge_dim': 4, 'gnn_num_layers': args.gnn_num_layers, 'gnn_hidden_dim': args.gnn_hidden_dim,
        'head_num': args.head_num, 'latent_dim': args.latent_dim, 'graph_lstm_hidden_dim': args.graph_lstm_hidden_dim,
        'graph_lstm_num_layers': args.graph_lstm_num_layers,'node_lstm_hidden_dim': args.node_lstm_hidden_dim, 
        'node_lstm_num_layers': args.node_lstm_num_layers, 'ground_motion_dim': 20,
        'output_dim': 18, 'device': device}
model = GraphLSTM(**model_kwargs).to(device)
print("--- Loading GraphLSTM from:", str(model_path))
model.load_state_dict(torch.load(model_path))
model.eval()


# load dataset
data_num = 20
dataset_name = "Nonlinear_Dynamic_Analysis_WorldGM_TaiwanSection_BSE-2"
dataset = GroundMotionDataset(folder=dataset_name,
                            graph_type=args.whatAsNode,
                            data_num=data_num,
                            timesteps=args.timesteps,
                            other_folders=[])


def cut_gm(gm):

    gm = gm[:, 0].numpy()
    timestep = gm.shape[0]
    intensity = np.zeros_like(gm)

    # normalize gm (avoid very big intensity)
    gm = gm / np.max(np.abs(gm))

    # calculate intensity
    intensity[0] = gm[0] ** 2
    for i in range(1, len(gm)):
        intensity[i] = intensity[i-1] + gm[i] ** 2

    # normalize intensity to percentage
    intensity = intensity / np.max(intensity) * 100

    # get 5% and 95% timestep index
    threshold = 0.01
    index_start = np.argmin(np.abs(intensity - threshold))
    index_end = np.argmin(np.abs(intensity - (100 - threshold)))

    # pad 5 second before and after
    pad = 5 * 20
    index_start = max(0, index_start - pad)
    index_end = min(timestep, index_end + pad)

    # plt.plot(gm)
    # plt.axvline(index_start, color='r', linewidth=3)
    # plt.axvline(index_end, color='r', linewidth=3)
    # plt.show()

    return index_start, index_end
    


# normalize
def normalize(graph, norm_dict):

    index_start, index_end = cut_gm(graph.ground_motion_1)

    graph.ground_motion_1 = graph.ground_motion_1[index_start:index_end]
    graph.ground_motion_2 = graph.ground_motion_2[index_start:index_end]

    graph.ground_motion_1 /= norm_dict['ground_motion']
    graph.ground_motion_2 /= norm_dict['ground_motion']

    graph.x[:, :3] /= norm_dict['grid_num']
    graph.x[:, 3:6] /= norm_dict['coord']
    graph.x[:, 11:14] /= norm_dict['period']
    graph.x[:, 14:23] /= norm_dict['modal_shape']
   
    graph.x[:, list(range(23, 35, 2))] /= norm_dict['elem_length']
    graph.x[:, list(range(24, 35, 2))] /= norm_dict['moment']

    graph.edge_attr[:, 0] /= norm_dict['elem_length']
    graph.edge_attr[:, 3] /= norm_dict['moment']

    graph.ground_motions = torch.cat([graph.ground_motion_1, graph.ground_motion_2], dim=1)
    graph.ground_motions = graph.ground_motions.unsqueeze(0)
    timesteps = graph.ground_motions.shape[1]
    graph.y = graph.y[:, :timesteps, :]

    return graph

dataset_norm = []
for i in range(len(dataset)):
    dataset_norm.append(normalize(dataset[i], norm_dict))
    print("--- Normalizing data:", i)


# predict
graphs = []
responses = []
for i in range(len(dataset)):
    loader = DataLoader([dataset[i]], batch_size=1, shuffle=False)
    graph = next(iter(loader)).to(device)

    with torch.no_grad():
        response, _ = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch, 
                            graph.ptr, [None], graph.ground_motions, sample_node=False)
        response = response.cpu().numpy()
        graph = graph.cpu()
        print(f"--- Response {i} shape:", response.shape)

    graphs.append(graph)
    responses.append(response)


def denormalize(graph, response, norm_dict):
    graph.ground_motion_1 *= norm_dict['ground_motion']
    graph.ground_motion_2 *= norm_dict['ground_motion']

    graph.x[:, :3] *= norm_dict['grid_num']
    graph.x[:, 3:6] *= norm_dict['coord']
    graph.x[:, 11:14] *= norm_dict['period']
    graph.x[:, 14:23] *= norm_dict['modal_shape']
   
    graph.x[:, list(range(23, 35, 2))] *= norm_dict['elem_length']
    graph.x[:, list(range(24, 35, 2))] *= norm_dict['moment']

    graph.edge_attr[:, 0] *= norm_dict['elem_length']
    graph.edge_attr[:, 3] *= norm_dict['moment']

    response[:, :, 0:2] *= norm_dict['acc']
    response[:, :, 2:4] *= norm_dict['vel']
    response[:, :, 4:6] *= norm_dict['disp']
    response[:, :, 6:12] *= norm_dict['moment']
    response[:, :, 12:18] *= norm_dict['shear']

    return graph, response


all_sections = beam_sections + column_sections
all_section_names = [sec['name'] for sec in all_sections]
all_section_Mys = np.array([sec['My_z(kN-mm)'] for sec in all_sections])
# print("all_seciton_names:", all_section_names)
# print("all_section_Mys:", all_section_Mys)
def get_section_name_from_My(My):
    residual = np.abs(My - all_section_Mys)
    nearest_index = np.argmin(residual)
    nearest_section_name = all_section_names[nearest_index]
    # print(f"My: {My}, section_name: {nearest_section_name}")
    return nearest_section_name


# convert current graph to geoemtric graph that contain more geo information
def to_geometric_graph(graph):
    x_grid_num = int(graph.x[0, 0].item())
    z_grid_num = int(graph.x[0, 2].item())
    node_per_story = x_grid_num * z_grid_num
    target_2F_node = node_per_story
    x_span_len = int(graph.x[target_2F_node, 25]) * 1000
    z_span_len = int(graph.x[target_2F_node, 33]) * 1000
    assert x_span_len != 0 and z_span_len != 0
    
    # node
    geo_x = np.zeros((graph.x.shape[0], 3))
    for i in range(graph.x.shape[0]):
        x = int(graph.x[i, 3].item()) * x_span_len
        z = int(graph.x[i, 5].item()) * z_span_len
        y_grid_index = int(graph.x[i, 4].item())
        if y_grid_index == 0:
            y = 0
        else:
            y = 4200 + (y_grid_index - 1) * 3200
        
        geo_x[i, 0] = x
        geo_x[i, 1] = y
        geo_x[i, 2] = z

    # edge
    geo_edge = []
    for i, j in graph.edge_index.T:
        if set([i.item(), j.item()]) not in geo_edge:
            geo_edge.append(set([i.item(), j.item()]))
    geo_edge = [list(s) for s in geo_edge]

    # member section
    section_names = []
    for (node_i, node_j) in geo_edge:
        if node_i > node_j:
            node_i, node_j = node_j, node_i
        # find relation of two node
        x1, y1, z1 = geo_x[node_i]
        x2, y2, z2 = geo_x[node_j]
        # print(f"i: {node_i}, j: {node_j}, x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}, z1: {z1}, z2: {z2}")
        assert x1<=x2 and y1<=y2 and z1<=z2

        if x1 != x2:
            moment1 = graph.x[node_i, 26] # xp
            moment2 = graph.x[node_j, 24] # xn
            assert moment1 == moment2
        elif y1 != y2:
            moment1 = graph.x[node_i, 30] # yp
            moment2 = graph.x[node_j, 28] # yn
            assert moment1 == moment2
        elif z1 != z2:
            moment1 = graph.x[node_i, 34] # zp
            moment2 = graph.x[node_j, 32] # zn
            assert moment1 == moment2
        
        My = moment1.item()
        section_name = get_section_name_from_My(My)
        section_names.append(section_name)

    geo_graph = {'geo_x': geo_x, 'geo_edge': geo_edge, 
                 'section_names': section_names, 
                 'node_per_story': node_per_story,
                 'gm1': graph.ground_motion_1.numpy() / 1000 / 9.8,
                 'gm2': graph.ground_motion_2.numpy() / 1000 / 9.8}
                 
    return geo_graph



# make animation
def make_one_structure_animation(graph, response, save_dir_i):
    # first denormalize
    graph, response = denormalize(graph, response, norm_dict)

    # convert to geo-graph
    geo_graph = to_geometric_graph(graph)

    # visualize displacement
    make_displacement_animation(graph, geo_graph, response, save_dir_i)


save_dir = model_folder / "animation"
save_dir.mkdir(parents=True, exist_ok=True)
for i in range(len(graphs)):
    save_dir_i = save_dir / str(i)
    make_one_structure_animation(graphs[i], responses[i], save_dir_i)
