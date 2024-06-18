import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import random
import os
from os import listdir
from os.path import join
from tqdm import tqdm
import sys
# sys.path.append("../")


class GroundMotionDataset(Dataset):
    def __init__(self, folder="Nonlinear_Dynamic_Analysis_WorldGM_TaiwanSection_BSE-1", graph_type="NodeAsNode", data_num=1000, timesteps=1000, time_patch=10, other_folders=[]):
        self.root = "D:\kyle_MD_project\LLM_decoder_only_acc\Data\\"
        folder = join(self.root, folder)
        other_folders = [join(self.root, other_folder) for other_folder in other_folders]
        self.folder = folder
        self.other_folders = other_folders
        self.data_num = data_num
        self.graphs = self.load(self.folder, graph_type, data_num, timesteps, time_patch)


    def load(self, folder, graph_type, data_num, timesteps, time_patch):
        max_steps = 20000
        graphs = []

        random.seed(731)
        all_folders = os.listdir(folder)
        all_folders = [join(folder, f) for f in all_folders]
        for other_folder in self.other_folders:
            other_folders = os.listdir(other_folder)
            other_folders = [join(other_folder, f) for f in other_folders]
            all_folders += other_folders
        random.shuffle(all_folders)
        selected_folders = all_folders[:data_num]
        for folder_name in tqdm(selected_folders):
            gm_path_1 = join(folder_name, "ground_motion_1.txt")
            gm_path_2 = join(folder_name, "ground_motion_2.txt")
            graph_path = join(folder_name, f"structure_graph_{graph_type}.pt")
            if os.path.exists(graph_path) == False:
                print(f"There's no {graph_path}!")
                continue
            ground_motion_1 = torch.zeros((max_steps, time_patch))
            ground_motion_2 = torch.zeros((max_steps, time_patch))
            count = 0
            with open(gm_path_1, "r") as f:
                for index, line in enumerate(f.readlines()):
                    i, j = index//time_patch, index%time_patch
                    ground_motion_1[i, j] = float(line.split()[1])
            with open(gm_path_2, "r") as f:
                for index, line in enumerate(f.readlines()):
                    i, j = index//time_patch, index%time_patch
                    ground_motion_2[i, j] = float(line.split()[1])
            graph = torch.load(graph_path)
            graph.ground_motion_1 = ground_motion_1[:timesteps, :]
            graph.ground_motion_2 = ground_motion_2[:timesteps, :]
            graphs.append(graph)

            # print(f' ground_motion_1: {graph.ground_motion_1.shape}, ground_motion_2: {graph.ground_motion_2.shape}')

            # AbsAcc = RelAcc + GroundMotionAcc (PISA outputs response every 10 steps: 0, 10, 20,...)
            new_y = torch.zeros(graph.y.shape)
            new_y[:, :timesteps, 0] = graph.y[:, :timesteps, 0] + graph.ground_motion_1[:, 0]  # torch.mean(graph.ground_motion_1, dim=1)
            new_y[:, :timesteps, 1] = graph.y[:, :timesteps, 1] + graph.ground_motion_2[:, 0]  # torch.mean(graph.ground_motion_2, dim=1)
            new_y[:, :timesteps, 2:] = graph.y[:, :timesteps, 2:]
            graph.y = new_y

        return graphs


    def __len__(self):
        return len(self.graphs)


    def __getitem__(self, i):
        return self.graphs[i]


