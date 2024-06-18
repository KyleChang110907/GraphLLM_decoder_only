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
    def __init__(self, folder="exp_graph_3", graph_type="NodeAsNode", data_num=50, timesteps=1000, other_folders=[]):
        self.root = "/home/kyle_chang/EXP/5F//"
        folder = join(self.root, folder)
        other_folders = [join(self.root, other_folder) for other_folder in other_folders]
        self.folder = folder
        self.other_folders = other_folders
        self.data_num = data_num
        self.graphs = self.load(self.folder, graph_type, data_num, timesteps)


    def load(self, folder, graph_type, data_num, timesteps):
        max_steps = 1500
        batch = 10
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
            gm_path_1 = join(folder_name, "ground_motion.txt")
            # gm_path_2 = join(folder_name, "ground_motion_2.txt")
            graph_path = join(folder_name, f"structure_graph_{graph_type}.pt")
            if os.path.exists(graph_path) == False:
                print(f"There's no {graph_path}!")
                continue
            ground_motion = torch.zeros((max_steps, batch))
            # ground_motion_2 = torch.zeros((max_steps, batch))
            count = 0
            with open(gm_path_1, "r") as f:
                for index, line in enumerate(f.readlines()):
                    if count == 0:
                        count+=1
                        continue
                    else:
                        i, j = index//10, index%10
                        ground_motion[i, j] = float(line.split(',')[1])
            # with open(gm_path_2, "r") as f:
            #     for index, line in enumerate(f.readlines()):
            #         i, j = index//10, index%10
            #         ground_motion_2[i, j] = float(line.split()[1])
            graph = torch.load(graph_path)
            graph.ground_motion = ground_motion[:timesteps, :]
            # graph.ground_motion_2 = ground_motion_2[:timesteps, :]
            graphs.append(graph)
        return graphs


    def __len__(self):
        return len(self.graphs)


    def __getitem__(self, i):
        return self.graphs[i]


