import torch
from torch.utils.data import random_split
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

import torch.optim as optim
import numpy as np
import time
import random
from datetime import datetime
import os
import json
import logging
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from pathlib import Path
from copy import deepcopy
from typing import List
import sys
sys.path.append("../")


from Models.GraphLLM import *
from Models.losses import *
from Utils import plot
from Utils import visualize
from Utils import accuracy
from Utils import dataset
from Utils import normalization
from Utils import utils




def parse_args() -> Namespace:
    parser = ArgumentParser()

    # comment
    parser.add_argument("--comment", type=str, default='')

    # checkpoint
    parser.add_argument("--ckpt_dir", type=Path, default="../Results/")

    # dataset
    parser.add_argument("--dataset_name", type=str, default='Nonlinear_Dynamic_Analysis_World_Full_BSE-2')
    parser.add_argument("--other_datasets", type=list, default=[])
    parser.add_argument("--whatAsNode", type=str, default='NodeAsNode')
    parser.add_argument("--data_num", type=int, default=300)
    parser.add_argument("--random_sample", type=bool, default=True)
    parser.add_argument("--timesteps", type=int, default=1400)
    parser.add_argument("--train_split_ratio", type=list, default=[0.10, 0.10, 0.80])

    # model
    parser.add_argument("--pretrain", type=Path, default=None)
    parser.add_argument("--model", type=str, default='GraphLSTM')
    parser.add_argument("--gnn_num_layers", type=int, default=1)
    parser.add_argument("--head_num", type=int, default=1)
    parser.add_argument("--gnn_hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--graph_lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--graph_lstm_num_layers", type=int, default=1)
    parser.add_argument("--node_lstm_hidden_dim", type=int, default=256)
    parser.add_argument("--node_lstm_num_layers", type=int, default=2)

    # training
    parser.add_argument("--loss_function", type=str, default='MSE')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sch_factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--target", type=str, default='acc_vel_disp_Mz_Sy')
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)

    # others
    parser.add_argument("--yield_factor", type=float, default=0.90)
    parser.add_argument("--plot_num", type=int, default=2)
    parser.add_argument("--training_time", type=float, default=0)

    args = parser.parse_args()
    return args



def get_loggings(ckpt_dir):
	logger = logging.getLogger(name='GraphLSTM')
	logger.setLevel(level=logging.INFO)
	# set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	# file handler
	# file_handler = logging.FileHandler(os.path.join(ckpt_dir, "record.log"))
	# file_handler.setFormatter(formatter)
	# logger.addHandler(file_handler)
	return logger


def set_random_seed(SEED: int):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True



def main(args):
    # args, logger setting
    args.ckpt_dir = args.ckpt_dir / args.dataset_name / "2022_11_15__18_17_03"
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = get_loggings(args.ckpt_dir)
    logger.info(f"ckpt_dir: {args.ckpt_dir}")
    logger.info(args)

    # set random seed
    SEED = 731
    set_random_seed(SEED)

    # device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    GPU_name = torch.cuda.get_device_name()
    logger.info(f"My GPU is {GPU_name}\n")

    # dataset
    dset = dataset.GroundMotionDataset(folder=args.dataset_name,
                                       graph_type=args.whatAsNode,
                                       data_num=args.data_num,
                                       timesteps=args.timesteps,
                                       other_folders=args.other_datasets)
    logger.info(f"Num of structure graph: {len(dset)}")
    logger.info(f"structure_1 graph data: {dset[0]}\n")

    # normalization
    y_start, y_finish, target_dict = utils.get_target_index(args.target)
    dataset_norm, norm_dict = normalization.normalize_dataset(dset, random_sample=args.random_sample)
    logger.info(f"Normlization structure_1 graph: {dataset_norm[0]}")
    logger.info(f"Normalized feastures: \n{norm_dict}\n")

    # spilt into train_dataset, valid dataset and test dataset
    data_num = len(dataset_norm)
    train_ratio, valid_ratio, test_ratio = args.train_split_ratio
    train_num, valid_num = int(data_num*train_ratio), int(data_num*valid_ratio)
    test_num = data_num - train_num - valid_num
    train_index, valid_index, test_index = random_split(list(range(data_num)), [train_num, valid_num, test_num])
    # train_dataset = [dataset_norm[i] for i in list(train_index)]
    # valid_dataset = [dataset_norm[i] for i in list(valid_index)]
    test_dataset = [dataset_norm[i] for i in list(test_index)]
    # logger.info(f"train data: {len(train_dataset)}")
    # logger.info(f"valid data: {len(valid_dataset)}")
    # logger.info(f"test data: {len(test_dataset)}")
    # logger.info("")

    # dataloader
    batch_size = args.batch_size
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # start index, end index, inedx of target feature dict
    y_start, y_finish, target_dict = utils.get_target_index(args.target)
    logger.info(f"Get predict target's index: start at {y_start}, end at {y_finish}")
    logger.info(f"Target index dictionary: {target_dict}")

    # num of features of data.x
    node_dim = dataset_norm[0].x.shape[1]
    edge_dim = dataset_norm[0].edge_attr.shape[1]
    ground_motion_dim = dataset_norm[0].ground_motions.shape[-1]
    output_dim = y_finish - y_start

    # save trainig arguments, norm_dict
    # args_temp = deepcopy(args)
    # args_temp.ckpt_dir = str(args_temp.ckpt_dir)
    # with open(args.ckpt_dir / 'training_args.json', 'w') as f:
    #     json.dump(vars(args_temp), f)

    # norm_dict_save = {}
    # for key in norm_dict.keys():
    #     norm_dict_save[key] = norm_dict[key][1]

    # with open(args.ckpt_dir / 'norm_dict.json', 'w') as f:
    #     json.dump(norm_dict_save, f)
        
    # save test dataset data path
    # test_data_paths = [graph.path for graph in test_dataset]
    # logger.info("")
    # logger.info("test data path:")
    # logger.info(test_data_paths)
    # with open(args.ckpt_dir / 'test_data_path.json', 'w') as f:
    #     json.dump(test_data_paths, f)

    # construct model
    model_constructor_args = {
        'node_dim': node_dim, 'edge_dim': edge_dim, 'gnn_num_layers': args.gnn_num_layers, 'gnn_hidden_dim': args.gnn_hidden_dim,
        'head_num': args.head_num, 'latent_dim': args.latent_dim, 'graph_lstm_hidden_dim': args.graph_lstm_hidden_dim,
        'graph_lstm_num_layers': args.graph_lstm_num_layers,'node_lstm_hidden_dim': args.node_lstm_hidden_dim, 
        'node_lstm_num_layers': args.node_lstm_num_layers, 'ground_motion_dim': ground_motion_dim,
        'output_dim': output_dim, 'device': device}
    model = globals()[args.model](**model_constructor_args).to(device)
    model.load_state_dict(torch.load(args.ckpt_dir / 'model.pt'))

    # plasticHingeClassifier = accuracy.PlasticHingeClassifier(yield_factor=args.yield_factor)
    # plasticHingeClassifier_1F = accuracy.PlasticHingeClassifier(yield_factor=args.yield_factor, specific_location=True, norm_dict=norm_dict)


    print("test dataset len:", len(test_dataset))
    import time
    start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            print("aa")
            batch = batch.to(device)
            output, keeped_indexes = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ptr, batch.sampled_index, batch.ground_motions)
    finish = time.time()
    print()
    print("use time:", finish - start)
    print("avg time:", (finish - start) / len(test_dataset))



    '''
    # plot
    worst_case_index, best_case_index = plot.plot_test_accuracy_distribution(test_dataset, model, args.ckpt_dir)
    logger.info(f"worst case index: {worst_case_index}, best case index: {best_case_index}")
    for case_name, case_index in zip(["Worst", "Best"], [worst_case_index, best_case_index]):
        logger.info(f"visualizing {case_name}")
        visualize.visualize_ground_motion(args.ckpt_dir, test_dataset, case_name, norm_dict, index=case_index)
        visualize.visualize_response(model, args.ckpt_dir, test_dataset, case_name, norm_dict, accuracy.R2_score, response="Displacement_X", index=case_index)
        visualize.visualize_response(model, args.ckpt_dir, test_dataset, case_name, norm_dict, accuracy.R2_score, response="Displacement_Z", index=case_index)

    # visualize graph embedding
    visualize.visualize_graph_embedding(model, args.ckpt_dir, train_dataset, "train", norm_dict, args.batch_size)
    visualize.visualize_graph_embedding(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, args.batch_size)
    visualize.visualize_graph_embedding(model, args.ckpt_dir, test_dataset, "test", norm_dict, args.batch_size)

    # visualize response
    for structure_index in range(args.plot_num):
        print(f"\n Plotting structure {structure_index}:")

        visualize.visualize_ground_motion(args.ckpt_dir, train_dataset, "train", norm_dict, index=structure_index)
        visualize.visualize_ground_motion(args.ckpt_dir, valid_dataset, "valid", norm_dict, index=structure_index)
        visualize.visualize_ground_motion(args.ckpt_dir, test_dataset, "test", norm_dict, index=structure_index)

        for response in ["Displacement_X", "Displacement_Z", "Moment_Z_Column", "Moment_Z_Xbeam", "Moment_Z_Zbeam"]:
            visualize.visualize_response(model, args.ckpt_dir, train_dataset, "train", norm_dict, accuracy.R2_score, response=response, index=structure_index)
            visualize.visualize_response(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, accuracy.R2_score, response=response, index=structure_index)
            visualize.visualize_response(model, args.ckpt_dir, test_dataset, "test", norm_dict, accuracy.R2_score, response=response, index=structure_index)
        
        # attention weight
        visualize.visualize_graph_attention(model, args.ckpt_dir, train_dataset, "train", norm_dict, index=structure_index)
        visualize.visualize_graph_attention(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, index=structure_index)
        visualize.visualize_graph_attention(model, args.ckpt_dir, test_dataset, "test", norm_dict, index=structure_index)
        
        # plastic hinge
        visualize.visualize_plasticHinge(model, args.ckpt_dir, train_dataset, "train", norm_dict, plasticHingeClassifier, index=structure_index)
        visualize.visualize_plasticHinge(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, plasticHingeClassifier, index=structure_index)
        visualize.visualize_plasticHinge(model, args.ckpt_dir, test_dataset, "test", norm_dict, plasticHingeClassifier, index=structure_index)
    '''



if __name__ == "__main__":
	args = parse_args()
	main(args)