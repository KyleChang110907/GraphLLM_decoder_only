import torch
print(torch.__version__)
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
import loralib as lora
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from transformers import GemmaModel, GemmaConfig
import bitsandbytes as bnb
from pathlib import Path
import shutil
sys.path.append("../")



from Models.GraphLLM import *
from Models import GraphLLM
from Models.losses import *
from Utils import plot
from Utils import visualize
from Utils import accuracy
from Utils import dataset
from Utils import normalization
from Utils import utils

torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# /home/kyle_chang/time-history-analysis-main/try_reproduce/results/pisa_results/Nonlinear_Dynamic_Analysis_WorldGM_TaiwanSection_BSE-2

def parse_args() -> Namespace:
    parser = ArgumentParser()

    # comment
    parser.add_argument("--comment", type=str, default='time sequence as dimension 1 ,(nodes*batch size, timesteps, features), Graph Pretrained-LLM with LoRA target acc only')

    # checkpoint
    parser.add_argument("--ckpt_dir", type=Path, default="./llm_results/")

    # dataset
    parser.add_argument("--dataset_name", type=str, default='Nonlinear_Dynamic_Analysis_WorldGM_TaiwanSection_BSE-1')
    parser.add_argument("--other_datasets", type=list, default=[])
    parser.add_argument("--whatAsNode", type=str, default='NodeAsNode')
    parser.add_argument("--data_num", type=int, default=50)
    parser.add_argument("--random_sample", type=bool, default=True)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--time_patch", type=int, default=20)
    parser.add_argument("--train_split_ratio", type=list, default=[0.70, 0.2, 0.1])

    # model
    parser.add_argument("--pretrain", type=Path, default=None)
    parser.add_argument("--model", type=str, default='GraphLLM')
    parser.add_argument("--LLM", type=str, default='google/gemma-7b')
    parser.add_argument("--LoRA_rank", type=int, default=32)
    parser.add_argument("--LoRA_alpha", type=int, default=8)
    parser.add_argument("--gnn_num_layers", type=int, default=1)
    parser.add_argument("--head_num", type=int, default=4)
    parser.add_argument("--gnn_hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--t2v_output_dim", type=int, default=64)
    # parser.add_argument("--GraphLLM_hidden_dim", type=int, default=2048)
    # parser.add_argument("--output_time_length", type=int, default=10)

    # parser.add_argument("--graph_lstm_hidden_dim", type=int, default=128)
    # parser.add_argument("--graph_lstm_num_layers", type=int, default=1)
    # parser.add_argument("--node_lstm_hidden_dim", type=int, default=256)
    # parser.add_argument("--node_lstm_num_layers", type=int, default=2)

    # training
    parser.add_argument("--loss_function", type=str, default='MSE_Acc')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sch_factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--target", type=str, default='acc')
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)

    # others
    parser.add_argument("--yield_factor", type=float, default=0.90)
    parser.add_argument("--plot_num", type=int, default=5)
    parser.add_argument("--training_time", type=float, default=0)

    args = parser.parse_args()
    return args



def get_loggings(ckpt_dir):
	logger = logging.getLogger(name='GraphLLM')
	logger.setLevel(level=logging.INFO)
	# set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	# file handler
	file_handler = logging.FileHandler(os.path.join(ckpt_dir, "record.log"))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


def set_random_seed(SEED: int):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def linear_layer_parameterization(layer, device, rank=50, lora_alpha=1):
    # Only add the parameterization to the weight matrix, ignore the Bias

    # From section 4.2 of the paper:
    #   We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity and parameter-efficiency.
    #   [...]
    #   We leave the empirical investigation of [...], and biases to a future work.
    
    features_in, features_out = layer.weight.shape
    return GraphLLM.LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )



def main(args):
    # args, logger setting
    date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    args.ckpt_dir = args.ckpt_dir / args.dataset_name / date_str
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
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
                                       time_patch=args.time_patch,
                                       other_folders=args.other_datasets)
    logger.info(f"Num of structure graph: {len(dset)}")
    logger.info(f"structure_1 graph data: {dset[0]}\n")

    # normalization
    y_start, y_finish, target_dict = utils.get_target_index(args.target)
    dataset_norm, norm_dict = normalization.normalize_dataset(dset, random_sample=args.random_sample, time_patch=args.time_patch)
    logger.info(f"Normlization structure_1 graph: {dataset_norm[0]}")
    logger.info(f"Normalized feastures: \n{norm_dict}\n")

    # spilt into train_dataset, valid dataset and test dataset
    data_num = len(dataset_norm)
    train_ratio, valid_ratio, test_ratio = args.train_split_ratio
    train_num, valid_num = int(data_num*train_ratio), int(data_num*valid_ratio)
    test_num = data_num - train_num - valid_num
    train_index, valid_index, test_index = random_split(list(range(data_num)), [train_num, valid_num, test_num])
    train_dataset = [dataset_norm[i] for i in list(train_index)]
    valid_dataset = [dataset_norm[i] for i in list(valid_index)]
    test_dataset = [dataset_norm[i] for i in list(test_index)]
    logger.info(f"train data: {len(train_dataset)}")
    logger.info(f"valid data: {len(valid_dataset)}")
    logger.info(f"test data: {len(test_dataset)}")
    logger.info("")

    # dataloader
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    miniBatch = next(iter(train_loader))
    logger.info(f"Batch size = {batch_size}")
    logger.info(f"One mini DataBatch for training:\n{miniBatch}")
    logger.info(f"graph sampled_index: {miniBatch.sampled_index}")
    logger.info(f"graph ptr: {miniBatch.ptr}")

    # start index, end index, inedx of target feature dict
    y_start, y_finish, target_dict = utils.get_target_index(args.target)
    logger.info(f"Get predict target's index: start at {y_start}, end at {y_finish}")
    logger.info(f"Target index dictionary: {target_dict}")

    # num of features of data.x
    node_dim = dataset_norm[0].x.shape[1]
    edge_dim = dataset_norm[0].edge_attr.shape[1]
    ground_motion_dim = dataset_norm[0].ground_motions.shape[-1]
    output_dim = y_finish - y_start

    output_time_length = args.timesteps

    # save trainig arguments, norm_dict
    args_temp = deepcopy(args)
    args_temp.ckpt_dir = str(args_temp.ckpt_dir)
    args_temp.pretrain = str(args_temp.pretrain)
    with open(args.ckpt_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args_temp), f)

    norm_dict_save = {}
    for key in norm_dict.keys():
        norm_dict_save[key] = norm_dict[key][1]

    with open(args.ckpt_dir / 'norm_dict.json', 'w') as f:
        json.dump(norm_dict_save, f)
        
    # save test dataset data path
    test_data_paths = [graph.path for graph in test_dataset]
    with open(args.ckpt_dir / 'test_data_path.json', 'w') as f:
        json.dump(test_data_paths, f)

    # LLM
    if args.LLM=='google/gemma-2b':
        model_name = args.LLM
        configs = GemmaConfig.from_pretrained(args.LLM)
        GraphLLM_hidden_dim = 2048
    elif args.LLM=='google/gemma-7b':
        configs = GemmaConfig.from_pretrained(args.LLM)
        GraphLLM_hidden_dim = 3072
    else:
        raise ValueError('No such LLM model or model not tested yet.')

    # construct model
    model_constructor_args = {
        'node_dim': node_dim, 'edge_dim': edge_dim, 'gnn_num_layers': args.gnn_num_layers, 'gnn_hidden_dim': args.gnn_hidden_dim,
        'head_num': args.head_num, 'latent_dim': args.latent_dim, 'GraphLLM_hidden_dim': GraphLLM_hidden_dim,'LoRA_rank': args.LoRA_rank,'LoRA_alpha': args.LoRA_alpha,
        'ground_motion_dim': ground_motion_dim, 'output_dim': output_dim,'output_time_length':output_time_length, 
        'configs' : configs,'model_name':args.LLM,'time_patch':args.time_patch, 'device': device}
    model = globals()[args.model](**model_constructor_args).to(device)
    logger.info(model)

    
    print(args.pretrain)
    if args.pretrain != None:
        print('load pretrained model')
        pretrain_path = Path(f"D:\kyle_MD_project\LLM_decoder_only_acc_2\llm_results\{ args.pretrain}") 
        # pretrain_path = args.pretrain
        print(pretrain_path)
        model.load_state_dict(torch.load(pretrain_path / 'model.pt'))
        logger.info(f"Loaded pretrain model: {pretrain_path}")
        src = pretrain_path/'model.pt'
        dst = args.ckpt_dir/'model.pt'
        shutil.copyfile(src, dst)

    #     # # visualize results of weights w/o lora
    #     # enable_disable_lora(enabled=False)
    #     # visualize response
    #     for structure_index in range(args.plot_num):
    #         print(f"\n Plotting structure {structure_index}:")

    #         visualize.visualize_ground_motion(args.ckpt_dir, test_dataset, "test_non_lora", norm_dict, index=structure_index)
            

    #         for response in ["Acceleration_Z"]:
    #             # visualize.visualize_response(model, args.ckpt_dir, train_dataset, "train", norm_dict, accuracy.R2_score, response=response, index=structure_index)
    #             # visualize.visualize_response(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, accuracy.R2_score, response=response, index=structure_index)
    #             visualize.visualize_response(model, args.ckpt_dir, test_dataset, "test_non_lora", norm_dict, accuracy.R2_score, response=response, index=structure_index)
        

    #     original_weights = {}
    #     for name, param in model.named_parameters():
    #         original_weights[name] = param.clone().detach()
    
    # # Print the size of the weights matrices of the network
    # # Save the count of the total number of parameters
    # # lora_layer = [model.nodeTimeSeriesDecoder.response_decoder.module_list[0],model.nodeTimeSeriesDecoder.response_decoder.module_list[1]]
    # lora_layer = [model.nodeTimeSeriesDecoder.node_encoder.module_list[0]]


    # total_parameters_original = 0
    # for index, layer in enumerate(lora_layer):
    #     total_parameters_original += layer.weight.nelement() + layer.bias.nelement()
    #     print(f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}')
    # print(f'Total number of parameters: {total_parameters_original:,}')
    
    # for layer in lora_layer:
    #     parametrize.register_parametrization(layer, "weight", linear_layer_parameterization(layer, device))
    # # parametrize.register_parametrization(model.nodeTimeSeriesDecoder.response_decoder.module_list[0], "weight", linear_layer_parameterization(model.nodeTimeSeriesDecoder.response_decoder.module_list[0], device))
    # # parametrize.register_parametrization(model.nodeTimeSeriesDecoder.response_decoder.module_list[1], "weight", linear_layer_parameterization(model.nodeTimeSeriesDecoder.response_decoder.module_list[1], device))
    # # parametrize.register_parametrization(model.linear2, "weight", linear_layer_parameterization(model.linear2, device))
    # # parametrize.register_parametrization(model.linear3, "weight", linear_layer_parameterization(model.linear3, device))

    # def enable_disable_lora(enabled=True):
    #     for layer in lora_layer:
    #         layer.parametrizations["weight"][0].enabled = enabled

    # total_parameters_lora = 0
    # total_parameters_non_lora = 0
    # for index, layer in enumerate(lora_layer):
    #     total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
    #     total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
    #     print(
    #         f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations["weight"][0].lora_A.shape} + Lora_B: {layer.parametrizations["weight"][0].lora_B.shape}'
    #     )
    # # The non-LoRA parameters count must match the original network
    # assert total_parameters_non_lora == total_parameters_original
    # print(f'Total number of parameters (original): {total_parameters_non_lora:,}')
    # print(f'Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
    # print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')
    # parameters_incremment = (total_parameters_lora / total_parameters_non_lora) * 100
    # print(f'Parameters incremment: {parameters_incremment:.3f}%')
    
    # for name, param in model.named_parameters():
    #     if 'lora' not in name:
    #         print(f'Freezing non-LoRA parameter {name}')
    #         param.requires_grad = False
    
    # # This sets requires_grad to False for all parameters without the string "lora_" in their name
    # lora.mark_only_lora_as_trainable(model)

    # check model and trainable parameters
    print(model)
    print('check trainable parameters')
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print (name, param.data)
            print (name)
    # loss function
    criterion = globals()[args.loss_function]()

    # optimizer
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.sch_factor, patience=args.patience, min_lr=1e-4)

    # accuracy metrics
    target_R2_record = utils.get_target_accuracy(args, target_dict)
    target_peak_record = utils.get_target_accuracy(args, target_dict)

    # acc, loss record
    epochs = args.epoch_num
    R2_acc_record = np.zeros((3, args.epoch_num))
    peak_acc_record = np.zeros((3, args.epoch_num))
    loss_record = np.zeros((3, args.epoch_num))
    
    best_loss = np.inf
    
    # train
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        R2_acc_train = 0
        peak_acc_train = 0
        loss_train, elem_train = 0, 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ptr, batch.ground_motions)
            x, y = batch.x, batch.y
            # print(f' allocated GPU memory {torch.cuda.memory_allocated()/1024/1024}GB')

            #only acceleration
            output = output[:, :, 0:1]
            y = y[:,:,0:1]
            # print(f'final output size {output.size()}')
            # print(f'final target size {y.size()}')
            # calculate loss
            loss = criterion(output, y)

            # calculate gradient and back propagation
            # clean the gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            # plasticHingeClassifier(x, output, y)
            # plasticHingeClassifier_1F(x, output, y)
            R2_acc = accuracy.R2_score(output, y)
            peak_acc = accuracy.peak_R2_score(output, y)
            target_R2_record = utils.accumulate_target_accuracy(y_start, target_R2_record, target_dict, epoch, accuracy.R2_score, output, y, t_v=0)
            target_peak_record = utils.accumulate_target_accuracy(y_start, target_peak_record, target_dict, epoch, accuracy.peak_R2_score, output, y, t_v=0)
            loss_train += loss.item()
            elem_train += 1
            R2_acc_train += R2_acc
            peak_acc_train += peak_acc
        
        # record accuracy and loss for each epoch
        R2_acc_record[0][epoch] = R2_acc_train / elem_train  
        peak_acc_record[0][epoch] = peak_acc_train / elem_train  
        loss_record[0][epoch] = loss_train / elem_train
        target_R2_record = utils.average_target_accuracy(target_R2_record, epoch, elem_train, t_v=0)
        target_peak_record = utils.average_target_accuracy(target_peak_record, epoch, elem_train, t_v=0)
        

        # Get validation loss
        model.eval()
        with torch.no_grad():
            for t_v, loader in zip([1, 2], [valid_loader, test_loader]):
                R2_acc_valid = 0
                peak_acc_valid = 0
                loss_valid, elem_valid = 0, 0
                for batch in loader:
                    batch = batch.to(device)
                    output= model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ptr, batch.ground_motions)
                    # x, y = batch.x[batch.sampled_index], batch.y[ batch.sampled_index]
                    x, y = batch.x, batch.y
                    #only acceleration
                    output = output[:, :, 0:1]
                    y = y[:,:,0:1]
                    # print(f'validation final output size {output.size()}')
                    # print(f'validation final target size {y.size()}')
                    # calculate loss
                    loss = criterion(output, y)
                    
                    # calculate accuracy

                    # plasticHingeClassifier(x, output, y)
                    # plasticHingeClassifier_1F(x, output, y)

                    R2_acc = accuracy.R2_score(output, y)
                    peak_acc = accuracy.peak_R2_score(output, y)
                    target_R2_record = utils.accumulate_target_accuracy(y_start, target_R2_record, target_dict, epoch, accuracy.R2_score, output, y, t_v=t_v)
                    target_peak_record = utils.accumulate_target_accuracy(y_start, target_peak_record, target_dict, epoch, accuracy.peak_R2_score, output, y, t_v=t_v)
                    loss_valid += loss.item()
                    elem_valid += 1
                    R2_acc_valid += R2_acc
                    peak_acc_valid += peak_acc

                # record accuracy and loss for each epoch
                R2_acc_record[t_v][epoch] = R2_acc_valid / elem_valid
                peak_acc_record[t_v][epoch] = peak_acc_valid / elem_valid
                loss_record[t_v][epoch] = loss_valid / elem_valid
                target_R2_record = utils.average_target_accuracy(target_R2_record, epoch, elem_valid, t_v=t_v)
                target_peak_record = utils.average_target_accuracy(target_peak_record, epoch, elem_valid, t_v=t_v)
                
                # if(t_v == 1):
                #     valid_ps_result_Mz, ps_record_Mz = plasticHingeClassifier.get_accuracy_and_reset(ps_record_Mz, epoch, t_v=t_v)
                #     valid_ps_result_Mz_1F, ps_record_Mz_1F = plasticHingeClassifier_1F.get_accuracy_and_reset(ps_record_Mz_1F, epoch, t_v=t_v)
                # elif(t_v == 2):
                #     test_ps_result_Mz, ps_record_Mz = plasticHingeClassifier.get_accuracy_and_reset(ps_record_Mz, epoch, t_v=t_v)
                #     test_ps_result_Mz_1F, ps_record_Mz_1F = plasticHingeClassifier_1F.get_accuracy_and_reset(ps_record_Mz_1F, epoch, t_v=t_v)

        # learning rate scheduler 
        scheduler.step(loss_record[1][epoch])

        # record trining process to log file
        text = f'Epo: {epoch:03d}, T_Acc: {R2_acc_record[0][epoch]:.4f}, V_Acc: {R2_acc_record[1][epoch]:.4f}, t_Acc: {R2_acc_record[2][epoch]:.4f}, '
        text += f'T_Peak_Acc: {peak_acc_record[0][epoch]:.4f}, V_Peak_Acc: {peak_acc_record[1][epoch]:.4f}, t_Peak_Acc: {peak_acc_record[2][epoch]:.4f}, '
        text += f'T_Loss: {loss_record[0][epoch]:.6f}, V_Loss: {loss_record[1][epoch]:.6f}, t_Loss: {loss_record[2][epoch]:.6f}, '    
        # text += f'T_hinge_node_Mz: {train_ps_result_Mz}, V_hinge_node_Mz: {valid_ps_result_Mz}, t_hinge_node_Mz: {test_ps_result_Mz},  '
        # text += f'T_hinge_node_Mz_1F: {train_ps_result_Mz_1F}, V_hinge_node_Mz_1F: {valid_ps_result_Mz_1F}, t_hinge_node_Mz_1F: {test_ps_result_Mz_1F}'

        logger.critical(text)
        
        # Save model if train_loss is better
        if loss_record[1][epoch] < best_loss:
            best_loss = loss_record[1][epoch]
            torch.save(model.state_dict(), args.ckpt_dir / 'model.pt',_use_new_zipfile_serialization=False)
            text = f'Trained model saved, valid loss: {best_loss:.6f}'
            logger.critical(text)

    # # Check that the frozen parameters are still unchanged by the finetuning
    # assert torch.all(model.nodeTimeSeriesDecoder.response_decoder.module_list.0.parametrizations.weight.original == original_weights['linear1.weight'])
    # assert torch.all(model.nodeTimeSeriesDecoder.response_decoder.module_list.0.parametrizations.weight.original == original_weights['linear2.weight'])

    # record
    finish_time = time.time()
    args.training_time = (finish_time - start_time)/60
    logger.info("")
    logger.info(f"Time spent: {(finish_time - start_time)/60:.2f} min")
    logger.info("Finish time: " + datetime.now().strftime('%b %d, %H:%M:%S'))

    # reload the best model
    model = globals()[args.model](**model_constructor_args).to(device)
    model.load_state_dict(torch.load(args.ckpt_dir / 'model.pt'),strict = False)

    # enable_disable_lora(enabled=True)
    # plot figures
    if args.epoch_num > 0:
        #------------------------------------------------------------
        plot.plot_learningCurve(R2_acc_record, args.ckpt_dir, title=', '.join([args.model, date_str]))
        plot.plot_lossCurve(loss_record, args.ckpt_dir, title=', '.join([args.model, args.loss_function, date_str]))
        plot.plot_target_accuracy(target_R2_record, args.epoch_num, args.ckpt_dir, evaluation="Overall_R2_Score")
        plot.plot_target_accuracy(target_peak_record, args.epoch_num, args.ckpt_dir, evaluation="Peak_R2_Score")
        # # plot.plot_plastic_hinge_accuracy(ps_record_Mz, args.ckpt_dir)
        # # plot.plot_plastic_hinge_accuracy(ps_record_Mz_1F, args.ckpt_dir, specific_location=True)
        worst_case_index, best_case_index = plot.plot_lora_test_accuracy_distribution(test_dataset, model, args.ckpt_dir)
        logger.info(f"worst case index: {worst_case_index}, best case index: {best_case_index}")
        for case_name, case_index in zip(["Worst", "Best"], [worst_case_index, best_case_index]):
            logger.info(f"visualizing {case_name}")
            visualize.visualize_ground_motion(args.ckpt_dir, test_dataset, case_name, norm_dict, index=case_index)
            # visualize.visualize_response(model, args.ckpt_dir, test_dataset, case_name, norm_dict, accuracy.R2_score, response="Displacement_X", index=case_index)
            visualize.visualize_response(model, args.ckpt_dir, test_dataset, case_name, norm_dict, accuracy.R2_score, response="Acceleration_X", index=case_index, output_time_length=output_time_length)
        #------------------------------------------------------------
    # # visualize graph embedding
    # visualize.visualize_graph_embedding(model, args.ckpt_dir, train_dataset, "train", norm_dict, args.batch_size)
    # visualize.visualize_graph_embedding(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, args.batch_size)
    # visualize.visualize_graph_embedding(model, args.ckpt_dir, test_dataset, "test", norm_dict, args.batch_size)

    # visualize response
    for structure_index in range(args.plot_num):
        print(f"\n Plotting structure {structure_index}:")

        visualize.visualize_ground_motion(args.ckpt_dir, train_dataset, "train", norm_dict, index=structure_index)
        visualize.visualize_ground_motion(args.ckpt_dir, valid_dataset, "valid", norm_dict, index=structure_index)
        visualize.visualize_ground_motion(args.ckpt_dir, test_dataset, "test", norm_dict, index=structure_index)

        for response in ["Acceleration_X"]:
            visualize.visualize_response(model, args.ckpt_dir, train_dataset, "train", norm_dict, accuracy.R2_score, response=response, index=structure_index, output_time_length=output_time_length)
            visualize.visualize_response(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, accuracy.R2_score, response=response, index=structure_index, output_time_length=output_time_length)
            visualize.visualize_response(model, args.ckpt_dir, test_dataset, "test", norm_dict, accuracy.R2_score, response=response, index=structure_index, output_time_length=output_time_length)
        
        # # attention weight
        # visualize.visualize_graph_attention(model, args.ckpt_dir, train_dataset, "train", norm_dict, args.head_num, index=structure_index)
        # visualize.visualize_graph_attention(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, args.head_num, index=structure_index)
        # visualize.visualize_graph_attention(model, args.ckpt_dir, test_dataset, "test", norm_dict, args.head_num, index=structure_index)
        
        # plastic hinge
        # visualize.visualize_plasticHinge(model, args.ckpt_dir, train_dataset, "train", norm_dict, plasticHingeClassifier, index=structure_index)
        # visualize.visualize_plasticHinge(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, plasticHingeClassifier, index=structure_index)
        # visualize.visualize_plasticHinge(model, args.ckpt_dir, test_dataset, "test", norm_dict, plasticHingeClassifier, index=structure_index)
    
    # # visualize results of weights w/o lora
    # # enable_disable_lora(enabled=False)
    #  # visualize response
    # for structure_index in range(args.plot_num):
    #     print(f"\n Plotting structure {structure_index}:")

    #     visualize.visualize_ground_motion(args.ckpt_dir, test_dataset, "test_w/o_lora", norm_dict, index=structure_index)
        

    #     for response in ["Acceleration_Z"]:
    #         # visualize.visualize_response(model, args.ckpt_dir, train_dataset, "train", norm_dict, accuracy.R2_score, response=response, index=structure_index)
    #         # visualize.visualize_response(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, accuracy.R2_score, response=response, index=structure_index)
    #         visualize.visualize_response(model, args.ckpt_dir, test_dataset, "test_w/o_lora", norm_dict, accuracy.R2_score, response=response, index=structure_index)
        
        # # attention weight
        # visualize.visualize_graph_attention(model, args.ckpt_dir, train_dataset, "train", norm_dict, args.head_num, index=structure_index)
        # visualize.visualize_graph_attention(model, args.ckpt_dir, valid_dataset, "valid", norm_dict, args.head_num, index=structure_index)
        # visualize.visualize_graph_attention(model, args.ckpt_dir, test_dataset, "test", norm_dict, args.head_num, index=structure_index)


if __name__ == "__main__":
	args = parse_args()
	main(args)