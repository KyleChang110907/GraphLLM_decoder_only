import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from torch_geometric.loader import DataLoader
import os
import json
from copy import deepcopy
import sys
sys.path.insert(0, "E:/TimeHistoryAnalysis/Time-History-Analysis/")
sys.path.append("../")
from Utils import dataset as dset
from Utils import normalization
from Models.GraphLLM import *


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)


def Hbeta(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 50 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    # this part may be some difference for complex eigenvalue
    # but complex eignevalue is meanless here, so they are replaced by their real part
    i = 0
    while i < d:
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 2
        else:
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 50 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y



def load_embedding(save_model_dir: str, data_number: int, location: str = "graph_encoder"):
    # random_seed
    SEED = 1021
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # device, args, norm_dict
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args_path = save_model_dir + "training_args.json"
    args = json.load(open(args_path, 'r'))
    args["data_num"] = int(data_number * 1.1)   # incase no graph for some cases
    args["batch_size"] = 1
    norm_path = save_model_dir + "norm_dict.json"
    norm_dict = json.load(open(norm_path, 'r'))

    # dataset, dataloder
    dataset = dset.GroundMotionDataset(folder=args["dataset_name"],
                                       graph_type=args["whatAsNode"],
                                       data_num=args["data_num"],
                                       other_folders=args["other_datasets"])
    dataset_norm, dataset_sampled, norm_dict = normalization.normalize_with_normDict(norm_dict, dataset, args["reduce_sample"], args["yield_factor"])
    final_dataset = dataset_sampled if args["sample_node"] else dataset_norm
    final_dataset = final_dataset[:data_number]
    print("dataset length:", len(final_dataset))
    dataloader = DataLoader(final_dataset, batch_size=args["batch_size"], shuffle=False)

    # model
    model_constructor_args = {
    'input_dim': 47, 'hidden_dim': args["hidden_dim"], 'output_dim': 8,
    'num_layers': args["num_layers"], 'device': device}
    model = LSTM(**model_constructor_args).to(device)
    model.load_state_dict(torch.load(save_model_dir + 'model.pt'))
    model.eval()

    # embedding, x
    embedding = torch.Tensor()
    x = torch.Tensor()

    for i, graph in enumerate(dataloader):
        graph = graph.to(device)
        graph.ground_motion_1 = graph.ground_motion_1.permute(1, 0)
        graph.ground_motion_2 = graph.ground_motion_2.permute(1, 0)

        with torch.no_grad():
            # embedding
            if location == "graph_encoder":
                gm_1 = graph.ground_motion_1[500]
                gm_2 = graph.ground_motion_2[500]
                embed = model.create_ground_motion_graph(gm_1, gm_2, graph.x, graph.ptr)

            elif location == "H_C":
                H_list = [None for i in range(args["num_layers"])]
                C_list = [None for i in range(args["num_layers"])]
                # loop for each time step
                for i, gm in enumerate(zip(graph.ground_motion_1, graph.ground_motion_2)):
                    if i == 500:
                        break
                    gm_1, gm_2 = gm
                    H_list, C_list, out = model(gm_1, gm_2, graph.x, graph.ptr, H_list, C_list)    
                embed = torch.cat([H_list[-1], C_list[-1]], dim=1)

        embedding = torch.cat([embedding, embed], dim=0)
        x = torch.cat([x, graph.x], dim=0)

    # x features
    features = normalization.denormalize(x, norm_dict)

    return embedding, features



def get_labels_single_mapping(Y, features, feature_num_without_gm=27):
    # check data number
    assert Y.shape[0] == features.shape[0]
    features = features[:, :feature_num_without_gm]

    # first normalize the features with mean=0, std=1 (normal distribution, gaussian)
    features_gaussian = deepcopy(features)
    for f in range(feature_num_without_gm):
        mean, std = torch.mean(features_gaussian[:, f]), torch.std(features_gaussian[:, f])
        features_gaussian[:, f] = (features_gaussian[:, f] - mean) / std

    # calculate score for each feature
    feature_scores = []
    for f in range(feature_num_without_gm):
        feat = features_gaussian[:, f]
        score = 0
        # calculate score for the feature
        for i in range(features_gaussian.shape[0]):
            feat_i = torch.full_like(feat, feat[i])
            Y_i = torch.full_like(Y, Y[i, 0])
            Y_i[:, 1] = Y[i, 1]
            # score_i_j = distance * val_diff
            score += torch.sum(torch.sqrt((Y_i[:, 0] - Y[:, 0]) ** 2 + (Y_i[:, 1] - Y[:, 1]) ** 2) * torch.abs(feat_i - feat))
        feature_scores.append(score)
        print(f"feature {f} has score: {score:.2f}")
    
    # sort the features according their scores
    feature_scores, feature_ranks = zip(*sorted(zip(feature_scores, np.arange(feature_num_without_gm))))
    feature_scores, feature_ranks = feature_scores[::-1], feature_ranks[::-1]
    best_fit_feature = feature_ranks[0]

    return features[:, best_fit_feature].cpu().numpy(), best_fit_feature




def get_labels_linear_mapping(Y, features, feature_num_without_gm=27):
    """Get best-fit feature combinaion with gradient descent."""
    # check data number
    assert Y.shape[0] == features.shape[0]
    features = features[:, :feature_num_without_gm]

    # first normalize the features with mean=0, std=1 (normal distribution, gaussian)
    features_gaussian = deepcopy(features)
    for f in range(feature_num_without_gm):
        mean, std = torch.mean(features_gaussian[:, f]), torch.std(features_gaussian[:, f])
        features_gaussian[:, f] = (features_gaussian[:, f] - mean) / std

    # variables and optimizer
    W = torch.rand((feature_num_without_gm, 1), requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=5e-1)

    # train
    for epoch in range(400):
        loss = 0
        for i in range(features_gaussian.shape[0]):
            feat_i = (features_gaussian[i:i+1, :] @ W).squeeze(dim=1)           # [1]
            feat_i = feat_i.repeat(features_gaussian.shape[0], 1).squeeze()     # [n_n]
            feat_j = (features_gaussian @ W).squeeze()                          # [n_n]
            Y_i = torch.full_like(Y, Y[i, 0])
            Y_i[:, 1] = Y[i, 1]
            distance = torch.sqrt((Y_i[:, 0] - Y[:, 0]) ** 2 + (Y_i[:, 1] - Y[:, 1]) ** 2)
            loss -= torch.sum(distance * torch.abs(feat_i - feat_j)) / 1e+9

        # regularization
        reg_coeff = 10
        val_loss = loss.item()
        reg_loss = (1 - W.abs().sum()) ** 2
        loss += reg_coeff * reg_loss

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch = {epoch:3d}, loss = {loss.item():5.3f}, MSE_loss: {val_loss:.3f}, reg_loss: {reg_loss.item():.3f}, weight_sum = {W.data.abs().sum():.3f}")
    
    W = W / W.abs().sum()
    print()
    print("W:", W.data.squeeze())

    labels = (features_gaussian @ W).squeeze().detach().cpu().numpy()
    W = W.data.squeeze().cpu().numpy()

    return labels, W



def plot_no_mapping(Y, data_number, location):
    plt.figure(figsize=(10, 8))
    plt.scatter(Y[:, 0], Y[:, 1], s=20, cmap='viridis')
    plt.colorbar(label="no feature mapping")
    plt.title(f"t-SNE embedding distribution (data_number = {data_number}, location = {location})")
    plt.savefig("0.png")
    plt.show()



def plot_single_mapping(Y, labels, feature_index, data_number, location):
    feature_names = ['x_grid_num', 'y_grid_num', 'z_grid_num', 'x_coord', 'y_coord', 'z_coord', 'period', 'if_bottom', 'if_top',
                     'if_side', 'tanh(1 / beta_x)', 'tanh(1 / beta_z)', 'Ux', 'Uz', 'Ry', 'x_n_len', 'x_n_My', 'x_p_len', 'x_p_My', 
                     'y_n_len', 'y_n_My', 'y_p_len', 'y_p_My', 'z_n_len', 'z_n_My', 'z_p_len', 'z_p_My']
    plt.figure(figsize=(10, 8))
    plt.scatter(Y[:, 0], Y[:, 1], s=20, c=labels, cmap='viridis')
    plt.colorbar(label=feature_names[feature_index])
    plt.title(f"t-SNE embedding distribution (data_number = {data_number}, location = {location})")
    plt.savefig("1.png")
    plt.show()



def plot_linear_mapping(Y, W, labels, data_number, location):
    feature_names = ['x_grid_num', 'y_grid_num', 'z_grid_num', 'x_coord', 'y_coord', 'z_coord', 'period', 'if_bottom', 'if_top',
                     'if_side', 'tanh(1 / beta_x)', 'tanh(1 / beta_z)', 'Ux', 'Uz', 'Ry', 'x_n_len', 'x_n_My', 'x_p_len', 'x_p_My', 
                     'y_n_len', 'y_n_My', 'y_p_len', 'y_p_My', 'z_n_len', 'z_n_My', 'z_p_len', 'z_p_My']
    W = np.abs(W)
    W, feature_ranks = zip(*sorted(zip(W, np.arange(len(W)))))
    W, feature_ranks = W[::-1], feature_ranks[::-1]
    print(W)
    print(feature_ranks)
    best_3 = f"{feature_names[feature_ranks[0]]}: {W[0]:.2f}, {feature_names[feature_ranks[1]]}: {W[1]:.2f}, {feature_names[feature_ranks[2]]}: {W[2]:.2f}"
    plt.figure(figsize=(10, 8))
    plt.scatter(Y[:, 0], Y[:, 1], s=20, c=labels, cmap='viridis')
    plt.colorbar(label="feature_combination")
    plt.title(f"t-SNE embedding distribution (data_number = {data_number}, location = {location}) \n {best_3}")
    plt.savefig("2.png")
    plt.show()





if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    # prepate data
    save_model_dir = "../Results/Nonlinear_Dynamic_Analysis_World_Taipei3_Design/2022_08_09__00_30_59/"
    data_number = 400
    location = 'graph_encoder'    # graph_encoder, H_C
    embedding, features = load_embedding(save_model_dir=save_model_dir, data_number=data_number, location=location)

    # run tsne
    with torch.no_grad():
        Y = tsne(embedding, no_dims=2, initial_dims=50, perplexity=20.0)
        
    # find the suitable feature as label
    # labels, feature_index = get_labels_single_mapping(Y, features, feature_num_without_gm=27)
    # labels, W = get_labels_linear_mapping(Y, features, feature_num_without_gm=27)
    

    # plot
    Y = Y.cpu().numpy()
    plot_no_mapping(Y, data_number, location)
    # plot_single_mapping(Y, labels, feature_index, data_number, location)
    # plot_linear_mapping(Y, W, labels, data_number, location)
    
