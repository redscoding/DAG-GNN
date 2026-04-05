
''''
Main function for traininng DAG-GNN

'''


from __future__ import division
from __future__ import print_function
import time
run_start_time = time.time()
import argparse
import pickle
import os
import datetime
import numpy as np
import networkx as nx
import csv
# import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import math

# import numpy as np
from utils import *
from modules import *

m = 11 #節點數量
# m =80
seed = 42
parser = argparse.ArgumentParser()

parser.add_argument('--run_id', type=int, default=0, help='用於標記這是第幾次獨立訓練 (Controller 專用)')
# -----------data parameters ------
# configurations
parser.add_argument('--data_type', type=str, default='discrete',# 'synthetic',
                    choices=['synthetic','real_DATA', 'discrete', 'real','coco','git_sachs_DATA'],
                    help='choosing which experiment to do.')
parser.add_argument('--data_filename', type=str, default= 'sachs.pkl',
                    help='data file name containing the discrete files.')
parser.add_argument('--data_dir', type=str, default= 'data/',
                    help='data file name containing the discrete files.')
parser.add_argument('--data_sample_size', type=int, default=7466,
                    help='the number of samples of data')
parser.add_argument('--data_variable_size', type=int, default=m,
                    help='the number of variables in synthetic generated data')
parser.add_argument('--graph_type', type=str, default='barabasi-albert',
                    help='the type of DAG graph by generation method')
parser.add_argument('--graph_degree', type=int, default=2,
                    help='the number of degree in generated DAG graph')
parser.add_argument('--graph_sem_type', type=str, default='linear-exp',
                    help='the structure equation model (SEM) parameter type')
parser.add_argument('--graph_linear_type', type=str, default='linear',
                    help='the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z, nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--x_dims', type=int, default=1, #changed here
                    help='The number of input dimensions: default 1.')
parser.add_argument('--z_dims', type=int, default=1,
                    help='The number of latent variable dimensions: default the same as variable size.')

# -----------training hyperparameters
parser.add_argument('--optimizer', type = str, default = 'Adam',
                    help = 'the choice of optimizer used')
parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                    help = 'threshold for learned adjacency matrix binarization')
parser.add_argument('--tau_A', type = float, default=0.0,
                    help='coefficient for L-1 norm of A.')
parser.add_argument('--lambda_A',  type = float, default= 0.,
                    help='coefficient for DAG constraint h(A).')
parser.add_argument('--c_A',  type = float, default= 1,
                    help='coefficient for absolute value h(A).')
parser.add_argument('--use_A_connect_loss',  type = int, default= 0,
                    help='flag to use A connect loss')
parser.add_argument('--use_A_positiver_loss', type = int, default = 0,
                    help = 'flag to enforce A must have positive values')


parser.add_argument('--no-cuda', action='store_true', default=False, #using cuda
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=seed, help='Random seed.')
parser.add_argument('--epochs', type=int, default= 300,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default = 100, # note: should be divisible by sample size, otherwise throw an error
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=3e-3,  # basline rate = 1e-3
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--k_max_iter', type = int, default = 1e2,
                    help ='the max iteration number for searching lambda and c')

parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp, or sem).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')


parser.add_argument('--h_tol', type=float, default = 1e-8,
                    help='the tolerance of error of h(A) to zero')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default= 1.0,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
args.factor = not args.no_factor
print(args)


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    save_folder = save_folder.replace(':', '_')
    os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")


# ================================================
# get data: experiments = {synthetic SEM, ALARM}
# ================================================
if args.data_type == 'discrete':
    train_loader, valid_loader, test_loader= load_data(args, args.batch_size, args.suffix)
    graph = {
        7: [6],
        8: [1, 2, 3, 4, 5, 11],
        9: [3, 4, 5, 8, 11],
        10: [6, 7],
        11: [4],
        4: [2],
        3: [8],
        2: [1],
    }
    ground_truth_G = nx.DiGraph(graph)
else:
    train_loader, valid_loader, test_loader, ground_truth_G = load_data( args, args.batch_size, args.suffix)


#===================================
# load modules
#===================================
# Generate off-diagonal interaction graph
off_diag = np.ones([args.data_variable_size, args.data_variable_size]) - np.eye(args.data_variable_size)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
rel_rec = torch.DoubleTensor(rel_rec)
rel_send = torch.DoubleTensor(rel_send)

# add adjacency matrix A
num_nodes = args.data_variable_size
if args.cuda:
    # device = torch.device("cuda")
    adj_A_np = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj_A = torch.from_numpy(adj_A_np).to('cuda')
else:
    adj_A = np.zeros((num_nodes, num_nodes))


if args.encoder == 'mlp':
    encoder = MLPEncoder(args.data_variable_size * args.x_dims, args.x_dims, args.encoder_hidden,
                         int(args.z_dims), adj_A,
                         batch_size = args.batch_size,
                         do_prob = args.encoder_dropout, factor = args.factor).double()
elif args.encoder == 'sem':
    encoder = SEMEncoder(args.data_variable_size * args.x_dims, args.encoder_hidden,
                         int(args.z_dims), adj_A,
                         batch_size = args.batch_size,
                         do_prob = args.encoder_dropout, factor = args.factor).double()

if args.decoder == 'mlp':
    decoder = MLPDecoder(args.data_variable_size * args.x_dims,
                         args.z_dims, args.x_dims, encoder,
                         data_variable_size = args.data_variable_size,
                         batch_size = args.batch_size,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout).double()
elif args.decoder == 'sem':
    decoder = SEMDecoder(args.data_variable_size * args.x_dims,
                         args.z_dims, 2, encoder,
                         data_variable_size = args.data_variable_size,
                         batch_size = args.batch_size,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout).double()

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

#===================================
# set up training parameters
#===================================
if args.optimizer == 'Adam':
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=args.lr)
elif args.optimizer == 'LBFGS':
    optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=args.lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.data_variable_size)
tril_indices = get_tril_offdiag_indices(args.data_variable_size)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.DoubleTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    # if args.cuda:
    #     log_prior = log_prior.cuda()

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1


def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr

#===================================
# training:
#===================================

def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []

    encoder.train()
    decoder.train()
    scheduler.step()


    # update optimizer
    optimizer, lr = update_optimizer(optimizer, args.lr, c_A)


    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data).double(), Variable(relations).double()

        # reshape data
        relations = relations.unsqueeze(2)
        data = data.unsqueeze(-1)

        optimizer.zero_grad()
        # return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data)  # logits is of size: [num_sims, z_dims]
        edges = logits

        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, args.data_variable_size * args.x_dims, origin_A, adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')

        target = data
        preds = output
        # print(f"preds={preds.shape}") #100,100,1
        variance = 0.

        # reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_A = origin_A # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))

        # other loss term
        if args.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if args.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        # compute h(A)
        h_A = _h_A(origin_A, args.data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss #+  0.01 * torch.sum(variance * variance)


        loss.backward()
        # 這裡會檢查所有參數的梯度
        all_params = list(encoder.named_parameters()) + list(decoder.named_parameters())

        # # 遍歷所有參數，檢查它們是否有梯度
        # for name, param in all_params:
        #     if param.grad is not None:
        #         # 只印出有梯度的參數，如果梯度為零可以特別標記
        #         grad_norm = param.grad.norm().item()
        #         if grad_norm > 1e-8:  # 設定一個閾值來避免印出極小的數值
        #             print(f"Epoch {epoch}, Parameter: {name}, Grad Norm: {grad_norm:.6f}")
        #     else:
        #         # 這是最關鍵的檢查：如果梯度為 None，代表計算圖斷了
        #         print(f"Epoch {epoch}, Parameter: {name}, Grad is None!")
        loss = optimizer.step()

        myA.data = stau(myA.data, args.tau_A*lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))


        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_trian.append(shd)

    print(h_A.item())
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    if 'graph' not in vars():
        print('error on assign')


    return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A

#===================================
# main
#===================================

t_total = time.time()
best_ELBO_loss = np.inf
best_NLL_loss = np.inf
best_MSE_loss = np.inf
best_epoch = 0
best_ELBO_graph = []
best_NLL_graph = []
best_MSE_graph = []
# optimizer step on hyparameters
c_A = args.c_A
lambda_A = args.lambda_A
h_A_new = torch.tensor(1.)
h_tol = args.h_tol
k_max_iter = int(args.k_max_iter)
h_A_old = np.inf

try:
    for step_k in range(k_max_iter):
        while c_A < 1e+20:
            for epoch in range(args.epochs):
                ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer)
                if ELBO_loss < best_ELBO_loss:
                    best_ELBO_loss = ELBO_loss
                    best_epoch = epoch
                    best_ELBO_graph = graph

                if NLL_loss < best_NLL_loss:
                    best_NLL_loss = NLL_loss
                    best_epoch = epoch
                    best_NLL_graph = graph

                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch
                    best_MSE_graph = graph

            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
            if ELBO_loss > 2 * best_ELBO_loss:
                break

            # update parameters
            A_new = origin_A.data.clone()
            h_A_new = _h_A(A_new, args.data_variable_size)
            if h_A_new.item() > 0.25 * h_A_old:
                c_A*=10
            else:
                break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break


    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

    # test()
    print (best_ELBO_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
    print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_NLL_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
    print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


    print (best_MSE_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
    print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph = origin_A.data.clone().cpu().numpy()
    graph[np.abs(graph) < 0.1] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.2] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.3] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


except KeyboardInterrupt:
    # print the best anway
    print(best_ELBO_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
    print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_NLL_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
    print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    print(best_MSE_graph)
    print(nx.to_numpy_array(ground_truth_G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
    print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph = origin_A.data.clone().cpu().numpy()
    graph[np.abs(graph) < 0.1] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.2] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    graph[np.abs(graph) < 0.3] = 0
    # print(graph)
    fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
    print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    # ==========================================
    # 異常中斷的存檔與報表區塊
    # ==========================================

    run_end_time = time.time()
    elapsed_seconds = run_end_time - run_start_time
    elapsed_formatted = f"{elapsed_seconds / 60:.2f} min"

    pool_dir = os.path.join('src', 'results')
    os.makedirs(pool_dir, exist_ok=True)
    current_run = args.run_id

    if current_run == 0:
        np.savetxt(os.path.join(pool_dir, 'trueG.txt'), nx.to_numpy_array(ground_truth_G), fmt='%.5f')

    final_continuous_A = origin_A.data.clone().cpu().numpy()

    csv_path = os.path.join(pool_dir, 'summary_report.csv')
    file_exists = os.path.isfile(csv_path)

    candidates = {}

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ['Filename', 'Run_ID', 'Threshold', 'SHD', 'TPR', 'FDR', 'NNZ', 'Duration', 'Status', 'Timestamp'])

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        for t in thresholds:
            g_thresh = final_continuous_A.copy()
            g_thresh[np.abs(g_thresh) < t] = 0
            g_thresh[g_thresh != 0] = 1

            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(g_thresh))

            # 💡 差異點 1：檔名加上 _interrupted 標記
            name_with_shd = f'thresh_{t}_shd_{shd}_interrupted'
            candidates[name_with_shd] = g_thresh

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_name = f'candidate_run{current_run}_{name_with_shd}.txt'

            # 💡 差異點 2：狀態標記為 'Interrupted'
            writer.writerow(
                [file_name, current_run, t, shd, tpr, fdr, nnz, elapsed_formatted, 'Interrupted', current_time])

    for name, graph_mat in candidates.items():
        file_name = f'candidate_run{current_run}_{name}.txt'
        file_path = os.path.join(pool_dir, file_name)
        np.savetxt(file_path, graph_mat, fmt='%d')

    print(f"\n[搶救成功] Run {current_run} 異常中斷 (耗時 {elapsed_formatted})！已存入 {len(candidates)} 張中斷候選圖。")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for t in thresholds:
        g_thresh = final_continuous_A.copy()
        g_thresh[np.abs(g_thresh) < t] = 0
        g_thresh[g_thresh != 0] = 1
        candidates[f'thresh_{t}'] = g_thresh

    # ==========================================
    # 將所有候選圖存成獨立的 txt 檔案 (防覆蓋機制)
    # ==========================================
    for name, graph_mat in candidates.items():
        # 【關鍵防護】在檔名中間安插 run_{current_run}
        file_name = f'candidate_run{current_run}_{name}.txt'
        file_path = os.path.join(pool_dir, file_name)
        np.savetxt(file_path, graph_mat, fmt='%d')  # 存成整數 0 和 1

    print(f"\n[成功] Run {current_run}：已將 {len(candidates)} 張候選圖存入資料夾：{pool_dir}/")
    import sys
    sys.exit(0)


f = open('trueG', 'w')
matG = np.matrix(nx.to_numpy_array(ground_truth_G))
for line in matG:
    np.savetxt(f, line, fmt='%.5f')
f.closed

f1 = open('results/predG', 'w')
matG1 = np.matrix(origin_A.data.clone().cpu().numpy())
for line in matG1:
    np.savetxt(f1, line, fmt='%.5f')
f1.closed



# ==========================================
# 停止碼錶，計算這一個 Run 總共花了多少時間
# ==========================================
run_end_time = time.time()
elapsed_seconds = run_end_time - run_start_time
elapsed_formatted = f"{elapsed_seconds / 60:.2f} min"  # 換算成分鐘，保留小數點後兩位

# 固定存在 src/results
pool_dir = os.path.join('src', 'results')
os.makedirs(pool_dir, exist_ok=True)
current_run = args.run_id

# 1. 存下真實的 Ground Truth
if current_run == 0:
    np.savetxt(os.path.join(pool_dir, 'trueG.txt'), nx.to_numpy_array(ground_truth_G), fmt='%.5f')

# 2. 獲取最後的連續權重矩陣
final_continuous_A = origin_A.data.clone().cpu().numpy()

# 準備 CSV 報表的標題
csv_path = os.path.join(pool_dir, 'summary_report.csv')
file_exists = os.path.isfile(csv_path)

candidates = {}

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        # 💡 表頭新增了 'Duration' 欄位！
        writer.writerow(['Filename', 'Run_ID', 'Threshold', 'SHD', 'TPR', 'FDR', 'NNZ', 'Duration', 'Timestamp'])

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for t in thresholds:
        g_thresh = final_continuous_A.copy()
        g_thresh[np.abs(g_thresh) < t] = 0
        g_thresh[g_thresh != 0] = 1

        # 計算指標
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(g_thresh))

        name_with_shd = f'thresh_{t}_shd_{shd}'
        candidates[name_with_shd] = g_thresh

        # 記錄當下時間
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_name = f'candidate_run{current_run}_{name_with_shd}.txt'

        # 💡 把 elapsed_formatted (花費時間) 寫進去！
        writer.writerow([file_name, current_run, t, shd, tpr, fdr, nnz, elapsed_formatted, current_time])

# 將所有候選圖存成獨立的 txt 檔案
for name, graph_mat in candidates.items():
    file_name = f'candidate_run{current_run}_{name}.txt'
    file_path = os.path.join(pool_dir, file_name)
    np.savetxt(file_path, graph_mat, fmt='%d')

print(f"\n[成功] Run {current_run} 結束 (耗時 {elapsed_formatted})！已將 {len(candidates)} 張圖與實驗數據存入 {pool_dir}/")

if log is not None:
    print(save_folder)
    log.close()

if log is not None:
    print(save_folder)
    log.close()
