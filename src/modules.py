import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax, preprocess_adj, preprocess_adj_new, preprocess_adj_new1, gauss_sample_z, my_normalize

_EPS = 1e-10


# class MLPEncoder(nn.Module):
#     """MLP encoder module."""
#     def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
#         super(MLPEncoder, self).__init__()
#
#         self.adj_A = nn.Parameter(adj_A.double())
#         self.factor = factor
#
#         self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
#         self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
#         self.fc2 = nn.Linear(n_hid, n_out, bias = True)
#         self.dropout_prob = do_prob
#         self.batch_size = batch_size
#         self.z = nn.Parameter(torch.tensor(tol))
#         self.z_positive = nn.Parameter(torch.ones_like(adj_A, dtype=torch.float32))
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight.data)
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
#     def forward(self, inputs):
#
#         if torch.sum(self.adj_A != self.adj_A):
#             print('nan error \n')
#
#         # print(self.adj_A.shape)
#
#         # to amplify the value of A and accelerate convergence.
#         adj_A1 = torch.sinh(3.*self.adj_A)
#
#         # adj_Aforz = I-A^T
#         adj_Aforz = preprocess_adj_new(adj_A1)
#         adj_A = torch.eye(adj_A1.size()[0]).double()
#         H1 = F.relu((self.fc1(inputs)))
#         x = (self.fc2(H1))
#         logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa
#
#         return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa
class MLPDEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol=0.1,z_dim=1):
        super(MLPDEncoder, self).__init__()
        self.adj_A = nn.Parameter((adj_A).double(), requires_grad=True)
        self.factor = factor

        self.Wa = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.fc1 = nn.Linear(n_hid, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, 2 * z_dim, bias= True)

        self.fc_mu = nn.Linear(n_hid, z_dim)
        self.fc_logvar = nn.Linear(n_hid, z_dim)  # 徹底分家

        n_var = adj_A.shape[0]
        self.embed = nn.Embedding(n_out, n_hid)
        self.dropout_prob = do_prob
        self.alpha =  nn.Parameter(Variable(torch.div(torch.ones(n_var, n_out),n_out)).double(), requires_grad = True)
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(adj_A).double())

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # ==========================================
        # 1. 處理因果圖矩陣 (Graph Matrices)
        # ==========================================
        adj_A1 = torch.sinh(3. * self.adj_A)

        # Encoder 使用的矩陣: adj_Aforz 對應數學上的 (I - A^T)
        adj_Aforz = preprocess_adj_new(adj_A1)

        # 【修正 ②】: 拔除錯誤的 identity 覆蓋！
        # Decoder 必須使用 (I - A^T)^(-1) 進行 Message Passing，因果圖才不會失效
        # 直接對 adj_Aforz 求逆矩陣，確保梯度能正確流回 adj_A
        adj_A = torch.inverse(adj_Aforz)

        # ==========================================
        # 2. 特徵提取 (Feature Extraction)
        # ==========================================
        bninput = self.embed(inputs.long().view(-1, inputs.size(2)))
        bninput = bninput.view(*inputs.size(), -1).squeeze()

        bninput = F.dropout(bninput, p=0.5, training=self.training)

        # print(f"DEBUG: bninput shape = {bninput.shape}, fc1 expects = {self.fc1.in_features}")
        H1 = F.leaky_relu((self.fc1(bninput)),negative_slope=0.1)  # 它餵的是 bninput！

        mu_raw = self.fc_mu(H1)
        logvar_raw = self.fc_logvar(H1)

        mu_raw = 10.0 * torch.tanh(mu_raw / 10.0)

        # 3. 分別進行圖卷積 (Graph Convolution)
        # 平均值 mu：遵循論文公式，加上 Wa 位移
        mu = torch.matmul(adj_Aforz, mu_raw + self.Wa) - self.Wa

        # 變異數 logvar：純粹的訊息傳遞，不加 Wa (避免數值爆炸)
        # logvar = torch.matmul(adj_Aforz, logvar_raw)

        # 4. 【核心保險】限制範圍，防止 KL 噴出 10^15
        # logvar = torch.clamp(logvar, -10, 10)
        # logvar = 10 * torch.tanh(logvar_raw / 10)
        logvar = torch.clamp(logvar_raw, min=-4.0)

        # 5. 重參數化
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa

# class MLPDEncoder(nn.Module):
#     def __init__(self, n_in, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
#         super(MLPDEncoder, self).__init__()
#
#         self.adj_A = nn.Parameter((adj_A).double(), requires_grad=True)
#         self.factor = factor
#
#         self.Wa = nn.Parameter(torch.tensor(0.0), requires_grad=True)
#         self.fc1 = nn.Linear(n_hid, n_hid, bias = True)
#         self.fc2 = nn.Linear(n_hid, n_out, bias = True)
#
#         n_var = adj_A.shape[0]
#         self.embed = nn.Embedding(n_out, n_hid)
#         self.dropout_prob = do_prob
#         self.alpha =  nn.Parameter(Variable(torch.div(torch.ones(n_var, n_out),n_out)).double(), requires_grad = True)
#         self.batch_size = batch_size
#         self.z = nn.Parameter(torch.tensor(tol))
#         self.z_positive = nn.Parameter(torch.ones_like(adj_A).double())
#
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight.data)
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, inputs):
#
#         if torch.sum(self.adj_A != self.adj_A):
#             print('nan error \n')
#
#         adj_A1 = torch.sinh(3.*self.adj_A)
#
#         adj_Aforz = preprocess_adj_new(adj_A1)
#         adj_A = torch.eye(adj_A1.size()[0]).double()
#         # print("!!! 準備查表前，inputs 的最大值:", inputs.max().item())
#         # print("!!! 準備查表前，inputs 的最小值:", inputs.min().item())
#         # print("!!! Embedding層的真面目:", self.embed)
#
#         bninput = self.embed(inputs.long().view(-1, inputs.size(2)))
#         bninput = bninput.view(*inputs.size(),-1).squeeze()
#         H1 = F.relu((self.fc1(bninput)))
#         x = (self.fc2(H1))
#
#
#         logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa
#
#         prob = my_softmax(logits, -1)
#         alpha = my_softmax(self.alpha, -1)
#
#         return x, prob, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa, alpha


class SEMEncoder(nn.Module):
    """SEM encoder module."""
    def __init__(self, n_in, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(SEMEncoder, self).__init__()

        self.factor = factor
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad = True))
        self.dropout_prob = do_prob
        self.batch_size = batch_size

    def init_weights(self):
        nn.init.xavier_normal(self.adj_A.data)

    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_A = I-A^T, adj_A_inv = (I-A^T)^(-1)
        adj_A = preprocess_adj_new((adj_A1))
        adj_A_inv = preprocess_adj_new1((adj_A1))

        meanF = torch.matmul(adj_A_inv, torch.mean(torch.matmul(adj_A, inputs), 0))
        logits = torch.matmul(adj_A, inputs-meanF)

        return inputs-meanF, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A


#[YY] delete it?
class MLPDDecoder(nn.Module):
    """MLP decoder module. OLD DON"T USE
    """

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.,z_dim=1):
        super(MLPDDecoder, self).__init__()

        self.bn0 = nn.BatchNorm1d(n_in_node * 1, affine=True)
        self.out_fc1 = nn.Linear(z_dim, n_hid, bias = False)
        self.out_fc2 = nn.Linear(n_hid, n_hid, bias = False)
        self.out_fc3 = nn.Linear(n_hid, n_out, bias = False)
        self.out_fc = nn.Linear(z_dim, n_in_node)
#        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        self.bn1 = nn.BatchNorm1d(n_in_node * 1, affine=True)
#         self.W3 = Variable(torch.from_numpy(W3).float())
#         self.W4 = Variable(torch.from_numpy(W4).float())

        self.ln1 = nn.LayerNorm(32)

        # TODO check if this is indeed correct
        #self.adj_A = encoder.adj_A
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            # 專門針對線性層 (Linear) 處理
            if isinstance(m, nn.Linear):
                # 1. 初始化權重 (Weight)
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)  # 官方寫法，完全不需要 .data

                # 2. 初始化偏差值 (Bias)
                # 只要你在定義層時寫了 bias=False，這裡 m.bias 就會是 None，完美跳過
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 官方歸零寫法，完全不需要 .data

            # 如果你的網路裡有 BatchNorm 層，順便更新成現代寫法保護起來
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs, input_z, n_in_node, origin_A, adj_A_tilt, Wa):

        # ==========================================
        # 1. 因果訊息傳遞 (Message Passing)
        # ==========================================
        # 這裡的邏輯非常完美：(I-A)^-1 * Z
        mat_z = torch.matmul(adj_A_tilt, input_z + Wa) - Wa
        mat_z_neighbors_only = mat_z - input_z
        # ==========================================
        # 2. MLP 解碼器 (打通梯度流動！)
        # ==========================================
        # 【關鍵修正】：將 ReLU 換成 LeakyReLU (設定 negative_slope=0.1)
        # 確保 input_z 裡的負數也能把梯度傳回給 adj_A_tilt
        H2 = self.out_fc1(mat_z_neighbors_only)
        H2 = self.ln1(H2)
        # 假設 z_dim=4: [batch, 11, 4] -> [batch, 11, 16]
        H2 = F.leaky_relu(H2, negative_slope=0.1)


        # [batch, 11, 16] -> [batch, 11, 16]
        # H4 = F.leaky_relu(self.out_fc2(H2), negative_slope=0.1)

        # 輸出最終的 3 類 Logits: [batch, 11, 16] -> [batch, 11, 3]
        out = self.out_fc3(H2)



        # 接下來用 "只有鄰居資訊" 的矩陣去預測
        # out = torch.clamp(out, min=-5.0, max=5.0)


        return mat_z, out, adj_A_tilt

#[YY] delete it?
class MLPDiscreteDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDiscreteDecoder, self).__init__()
#        self.msg_fc1 = nn.ModuleList(
#            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
#        self.msg_fc2 = nn.ModuleList(
#            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
#        self.msg_out_shape = msg_out
#        self.skip_first_edge_type = skip_first

        self.bn0 = nn.BatchNorm1d(n_in_node * 1, affine=True)
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_hid, bias = True)
#        self.out_fc4 = nn.Linear(n_hid, n_hid, bias=True)
        self.out_fc3 = nn.Linear(n_hid, n_out, bias = True)
#        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        self.bn1 = nn.BatchNorm1d(n_in_node * 1, affine=True)
#         self.W3 = Variable(torch.from_numpy(W3).float())
#         self.W4 = Variable(torch.from_numpy(W4).float())

        # TODO check if this is indeed correct
        #self.adj_A = encoder.adj_A
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size
        self.softmax = nn.Softmax(dim=2)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):

        # # copy adj_A batch size
        # adj_A = self.adj_A.unsqueeze(0).repeat(self.batch_size, 1, 1)

        adj_A_new = torch.eye(origin_A.size()[0]).double()#preprocess_adj(origin_A)#
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa #.unsqueeze(2) #.squeeze(1).unsqueeze(1).repeat(1, self.data_variable_size, 1) # torch.repeat(torch.transpose(input_z), torch.ones(n_in_node), axis=0)

        adj_As = adj_A_new

        #mat_z_max = torch.matmul(adj_A_new, my_normalize(mat_z))

#        mat_z_max = (torch.max(mat_z, torch.matmul(adj_As, mat_z)))
        H3 = F.relu(self.out_fc1((mat_z)))

        #H3_max = torch.matmul(adj_A_new, my_normalize(H3))
#        H3_max = torch.max(H3, torch.matmul(adj_As, H3))

#        H4 = F.relu(self.out_fc2(H3))

        #H4_max = torch.matmul(adj_A_new, my_normalize(H4))
#        H4_max = torch.max(H4, torch.matmul(adj_As, H4))

#        H5 = F.relu(self.out_fc4(H4_max)) + H3

        #H5_max = torch.max(H5, torch.matmul(adj_As, H5))

        # mu and sigma
        out = self.softmax(self.out_fc3(H3)) # discretized log

        return mat_z, out, adj_A_tilt#, self.adj_A


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node,  origin_A, adj_A_tilt, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt

class SEMDecoder(nn.Module):
    """SEM decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(SEMDecoder, self).__init__()

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):

        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa)
        out = mat_z

        return mat_z, out-Wa, adj_A_tilt

