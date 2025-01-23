import torch
import torch.nn as nn
import math
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        #assert q.size(0) == batch_size
        #assert q.size(2) == input_dim
        #assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            # U = U.masked_fill(mask == 1, -np.inf)
            U[mask.bool()] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            # attnc = attnc.masked_fill(mask == 1, 0)
            attention = attnc

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())

'''
class Normalization1(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization1, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        #print("003==========================input.size()=",*input.size())
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
'''

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.embedding = nn.Linear(20, embedding_dim)
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(inplace=True),
                                         nn.Linear(256, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)
        self.embedding1 = nn.Linear(embedding_dim, 20)

    def forward(self, src, mask=None):
        src = self.embedding(src)
        #print("001==========================src.size()=",src.size())
        h0 = src # 128
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        #print("002==========================h.size()=",h.size())
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        h2 = self.embedding1(h2)
        return h2

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.embedding = nn.Linear(20, embedding_dim)
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(inplace=True),
                                         nn.Linear(256, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)
        self.embedding1 = nn.Linear(embedding_dim, 20)

    def forward(self, src, mask=None):
        src = self.embedding(src)
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        h2 = self.embedding1(h2)
        return h2

# 这个环节将sample_size加到batch_size上。
def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats

class WAE_Model(nn.Module):
    def __init__(self):
        super(WAE_Model, self).__init__()
        self.wae_encoder = EncoderLayer(embedding_dim=128, n_head=4)
        self.wae_decoder = DecoderLayer(embedding_dim=128, n_head=4)
    def forward(self, x):
        z = self.wae_encoder(x)
        x_recon = self.wae_decoder(z)
        #z_real = self.wae_encoder(x)
        
        #return z, x_recon, z_real
        return z, x_recon

class Forward_Model(nn.Module):
    def __init__(self, embedding_dim=128, n_head=4):
        super(Forward_Model, self).__init__()
        self.embedding = nn.Linear(40, embedding_dim)
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(inplace=True),
                                         nn.Linear(256, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)
        self.embedding1 = nn.Linear(embedding_dim, 4)

    def forward(self, src, mask=None):
        src = self.embedding(src)
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        h2 = self.embedding1(h2)
        return h2


def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 获取样本的特征维度
    z_dim = x.shape[2]

    # 计算高斯核矩阵
    return torch.exp(-((x[:, :, None, :] - y[:, None, :, :]) ** 2).mean(dim=3) / z_dim)

def mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 计算核矩阵并在batch维度上取平均
    return kernel(x, x).mean(dim=[1, 2]) + kernel(y, y).mean(dim=[1, 2]) - 2 * kernel(x, y).mean(dim=[1, 2])



class EncoderLayer_VAE_A(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer_VAE_A, self).__init__()
        self.embedding = nn.Linear(20, embedding_dim)
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(inplace=True),
                                         nn.Linear(256, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)
        #self.embedding1 = nn.Linear(embedding_dim, 20)
        self.FC_mean  = nn.Linear(embedding_dim, 20)
        self.FC_var   = nn.Linear (embedding_dim, 20)

    def forward(self, src, mask=None):
        src = self.embedding(src)
        #print("001==========================src.size()=",src.size())
        h0 = src # 128
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        #print("002==========================h.size()=",h.size())
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        #h2 = self.embedding1(h2)
        mean    = self.FC_mean(h2)
        log_var = self.FC_var(h2)
        
        return mean, log_var


class VAE_Model_A(nn.Module):
    def __init__(self):
        super(VAE_Model_A, self).__init__()
        self.wae_encoder = EncoderLayer_VAE_A(embedding_dim=128, n_head=4)
        self.wae_decoder = DecoderLayer(embedding_dim=128, n_head=4)
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)#.to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                             # reparameterization trick
        return z
    
    def forward(self, x):
        mean, log_var = self.wae_encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_recon = self.wae_decoder(z)
        
        return x_recon, z, mean, log_var



