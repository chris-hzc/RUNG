import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, adj, input):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        #attention = torch.where(adj > 0, e, zero_vec)
        attention = adj * e + (1-adj) * zero_vec
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义多头注意力层
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 输出层
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(adj, x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(adj, x))
        return F.log_softmax(x, dim=1)
    
    
    
    
class RobustGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(RobustGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.robust_sum = RobustSum(L=3, norm="MCP", epsilon =1e-2, gamma=4.0)

    def forward(self, adj, input):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        #attention = torch.where(adj > 0, e, zero_vec)
        attention = adj * e + (1-adj) * zero_vec
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        
        #h_prime = torch.matmul(attention, h)
        h_prime = self.robust_sum(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class RGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(RGAT, self).__init__()
        self.dropout = dropout

        # 定义多头注意力层
        self.attentions = [RobustGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 输出层
        self.out_att = RobustGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(adj, x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(adj, x))
        return F.log_softmax(x, dim=1)







class RobustSum(nn.Module):
    def __init__(self, L=3, norm="L2", epsilon =1e-2, gamma=4.0, t=1.0, delta=4.0):
        super().__init__()
        self.L=L
        self.norm=norm
        self.epsilon=epsilon
        self.gamma=gamma
        self.t=t
        self.delta=delta



    def forward(self, A, V):

        M = torch.matmul(A, V)

        
        if self.norm == 'L2':
            return M


        for _ in range(self.L):
            # 计算差值的平方，然后求和并开方
            #dist = ((V.unsqueeze(2) - M.unsqueeze(3))**2).sum(-1).sqrt()
            #dist = torch.cdist(M,V)
            dist = torch.cdist(M.detach(),V.detach())

            #dist = torch.square(V.unsqueeze(2).repeat(1,1,A.shape[2],1,1) - M.unsqueeze(3).repeat(1,1,1,A.shape[3],1)).sum(-1).sqrt()
            #dist = torch.cat([(V- M[:,:,i].unsqueeze(2).repeat(1,1,M.shape[2],1)).square().sum(-1).sqrt().unsqueeze(-2) for i in range(M.shape[2])],axis=-2)
            
            if self.norm == 'L1':
                w = 1/(dist+self.epsilon)
                
            elif  self.norm == 'MCP':
                w = 1/(dist + self.epsilon) - 1/self.gamma
                w[w<self.epsilon]=self.epsilon
                
            elif self.norm == 'Huber':
                w = self.delta/(dist + self.epsilon)
                w[w>1.0] = 1.0
            #del dist
            
            ww = w * A
            
            #del w
            
            #ww_norm = ww/ww.sum(-1).unsqueeze(-1)
            
            ww_norm = torch.nn.functional.normalize(ww,p=1,dim=-1)
            
            #del ww
            
            
            
            #M = torch.matmul(ww_norm.unsqueeze(3),V.unsqueeze(2).repeat(1,1,A.shape[2],1,1)).squeeze(-2)
            
            #M = torch.cat([torch.matmul(ww_norm[:,:,i].unsqueeze(-2),V) for i in range(A.shape[2])],axis=-2)
            
            M = (1.0 - self.t) * M + self.t * torch.matmul(ww_norm,V)
            #del ww_norm
            torch.cuda.empty_cache()
            
        return M