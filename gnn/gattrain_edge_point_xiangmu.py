import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import numpy as np
import os
import random
import datetime
from torch_geometric.data import Data
# x=torch.tenser([[1,2,3],[2,3,2],[1,2,1]],dtype=torch.float)
# y=torch.tensor([0,1,0],dtype=float)
# edge_index=torch.tensor([[1,2,3],[2,3,1]],dtype=torch.long)

def dataset(class_num,datapath,edge_index):
    # datapath='../Pointnet2/data/hand5000'
    # edge_index= [[1, 2, 3, 4, 5, 22, 5, 6, 7, 8,22,21, 9, 9,  10, 11, 12, 21, 20, 13, 13, 14, 15, 16, 20, 17, 17, 18, 19],
    #              [0, 1, 2, 0, 4,  4, 22,5, 6, 0, 8, 8, 22,21, 9,  10, 0,  12, 12, 20, 21, 13, 14,  0, 16, 16, 20, 17, 18]]
    datasets=[]
    for i in os.listdir(datapath):
        y=[[0 for i in range(class_num)]]
        if len(i.split('.'))==1:
        # i 就是类别
            y[0][int(i)]=1
        #    print(y)
            txt=os.path.join(datapath+'/'+i)
            for j in os.listdir(txt):
                    #j是i类别里的txt文本
                    x=[]
                    edge_attr=[]
                    with open(txt+'/'+j,'r',encoding='utf-8') as f:
                        for m in f.readlines():
                            data=[]
                            for num in m.strip('\n').split(','):
                                data.append(float(num))
                                # print(data)
                            x.append(data)
                        for n in range(len(edge_index[0])):
                            # print(np.array(x[edge_index[0][n]]))
                            attr=(np.array(x[edge_index[0][n]])-np.array(x[edge_index[1][n]]))*10
                            # print(attr)
                            edge_attr.append(list(attr))
                    dataset = Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long),
                            edge_attr=torch.tensor(edge_attr, dtype=torch.float))
                    # print(len(edge_index),len(x),len(edge_attr),len(y))
                    # print(len(edge_index[0]),len(x[0]),len(edge_attr[0]),len(y))
                    # print(dataset)
                    datasets.append(dataset)
    random.shuffle(datasets)
    return datasets[:int(len(datasets)*0.8)],datasets[int(len(datasets)*0.8):]

def dataset_val(class_num,datapath,edge_index):
    # datapath='../Pointnet2/data/hand5000'
    # edge_index= [[1, 2, 3, 4, 5, 22, 5, 6, 7, 8,22,21, 9, 9,  10, 11, 12, 21, 20, 13, 13, 14, 15, 16, 20, 17, 17, 18, 19],
    #              [0, 1, 2, 0, 4,  4, 22,5, 6, 0, 8, 8, 22,21, 9,  10, 0,  12, 12, 20, 21, 13, 14,  0, 16, 16, 20, 17, 18]]
    datasets=[]
    for i in os.listdir(datapath):
        y=[[0 for i in range(class_num)]]
        if len(i.split('.'))==1:
        # i 就是类别
            y[0][int(i)]=1
        #    print(y)
            txt=os.path.join(datapath+'/'+i)
            for j in os.listdir(txt):
                    #j是i类别里的txt文本
                    x=[]
                    edge_attr=[]
                    with open(txt+'/'+j,'r',encoding='utf-8') as f:
                        for m in f.readlines():
                            data=[]
                            for num in m.strip('\n').split(','):
                                data.append(float(num))
                                # print(data)
                            x.append(data)
                        for n in range(len(edge_index[0])):
                            # print(np.array(x[edge_index[0][n]]))
                            attr=(np.array(x[edge_index[0][n]])-np.array(x[edge_index[1][n]]))*10
                            # print(attr)
                            edge_attr.append(list(attr))
                    dataset = Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long),
                            edge_attr=torch.tensor(edge_attr, dtype=torch.float))
                    # print(len(edge_index),len(x),len(edge_attr),len(y))
                    # print(len(edge_index[0]),len(x[0]),len(edge_attr[0]),len(y))
                    # print(dataset)
                    datasets.append(dataset)
    random.shuffle(datasets)
    return datasets 

def get_weights(size, gain=1.414):
    weights = nn.Parameter(torch.zeros(size=size))
    nn.init.xavier_uniform_(weights, gain=gain)
    return weights
 
class GraphAttentionLayer(nn.Module):
    '''
    Simple GAT layer 图注意力层 (inductive graph)
    '''
    def __init__(self, in_features, out_features, dropout, alpha, concat = True, head_id = 0):
        ''' One head GAT '''
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  #节点表示向量的输入特征维度
        self.out_features = out_features    #节点表示向量的输出特征维度
        self.dropout = dropout  #dropout参数
        self.alpha = alpha  #leakyrelu激活的参数
        self.concat = concat    #如果为true，再进行elu激活
        self.head_id = head_id  #表示多头注意力的编号
 
        self.W_type = nn.ParameterList()
        self.a_type = nn.ParameterList()
        self.n_type = 1 #表示边的种类
        for i in range(self.n_type):
            self.W_type.append(get_weights((in_features, out_features)))
            self.a_type.append(get_weights((out_features * 2, 1)))
 
        #定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size = (in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain = 1.414)  #xavier初始化
        self.a = nn.Parameter(torch.zeros(size = (2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain = 1.414)  #xavier初始化
 
        #定义dropout函数防止过拟合
        self.dropout_attn = nn.Dropout(self.dropout)
        #定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
 
    def forward(self, node_input, adj ,node_mask = None):
        '''
        node_input: [batch_size, node_num, feature_size] feature_size 表示节点的输入特征向量维度
        adj: [batch_size, node_num, node_num] 图的邻接矩阵
        node_mask:  [batch_size, node_mask]
        '''
 
        zero_vec = torch.zeros_like(adj)
        scores = torch.zeros_like(adj)
 
        for i in range(self.n_type):
            h = torch.matmul(node_input, self.W_type[i])
            h = self.dropout_attn(h)
            # print(h.shape)
            N, E, d = h.shape   # N == batch_size, E == node_num, d == feature_size
 
            a_input = torch.cat([h.repeat(1, 1, E).view(N, E * E, -1), h.repeat(1, E, 1)], dim = -1)
            a_input = a_input.view(-1, E, E, 2 * d)     #([batch_size, E, E, out_features])
 
            score = self.leakyrelu(torch.matmul(a_input, self.a_type[i]).squeeze(-1))   #([batch_size, E, E, 1]) => ([batch_size, E, E])
            #图注意力相关系数（未归一化）
 
            zero_vec = zero_vec.to(score.dtype)
            scores = scores.to(score.dtype)
            scores += torch.where(adj == i+1, score, zero_vec.to(score.dtype))
 
        zero_vec = -1*30 * torch.ones_like(scores)  #将没有连接的边置为负无穷
        attention = torch.where(adj > 0, scores, zero_vec.to(scores.dtype))    #([batch_size, E, E])
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，则该位置的注意力系数保留；否则需要mask并置为非常小的值，softmax的时候最小值不会被考虑
 
        if node_mask is not None:
            node_mask = node_mask.unsqueeze(-1)
            h = h * node_mask   #对结点进行mask
 
        attention = F.softmax(attention, dim = 2)   #[batch_size, E, E], softmax之后形状保持不变，得到归一化的注意力权重
        h = attention.unsqueeze(3) * h.unsqueeze(2) #[batch_size, E, E, d]
        h_prime = torch.sum(h, dim = 1)             #[batch_size, E, d]
 
        # h_prime = torch.matmul(attention, h)    #[batch_size, E, E] * [batch_size, E, d] => [batch_size, N, d]
 
        #得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
 
class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, n_heads,dropout, alpha, concat = True):
        '''
        Dense version of GAT
        in_dim输入表示的特征维度、hid_dim输出表示的特征维度
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似于self-attention从不同的子空间进行抽取特征
        '''
        super(GAT, self).__init__()
        assert hid_dim % n_heads == 0
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
 
        self.attn_funcs = nn.ModuleList()
        for i in range(n_heads):
            self.attn_funcs.append(
                #定义multi-head的图注意力层
                GraphAttentionLayer(in_features = in_dim, out_features = hid_dim // n_heads,
                                    dropout = dropout, alpha = alpha, concat = concat, head_id = i)
            )
 
        self.dropout = nn.Dropout(self.dropout)
 
    def forward(self, node_input, adj, node_mask = None):
        '''
        node_input: [batch_size, node_num, feature_size]    输入图中结点的特征
        adj:    [batch_size, node_num, node_num]    图邻接矩阵
        node_mask:  [batch_size, node_num]  表示输入节点是否被mask
        '''
        hidden_list = []
        for attn in self.attn_funcs:
            h = attn(node_input, adj, node_mask = node_mask)
            hidden_list.append(h)
 
        h = torch.cat(hidden_list, dim = -1)
        h = self.dropout(h) #dropout函数防止过拟合
        x = F.elu(h)     #激活函数
        return x

class GAT_GCN(torch.nn.Module):
    def __init__(self, in_dim,n_heads,dropout,num_classes,):
        super(GAT_GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv0 = GAT(in_dim, 16, n_heads,dropout, alpha = 0.2, concat = True).cuda() 
        # self.conv1 = GAT(16, 32, n_heads,dropout, alpha = 0.2, concat = True).cuda() 
        # self.conv2 = GAT(32, 64, n_heads,dropout, alpha = 0.2, concat = True).cuda() 
        self.conv1 = GCNConv(16, 32)
        # self.conv0 = GAT(16, 32, n_heads,dropout, alpha = 0.2, concat = True).cuda() 
        self.conv2 = GCNConv(32, 64)
        # self.conv0 = GAT(32, 64, n_heads,dropout, alpha = 0.2, concat = True).cuda() 
        # self.conv3 = GCNConv(gcn_hid_dim, gcn_hid_dim)
        self.lin = Linear(64, num_classes)

    def forward(self, x, edge_index, batch, adj):

        # # 1.对各节点进行编码
        x=x.view([-1, 21, 3])
        x =self.conv0(x, adj)
        x=x.view([-1, 16])

        x = self.conv1(x, edge_index)
        x = x.relu()
        # x=x.view([-1, 21, 16])
        # x =self.conv0(x, adj)
        # x=x.view([-1, 32])
        x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. 平均操作
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 输出
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
def adj_(edge_index,batch_size,adj_size):
    a=np.zeros((adj_size,adj_size))
    for i in range(len(edge_index[0])):
        a[edge_index[0][i]][edge_index[1][i]]=1
        a[edge_index[1][i]][edge_index[0][i]]=1
    adj = torch.tensor(a).cuda()
    adj.unsqueeze(0)
    adj = adj.repeat(batch_size, 1, 1).cuda()
    return adj
def train(train_loader,edge_index,batch_size,adj_size):
    model.train()
    adj=adj_(edge_index,batch_size,adj_size)
    for data in train_loader:  # Iterate in batches over the training dataset.
        # print('1',data.x.cuda().shape)
        out = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda(),adj)  # Perform a single forward pass.
        loss = criterion(out, data.y.cuda())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader,edge_index,batch_size,adj_size):
    model.eval()
    adj=adj_(edge_index,batch_size,adj_size)
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda(),adj)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # print(pred)
        # print(data.y)
        sum=0
        for i in range(len(pred)):
            sum+=data.y[i][pred[i]]
        # print(sum)
        correct+=sum 
        # correct += int((pred == data.y.cuda()).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.






if __name__=="__main__":
    ##### 设置训练参数
    # 训练记录
    # 0 1层
    # 1 2层
    # 2 3层
    # def __init__(self, in_dim,gat_hid_dim,n_heads,dropout,gcn_hid_dim,num_classes,):
    in_dim=3
    n_heads=4
    dropout=0.1
    num_classes=15
    batch_size=50
    lr=0.001
    adj_size=21
    epoch=300
    datapath='../Pointnet2/data/hand10000_change'
    datapath_val='../Pointnet2/data/hand5000_10000_val'
    txt_path='./run/hand10000_change_gat'
    # edge_index0=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20],
    #              [0, 1, 2, 3, 0, 5, 6, 7, 5,  9, 10, 11,  9, 13, 14, 15, 13,  0, 17, 18, 19]]
    edge_index= [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 19, 20],
                 [0, 1, 2, 0, 4, 5, 6, 5, 8,  9, 10,  9, 12, 13, 14, 13, 16, 16, 17, 18, 19]]
    # datapath='../Pointnet2/data/hand5000'
    # datapath_val='../Pointnet2/data/hand5000_val'
    # txt_path='./run/hand5000_gat'
    # edge_index= [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 13, 9, 5],
    #              [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0,  13, 14, 15, 0,  17, 18, 19, 17, 13, 9]]

     ####加载数据
    print("加载数据开始！")
    train_dataset,test_dataset=dataset(num_classes,datapath,edge_index)
    val_dataset=dataset_val(num_classes,datapath_val,edge_index)
    train_loader= DataLoader(train_dataset, batch_size, shuffle=True,drop_last=False)
    test_loader= DataLoader(test_dataset, batch_size, shuffle=True,drop_last=False)
    val_loader=DataLoader(val_dataset, batch_size, shuffle=True,drop_last=False)
    print("加载数据完成！！！！！！")
    model = GAT_GCN(in_dim,n_heads,dropout,num_classes).cuda()
    print("加载模型完成！！！！！！！！！")
    print(model)

    ###加载模型和loss，优化器
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    ###### 设置保存文件夹
    if os.path.isdir(txt_path):
        pass
    else:
        os.mkdir(txt_path)
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    log_path=txt_path+'/'+timestr
    if os.path.isdir(log_path):
        pass
    else:
        os.mkdir(log_path)
    print(f'文件储存地址：{log_path}')
    f=open(log_path+'/'+'train.txt','w',encoding='utf-8')
    f.write('epoch\ttrain_acc\ttest_acc\tval_acc\n')
    print("文件夹设置完成：",log_path)
    
    ######开始训练
    best=0
    print("开始训练")
    for epoch in range(epoch):
        train(train_loader,edge_index,batch_size,adj_size)
        train_acc = test(train_loader,edge_index,batch_size,adj_size)
        test_acc = test(test_loader,edge_index,batch_size,adj_size)
        val_acc = test(val_loader,edge_index,batch_size,adj_size)
        if val_acc>best:
           torch.save(model.state_dict(), log_path+'/'+'best.params')
           best=val_acc
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f},Test Acc: {test_acc:.4f},val Acc: {val_acc:.4f}')
        f.write(f'{epoch}\t{train_acc}\t{test_acc}\t{val_acc}\n')
    torch.save(model.state_dict(), log_path+'/'+'last.params')
    f.close()