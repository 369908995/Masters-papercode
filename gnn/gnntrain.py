import torch
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import numpy as np
import os
import random
import datetime

# x=torch.tenser([[1,2,3],[2,3,2],[1,2,1]],dtype=torch.float)
# y=torch.tensor([0,1,0],dtype=float)
# edge_index=torch.tensor([[1,2,3],[2,3,1]],dtype=torch.long)
from torch_geometric.data import Data
def dataset():
    datapath='../Pointnet2/data/hand'
    edge_index =[[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,17,18,19,20],
                [0, 1, 2, 3, 0,5,6,7,5, 9,10,11, 9,13,14,15,13, 0,17,18,19]]
    # edge_index =[[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,17,18,19,20,8,12,16,20,4,8,12,16,20],
    #             [0, 1, 2, 3, 0,5,6,7,5, 9,10,11, 9,13,14,15,13, 0,17,18,19,4,8,12,16,0,0,0,0,0]]
    datasets=[]
    for i in os.listdir(datapath):
        y=[[0,0,0,0,0]]
        if len(i.split('.'))==1:
        # i 就是类别
            y[0][int(i)-1]=1
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

def dataset1():
    datapath='../Pointnet2/data/hand5000'
    # edge_index =[[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,17,18,19,20],
    #             [0, 1, 2, 3, 0,5,6,7,5, 9,10,11, 9,13,14,15,13, 0,17,18,19]]
    edge_index =[[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,17,18,19,20,8,12,16,20,4,8,12,16,20],
                [0, 1, 2, 3, 0,5,6,7,5, 9,10,11, 9,13,14,15,13, 0,17,18,19,4,8,12,16,0,0,0,0,0]]
    datasets=[]
    for i in os.listdir(datapath):
        y=[[0,0,0,0,0]]
        if len(i.split('.'))==1:
        # i 就是类别
            y[0][int(i)-1]=1
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
class GCN(torch.nn.Module):
    def __init__(self, num_node_features,num_classes,hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1.对各节点进行编码
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. 平均操作
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 输出
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())  # Perform a single forward pass.
        loss = criterion(out, data.y.cuda())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())  
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
    ####加载数据
    train_dataset,test_dataset=dataset()
    # test_dataset1=dataset1()
    train_loader= DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader= DataLoader(test_dataset, batch_size=64, shuffle=True)

    ####打印数据
# # for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()

    ###### 设置训练参数 加载模型和loss，优化器
    num_node_features=3
    num_classes=5
    model = GCN(num_node_features=3,num_classes=5,hidden_channels=64).cuda()
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    ###### 设置保存文件夹
    txt_path='./run'
    if os.path.isdir(txt_path):
        pass
    else:
        os.mkdir(txt_path)
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    log_path='./run'+'/'+timestr
    if os.path.isdir(log_path):
        pass
    else:
        os.mkdir(log_path)
    f=open(log_path+'/'+'train.txt','w',encoding='utf-8')
    f.write('epoch\ttrain_acc\ttest_acc\n')


    #######开始训练
    best=0
    for epoch in range(200):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        if train_acc>best:
           torch.save(model.state_dict(), log_path+'/'+'best.params')
           best=test_acc
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f},Test Acc: {test_acc:.4f}')
        f.write(f'{epoch}\t{train_acc}\t{test_acc}\n')
    torch.save(model.state_dict(), log_path+'/'+'last.params')
    f.close()