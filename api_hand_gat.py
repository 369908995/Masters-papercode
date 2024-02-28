import cv2
import mediapipe as mp
import cmath
from utils.torch_utils import select_device, time_sync
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from gnn.gnntrain import GCN
from gnn.gattrain_edge_point_xiangmu import GAT_GCN,adj_
from torch_geometric.data import Data, Batch
import math

def main():
    flag = 0
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
    in_dim=3
    n_heads=4
    dropout=0.1
    num_classes=15
    edge_index0=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20],
                 [0, 1, 2, 3, 0, 5, 6, 7, 5,  9, 10, 11,  9, 13, 14, 15, 13,  0, 17, 18, 19]]
    edge_index= [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 19, 20],
                 [0, 1, 2, 0, 4, 5, 6, 5, 8,  9, 10,  9, 12, 13, 14, 13, 16, 16, 17, 18, 19]]
    class_dict= {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',
                9:'nine',10:'ok',11:'good',12:'dislik',13:'like',14:'none'}
    model = GAT_GCN(in_dim,n_heads,dropout,num_classes).cuda()
    print("加载模型完成！！！！！！！！！")

    model.load_state_dict(torch.load('./runs/hand10000_change_gat/2023-06-13_10-18/best.params'))
    print(model)
    cap = cv2.VideoCapture(0)
    while True:
        t1 = time_sync()
        flag = 0
        ret,frame0 = cap.read()
        frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        # 因为摄像头是镜像的，所以将摄像头水平翻转
        # 不是镜像的可以不翻转
        frame= cv2.flip(frame,1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)       
        if results.multi_hand_landmarks:
            x=[]
            for hand_landmarks in results.multi_hand_landmarks:
                # 关键点可视化
                datas=[]
                mp_drawing.draw_landmarks(frame, 
                                        hand_landmarks, 
                                        mp_hands.HAND_CONNECTIONS)
                # 处理数据为tensor格式tenser[[  [1,2,3],[4,5,6],........   ]]   1,21,3
                x_0,y_0,z_0=hand_landmarks.landmark[0].x,hand_landmarks.landmark[0].y,hand_landmarks.landmark[0].z
                x_5,y_5,z_5=hand_landmarks.landmark[5].x,hand_landmarks.landmark[5].y,hand_landmarks.landmark[5].z
                len_4=math.sqrt((x_5-x_0)**2+(y_5-y_0)**2+(z_5-z_0)**2)

                for j in range(len(edge_index0[0])):
                    data=[]
                    end=edge_index0[0][j]
                    start=edge_index0[1][j]
                    data.append((hand_landmarks.landmark[end].x-hand_landmarks.landmark[start].x)/len_4)
                    data.append((hand_landmarks.landmark[end].y-hand_landmarks.landmark[start].y)/len_4)
                    data.append((hand_landmarks.landmark[end].z-hand_landmarks.landmark[start].z)/len_4)
                    datas.append(data)
                x.append(datas)
            edge_attr=[]
            for i in x:
                ed=[]
                for n in range(len(edge_index[0])):
                                # print(np.array(x[edge_index[0][n]]))
                                attr=(np.array(i[edge_index[0][n]])-np.array(i[edge_index[1][n]]))*10
                                # print(attr)
                                ed.append(list(attr))
                edge_attr.append(ed)



            for i in range(len(x)):
                adj=adj_(edge_index,len(x),21)
                x,edge_attr,edge_index=torch.tensor(x[i]).cuda(),torch.tensor(edge_attr[i]).cuda(),torch.tensor(edge_index).cuda()
                # print(x.shape,edge_index.shape,edge_attr.shape)
                batch=torch.tensor([0 for i in range(21)]).cuda()
                if x.shape==torch.Size([21, 3]):
                    # print('ok')
                    pred=model(x,edge_index,batch,adj)
                    pred=int(pred.argmax().tolist())
                    se=class_dict[pred]
                    txt='CLASS:'+se
                    cv2.putText(frame,txt,(40,30),0,1,(0,0,255),3)
        t2 = time_sync()
        fps=int(1/(t2-t1))
        f='FPS:'+str(fps)
        cv2.putText(frame,f,(350,30),0,1,(0,0,255),3)
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('./0-1.png',frame0)
            cv2.imwrite('./7.png',frame)
            break
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
     main()
