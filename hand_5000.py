import cv2
import mediapipe as mp
import os
import cmath
import random
import math
import numpy as np
import shutil
def main():
    edge_index=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 13, 14, 15, 16, 17, 17, 18, 19, 20],
                [0, 1, 2, 3, 0, 5, 6, 7, 0, 5,  9, 10, 11,  0, 9, 13, 14, 15, 13,  0, 17, 18, 19]]
    in_path='./hand5000'
    out_path='./hand5000_change'
    if os.path.exists(out_path):
        pass
    else:
        os.makedirs(out_path)
    for i in os.listdir(in_path):
        if len(i.split('.'))==1:
            read_path=in_path+'/'+i
            wirt_path=out_path+'/'+i
            if os.path.exists(read_path):
                pass
            else:
                os.makedirs(read_path)
            if os.path.exists(wirt_path):
                    pass
            else:
                os.makedirs(wirt_path)
            for j in os.listdir(read_path):
                read_txt=read_path+'/'+j
                wirt_txt=wirt_path+'/'+j
                # print(read_txt,wirt_txt)
                f_w=open(wirt_txt,'w',encoding='utf-8')
                with open(read_txt,'r',encoding='utf-8') as f:
                    data0=[]
                    for d in f.readlines():
                        data=d[:-1].split(',')
                        da=[]
                        for n in data:
                            da.append(float(n))
                        data0.append(np.asarray(da))
                    # print(len(np.asarray(data0)))
                    x_0,y_0,z_0=data0[0][0],data0[0][1],data0[0][2]
                    x_5,y_5,z_5=data0[5][0],data0[5][1],data0[5][2]
                    len_4=math.sqrt((x_5-x_0)**2+(y_5-y_0)**2+(z_5-z_0)**2)
                    for j in range(len(edge_index[0])):
                        end=edge_index[0][j]
                        start=edge_index[1][j]
                        f_w.write( str((data0[end][0]-data0[start][0])/len_4)+','
                                    +str((data0[end][1]-data0[start][1])/len_4)+','
                                    +str((data0[end][2]-data0[start][2])/len_4)+'\n')
                # return
                f_w.close()
        else:
            shutil.copyfile(in_path+'/'+i,out_path+'/'+i)
main()
