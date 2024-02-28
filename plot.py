# import cv2
# #获取摄像头
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# #打开摄像头
# while (1):
     
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)  # 将图像左右调换回来正常显示
 
#     cv2.imshow("capture", frame)  # 摄像头窗口
 
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q就截图保存并退出
#         cv2.imwrite("../data-master/PM2.5/test.png", frame)  # 保存路径
#         break
 
# cap.release()
# cv2.destroyAllWindows()

# import torch
# print(torch.cuda.is_available() )

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
# myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttf')
# plt.rcParams['font.sans-serif'] = ['SimHei']  #解决中文显示乱码问题
# plt.rcParams['font.family'] = 'sans-serif' #设置字体格式
# plt.rcParams['axes.unicode_minus'] = False 



def read(path):
    x=[]
    train=[]
    test=[]
    val=[]
    with open(path,'r',encoding='utf-8') as f:
        datas=f.readlines()
        for i in range(1,len(datas)):
            x.append(int(datas[i].lstrip('\n').split('\t')[0]))
            train.append(float(datas[i].lstrip('\n').split('\t')[1])*100)
            test.append(float(datas[i].lstrip('\n').split('\t')[2])*100)
            val.append(float(datas[i].lstrip('\n').split('\t')[3])*100)
    return x,train,test,val

pc='./run/hand5000_change/GC/train.txt'#gcca
pcc='./run/hand5000_change/GCC/train.txt'#gcca
pccc='./run/hand5000_change/GCCC/train.txt'#gcac
pcccc='./run/hand5000_change/GCCCC/train.txt'#gcac

pa='./run/hand5000_change/GA/train.txt'#gacc
paa='./run/hand5000_change/GAA/train.txt'#gat-3
paaa='./run/hand5000_change/GAAA/train.txt'#gat-3
paaaa='./run/hand5000_change/GAAAA/train.txt'#gat-3

pcca='./run/hand5000_change/GCCA/train.txt'#gcn-3
pcac='./run/hand5000_change/GCAC/train.txt'#gcn-3
pacc='./run/hand5000_change/GACC/train.txt'#gcn-3


path0='./run/hand5000/GACC/train.txt'#no CHANGE

xc,trainc,testc,valc=read(pc)
xcc,traincc,testcc,valcc=read(pcc)
xccc,trainccc,testccc,valccc=read(pccc)
xcccc,traincccc,testcccc,valcccc=read(pcccc)

xa,traina,testa,vala=read(pa)
xaa,trainaa,testaa,valaa=read(paa)
xaaa,trainaaa,testaaa,valaaa=read(paaa)
xaaaa,trainaaaa,testaaaa,valaaaa=read(paaaa)

x0,train0,test0,val0=read(path0)

xacc,trainacc,testacc,valacc=read(pacc)

x=np.array(xc)
aC=np.array(trainc)
bC=np.array(testc)
cC=np.array(valc)

aCC=np.array(traincc)
bCC=np.array(testcc)
cCC=np.array(valcc)

aCCC=np.array(trainccc)
bCCC=np.array(testccc)
cCCC=np.array(valccc)

aCCCC=np.array(traincccc)
bCCCC=np.array(testcccc)
cCCCC=np.array(valcccc)
# ################################
aA=np.array(traina)
bA=np.array(testa)
cA=np.array(vala)

aAA=np.array(trainaa)
bAA=np.array(testaa)
cAA=np.array(valaa)

aAAA=np.array(trainaaa)
bAAA=np.array(testaaa)
cAAA=np.array(valaaa)

aAAAA=np.array(trainaaaa)
bAAAA=np.array(testaaaa)
cAAAA=np.array(valaaaa)
########################################

aACC=np.array(trainacc)
bACC=np.array(testacc)
cACC=np.array(valacc)

##############################
a0=np.array(train0)
b0=np.array(test0)
c0=np.array(val0)

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
# ax[0].plot(x, a0,color='b', label=u'Train-Acc')
# ax[0].plot(x, b0,color='k', label=u'Test-Acc')
# ax[0].plot(x, c0,color='r', label=u'Validate-Acc')
# ax[0].legend(loc=1, bbox_to_anchor=(1,0.2),borderaxespad = 0)
# # ax[0].set_title('Train_Acc-Epoch')
# ax[0].set_title('3D-PHGRID(Acc-Epoch)')

# ax[1].plot(x, aACC,color='b', label=u'Train-Acc')
# ax[1].plot(x, bACC,color='k', label=u'Test-Acc')
# ax[1].plot(x, cACC,color='r', label=u'Validate-Acc')
# ax[1].legend(loc=1, bbox_to_anchor=(1,0.2),borderaxespad = 0)
# # ax[1].set_title('Validate_Acc-Epoch')
# ax[1].set_title('3D-EHGRID(Acc-Epoch)')
# plt.savefig('./0.png')
# plt.show(block=True)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
ax[0].plot(x, aACC,color='b', label=u'G-ACC')
ax[0].plot(x, aCCC,color='k', label=u'G-CCC')
ax[0].plot(x, aAAA,color='r', label=u'G-AAA')
ax[0].legend(loc=1, bbox_to_anchor=(1,0.2),borderaxespad = 0)
ax[0].set_title('3D-PHGRID(Acc-Epoch)')

ax[1].plot(x, cACC,color='b', label=u'G-ACC')
ax[1].plot(x, cCCC,color='k', label=u'G-CCC')
ax[1].plot(x, cAAA,color='r', label=u'G-AAA')
ax[1].legend(loc=1, bbox_to_anchor=(1,0.2),borderaxespad = 0)
ax[1].set_title('3D-EHGRID(Acc-Epoch)')
plt.savefig('./1.png')
plt.show(block=True)



# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,10),sharex=True, sharey=True)

# ax[0,0].plot(x, aCCC,color='k', label=u'G-CCC')
# ax[0,0].plot(x, aAAA,color='m', label=u'G-AAA')
# ax[0,0].legend(loc=1, bbox_to_anchor=(1,0.32),borderaxespad = 0)
# ax[0,0].set_title('Train_Acc-Epoch')

# ax[0,1].plot(x, bCCC,color='k', label=u'G-CCC')
# ax[0,1].plot(x, bAAA,color='m', label=u'G-AAA')
# ax[0,1].legend(loc=1, bbox_to_anchor=(1,0.32),borderaxespad = 0)
# ax[0,1].set_title('Test_Acc-Epoch')

# ax[1,0].plot(x, cCCC,color='k', label=u'G-CCC')
# ax[1,0].plot(x, cAAA,color='m', label=u'G-AAA')
# ax[1,0].legend(loc=1, bbox_to_anchor=(1,0.32),borderaxespad = 0)
# ax[1,0].set_title('Validate_Acc-Epoch')

# ax[1,1].plot(x, cC,color='g', label=u'G-C')
# ax[1,1].plot(x, cCC,color='r', label=u'G-CC')
# ax[1,1].plot(x, cCCC,color='k', label=u'G-CCC')
# # ax[1,1].plot(x, cCCCC,color='g', label=u'G-CCCC')
# ax[1,1].plot(x, cA,color='c', label=u'G-A')
# ax[1,1].plot(x, cAA,color='y', label=u'G-AA')
# ax[1,1].plot(x, cAAA,color='m', label=u'G-AAA')
# ax[1,1].plot(x, cAAAA,color='sienna', label=u'G-AAAA')
# ax[1,1].legend(loc=1, bbox_to_anchor=(1,0.45),borderaxespad = 0)
# ax[1,1].set_title('Validate_Acc-Epoch')

# plt.show()





# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5),sharex=True, sharey=True)

# ax[0].plot(x1, a3,color='k', label=u'GACC-train')
# ax[0].plot(x1, a2,color='r', label=u'GCAC-train')
# ax[0].plot(x1, a1,color='g', label=u'GCAC-train')
# ax[0].plot(x1, a4,color='b', label=u'GCN_3-train')
# ax[0].legend(loc=1, bbox_to_anchor=(1,0.24),borderaxespad = 0)
# ax[0].set_title('Train_Acc-Epoch')

# ax[1].plot(x1, b3,color='k', label=u'GACC-test')
# ax[1].plot(x1, b2,color='r', label=u'GCAC-test')
# ax[1].plot(x1, b1,color='g', label=u'GCCA-test')
# ax[1].plot(x1, b4,color='b', label=u'GCN_3-test')
# ax[1].legend(loc=1, bbox_to_anchor=(1,0.24),borderaxespad = 0)
# ax[1].set_title('Test_Acc-Epoch')
# plt.show()

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5),sharex=True, sharey=True)
# ax[0].scatter(x1, c3,color='k', label=u'GACC-validate')
# ax[0].scatter(x1, c2,color='r', label=u'GCAC-validate')
# ax[0].scatter(x1, c1,color='g', label=u'GCCA-validate')
# ax[0].scatter(x1, c4,color='b', label=u'GCN_3-validate')
# ax[0].legend(loc=1, bbox_to_anchor=(1,0.24),borderaxespad = 0)
# ax[0].set_title('Validate_Acc-Epoch')

# ax[1].plot(x1, a3,color='k', label=u'GACC-train')
# ax[1].plot(x1, b3,color='r', label=u'GACC-test')
# ax[1].scatter(x1, c3,color='b', label=u'GACC-validate')
# ax[1].legend(loc=1, bbox_to_anchor=(1,0.24),borderaxespad = 0)
# ax[1].set_title('GACC_Acc-Epoch')
# plt.show()

# yL='acc'
# xL='epoch'
# plt.title(xL+'-'+yL, size=12)  #曲线标题
#     # 其中参数loc用于设置legend的位置，bbox_to_anchor用于在bbox_transform坐标（默认轴坐标）中为图例指定任意位置。
# plt.legend(loc=1, bbox_to_anchor=(1,0.2),borderaxespad = 0)
# plt.xlabel(xL, size=12)
#     #设置x轴标签旋转角度和字体大小
# # plt.xticks(rotation=0, fontsize=8)
# plt.ylabel(yL+'/%', size=12)
# # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(80))# 设置横坐标间隔（每隔80个横坐标显示一个横坐标，解决横坐标显示重叠问题）
# # plt.bar(['fall','other'],[4800,7800],color=['green','blue'],width=0.8)
# # plt.ylabel('instances')

# plt.show(block=True)