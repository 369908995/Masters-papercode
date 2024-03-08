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


import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import pandas as pd
import numpy as np
myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttf')


plt.rcParams['font.sans-serif'] = ['SimHei']  #解决中文显示乱码问题
plt.rcParams['font.family'] = 'sans-serif' #设置字体格式
plt.rcParams['axes.unicode_minus'] = False 


#读文件
yolov5s= pd.read_csv('../yolov5-master/runs/hand/yolov5s/results.csv')
# # yolov5s_se=pd.read_csv('../yolov5-master/runs/hand/yolov5s-se/results.csv')
# # yolov5s_ca=pd.read_csv('../yolov5-master/runs/hand/yolov5s-ca/results.csv')
# # yolov5s_cabm=pd.read_csv('../yolov5-master/runs/hand/yolov5s-cabm/results.csv')
# # yolov5s_gnconv=pd.read_csv('../yolov5-master/runs/hand/yolov5s-gnconv/results.csv')
# yolov5s_cagn= pd.read_csv('../yolov5-master/runs/hand/yolov5s-cagn/results.csv')
# yolov5s_cagn_gsconv= pd.read_csv('../yolov5-master/runs/hand/yolov5s-cagn-gsconv/results.csv')

# yolov5s= pd.read_csv('../yolov5-master/runs/train/s_/results.csv')
# # yolov5s_cagn= pd.read_csv('../yolov5-master/runs/train/s_cagn/results.csv')
# yolov5s_cagn_gsconv= pd.read_csv('../yolov5-master/runs/train/yolov5s-cagn-gsconv/results.csv')
# yolov5s_C2F_gsconv=pd.read_csv('../yolov5-master/runs/train/yolov5s-C2F-gsconv/results.csv')
# yolov5s_C3HB_gsconv=pd.read_csv('../yolov5-master/runs/train/yolov5s-C3HB-gsconv/results.csv')
# yolov5s_C3TR_gsconv=pd.read_csv('../yolov5-master/runs/train/yolov5s-C3TR-gsconv/results.csv')
# yolov5s_ConvNextBlock_gsconv=pd.read_csv('../yolov5-master/runs/train/yolov5s-ConvNextBlock-gsconv/results.csv')
# yolov5s_CoT3_gsconv=pd.read_csv('../yolov5-master/runs/train/yolov5s-CoT3-gsconv/results.csv')
# yolov5s_HorBlock_gsconv=pd.read_csv('../yolov5-master/runs/train/yolov5s-HorBlock-gsconv/results.csv')
# yolov5s_MobileOne_gsconv=pd.read_csv('../yolov5-master/runs/train/yolov5s-MobileOne-gsconv/results.csv')

yolov5s_cagn_1= pd.read_csv('../yolov5-master/runs/hand/yolov5s-cagn/results.csv')
yolov5s_cagn_2= pd.read_csv('../yolov5-master/runs/hand/yolov5s-cagn-1/results.csv')
yolov5s_cagn_3= pd.read_csv('../yolov5-master/runs/hand/yolov5s-cagn-2/results.csv')
# print(x.head())s


t1=0
p=7
# xdata = yolov5s.iloc[t1:,0] #横坐标 时间是列名
# y_yolov5s=yolov5s.iloc[t1:,p]*100
# # y_yolov5s_se=yolov5s_se.iloc[t1:,p]*100
# # y_yolov5s_ca=yolov5s_ca.iloc[t1:,p]*100
# # y_yolov5s_cabm=yolov5s_cabm.iloc[t1:,p]*100
# # y_yolov5s_gnconv=yolov5s_gnconv.iloc[t1:,p]*100
# y_yolov5s_cagn=yolov5s_cagn.iloc[t1:,p]*100
# y_yolov5s_cagn_gsconv=yolov5s_cagn_gsconv.iloc[t1:,p]*100


xdata = yolov5s.iloc[t1:,0] #横坐标 时间是列名
y_yolov5s=yolov5s.iloc[t1:,p]*100
# y_yolov5s_cagn_gsconv=yolov5s_cagn_gsconv.iloc[t1:,p]*100
# y_yolov5s_C2F_gsconv=yolov5s_C2F_gsconv.iloc[t1:,p]*100
# y_yolov5s_C3HB_gsconv=yolov5s_C3HB_gsconv.iloc[t1:,p]*100
# y_yolov5s_C3TR_gsconv=yolov5s_C3TR_gsconv.iloc[t1:,p]*100
# y_yolov5s_ConvNextBlock_gsconv=yolov5s_ConvNextBlock_gsconv.iloc[t1:,p]*100
# y_yolov5s_CoT3_gsconv=yolov5s_CoT3_gsconv.iloc[t1:,p]*100
# y_yolov5s_HorBlock_gsconv=yolov5s_HorBlock_gsconv.iloc[t1:,p]*100
# y_yolov5s_MobileOne_gsconv=yolov5s_MobileOne_gsconv.iloc[t1:,p]*100

y_yolov5s_cagn_1=yolov5s_cagn_1.iloc[t1:,p]*100
y_yolov5s_cagn_2=yolov5s_cagn_2.iloc[t1:,p]*100
y_yolov5s_cagn_3=yolov5s_cagn_3.iloc[t1:,p]*100
# # color可自定义折线颜色，marker可自定义点形状，label为折线标注
# plt.plot(xdata, y_yolov5s, color='r', label=u'YOLOv5s')#点标记,红色
# plt.plot(xdata, y_yolov5s_cagn, color='b', label=u'YOLOv5-CAGn-1')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_cagn_gsconv, color='g', label=u'YOLOv5-CAGn-gsconv')#星形标记,蓝色
# # plt.plot(xdata, y_yolov5s_cabm, color='k', label=u'YOLOv5-cabm')#星形标记,蓝色
# # plt.plot(xdata, y_yolov5s_gnconv, color='y', label=u'YOLOv5-gnconv')#星形标记,蓝色

# color可自定义折线颜色，marker可自定义点形状，label为折线标注
plt.plot(xdata, y_yolov5s, color='k', label=u'YOLOv5s')#点标记,红色plt.plot(xdata, y_yolov5s_cagn_gsconv, color='r', label=u'YOLOv5-CAGn-GSconv')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_C2F_gsconv, color='g', label=u'YOLOv5-C2F-GSconv')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_C3HB_gsconv, color='y', label=u'YOLOv5-C3HB-GSconv')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_C3TR_gsconv, color='b', label=u'YOLOv5-C3TR-GSconv')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_ConvNextBlock_gsconv, color='c', label=u'YOLOv5-ConvNextBlock-GSconv')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_MobileOne_gsconv, color='lime', label=u'YOLOv5-MobileOne-GSconv')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_HorBlock_gsconv, color='tan', label=u'YOLOv5-HorBlock-GSconv')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_CoT3_gsconv, color='grey', label=u'YOLOv5-CoT3-GSconv')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_HorBlock_gsconv, color='pink', label=u'YOLOv5-HorBlock-GSconv')#星形标记,蓝色plt.plot(xdata, y_yolov5s_MobileOne_gsconv, color='lime', label=u'YOLOv5-MobileOne-GSconv')#星形标记,蓝色
# plt.plot(xdata, y_yolov5s_cagn_gsconv, color='r', label=u'YOLOv5-CAGn-GSconv')#星形标记,蓝色
plt.plot(xdata, y_yolov5s_cagn_3, color='b', label=u'YOLOv5-CAGn-3')#星形标记,蓝色
plt.plot(xdata, y_yolov5s_cagn_2, color='g', label=u'YOLOv5-CAGn-2')#星形标记,蓝色
plt.plot(xdata, y_yolov5s_cagn_1, color='r', label=u'YOLOv5-CAGn-1')#星形标记,蓝色



# plt.title(u"mAP@.5:.95-epoch", size=12)  #曲线标题
    #其中参数loc用于设置legend的位置，bbox_to_anchor用于在bbox_transform坐标（默认轴坐标）中为图例指定任意位置。
plt.legend(loc=1, bbox_to_anchor=(1,0.4),borderaxespad = 0)
plt.xlabel(u'epoch', size=12)
#     #设置x轴标签旋转角度和字体大小
# plt.xticks(rotation=0, fontsize=8)
plt.ylabel(u'mAP@.5:.95/%', size=12)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))# 设置横坐标间隔（每隔80个横坐标显示一个横坐标，解决横坐标显示重叠问题）
# plt.bar(['fall','other'],[4800,7800],color=['green','blue'],width=0.8)
# plt.ylabel('instances')
plt.savefig(r"./2.png")
plt.show()
