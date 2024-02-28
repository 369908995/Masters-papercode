import cv2
from utils import detect
import torch
import numpy
import random
import os
from utils.torch_utils import select_device, time_sync
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)# 0
# a = api_detect.detectapi_yolov7(weights='../yolov5-master/runs/train/fall_lexp/weights/best.pt')
b=detect.detectapi_yolov5(weight='./runs/yolov5s-cagn-gsconv/weights/best.pt')



def main():
    while 1:
        ret,img= cap.read()
        # if not ret:break
        t1 = time_sync()
        print(img)
        result,names=b.run([img])
        # print(result,names)
        if len(result):
            img=result[0][0]
            # print('检测到手势数量：',len(result))
        lis=[]
        for i in range(len(result)):
            ls=[]
            ls.append(result[0][1][i][0])
            ls.append(result[0][1][i][1])
            ls.append(result[0][1][i][2])
            lis.append(ls)
            # print('手势类别：',names[result[0][1][i][0]],'位置：',str(result[0][1][i][1]),'置信度：',result[0][1][i][2])
        # print('\n')
        t2 = time_sync()
        fps=int(1/(t2-t1))
        txt='FPS:'+str(fps)
        cv2.putText(img,txt,(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),)
        cv2.imshow("hand-gesture",img)
        
        # if cv2.waitKey(1)==ord('q'):
        #        cv2.destroyAllWindows()
        # ret1,buffer = cv2.imencode('.jpg',img)
        # frame = buffer.tobytes()
        # yield(num,lis,fps,img,frame)
if __name__=='__main__':
    main()
    # print()
