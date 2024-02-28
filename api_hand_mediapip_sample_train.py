import cv2
import mediapipe as mp
import os
import cmath
from utils.torch_utils import select_device, time_sync
from handneo4j.graph_build import neo4j_build
import random

def main():
    path='./Pointnet2/data/hand5000/'
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)

    cap = cv2.VideoCapture(0)
    total_name=[]
    name=input("请输入姓名：")
    names=[]
    names.append(name)
    sema=[]
    datas=[]
    while True:
        semanteme=input("请输入手势语义或输入end退出程序:")
        if semanteme!='end':
            paths=path+str(semanteme)
            if os.path.isdir(paths):
                pass
            else:
                os.makedirs(paths)
            nametxt=path+'hand.txt'
            with open(nametxt,'a',encoding='utf-8')as f:
                f.write(str(semanteme)+'\n')
        elif semanteme=='end':
            train_txt=path+'hand_train.txt'
            test_txt=path+'hand_test.txt'
            random.shuffle(total_name)
            lens=len(total_name)
            with open(train_txt,'a',encoding='utf-8')as f:
                for i in total_name[:int(lens*0.8)]:
                    f.write(i+'\n')
            with open(test_txt,'a',encoding='utf-8')as f:
                for i in total_name[int(lens*0.8):]:
                    f.write(i+'\n')
            return names,sema,datas


        sema.append(semanteme)
        data=[]
        num=5000
        t=1
        c=0
        i=0
        while True:
            ret,frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 因为摄像头是镜像的，所以将摄像头水平翻转
            # 不是镜像的可以不翻转
            frame= cv2.flip(frame,1)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # print(results.multi_hand_landmarks)
        #    if results.multi_handedness:
        #        for hand_label in results.multi_handedness:
        #            print(hand_label)
                    
            if results.multi_hand_landmarks:
                c+=1
                for hand_landmarks in results.multi_hand_landmarks:
                    # 关键点可视化
                    mp_drawing.draw_landmarks(frame, 
                                            hand_landmarks, 
                                            mp_hands.HAND_CONNECTIONS)
                                            
            if results.multi_hand_landmarks and len(data)<num and c%t==0:
                txtname=str(semanteme)+'_'+str(i)
                files=path+str(semanteme)+'/'+txtname+'.txt'
                total_name.append(txtname)
                f=open(files,'w',encoding='utf-8')

                for j in range(21):
                    f.write(str(hand_landmarks.landmark[j].x)+','
                            +str(hand_landmarks.landmark[j].y)+','
                            +str(hand_landmarks.landmark[j].z)+'\n')
                f.close()
                # print(len(list))
                data.append(paths)
                print(len(data))
                i+=1        
            elif len(data)==num:   
                datas.append(data)
                print(semanteme,'样本采集数量:',len(datas))
                break
            elif c>t:
                c=0  
            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        # cap.release()
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    name,sema,datas=main() #采集手势点云数据

    #  neo4j_build(name,sema,datas)# 构建知识图谱
    
