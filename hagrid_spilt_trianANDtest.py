import argparse
from omegaconf import DictConfig, ListConfig, OmegaConf
import os
import json
import pandas as pd
import cv2
import sys

def run_convert(args):
    conf = OmegaConf.load(args.cfg)
    bbox_format = args.bbox_format
    labels = {label: num for (label, num) in zip(conf.dataset.targets, range(len(conf.dataset.targets)))}
    # print(labels)
    dataset_folder = conf.dataset.dataset_folder
    dataset_annotations=conf.dataset.dataset_annotations
    annotations_all = None
    exists_images = []
    # print(bbox_format,labels,dataset_folder,dataset_annotations)
    for i in os.listdir(dataset_folder):
        target_json=dataset_annotations+'/'+i+'.json'
        # print(target_json)
        exists_images.extend( i.split('.')[0] for i in os.listdir(dataset_folder+'/'+i))
        # print(exists_images)
        if os.path.exists(target_json):
            print(1)
            json_annotation = json.load(open(os.path.join(target_json)))
            for j in json_annotation:
                if j in exists_images:
                    txt_dir=os.path.join(dataset_folder+'/'+i+'/'+j)
                    print(txt_dir)
                    image = cv2.imread(txt_dir+'.jpg')
                    height,width,_=image.shape
                    print(height,width)
                    f=open(txt_dir+'.txt','w',encoding='utf-8')
                    for index in range(len(json_annotation[j]['bboxes'])):
                        box=json_annotation[j]['bboxes'][index]
                        label=labels[json_annotation[j]['labels'][index]]
                        print(box,label)
                        x,y,w,h=box
                        x=x+w/2
                        y=y+h/2
                        f.write(str(label)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')
                    f.close()
    #     else:
    #         logging.warning(f"Database for {j} not found")
    # print(len(exists_images))
def run_convert2023(args):
    conf = OmegaConf.load(args.cfg)
    bbox_format = args.bbox_format
    labels = {label: num for (label, num) in zip(conf.dataset.targets, range(len(conf.dataset.targets)))}
    # print(labels)
    dataset_folder = conf.dataset.dataset_folder
    dataset_annotations=conf.dataset.dataset_annotations
    annotations_all = None
    exists_images = []
    # print(bbox_format,labels,dataset_folder,dataset_annotations)
    for i in os.listdir(dataset_folder):
        target_json=[dataset_annotations+'/'+'train/'+i+'.json',
                    dataset_annotations+'/'+'test/'+i+'.json',
                    dataset_annotations+'/'+'val/'+i+'.json']
        # print(target_json)
        exists_images.extend( i.split('.')[0] for i in os.listdir(dataset_folder+'/'+i))
        # print(exists_images)
        for target_json1 in target_json:
            if os.path.exists(target_json1):
                print(1)
                json_annotation = json.load(open(os.path.join(target_json1)))
                for j in json_annotation:
                    if j in exists_images:
                        txt_dir=os.path.join(dataset_folder+'/'+i+'/'+j)
                        print(txt_dir)
                        image = cv2.imread(txt_dir+'.jpg')
                        height,width,_=image.shape
                        print(height,width)
                        f=open(txt_dir+'.txt','w',encoding='utf-8')
                        for index in range(len(json_annotation[j]['bboxes'])):
                            box=json_annotation[j]['bboxes'][index]
                            label=labels[json_annotation[j]['labels'][index]]
                            print(box,label)
                            x,y,w,h=box
                            x=x+w/2
                            y=y+h/2
                            f.write(str(label)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')
                        f.close()
            # sys.exit()
    #     else:
    #         logging.warning(f"Database for {j} not found")
    # print(len(exists_images))
import random       
def split_dataset(args):
    conf = OmegaConf.load(args.cfg)
    exists_images=[conf.dataset.data_keep+'/'+i for i in os.listdir(conf.dataset.data_keep) if i.split('.')[1]!='txt']
    # print(exists_images)
    index=[i for i in range(len(exists_images))]
    random.shuffle(index)
    val=[exists_images[i] for i in index[0:int(len(index)*0.2)]]
    train=[exists_images[i] for i in index[int(len(index)*0.2):len(index)]]
    with open('val.txt','w',encoding='utf-8') as f:
        for i in val:
           f.write(i+'\n')
    with open('train.txt','w',encoding='utf-8') as f:
        for i in train:
           f.write(i+'\n')
    print(len(val),len(train))
import shutil
def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))

def mymovefile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.move(srcfile, dstpath + '/'+fname)          # move文件
        print ("move %s -> %s"%(srcfile, dstpath +'/'+fname))
def creat_data(args):
    conf = OmegaConf.load(args.cfg)
    dataset_folder = conf.dataset.dataset_folder
    data_keep=conf.dataset.data_keep
    exists_images = []
    for i in os.listdir(dataset_folder):
        exists_images.extend( dataset_folder+'/'+i+'/'+j for j in os.listdir(dataset_folder+'/'+i) if j.split('.')[1]!='txt')
   
    # print(exists_images)
    if not os.path.exists(data_keep):
         os.makedirs(data_keep)
    for i in exists_images:
       txt_path=i.split('.jpg')[0]+'.txt'
       mymovefile(i,data_keep)
       mymovefile(txt_path,data_keep)

   




if __name__=='__main__':
    parser = argparse.ArgumentParser("Convert Hagrid annotations to Yolo annotations format", add_help=False)
    parser.add_argument("--bbox_format", default="cxcywh", type=str, help="bbox format: xyxy, cxcywh, xywh")
    parser.add_argument("--cfg", default="hand.yaml", type=str, help="path to data config")
    args = parser.parse_args()
    # run_convert2023(args)
    # creat_data(args)####将所有的jpg和txt全部转移到image—txt中
    split_dataset(args)