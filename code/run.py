import os,sys
import cv2
import torch

import numpy as np
# from tqdm import tqdm
from network import Facenet_V2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


curr_path = os.path.dirname(os.path.abspath(__file__))
mode_path = os.path.join(curr_path,'model-best.pth')
net = Facenet_V2(model_path=mode_path)
net = net.to(device)
# net.load_state_dict(torch.load(mode_path))
net.eval()



def load_img(img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img

@torch.no_grad()
def predict(file1,file2):
    
    with torch.no_grad(): #不需要梯度计算
        img1 = load_img(file1).to(device)
        img2 = load_img(file2).to(device)
        
        similarity_score = net(img1,img2).cpu().numpy()[0]

        print(file1,file2,similarity_score)
    return similarity_score

def main(to_pred_dir,result_save_path):
    subdirs = os.listdir(to_pred_dir) # name
    labels = []
    # coses = []
    # i = 0
    for subdir in subdirs:
        result = predict(os.path.join(to_pred_dir,subdir,"a.jpg"),os.path.join(to_pred_dir,subdir,"b.jpg"))
        labels.append(result)

    fw = open(result_save_path,"w")
    fw.write("id,label\n")
    for subdir,label in zip(subdirs,labels):
        fw.write("{},{}\n".format(subdir,label))
    fw.close()

    

def cal_accuracy_test(csv_pred, csv_label):
    import pandas as pd
    # 0. 划分数据集
    
    df_pred = pd.read_csv(csv_pred)
    pred_data_dict = {}
    for i in range(len(df_pred)):
        folder_name,cos = df_pred.iloc[i,0],df_pred.iloc[i,1]
        pred_data_dict[str(folder_name)] = cos

    df_label = pd.read_csv(csv_label)
    label_data_dict = {}
    for i in range(len(df_label)):
        folder_name,label = df_label.iloc[i,0],df_label.iloc[i,1]
        label_data_dict[str(folder_name)] = label

    best_acc = 0
    best_th = 0
    for k,v in pred_data_dict.items():
        th = v
        correct_num = 0
        for ki,vi in pred_data_dict.items():
            
            y_test = (vi >= th)
            if y_test == label_data_dict[ki]:
                 correct_num += 1
        acc = correct_num/len(pred_data_dict) 

        if acc > best_acc:
            best_acc = acc
            best_th = th
    # print
    print("测试阈值准确率",best_acc,best_th)

if __name__ == "__main__":
    to_pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    # to_pred_dir =  "../init_data/toUser/train/data"
    # result_save_path = "x.csv"
    main(to_pred_dir, result_save_path)
 

    # cal_accuracy_test(result_save_path,csv_label = "../init_data/toUser/train/annos.csv")
  