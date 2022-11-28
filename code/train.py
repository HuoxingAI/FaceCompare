import torch
import os,sys
from data_utils import my_data_loader
import numpy as np
from network import Facenet_V2
import pandas as pd
import random 
from tqdm import tqdm
from iresnet import iresnet100
from sklearn.metrics import roc_auc_score

epochs = 10
batch_size = 32
img_size = 112
pretrained = True
# backbone = "mobilenet"   # mobilenet   inception_resnetv1
embedding_size = 512






def gen_train_test_list(csv_path):
    df = pd.read_csv(csv_path)
    # sample2 = random.sample(img_pathDir, val_picknumber) 
    data_list = []
    for i in range(len(df)):
        folder_name,label = df.iloc[i,0],df.iloc[i,1]
        data_list.append([folder_name,label])
    random.shuffle(data_list)
    # 获取列表的label
    def takelabel(elem):
        return elem[1]
    data_list.sort(key=takelabel)
    
    sample1_num = sum( ele [1] for ele in data_list)
    sample0_num = len(data_list) - sample1_num

    mid1 = int(sample0_num*0.2)
    mid2 = sample0_num + int(sample1_num*0.8)

    train_list = data_list[mid1:mid2]
    test_list = data_list[0:mid1] + data_list[mid2:len(data_list)]


    fw = open(os.path.dirname(csv_path)+'/4train.csv',"w")
    fw.write("id,label\n")
    for data in train_list:
        fw.write("{},{}\n".format(data[0],data[1]))
    fw.close()

    fw = open(os.path.dirname(csv_path)+'/4test.csv',"w")
    fw.write("id,label\n")
    for data in test_list:
        fw.write("{},{}\n".format(data[0],data[1]))
    fw.close()
    return train_list,test_list


def train(trainSetDir,modelPath):
    """训练函数"""
    # 0. 划分数据集
    
    csv_path = os.path.join(trainSetDir,'annos.csv')
    data_path = os.path.join(trainSetDir,'data')
    train_list,test_list = gen_train_test_list(csv_path)
    
    # 1 读取数据
    train_data = my_data_loader(root_dir=data_path,
                                data_list=train_list,
                                batch_size=batch_size,img_size=img_size,train=True)

    test_data = my_data_loader(root_dir=data_path,
                               data_list=test_list,
                               batch_size=batch_size,img_size=img_size,train=False)

    # 2 创建模型
    
    model = Facenet_V2()
    # 3 计算设备：GPU或者CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device=",device)

    # 将模型放到指定计算设备：GPU或者CPU
    model = model.to(device)

 
    loss_fc = torch.nn.MSELoss()

    opt = "sgd"
    if opt=="sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    if opt=="adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    # 5 迭代（循环）
    last_score = 0
    for epoch in range(epochs):
        # 训练模型
        model.train() # 这行代码意味着模型会产生梯度，计算bn

        train_epoch_loss = 0
        train_bn = 0

        pred,label = None,None
        for sample_batch in tqdm(train_data): # sample_batch shape: [b,c,h,w] b=batch size c=channels
            x1 = sample_batch[0].to(device)  # 输入样本，序列化图像数据
            x2 = sample_batch[1].to(device)  
            y = sample_batch[2].to(device,dtype=torch.long) # 标签
            #梯度清零
            optimizer.zero_grad()
            train_outputs = model(x1,x2) # 前向推理, out shape：[1,num_classes]
            loss = loss_fc(train_outputs, y.float()) # 计算损失
           
            train_epoch_loss += loss.cpu().detach().numpy() # 将损失值从GPU上拷贝到CPU上
            train_bn += 1 # 统计有多少个批次的 比如50个样本，每批次10，那么就有5个批次

            # loss求导，反向
            loss.backward()
            optimizer.step()
            if pred is None:
                pred = train_outputs.cpu().detach().numpy()
            else:
                pred = np.append(pred,train_outputs.cpu().detach().numpy())

            if label is None:
                label = y.cpu().detach().numpy()
            else:
                label = np.append(label,y.cpu().detach().numpy())
        score = roc_auc_score(label,pred)
        print('train:',score)

        train_lo = train_epoch_loss / train_bn
       
        # # 验证模型
        model.eval() # 验证模式，不需要计算bn
        test_epoch_loss = 0
        test_bn = 0
        with torch.no_grad(): #不需要梯度计算
            pred , label = None,None
            for sample_batch in tqdm(test_data):
                tx1 = sample_batch[0].to(device)  # 输入样本，序列化图像数据
                tx2 = sample_batch[1].to(device) 
                
                ty = sample_batch[2].to(device,dtype=torch.long) # 标签
              
                test_outputs = model(tx1,tx2)

                # 计算损失
                loss2 = loss_fc(test_outputs, ty.float())

                if pred is None:
                    pred = test_outputs.cpu().detach().numpy()
                else:
                    pred = np.append(pred,test_outputs.cpu().detach().numpy())

                if label is None:
                    label = ty.cpu().detach().numpy()
                else:
                    label = np.append(label,ty.cpu().detach().numpy())
                test_epoch_loss += loss2.item()
                test_bn += 1
   
        test_score = roc_auc_score(label,pred)
        print('test',test_score)
       
        test_lo = test_epoch_loss / test_bn
        # test_loss.append(test_lo)

        print("【{}/{}】【训练损失{:.4f}】【测试损失{:.4f}】".format(
            epoch+1,epochs,train_lo,test_lo))
        print("model saved {}".format(modelPath))
        if test_score > last_score:
            torch.save(model.backbone.state_dict(),
                    modelPath)
        

    print('training finish !')







if __name__ == '__main__':
    trainSetDir = sys.argv[1]
    modelPath = sys.argv[2]
    train(trainSetDir,modelPath)
    # predict()