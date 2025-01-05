import torch
import torch.nn as nn
import wandb
import numpy as np
from tqdm import tqdm
from torch.optim import adam
import wandb
from argparse import Namespace
from argparse import ArgumentParser
import os
from datetime import datetime
def init_wandb(key:str):
    """
    params: key: wandb api key
    return:project name
    """
    # log in
    os.environ["WANDB_API_KEY"] =key
    wandb.login(key=os.environ['WANDB_API_KEY'])
    
    # name
    current_time = datetime.now()
    standard_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    name=standard_time

    return name


def train_epoch(model, criterion, optimizer, data_iter, confusion_matrix,num_classes=10, is_train=True, device='cuda'):
    total_sample = 0
    total_correct_sample = 0
    total_loss = 0
    
    if is_train:
        model.train()
    else:
        model.eval()

    # 开始训练或验证
    with torch.set_grad_enabled(is_train):  # 根据模式开启或禁用梯度计算
        for x, label in data_iter:
            x, label = x.to(device), label.to(device)

            y_pred = model(x)  # [bs, n_classes]
            loss = criterion(y_pred, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            _, prediction = torch.max(y_pred, dim=1)  
            total_sample += x.size(0)
            total_correct_sample += (prediction == label).sum().item()

            # 更新混淆矩阵
            for t, p in zip(label.cpu().numpy(), prediction.cpu().numpy()):
                confusion_matrix[t, p] += 1

    # 计算整体准确率
    accuracy = total_correct_sample / (total_sample + 1e-6)
    average_loss = total_loss / (len(data_iter) + 1e-6)

    # 计算每类的精确率和召回率
    precision = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=0) + 1e-6) # [n_classes]
    recall = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + 1e-6)

    return accuracy, average_loss, confusion_matrix, precision, recall








def train(model,criterion,optimizer,nums_epoch,nums_classes,train_iter,test_iter,config,device='cuda'):
    # init wandb
    wandb.init(project=config.project_name,name=config.name,config=config.__dict__)# 转换为dict    
    model_run_id=wandb.run.id

    model.train()
    train_confusion_matrix=np.zeros((nums_classes,nums_classes))# 用于最终统计
    test_confusion_matrix=np.zeros((nums_classes,nums_classes))
    for epoch in range(nums_epoch):
        epoch_train_accuracy,epoch_train_average_loss,\
        train_confusion_matrix,epoch_train_precision,\
        epoch_train_recall=train_epoch(model,criterion,optimizer,train_iter,confusion_matrix=train_confusion_matrix,\
                                       nums_classes=nums_classes,is_train=True,device=device)

        if test_iter is not None:
            epoch_test_accuracy,epoch_test_average_loss,\
            test_confusion_matrix,epoch_test_precision,\
            epoch_test_recall=train_epoch(model,criterion,optimizer,test_iter,confusion_matrix=test_confusion_matrix,
                                          nums_classes=nums_classes,is_train=False,device=device)
            
        wandb.log({'epoch':epoch+1,'train_accuracy':epoch_train_accuracy,'train_average_loss':epoch_train_average_loss,epoch_t})
    return model,model_run_id