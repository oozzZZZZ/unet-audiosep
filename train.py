#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:38:36 2020

@author: t.yamamoto
"""

import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import os
from tqdm import tqdm
import numpy as np

import parameter as C
import myutils as ut
import network

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA is available:", torch.cuda.is_available())
    
    if not os.path.exists(C.MODEL_PATH):
        os.mkdir(C.MODEL_PATH)
    
    now = datetime.datetime.now()
    model_path = C.MODEL_PATH+"/model_"+now.strftime('%Y%m%d_%H%M%S')+".pt"
    print(model_path)
    
    train_loader,val_loader = ut.MyDataLoader()
    
    model = network.UnetConv2().to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=C.learning_rate)
    
    train_loss_list = []
    val_loss_list = []
    
    for epoch in tqdm(range(1, C.epochs+1),desc='[Training..]',leave=True):
        
        model.train()
        train_loss = 0
    
        for batch_idx, (speech, addnoise) in enumerate(train_loader):
            speech= speech.to(device)
            addnoise=addnoise.to(device)
            mask = model(addnoise).to(device)
            enhance = addnoise * mask
            
            loss = criterion(enhance, speech)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
    
        model.eval()
        eval_loss = 0
    
        with torch.no_grad():
            for speech, addnoise in val_loader:
                speech= speech.to(device)
                addnoise=addnoise.to(device)
                mask = model(addnoise).to(device)
                enhance = addnoise * mask
    
                loss = criterion(enhance, speech)
                eval_loss += loss.item()
            eval_loss /= len(val_loader)
            val_loss_list.append(eval_loss)
            tqdm.write('\nTrain set: Average loss: {:.6f}\nVal set:  Average loss: {:.6f}'
                               .format(train_loss,eval_loss))
    
        if epoch == 1:
                best_loss = eval_loss
                torch.save(model.state_dict(), model_path)
    
        else:
            if best_loss > eval_loss:
                torch.save(model.state_dict(), model_path)
                best_loss = eval_loss
    
        if epoch % 10 == 0: #10回に１回定期保存
            save_time = now.strftime('%Y%m%d_%H%M%S')
            epoch_model_path = C.MODEL_PATH+"/model_"+save_time+"_Epoch"+str(epoch)+".pt"
            torch.save(model.state_dict(), epoch_model_path)
            np.savez(C.MODEL_PATH+"/loss_"+save_time+".npz",train_loss=train_loss_list, val_loss=val_loss_list)
            
            
if __name__ == "__main__":
    main()
