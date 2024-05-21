""" Run as python3 train_DA.py model_directory"""
import copy
import numpy as np
import matplotlib.pyplot as plt
from model import MultiviewNet
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyDataset, MyDatasetNormalRotationAndFlip
from metrics import accuracy, run_metrics, predict, criterion_DA, da_val_loss
from datetime import datetime
import os
import glob
import pandas as pd
import ast #json
import sys



#################################################################
############ Set run parameters #################################
#################################################################

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Running on GPU")
else:
    print("Running on CPU")

basepath = "/mnt/disk3/regina_data/DL/"

with open(basepath+'Segmentation/'+sys.argv[1]+'/hyperparams.txt') as f: 
    data = f.read() 

hyperparams = ast.literal_eval(data) 
#hyperparams = json.loads(data)   
hyperparams['learning_rate'] = hyperparams['learning_rate']*1e-1
hyperparams['epochs'] = 10
hyperparams['batch_size'] = 8
hyperparams['da_loss_factor'] = 0.1
hyperparams['seg_model_path'] = basepath+'Segmentation/'+sys.argv[1]+'/'
hyperparams['version'] = 'last_model-'


#################################################################
############ Load datsets #######################################
#################################################################

tv_id, _ = train_test_split(glob.glob(hyperparams["source_data_path"]), test_size=0.15, random_state=12345)
train_id, validation_id = train_test_split(tv_id, test_size=0.15, random_state=12345)
train = MyDatasetNormalRotationAndFlip(train_id, filters=hyperparams['filters'])
valid = MyDataset(validation_id, filters=hyperparams['filters'])
trainloader_source = DataLoader(train, batch_size=hyperparams['batch_size'], shuffle=True, drop_last=True, num_workers=4)
validloader_source = DataLoader(valid, batch_size=hyperparams['batch_size'], shuffle=True, drop_last=True, num_workers=4)

print('Train source dataloader size:', trainloader_source.dataset.__len__())

tv_target_id, _ = train_test_split(glob.glob(hyperparams["target_data_path"]), test_size=0.2, random_state=12345)
train_target_id, validation_target_id = train_test_split(tv_target_id, test_size=0.2, random_state=12345)
train_target = MyDatasetNormalRotationAndFlip(train_target_id, sim=False, filters=hyperparams['filters'])
valid_target = MyDataset(validation_target_id, sim=False, filters=hyperparams['filters'])
trainloader_target = DataLoader(train_target, batch_size=hyperparams['batch_size'], shuffle=True, drop_last=True, num_workers=4)
validloader_target = DataLoader(valid_target, batch_size=hyperparams['batch_size'], shuffle=True, drop_last=True, num_workers=4)

print('Train target dataloader size:', trainloader_target.dataset.__len__())

#################################################################
############ Compile Neural Network & train #####################
#################################################################

pretrain_path = hyperparams['seg_model_path'] + hyperparams['version'] + hyperparams['architecture']

neural_network = MultiviewNet(hyperparams['architecture'], len(hyperparams['filters'])).to(device)
neural_network.load_state_dict(torch.load(pretrain_path))
neural_network.freeze_decoder()

class_weights = torch.FloatTensor(hyperparams['weights']).cuda()#to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = torch.optim.Adam(neural_network.parameters(), lr = hyperparams['learning_rate'], weight_decay = hyperparams['weight_decay'])


def perform_train_DA(neural_network):

    t = time.time()
    best_model, best_epoch, best_dice = None, None, 0 #best model according to source validation
    loss_train, loss_train_seg, loss_train_da, acc_train = [], [], [], [] #training set metrics
    loss_val_da = [] #domain adaptation validation metrics
    loss_val, acc_val, dice_val, pres_val, recall_val= [], [], [], [], [] #source validation
    loss_val_t, acc_val_t, dice_val_t, pres_val_t, recall_val_t= [], [], [], [], [] #target validation

    for epoch in range(hyperparams['epochs']):

        neural_network.train()

        mean_loss_train, mean_accuracy_train_seg, mean_loss_train_seg,\
             mean_loss_train_da, train_steps = 0, 0, 0, 0, 0

        for data_s, data_t in zip(trainloader_source, trainloader_target):
            #neural_network.half()

            inputs_s, labels_s = data_s[0].to(device), data_s[1].to(device)
            inputs_t = data_t[0].to(device)#, data_t[1].to(device)

            optimizer.zero_grad()

            outputs_s, pred_s = neural_network(inputs_s)
            loss_seg = criterion(outputs_s.float(), labels_s)

            _, pred_t  = neural_network(inputs_t)
            loss_da = criterion_DA(pred_s, pred_t, device)
            loss = loss_seg + hyperparams['da_loss_factor']*loss_da

            mean_loss_train_seg += loss_seg.item()
            mean_loss_train_da += hyperparams['da_loss_factor']*loss_da.item()
            mean_loss_train += mean_loss_train_seg + mean_loss_train_da
            mean_accuracy_train_seg += accuracy(outputs_s, labels_s)
            train_steps += 1
            loss.backward()
            #neural_network.float()
            optimizer.step()

        dice, mean_loss_validation, mean_accuracy_validation, \
            precision, recall = run_metrics(neural_network,
                                            validloader_source, 
                                            criterion, 
                                            device)
        
        dice_t, mean_loss_validation_t, mean_accuracy_validation_t, \
            precision_t, recall_t = run_metrics(neural_network,
                                            validloader_target, 
                                            criterion, 
                                            device)
        
        mean_loss_val_da = da_val_loss(neural_network, validloader_source, \
                                  validloader_target, device)
        
        #metrics for the training set
        loss_train_seg.append(mean_loss_train_seg/train_steps)
        loss_train_da.append(mean_loss_train_da/train_steps)
        loss_train.append(mean_loss_train/train_steps)
        acc_train.append(mean_accuracy_train_seg/train_steps)

        #metrics for the source validation set
        loss_val.append(mean_loss_validation)
        acc_val.append(mean_accuracy_validation)
        dice_val.append(dice)
        pres_val.append(precision)
        recall_val.append(recall)

        #metrics for the target validation set
        loss_val_t.append(mean_loss_validation_t)
        acc_val_t.append(mean_accuracy_validation_t)
        dice_val_t.append(dice_t)
        pres_val_t.append(precision_t)
        recall_val_t.append(recall_t)

        loss_val_da.append(mean_loss_val_da)

        if len(loss_val)>11 and np.sum(np.diff(loss_val_da[-11:]+loss_val[-11:]))>0:
            break

        if dice > best_dice:

            best_model = copy.deepcopy(neural_network)
            best_epoch = epoch
            best_dice = dice

        print('Epoch ', epoch + 1, 'finished.')

    print('Finished Training')
    print('Time required:', time.time() - t)
    print('Best model obtained in epoch ', best_epoch, ' with a validation dice of ', best_dice)

    torch.save(best_model.state_dict(), basepath+'DomAdap/'+dt_string+'/DA_best_model-' + hyperparams['architecture'])
    torch.save(neural_network.state_dict(), basepath+'DomAdap/'+dt_string+'/DA_last_model-' + hyperparams['architecture'])

    hyperparams['best_epoch_da'] = best_epoch

    with open(basepath+'DomAdap/'+dt_string+'/DA_hyperparams.txt', 'w') as f:
        print(hyperparams, file=f)

    df = pd.DataFrame({'loss_train': loss_train,
                       'loss_train_seg': loss_train_seg,
                       'loss_train_da': loss_train_da,
                       'loss_val': loss_val, 
                       'acc_train': acc_train, 
                       'acc_val': acc_val, 
                       'dice_val': dice_val,
                       'pres_val': pres_val, 
                       'rec_val': recall_val,
                       'loss_val_t': loss_val_t, 
                       'acc_val_t': acc_val_t, 
                       'dice_val_t': dice_val_t,
                       'pres_val_t': pres_val_t, 
                       'rec_val_t': recall_val_t,
                       'loss_val_da': loss_val_da,
                       })
    
    df.to_csv(basepath+'DomAdap/'+dt_string+'/da_train.csv', index=False) 


#################################################################
############ Plot metrics of the run ############################
#################################################################

perform_train_DA(neural_network)
