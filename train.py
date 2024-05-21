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
from metrics import accuracy, run_metrics, predict
from datetime import datetime
import os
import glob
import pandas as pd
import sys



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Running on GPU")

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
basepath = "/mnt/disk3/regina_data/DL/Segmentation/"

hyperparams = {
    'architecture' : sys.argv[2], #'timm-efficientnet-b1',
    'batch_size' : 16,
    'source_data_path' : '/mnt/disk3/regina_data/imfit_sim/mocks/*.fits.gz',
    'target_data_path' : '/mnt/disk3/regina_data/stripe82/chamba_stamps/*.fits.gz',
    'epochs' : 100,
    'learning_rate' : 1e-4,#10**(-4.5),
    'weight_decay' : 1e-5,
    'weights' : [0.075, 0.925],
    'outpath' : dt_string+"/",
    'filters' : [np.int32(sys.argv[1])], # 0=u, 1=g, 2=r, 3=i, 4=z
    'norm' : sys.argv[3], # "minmax", "minperc", "asinh" 
}


tv_id, test_id = train_test_split(glob.glob(hyperparams["source_data_path"]), test_size=0.15, random_state=12345)
train_id, validation_id = train_test_split(tv_id, test_size=0.15, random_state=12345)
train = MyDatasetNormalRotationAndFlip(train_id, filters=hyperparams['filters'], norm=hyperparams['norm'])
valid = MyDataset(validation_id, filters=hyperparams['filters'], norm=hyperparams['norm'])
trainloader = DataLoader(train, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=20)
validloader = DataLoader(valid, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=20)


print('Train dataloader size:', trainloader.dataset.__len__())
print('Validation dataloader size:', validloader.dataset.__len__())


neural_network = MultiviewNet(hyperparams['architecture'], len(hyperparams['filters'])).to(device)
class_weights = torch.FloatTensor(hyperparams['weights']).to(device) #.cuda()
criterion = nn.CrossEntropyLoss(weight = class_weights)#nn.BCELoss()##WithLogits pos_weight = class_weights[0]
optimizer = torch.optim.Adam(neural_network.parameters(), lr = hyperparams['learning_rate'], weight_decay = hyperparams['weight_decay'])

#summary(neural_network, (5, 576, 576))

def perform_train(neural_network):

    t = time.time()
    best_model, best_epoch, best_dice = None, None, 0 #best model according to source validation
    loss_train = []
    acc_train = []
    loss_val, acc_val, dice_val, pres_val, recall_val= [], [], [], [], [] #source validation



    for epoch in range(hyperparams['epochs']):

        neural_network.train()

        mean_loss_train, mean_accuracy_train, train_steps = 0, 0, 0

        for data in trainloader:
            #neural_network.half()

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs,_ = neural_network(inputs)
            #print(outputs)#inputs, labels, 
            #print(outputs.type(), labels.type())
            loss = criterion(outputs.float(), labels)
            mean_loss_train += loss.item()
            #print(loss.item())
            mean_accuracy_train += accuracy(outputs, labels)
            train_steps += 1
            loss.backward()
            #neural_network.float()
            optimizer.step()

        dice, mean_loss_validation, mean_accuracy_validation, \
            precision, recall = run_metrics(neural_network,
                                            validloader, 
                                            criterion, 
                                            device)

        loss_train.append(mean_loss_train/train_steps)
        loss_val.append(mean_loss_validation)
        acc_train.append(mean_accuracy_train/train_steps)
        acc_val.append(mean_accuracy_validation)
        dice_val.append(dice)
        pres_val.append(precision)
        recall_val.append(recall)

        print("Loss train:",loss_train[-1],", Loss val:", loss_val[-1])
        print("Acc train:",acc_train[-1]," , Acc val:", acc_val[-1])
        print("                                 Dice val:", dice_val[-1])

        if len(loss_val)>11 and np.all(np.diff(loss_val[-11:])>0):
            break

        if dice > best_dice:

            best_model = copy.deepcopy(neural_network)
            best_epoch = epoch
            best_dice = dice

        print('Epoch', epoch + 1, 'finished.\n')


    print('Finished Training')
    print('Time required:', (time.time() - t)/3600, 'hs')
    print('Best model obtained in epoch ', best_epoch, ' with a validation dice of ', best_dice)

    if not os.path.exists(basepath+hyperparams['outpath']):
        os.system("mkdir "+basepath+hyperparams['outpath'])

    torch.save(best_model.state_dict(), basepath+hyperparams['outpath']+'best_model-' + hyperparams['architecture'])
    torch.save(neural_network.state_dict(), basepath+hyperparams['outpath']+'last_model-' + hyperparams['architecture'])

    hyperparams['best_epoch_seg'] = best_epoch

    with open(basepath+hyperparams['outpath']+'hyperparams.txt', 'w') as f:
        print(hyperparams, file=f)


    df = pd.DataFrame({'loss_train': loss_train,
                    'loss_val': loss_val, 
                    'acc_train': acc_train, 
                    'acc_val': acc_val, 
                    'dice_val': dice_val,
                    'pres_val': pres_val, 
                    'rec_val': recall_val,
                    })
    
    df.to_csv(basepath+hyperparams['outpath']+'seg_train.csv', index=False) 


perform_train(neural_network)








