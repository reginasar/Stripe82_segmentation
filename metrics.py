import numpy as np
import torch
import torch.nn as nn

def criterion_DA(x_s, x_t, device):

    labels_rand = torch.randint(2, size=(2*x_s.shape[0],), device=device)
    crit = nn.CrossEntropyLoss()
    loss_da = crit(torch.cat((x_s, x_t), 0), labels_rand)

    return loss_da


def accuracy(predb, yb):

    metric = 0

    for i in range(yb.shape[0]):

        metric += (predb[i,:,:,:].argmax(dim=0) == yb[i,:,:]).float().mean().item()

    return(metric/yb.shape[0])



def run_metrics(neural_network, dataloader, criterion, device):

    neural_network.eval()

    mean_loss_validation, mean_accuracy_validation, validation_steps = 0, 0, 0
    tpl, fnl, fpl = [], [], []

    with torch.no_grad():

        for data in dataloader:
            #neural_network.half()

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs, _ = neural_network(inputs)
            loss_validation = criterion(outputs.float(), labels)
            mean_loss_validation += loss_validation.item()
            mean_accuracy_validation += accuracy(outputs, labels)
            validation_steps += 1

            for m in range(labels.shape[0]):

                mask = labels[m]
                #print(mask.shape, outputs[m,:,:,:].shape)
                output = outputs[m,:,:,:].argmax(dim=0)
                #print(output.shape, [mask == 1])
                tpl.append(torch.sum(output[mask == 1] == 1).item())
                fnl.append(torch.sum(output[mask == 1] == 0).item())
                fpl.append(torch.sum(output[mask == 0] == 1).item())

    precision = np.mean(np.array(tpl)/(np.array(tpl) + np.array(fpl)))
    recall = np.mean(np.array(tpl)/(np.array(tpl) + np.array(fnl)))
    dice = np.mean(2*np.array(tpl)/(2*np.array(tpl)+np.array(fpl)+np.array(fnl)))

    return dice, mean_loss_validation/validation_steps, \
            mean_accuracy_validation/validation_steps, precision, recall

def run_metrics2(neural_network, dataloader, criterion, device):
    with torch.no_grad():

        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs, _ = neural_network(inputs)
            output = outputs.argmax(dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(output, 
                                                   labels, 
                                                   mode='binary', 
                                                   threshold=0.5)
            
        f1_score = smp.metrics.f1_score(tp,fp,fn,tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp,fp,fn,tn, reduction="macro")
        recall = smp.metrics.recall(tp,fp,fn,tn, reduction="micro-imagewise")


def predict(neural_network, testloader, device):
    neural_network.eval()
    neural_network.half()
    predicted_seg = []
    mask_seg = []
    data_seg = []

    with torch.no_grad():

        for data in testloader:

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs, _ = neural_network(inputs)

            for m in range(labels.shape[0]):

                mask = labels[m]
                #print(mask.shape, outputs[m,:,:,:].shape)
                output = outputs[m,:,:,:].argmax(dim=0)
                predicted_seg.append(output.cpu())
                mask_seg.append(mask.cpu())
                data_seg.append(inputs[m].cpu())
                #print(output.shape, [mask == 1])


    return data_seg, mask_seg, predicted_seg

def da_val_loss(neural_network, source_loader, target_loader, device):
    neural_network.eval()
    validation_steps = 0
    mean_loss_da = 0

    with torch.no_grad():

        for data_s, data_t in zip(source_loader, target_loader):
            neural_network.half()

            inputs_s = data_s[0].to(device)
            inputs_t = data_t[0].to(device)
            _, pred_s = neural_network(inputs_s)
            _, pred_t  = neural_network(inputs_t)

            loss_da = criterion_DA(pred_s, pred_t, device)
            mean_loss_da += loss_da.item()
            validation_steps += 1

    return mean_loss_da/validation_steps

    



def get_embeddings(neural_network, dataloader, criterion, device):
    neural_network.eval()

    predicted_seg = []
    mask_seg = []
    data_seg = []

    with torch.no_grad():

        for data in dataloader:

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs, _ = neural_network(inputs)

            for m in range(labels.shape[0]):

                mask = labels[m]
                #print(mask.shape, outputs[m,:,:,:].shape)
                output = outputs[m,:,:,:].argmax(dim=0)
                predicted_seg.append(output.cpu())
                mask_seg.append(mask.cpu())
                data_seg.append(inputs[m].cpu())
                #print(output.shape, [mask == 1])