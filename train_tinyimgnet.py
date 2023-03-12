import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from livelossplot import PlotLosses
import datetime

import sys
import splinecam

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_filename', type=str,
                    help='unique filename for experiments', 
                    default=str(datetime.datetime.now()).replace(' ','-'))
parser.add_argument('--epochs', type=int,default=20)
parser.add_argument('--ngpus', type=int, help='number of gpus to use for training', default=1)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model_name', type=str, default='vgg11')

params = parser.parse_args()


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
                num_epochs=25, return_best_val=False):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    liveloss = PlotLosses()
    
    if return_best_val:
        best_model_wts = copy.deepcopy(model.state_dict())
        
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device) 
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")

#                 print( (i+1)*100. / len(dataloaders[phase]), "% Complete" )
                sys.stdout.flush()
                
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
            
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if return_best_val:
                    best_model_wts = copy.deepcopy(model.state_dict())
                
        liveloss.update({
            'log loss': avg_loss,
            'val_log loss': val_loss,
            'accuracy': t_acc.cpu(),
            'val_accuracy': val_acc.cpu()
        })
                
        liveloss.draw()
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print(  'Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print('Best Val Accuracy: {}'.format(best_acc))
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    
    if return_best_val:
        model.load_state_dict(best_model_wts)
    
    return model


data_transforms = { 'train': transforms.Compose([transforms.ToTensor()]),
                    'val'  : transforms.Compose([transforms.ToTensor(),]) }

data_dir = './data/tiny-imagenet-200/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=params.batch_size, shuffle=True, pin_memory=True,
                                              num_workers=24, drop_last=False)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


if params.model_name == 'vgg11':
    model_ft = splinecam.models.vgg11_bn(input_res=64,n_class=200)
elif params.model_name == 'vgg16':
    model_ft = splinecam.models.vgg16_bn(input_res=64,n_class=200)
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

#Multi GPU
if params.ngpus > 1:
    model_ft = torch.nn.DataParallel(model_ft,
                                     device_ids=list(range(params.ngpus))
                                    )

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=params.lr)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print('training...')
model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=params.epochs,return_best_val=False)

torch.save(model_ft,f'./models/tinyimagenet_{params.model_name}_{params.log_filename}.pt')
