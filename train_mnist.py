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
from torchvision.datasets import MNIST
import datetime

import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',type=str,default='./models/mlp-mnist')
parser.add_argument('--width',type=int,default=64)
parser.add_argument('--depth',type=int,default=5)

params = parser.parse_args()


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
                num_epochs=25, save_checkpoints=None, return_best_val=False, checkpoint_path=None):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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
                        scheduler.step(loss)

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
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if save_checkpoints is not None and phase== 'val':
                if epoch in save_checkpoints:
                    save_model = copy.deepcopy(model)
                    torch.save(save_model,f'{checkpoint_path}_{epoch}_{t_acc.cpu():.3f}_{val_acc.cpu():.3f}.pt')
                    
                
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

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
#         transforms.RandomHorizontalFlip(.5),
#         transforms.RandomResizedCrop((64,64)),
#         transforms.Resize(224, transforms.InterpolationMode.BICUBIC)
#         transforms.RandomRotation(10),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         transforms.RandomHorizontalFlip(.5),
#         transforms.RandomResizedCrop((64,64)),
#         transforms.RandomRotation(10),
#         transforms.Resize(224, transforms.InterpolationMode.BICUBIC)
    ])

data_transforms = { 'train': train_transform,
                    'val'  : val_transform }


image_datasets = {'train': MNIST('./data',train=True,transform=data_transforms['train'],download=True),
                  'val': MNIST('./data',train=False,transform=data_transforms['val'],download=True)
                 }
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True, pin_memory=True,
                                              num_workers=24, drop_last=False)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


width = params.width
depth = params.depth

model_ft = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,width),
    nn.ReLU(),
    *[nn.Linear(width,width),nn.ReLU()]*(depth-2),
    nn.Linear(width,10),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# optimizer = optim.SGD(model_ft.parameters(), lr=0.1,
#                       momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


dir_name = params.save_dir
try:
    os.mkdir(dir_name)
except FileExistsError:
    pass

timestamp = str(datetime.datetime.now()).replace(' ','-')

torch.save(model_ft,f'{dir_name}/{width}x{depth}_{0}_-1_-1.pt')
model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=51, checkpoint_path=f'{dir_name}/{width}x{depth}', save_checkpoints=[2,5,10,20,50])

