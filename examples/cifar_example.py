# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
import torchvision
import statopt  
import argparse

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--opt', choices=['sgd', 'sasa', 'salsa'], default='sgd')
args = parser.parse_args()

#----------------------------------------------------
# Prepare datasets, download to the directory ../data 
print('Preparing data ...')
batch_size = 128
normalizer = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2023, 0.1994, 0.2010))
transform_train = torchvision.transforms.Compose(
                 [torchvision.transforms.RandomCrop(32, padding=4),
                  torchvision.transforms.RandomHorizontalFlip(),
                  torchvision.transforms.ToTensor(), normalizer,])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, 
                                        transform=transform_train)
sampler = torch.utils.data.sampler.RandomSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          sampler=sampler, num_workers=4)
transform_test = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(), normalizer,])
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, 
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

#-----------------------------------------------
# Choose device, network model and loss function
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = torchvision.models.resnet18().to(device)
loss_func = torch.nn.CrossEntropyLoss()

#--------------------------------------------------------
# Choose optimizer from the list ['sgd', 'sasa', 'salsa']
optimizer_name = args.opt
print('Using optimier {}'.format(optimizer_name))

if optimizer_name == 'sasa':
    testfreq = min(1000, len(trainloader))
    optimizer = statopt.SASA(net.parameters(), lr=1.0, 
                             momentum=0.9, weight_decay=5e-4, 
                             testfreq=testfreq)
elif optimizer_name == 'salsa':
    gamma = math.sqrt(batch_size/len(trainset))     
    testfreq = min(1000, len(trainloader))
    optimizer = statopt.SALSA(net.parameters(), lr=1e-3, 
                              momentum=0.9, weight_decay=5e-4, 
                              gamma=gamma, testfreq=testfreq)
else:
    optimizer_name = 'sgd'  # SGD with a Step learning rate scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50*len(trainloader),
                                                gamma=0.1, last_epoch=-1)

#----------------------------------
# Training the neural network model
print('Start training ...')

for epoch in range(150):
    # Reset accumulative running loss at beginning or each epoch
    running_loss = 0.0

    for (images, labels) in trainloader:
        # switch to train mode each time due to potential use of eval mode
        net.train()
    
        # Compute model outputs and loss function 
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = loss_func(outputs, labels)
    
        # Compute gradient with back-propagation 
        optimizer.zero_grad()
        loss.backward()
    
        # Call the step() method of different optimizers
        if optimizer_name == 'sgd':
            optimizer.step()
            scheduler.step()
        elif optimizer_name == 'sasa':
            optimizer.step()
        elif optimizer_name == 'salsa':
            def eval_loss(eval_mode=True):
                if eval_mode:
                    net.eval()
                with torch.no_grad():
                    loss = loss_func(net(images), labels)
                return loss
            optimizer.step(closure=eval_loss)

        # Accumulate running loss during each epoch
        running_loss += loss.item()
    print('    epoch {:3d}: average loss {:.3f}'.format(
           epoch + 1, running_loss / len(trainset))) 

print('Finished training.')
            
#-------------------------------------
# Compute accuracy on the test dataset
n_correct = 0
n_testset = 0
with torch.no_grad():
    for (images, labels) in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        n_testset += labels.size(0)
        n_correct += (predicted == labels).sum().item()

print('Accuracy of the model on {} test images: {} %'.format(
      n_testset, 100 * n_correct / n_testset))
