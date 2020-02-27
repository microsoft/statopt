# Statistical Adaptive Stochastic Gradient Methods

A package of PyTorch optimizers that can automatically schedule learning rates based on online statistical tests.

* main algorithms: SALSA and SASA
* auxiliary codes: QHM and SSLS

Companion paper: [Statistical Adaptive Stochastic Gradient Methods](https://www.microsoft.com/en-us/research/publication/statistical-adaptive-stochastic-gradient-methods) by Zhang, Lang, Liu and Xiao, 2020.

## Install

    pip install statopt

Or from Github:

    pip install git+git://github.com/microsoft/statopt.git#egg=statopt


## Usage of SALSA and SASA

Here we outline the key steps on CIFAR10.
Complete Python code is given in [examples/cifar_example.py](examples/cifar_example.py). 

### Common setups

First, choose a batch size and prepare the dataset and data loader as in [this PyTorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html):

```python
import torch, torchvision

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, ...)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, ...)
```

Choose device, network model, and loss function:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = torchvision.models.resnet18().to(device)
loss_func = torch.nn.CrossEntropyLoss()
```

### SALSA
Import ```statopt```, and initialize SALSA with a small learning rate and two extra parameters:

```python
import statopt

gamma = math.sqrt(batch_size/len(trainset))             # smoothing parameter for line search
testfreq = min(1000, len(trainloader))                  # frequency to perform statistical test 

optimizer = statopt.SALSA(net.parameters(), lr=1e-3,            # any small initial learning rate 
                          momentum=0.9, weight_decay=5e-4,      # common choices for CIFAR10/100
                          gamma=gamma, testfreq=testfreq)       # two extra parameters for SALSA
```

Training code using SALSA

```python
for epoch in range(100):
    for (images, labels) in trainloader:
        net.train()	# always switch to train() mode
    
        # Compute model outputs and loss function 
        images, labels = images.to(device), labels.to(device)
        loss = loss_func(net(images), labels)
    
        # Compute gradient with back-propagation 
        optimizer.zero_grad()
        loss.backward()
    
        # SALSA requires a closure function for line search
        def eval_loss(eval_mode=True):
            if eval_mode:
                net.eval()
            with torch.no_grad():
                loss = loss_func(net(images), labels)
            return loss

        optimizer.step(closure=eval_loss)

```

### SASA

SASA requires a good (hand-tuned) initial learning rate like most other optimizers, but do not use line search:

```python
optimizer = statopt.SASA(net.parameters(), lr=1.0,              # need a good initial learning rate 
                         momentum=0.9, weight_decay=5e-4,       # common choices for CIFAR10/100
                         testfreq=testfreq)                     # frequency for statistical tests
```

Within the training loop: ```optimizer.step()``` does NOT need any closure function.
