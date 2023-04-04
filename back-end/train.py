import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms
import os
from torchvision.utils import save_image

base_dir = r'/kaggle/input/weatherdata/dataset3'
# base_dir = r'./dataset'
train_dir = os.path.join(base_dir , 'train')
test_dir = os.path.join(base_dir , 'test')   

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

train_ds =  torchvision.datasets.ImageFolder(
        train_dir, transform=train_augs)
test_ds =  torchvision.datasets.ImageFolder(
        test_dir, transform=test_augs)

id_to_class= dict((v, k) for k, v in train_ds.class_to_idx.items())

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
nn.init.xavier_uniform_(model.fc.weight)
if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.CrossEntropyLoss()
# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
learning_rate = 5e-5
params_1x = [param for name, param in model.named_parameters()
     if name not in ["fc.weight", "fc.bias"]]
optimizer = torch.optim.SGD([{'params': params_1x},
                           {'params': model.fc.parameters(),
                            'lr': learning_rate * 10}],
                        lr=learning_rate, weight_decay=0.001)

def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
#    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total
        
        
    test_correct = 0
    test_total = 0
    test_running_loss = 0 
    
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            # print(y,y_pred)
            test_total += y.size(0)
            test_running_loss += loss.item()
    
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total
    
        
    print('epoch: ', epoch, 
          'loss: ', round(epoch_loss, 3),
          'accuracy: ', round(epoch_acc, 3),
          'test_loss: ', round(epoch_test_loss, 3),
          'test_accuracy: ', round(epoch_test_acc, 3)
             )
        
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

train_loss = []
train_acc = []
test_loss = []
test_acc = []
batch_size = 32

train_dl = torch.utils.data.DataLoader(train_ds,
        batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds,
        batch_size=batch_size)

epochs = 20
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 model,
                                                                 train_dl,
                                                                 test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

plt.plot(range(1, 293), train_loss, label='train_loss')
plt.plot(range(1, 293), test_loss, label='test_loss')
plt.legend()

plt.plot(range(1, 293), train_acc, label='train_acc')
plt.plot(range(1, 293), test_acc, label='test_acc')
plt.legend()

weight_path = '/kaggle/working/resnet2.pth'
torch.save(model.state_dict(), weight_path)
os.mkdir('/kaggle/working/outing/')

model.eval()
weight_path = '/kaggle/working/outing/resnet.pth'
torch.save(model.state_dict(), weight_path)

from pathlib import Path
import zipfile
img_root = Path('/kaggle/working/outing')
with zipfile.ZipFile('resnet.zip', 'w') as z:
    for img_name in img_root.iterdir():
        z.write(img_name)

torch.save(model, './out/model.pkl') 