import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms

# 准备数据；定义模型和优化器；循环迭代：前向传播，计算损失函数，清空梯度，反向传播，更新梯度，打印损失函数值，存储模型

#   定义类MNISTDataset来封装数据集
from torch import Tensor
class MNISTDataset(torch.utils.data.Dataset):
    #构造函数
    def __init__(self, transform, data, label):
        super(MNISTDataset, self).__init__() #调用父类构造函数
        self.transform = transform
        self.images = data
        self.labels = label

    #相当于重载[]运算符
    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

    #实现了这个方法之后，对于一个这个类的对象obj，可以用len(obj)来获取长度
    def __len__(self):
        return len(self.images)

#   定义transform
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
 ])

#   加载已经准备好的数据
train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_label.npy')

#   封装训练集及其加载器、测试集及其加载器
trainset = MNISTDataset(transform=transform, data=train_data, label=train_label)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True
)
testset = MNISTDataset(transform=transform, data=test_data, label=test_label)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False
)

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


#网络结构
import torch.nn as nn
import torch.nn.functional as F
#   继承一个nn.Module，实现构造函数和forward方法，就是一个网络模型
class Net(nn.Module):
    #构造函数
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#迭代训练
net = Net()

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
running_loss = 0.0

for epoch in range(10):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        inputs = torch.tensor(inputs, dtype=torch.float32)##########
        outputs = net(inputs)
        #print(outputs, labels.squeeze())
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            print(epoch)
            print('[%d, %5d] loss: %.5f ' %(epoch+1, i+1, running_loss/1000))
            running_loss = 0.0


#存模型
PATH = './mnist_net.pth'
torch.save(net.state_dict(),PATH)


#测试
net = Net()
net.load_state_dict(torch.load(PATH))
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = torch.tensor(images, dtype=torch.float32)  ##########
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.squeeze())
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %4s : %2d %%'%(classes[i], 100*class_correct[i]/class_total[i]))
