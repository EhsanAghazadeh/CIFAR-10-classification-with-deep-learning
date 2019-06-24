
import torchvision
import torchvision.transforms as transforms

import torch

device = torch.device("cpu")
device

def test_it(testloader, net):
  correct = 0
  total = 0
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))
  
  return 100 * correct / total

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

def init_weights(m):
  if type(m) == nn.Linear or type(m) == nn.Conv2d:
    torch.nn.init.xavier_uniform(m.weight)

net.apply(init_weights)

%time losses = train_data(trainloader, net, 5)

test_it(trainloader, net)

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def my_imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
images.size()

labels

my_imshow(torchvision.utils.make_grid(images))

import torch.optim as optim

def train_data(trainloader, net, epoch_numbers,
                criterion = nn.CrossEntropyLoss(),
                optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)):
    
    losses = []
    for epoch in range(epoch_numbers):
      
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
          inputs, labels = data
          optimizer.zero_grad()
        
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          losses.append(loss.item())
        
          running_loss += loss.item()
          if i % 500 == 499:
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0
         
    print("Finished Training")  
    return losses   

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
my_imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(32)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(32)))


%time losses = train_data(trainloader, net, 5)

test_it(trainloader, net)

net

def init_weights(m):
  if type(m) == nn.Linear or type(m) == nn.Conv2d:
    m.weight.data.fill_(0)
   
net.apply(init_weights)

losses = train_data(trainloader, net, 5)

test_it(testloader, net)

from math import floor

def make_net_2_layer(channels_number=6, filter_size=5, pooling_window_size=2):
  
  output_shape = ((32 - filter_size + 1) - pooling_window_size) / 2 + 1
  if int(((32 - filter_size + 1) - pooling_window_size) / 2 + 1) != float(((32 - filter_size + 1) - pooling_window_size) / 2 + 1):
    return None
  
  class Net(nn.Module):
    
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(3, channels_number, filter_size)
      self.pool = nn.MaxPool2d(pooling_window_size)
      self.fc1 = nn.Linear(channels_number * int(output_shape) * int(output_shape), 10)
      
    def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = x.view(-1, channels_number * int(output_shape) * int(output_shape))
      x = self.fc1(x)
      return x
  
  return Net()

channels_number = 5
max_accuracy = 0

for curr_channels_number in range(5, 16):
  curr_net = make_net_2_layer(curr_channels_number)
  if curr_net != None:
    curr_net.apply(init_weights)
    train_data(trainloader, curr_net, 5)
    curr_accuracy = test_it(testloader, curr_net)
  
    if curr_accuracy > max_accuracy:
      channels_number = curr_channels_number
      max_accuracy = curr_accuracy
    
  
print('Max Accuracy of the network on the 10000 test images: %d %%' % (
      max_accuracy))
print(channels_number)

filter_size = 3
max_accuracy = 0

for curr_filter_size in range(3, 10):
  curr_net = make_net_2_layer(channels_number=10, filter_size=curr_filter_size)
  if curr_net != None:
    curr_net.apply(init_weights)
    train_data(trainloader, curr_net, 5)
    curr_accuracy = test_it(testloader, curr_net)
  
    if curr_accuracy > max_accuracy:
      filter_size = curr_filter_size
      max_accuracy = curr_accuracy
    
  
print('Max Accuracy of the network on the 10000 test images: %d %%' % (
      max_accuracy))
print(filter_size)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

def init_weights(m):
  if type(m) == nn.Linear or type(m) == nn.Conv2d:
    torch.nn.init.xavier_uniform(m.weight)

net.apply(init_weights)

%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9))
test_it(testloader, net)

import torch.nn as nn
import torch.nn.functional as F


class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net_1()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

learning_rates = [3, 0.03, 0.0003]

losses_function = []

for i, _lr in enumerate(learning_rates):
  
#   net.apply(init_weights)
  net = Net_1()
  
  %time losses_function.append(train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=_lr, momentum=0.9))) 
  test_it(testloader, net)
  


for losses in losses_function:
  plt.plot(losses)
  plt.show()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net_1()
net.apply(init_weights)

%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9))
test_it(testloader, net)

%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9))
test_it(testloader, net)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9))
test_it(testloader, net)

%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9))
test_it(testloader, net)

import torch.nn as nn
import torch.nn.functional as F


class Net_2(nn.Module):
    def __init__(self, _aFunc):
        self.aFunc = _aFunc
        super(Net_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.aFunc(self.conv1(x)))
        x = self.pool(self.aFunc(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.aFunc(self.fc1(x))
        x = self.aFunc(self.fc2(x))
        x = self.fc3(x)
        return x


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net_2(F.tanh)
%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9))
test_it(testloader, net)

net = Net_2(F.leaky_relu)
%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9))
test_it(testloader, net)

net = Net_2(F.softplus)
%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9))
test_it(testloader, net)

import torch.nn as nn
import torch.nn.functional as F


class Net_3(nn.Module):
    def __init__(self, _aFunc):
        self.aFunc = _aFunc
        super(Net_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.aFunc(self.conv1(x)))
        x = self.pool(self.aFunc(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = self.aFunc(self.fc1(x))
        x = self.aFunc(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net_2(F.leaky_relu)
%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9))
test_it(testloader, net)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net_2(F.leaky_relu)
%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9))
test_it(testloader, net)

net = Net_2(F.leaky_relu)
%time train_data(trainloader, net, 5, optimizer = optim.SGD(net.parameters(), lr=0.01))
test_it(testloader, net)
