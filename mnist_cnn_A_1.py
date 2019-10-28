import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

n_epochs = 50
batch_size = 100
learning_rate = 0.005
momentum = 0.9
interval = 100

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)


### change the targets(class to "1" if the image represents the number 3, and "0" otherwise)
y_new = np.zeros(train_loader.dataset.targets.shape)
y_new[np.where(train_loader.dataset.targets==3)] = 1
train_loader.dataset.targets = y_new

y_new = np.zeros(test_loader.dataset.targets.shape)

y_new[np.where(test_loader.dataset.targets==3)] = 1
test_loader.dataset.targets = y_new

train_loader.dataset.targets = train_loader.dataset.targets.T
test_loader.dataset.targets = test_loader.dataset.targets.T


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)

network = Net()

# optimizer
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# define loss binary cross entropy
B_loss = nn.BCELoss()

train_losses = []
train_counter = []
train_accuracy = []

# network training
def train(epoch):
  network.train()
  correct = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)

    #print(output)
    output = output.view(-1)
    #print(output)
    #print(target)
    loss = B_loss(output, target.float())
    loss.backward()
    optimizer.step()
    pred = output.data>0.8
    #correct += pred.eq(target.data.view_as(pred)).sum()
    correct += (target == pred).sum()
    acurracy = (float(correct*100) / float(batch_size*(batch_idx+1)))

    if batch_idx % interval == 0:
      print('\tEpoch : ', epoch, '\t [',batch_idx*len(data), '/', len(train_loader.dataset), '', round(100*batch_idx / len(train_loader),1), '% ]', '\t\tTrain Loss: ', round(loss.item(),6), '\tTrain Accuracy: ',round(acurracy,5),'% ')
     
      '''    
      train_losses.append(loss.item())
      train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      train_accuracy.append(acurracy)
      
      torch.save(network.state_dict(), 'results/model_c1.pth')
      torch.save(optimizer.state_dict(), 'results/optimizer_c1.pth')
      '''
  train_losses.append(loss.item())
  train_counter.append(epoch)
  train_accuracy.append(acurracy)
def evaluate():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      output = output.view(-1)
      test_loss += B_loss(output, target.float()).item()
      #pred = output.data.max(1, keepdim=True)[1]
      pred = output.data>0.8
      #correct += pred.eq(target.data.view_as(pred)).sum()
      correct += (target == pred).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: ',  test_loss  , '\t[ Accuracy: ', correct.item(), '/', len(test_loader.dataset), ' ( ', 100 * correct.item() / len(test_loader.dataset) ,'%)')
  
# train the model
for epoch in range(1, n_epochs + 1):
  train(epoch)

# evaluate the model
evaluate() 

# plot loss of the model during training
plt.figure()
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.plot(train_counter, train_losses, color='blue')
plt.legend(['Train Loss'], loc='upper right')
plt.xlabel('number of epochs')
plt.ylabel('model loss')

# plot accuracy of the model during training 
plt.figure()
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.plot(train_counter, train_accuracy, color='red')
plt.legend(['Train Accuracy'], loc='upper right')
plt.xlabel('number of epochs')
plt.ylabel('model Accuracy')

plt.show()