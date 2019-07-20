
import torch
from torch import  nn
import  torchvision
from torch import optim
from torch.nn import  functional as F

from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot

batch_size = 512
lr=0.1
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data',train=True,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size = batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data',train=False,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081,))
                               ])),
    batch_size = batch_size,shuffle=False)

# x, y = next(iter(train_loader))
# print(x.shape, y.shape, x.min(), x.max())
# plot_image(x, y, 'image_MNIST')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #[b, 1, 28, 28]
        #h1=w1*x+b1
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256, 128)
        #h2=w2*h1+b2
        self.fc3 = nn.Linear(128,64)
        #h3=w3*h2+b3
        self.fc4 = nn.Linear(64,10)

    def forward(self, x):
        # [b, 1, 28, 28]
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)

        return x

net = Net()
optimize = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
train_loss = []
for epoch in range(3):
    for bach_idx ,(x, y) in enumerate(train_loader):
        # print(x.shape, y.shape)
        # break
        #x:[b, 1, 28, 28] y:[512]
        #x:[b, 1, 28, 28]=>[b,28*28]
        x = x.view(x.size(0),-1)
        #b[b,10]
        x = net(x)
        #y =>[b,10] one hot
        y_onehot = one_hot(y)
        loss = F.mse_loss(x, y_onehot)

        optimize.zero_grad()
        loss.backward()
        optimize.step()
        train_loss.append(loss.item())
        if bach_idx % 10 == 0:
            print(epoch, bach_idx, loss.item())


plot_curve(train_loss)

correct_num = 0
for x,y in test_loader:
    #[b, 1, 28, 28]
    x = x.view(x.size(0),-1)
    #[b,10]
    out = net(x)
    pred=out.argmax(dim=1)
    correct_num += pred.eq(y).sum().float().item()
print('x:', x.shape, 'out:', out.shape, 'pred:',pred.shape, 'y:', y.shape)
total_num = len(test_loader.dataset)
acc = correct_num / total_num
print('acc:', acc, 'ir:',lr,'loss:',loss.item())

x, y =next(iter(test_loader))
out = net(x.view(x.size(0),-1))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
