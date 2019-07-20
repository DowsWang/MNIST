
import torch
from torch import  nn
import  torchvision
from torch import optim
from torch.nn import  functional as F

from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot

batch_size = 190
lr=0.05
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

w1, b1 = torch.randn(200, 784, requires_grad=True),\
         torch.randn(200, requires_grad=True)
w2, b2 = torch.randn(64, 200, requires_grad=True),\
         torch.randn(64, requires_grad=True)
w3, b3 = torch.randn(10, 64,requires_grad=True),\
         torch.randn(10, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

def forward(x):
    #[b,784] @ [200, 784]t = [b, 200]
    x = x@w1.t() + b1
    x = F.relu(x)
    #[b,200] @ [200, 64]t  = [b,64]
    x = x@w2.t() + b2
    x = F.relu(x)
    #[b,64] @ [64, 10]t = [b,10]
    x = x@w3.t() + b3
    x = F.relu(x)

    return x

optimizer = optim.SGD([w1, b1, w2, b2, w3, b3],lr=lr,momentum=0.9)
criteon = nn.CrossEntropyLoss()
for epoch in range(1000):
    for batch_idx, (x, y) in enumerate(train_loader):
        raw = x.view(-1, 28*28)
        logit = forward(raw)
        loss = criteon(logit, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            # print('lenx:', len(x), 'train_loader.dataset:', len(train_loader.dataset), 'train_loader:',len(train_loader))
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(x), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item()))
            print('Train Eporch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x),len(train_loader.dataset),
                       100 * batch_size / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0

    for x,y in test_loader:
        raw = x.view(-1, 28*28)
        logit = forward(raw)
        pred = logit.argmax(dim=1)
        #print(logit.shape,y.shape,pred.shape)
        test_loss += criteon(logit,y).item()
        correct += pred.eq(y).sum().float().item()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



