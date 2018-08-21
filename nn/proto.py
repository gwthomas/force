import torch; nn = torch.nn; F = torch.nn.functional
from gtml.nn.util import batch_flatten


class MLP(nn.Module):
    def __init__(self, sizes, softmax_out=False):
        nn.Module.__init__(self)
        self.sizes = sizes
        self.layers = []
        for i in range(len(sizes)-1):
            layer = nn.Linear(sizes[i], sizes[i+1])
            setattr(self, 'fc' + str(i), layer)
            self.layers.append(layer)
        self.softmax_out = softmax_out

    def forward(self, x):
        for fc in self.layers[:-1]:
            x = F.relu(fc(x))
        x = self.layers[-1](x)
        if self.softmax_out:
            x = F.softmax(x)
        return x


class LeNet(nn.Module):
    def __init__(self, in_channels):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = None
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool(x)
        x = F.tanh(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = batch_flatten(x)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size()[1], 84)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AtariNet(nn.Module):
    def __init__(self, actions, in_channels=4):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = None
        self.fc2 = nn.Linear(512, actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = batch_flatten(x)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size()[1], 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LinearFunction(nn.Module):
    def __init__(self, input):
        nn.Module.__init__(self)
        self.fc = None

    def forward(self, x):
        x = batch_flatten(x)
        if self.fc is None:
            self.fc = nn.Linear(x.size()[1], 1)
        x = self.fc(x)
        return x
