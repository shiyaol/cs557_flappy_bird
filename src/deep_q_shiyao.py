import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2), nn.ReLU(inplace=True))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2,padding=1), nn.ReLU(inplace=True))
        #self.max_pool_2 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1), nn.ReLU(inplace=True))
        #self.max_pool_3 = nn.MaxPool2d(kernel_size=2,padding=1)
        self.fc1 = nn.Sequential(nn.Linear(1600, 256), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(256, 2)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv1(input)
        #print(output.size())
        output = self.max_pool_1(output)
        #print(output.size())
        output = self.conv2(output)
        #print(output.size())
        #output = self.max_pool_2(output)
        #print(output.size())
        output = self.conv3(output)
        #output = self.max_pool_3(output)
        #print(output.size())
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)

        return output
