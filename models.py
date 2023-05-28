from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, BatchNorm1d
from torchvision.models import resnet18


# Create CNN Model - Q2
class CNNModel(Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.conv1 = Conv2d(3, 32, kernel_size=(3, 3), padding=1)

        self.conv2 = Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.batch1 = BatchNorm2d(32)
        self.relu1 = ReLU()

        self.pool1 = MaxPool2d(2, stride=2)

        self.conv3 = Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.batch2 = BatchNorm2d(64)
        self.relu2 = ReLU()

        self.pool2 = MaxPool2d(2, stride=2)

        self.fc1 = Linear(64 * 56 * 56, 1024)
        self.batch3 = BatchNorm1d(1024)
        self.relu3 = ReLU()

        self.fc2 = Linear(1024, num_classes)
        # self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.batch2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = out.view(-1, 64 * 56 * 56)
        out = self.fc1(out)
        out = self.batch3(out)
        out = self.relu3(out)
        out = self.fc2(out)
        '''out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.drop(out)
        print(out.shape)'''
        return out


# Modified pre-trained ResNet 18 model for Q3
class ModifiedResnet18(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.freeze()
        # Modify last layer for fine tuning
        self.model.fc = Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
