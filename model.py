import torch.nn as nn
import torch.nn.functional as F

# S8 Models

class Model_S8(nn.Module):
    def __init__(self, method='batch'):
        super(Model_S8, self).__init__()

        self.dropout_value = 0.05

        # CONVOLUTION BLOCK 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.get_norm_layer(method, 16),  
            nn.Dropout(self.dropout_value),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.get_norm_layer(method, 16),    
            nn.Dropout(self.dropout_value),  
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 

        self.pool1 = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.get_norm_layer(method, 16),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.get_norm_layer(method, 16),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.get_norm_layer(method, 32),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)
        ) 

        self.pool2 = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.get_norm_layer(method, 32),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.get_norm_layer(method, 32),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.get_norm_layer(method, 32),
            nn.Dropout(self.dropout_value),
        ) 

        self.pool3 = nn.AvgPool2d(kernel_size=8)
        
        # CONVOLUTION BLOCK 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

    def get_norm_layer(self, method, out_channels):
        if method == 'batch':
            return nn.BatchNorm2d(out_channels)
        elif method == 'group':
            return nn.GroupNorm(4, out_channels)
        elif method == 'layer':
            return nn.GroupNorm(1, out_channels)

Model_S8_batch = Model_S8('batch')
Model_S8_group = Model_S8('group')
Model_S8_layer = Model_S8('layer')

# S7 Models

class Model_S7_1(nn.Module):
    def __init__(self):
        super(Model_S7_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x


class Model_S7_2(nn.Module):
    def __init__(self):
        super(Model_S7_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(9, 10, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x


class Model_S7_3(nn.Module):
    def __init__(self):
        super(Model_S7_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 10, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x


class Model_S7_4(nn.Module):
    def __init__(self):
        DROP = 0.01
        super(Model_S7_4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 10, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.transition1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.transition2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x
    

# S6 Models

class Model_S6(nn.Module):
    #This defines the structure of the NN.

    """
    WRITE IT AGAIN SUCH THAT IT ACHIEVES
    99.4% validation accuracy against MNIST
    Less than 20k Parameters
    You can use anything from above you want. 
    Less than 20 Epochs
    Have used BN, Dropout,
    (Optional): a Fully connected layer, have used GAP
    """

    def __init__(self):
        super(Model_S6, self).__init__()

        drop_out_value = 0.05

        # Input Block

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Input = 28x28x1 | Output = 26x26x16 | RF = 3x3

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Input = 26x26x16 | Output = 24x24x16 | RF = 5x5

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Input = 24x24x16 | Output = 22x22x16 | RF = 7x7

            nn.MaxPool2d(2, 2),

            # Input = 22x22x16 | Output = 11x11x16 | RF = 14x14
        )

        # CONVOLUTION BLOCK 1

        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 11x11x16 | Output = 11x11x8 | RF = 14x14

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 11x11x16 | Output = 9x9x16 | RF = 16x16

        )

        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 9x9x16 | Output = 7x7x16 | RF = 18x18

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

        )

        self.conv4 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 7x7x16 | Output = 7x7x16 | RF = 20x20

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 7x7x16 | Output = 5x5x16 | RF = 22x22

        )

        self.gap = nn.AvgPool2d(kernel_size=5)

        self.fc = nn.Linear(in_features=16, out_features=10, bias=False)

    def forward(self, x):
            
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
    
            x = self.gap(x)

            x = x.view(-1, 16)

            x = self.fc(x)

            return F.log_softmax(x, dim=-1)


# S5 Models

class Model_S5(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Model_S5, self).__init__()
        # The convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        # The fully connected layers
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # This defines the forward pass of the NN.
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
