from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(torch.nn.Module):
    def __init__(self, output_num):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, 2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, output_num))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


class AdaptedModel(ABC):
    def __init__(self):
        super(AdaptedModel).__init__()

    @abstractmethod
    def adapted_forward(self, x):
        raise NotImplementedError

    def set_adapted_para(self, name, val):
        self.adapted_model_para[name] = val

    def del_adapted_para(self):
        for key, val in self.adapted_model_para.items():
            if self.adapted_model_para[key] is not None:
                self.adapted_model_para[key].grad = None
                self.adapted_model_para[key] = None


class LeafCNN1(torch.nn.Module, AdaptedModel):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """

    def __init__(self, num_classes):
        super(LeafCNN1, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 2048)
        self.output = torch.nn.Linear(2048, num_classes)

        # adapted_model_para is used to make self-model a non-leaf computational graph,
        # such that other trainable components using self-model can track the grad passing self-model,
        # e.g. a gating layer that changes the weights of self-model
        self.adapted_model_para = {name: None for name, val in self.named_parameters()}
        # ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
        # 'fc1.weight', 'fc1.bias', 'output.weight', 'output.bias']

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

    def adapted_forward(self, x):
        # forward using the adapted parameters
        x = self.pool(F.relu(self.conv1._conv_forward(
            x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para["conv1.bias"])))
        x = self.pool(F.relu(self.conv2._conv_forward(
            x, weight=self.adapted_model_para["conv2.weight"], bias=self.adapted_model_para["conv2.bias"])))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["fc1.weight"], bias=self.adapted_model_para["fc1.bias"]))
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["output.weight"], bias=self.adapted_model_para["output.bias"]))
        return x


class LeNet(LeafCNN1):
    """
    CNN model used in "(ICML 21)  Personalized Federated Learning using Hypernetworks":
    a LeNet-based (LeCun et al., 1998) network with two convolution and two fully connected layers.
    """

    def __init__(self, num_classes, n_kernels=32, in_channels=3, fc_factor=1, fc_factor2=1):
        super(LeNet, self).__init__(num_classes)
        in_channels = in_channels
        self.n_kernels = n_kernels
        self.fc_factor = fc_factor
        self.fc_factor2 = fc_factor2
        self.conv1 = torch.nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = torch.nn.Linear(2 * n_kernels * 5 * 5, 120 * self.fc_factor)
        self.fc2 = torch.nn.Linear(120 * self.fc_factor, 84 * self.fc_factor2)
        self.output = torch.nn.Linear(84 * self.fc_factor2, num_classes)

        # def __init__(self, num_classes, n_kernels=32, in_channels=3, fc_factor=1):
        #     super(AdaptedLeafCNN3, self).__init__(num_classes)
        #     self.n_kernels = n_kernels
        #     self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        #     self.pool = nn.MaxPool2d(2, 2)
        #     self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        #     self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 2048 * fc_factor)
        #     self.output = nn.Linear(2048 * fc_factor, num_classes)

        # adapted_model_para is used to make self-model a non-leaf computational graph,
        # such that other trainable components using self-model can track the grad passing self-model,
        # e.g. a gating layer that changes the weights of self-model

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2 * self.n_kernels * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

    def adapted_forward(self, x):
        # forward using the adapted parameters
        x = self.pool(F.relu(self.conv1._conv_forward(
            x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para["conv1.bias"])))
        x = self.pool(F.relu(self.conv2._conv_forward(
            x, weight=self.adapted_model_para["conv2.weight"], bias=self.adapted_model_para["conv2.bias"])))
        x = x.view(-1, 2 * self.n_kernels * 5 * 5)
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["fc1.weight"], bias=self.adapted_model_para["fc1.bias"]))
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["fc2.weight"], bias=self.adapted_model_para["fc2.bias"]))
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["output.weight"], bias=self.adapted_model_para["output.bias"]))
        return x


class AlexNet(torch.nn.Module, AdaptedModel):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.adapted_model_para = {name: None for name, _ in self.named_parameters()}

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def adapted_forward(self, x):
        x = F.relu(self.features[0]._conv_forward(
            x, weight=self.adapted_model_para["features.0.weight"], bias=self.adapted_model_para["features.0.bias"]))
        x = self.features[2](x)
        x = F.relu(self.features[3]._conv_forward(
            x, weight=self.adapted_model_para["features.3.weight"], bias=self.adapted_model_para["features.3.bias"]))
        x = self.features[5](x)
        x = F.relu(self.features[6]._conv_forward(
            x, weight=self.adapted_model_para["features.6.weight"], bias=self.adapted_model_para["features.6.bias"]))
        x = F.relu(self.features[8]._conv_forward(
            x, weight=self.adapted_model_para["features.8.weight"], bias=self.adapted_model_para["features.8.bias"]))
        x = F.relu(self.features[10]._conv_forward(
            x, weight=self.adapted_model_para["features.10.weight"], bias=self.adapted_model_para["features.10.bias"]))
        x = self.features[12](x)

        x = torch.flatten(x, 1)
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["classifier.1.weight"],
            bias=self.adapted_model_para["classifier.1.bias"]))
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["classifier.4.weight"],
            bias=self.adapted_model_para["classifier.4.bias"]))
        x = F.linear(
            x, weight=self.adapted_model_para["classifier.6.weight"], bias=self.adapted_model_para["classifier.6.bias"])

        return x
