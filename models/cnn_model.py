from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights, alexnet, AlexNet_Weights, resnet50, \
    ResNet50_Weights


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
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.adapted_model_para = {name: None for name, _ in self.named_parameters()}

    def forward(self, x):
        return self.model(x)

    def adapted_forward(self, x):
        # 执行特征提取部分的自适应前向传播
        for i, layer in enumerate(self.model.features):
            if isinstance(layer, nn.Conv2d):
                x = F.conv2d(x, self.adapted_model_para[f'features.{i}.weight'],
                             self.adapted_model_para[f'features.{i}.bias'], stride=layer.stride, padding=layer.padding)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
            elif isinstance(layer, nn.MaxPool2d):
                x = F.max_pool2d(x, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)

        # 准备输入到分类器
        x = torch.flatten(x, 1)

        # 执行分类部分的自适应前向传播
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(
            F.linear(x, self.adapted_model_para['classifier.1.weight'], self.adapted_model_para['classifier.1.bias']))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(
            F.linear(x, self.adapted_model_para['classifier.4.weight'], self.adapted_model_para['classifier.4.bias']))
        x = F.linear(x, self.adapted_model_para['classifier.6.weight'], self.adapted_model_para['classifier.6.bias'])

        return x


class ResNet18(torch.nn.Module, AdaptedModel):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)

        # 创建一个字典来保存自适应参数
        self.adapted_model_para = {name: None for name, val in self.named_parameters()}

    def forward(self, x):
        return self.model(x)

    def adapted_forward(self, x):
        # Begin with the initial convolution and batch norm
        x = F.conv2d(x, self.adapted_model_para['model.conv1.weight'], None, stride=2, padding=3)
        x = F.batch_norm(x, self.model.bn1.running_mean, self.model.bn1.running_var, self.model.bn1.weight,
                         self.model.bn1.bias,
                         training=self.model.bn1.training, momentum=self.model.bn1.momentum, eps=self.model.bn1.eps)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # Process each block of the ResNet18
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for block_index in range(2):  # Each layer in ResNet18 has 2 blocks
                block = getattr(self.model, layer_name)[block_index]
                identity = x

                # First sub-layer of the block
                out = F.conv2d(x, block.conv1.weight, None,
                               stride=block.conv1.stride, padding=block.conv1.padding)
                out = F.batch_norm(out, block.bn1.running_mean, block.bn1.running_var, block.bn1.weight, block.bn1.bias,
                                   training=block.bn1.training, momentum=block.bn1.momentum, eps=block.bn1.eps)
                out = F.relu(out)

                # Second sub-layer of the block
                out = F.conv2d(out, block.conv2.weight, None,
                               stride=block.conv2.stride, padding=block.conv2.padding)
                out = F.batch_norm(out, block.bn2.running_mean, block.bn2.running_var, block.bn2.weight, block.bn2.bias,
                                   training=block.bn2.training, momentum=block.bn2.momentum, eps=block.bn2.eps)

                # Shortcut connection
                if block.downsample is not None:
                    identity = F.conv2d(x, block.downsample[0].weight, None, stride=block.downsample[0].stride)
                    identity = F.batch_norm(identity, block.downsample[1].running_mean, block.downsample[1].running_var,
                                            block.downsample[1].weight, block.downsample[1].bias,
                                            training=block.downsample[1].training,
                                            momentum=block.downsample[1].momentum,
                                            eps=block.downsample[1].eps)

                x = F.relu(out + identity)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = F.linear(x, self.adapted_model_para['model.fc.weight'], self.adapted_model_para['model.fc.bias'])
        return x


class ResNet50(nn.Module, AdaptedModel):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # 使用预训练权重初始化 ResNet50 模型
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # ResNet50 最后一层全连接层的输入通道数为 2048
        self.model.fc = nn.Linear(2048, num_classes)
        # 创建一个字典来保存自适应参数
        self.adapted_model_para = {name: None for name, _ in self.named_parameters()}

    def forward(self, x):
        return self.model(x)

    def adapted_forward(self, x):
        # 开始：初始卷积层 + BN + ReLU + 最大池化
        x = F.conv2d(x, self.adapted_model_para['model.conv1.weight'], None, stride=2, padding=3)
        x = F.batch_norm(x,
                         self.model.bn1.running_mean,
                         self.model.bn1.running_var,
                         self.model.bn1.weight,
                         self.model.bn1.bias,
                         training=self.model.bn1.training,
                         momentum=self.model.bn1.momentum,
                         eps=self.model.bn1.eps)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # 遍历四个层，每个层中包含若干个 Bottleneck block
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.model, layer_name)
            for block in layer:
                identity = x

                # 第1个卷积层（1x1 卷积）
                out = F.conv2d(x, block.conv1.weight, None,
                               stride=block.conv1.stride,
                               padding=block.conv1.padding)
                out = F.batch_norm(out,
                                   block.bn1.running_mean,
                                   block.bn1.running_var,
                                   block.bn1.weight,
                                   block.bn1.bias,
                                   training=block.bn1.training,
                                   momentum=block.bn1.momentum,
                                   eps=block.bn1.eps)
                out = F.relu(out)

                # 第2个卷积层（3x3 卷积）
                out = F.conv2d(out, block.conv2.weight, None,
                               stride=block.conv2.stride,
                               padding=block.conv2.padding)
                out = F.batch_norm(out,
                                   block.bn2.running_mean,
                                   block.bn2.running_var,
                                   block.bn2.weight,
                                   block.bn2.bias,
                                   training=block.bn2.training,
                                   momentum=block.bn2.momentum,
                                   eps=block.bn2.eps)
                out = F.relu(out)

                # 第3个卷积层（1x1 卷积）
                out = F.conv2d(out, block.conv3.weight, None,
                               stride=block.conv3.stride,
                               padding=block.conv3.padding)
                out = F.batch_norm(out,
                                   block.bn3.running_mean,
                                   block.bn3.running_var,
                                   block.bn3.weight,
                                   block.bn3.bias,
                                   training=block.bn3.training,
                                   momentum=block.bn3.momentum,
                                   eps=block.bn3.eps)

                # 如果 block 存在 downsample，则对 shortcut 进行处理
                if block.downsample is not None:
                    identity = F.conv2d(x, block.downsample[0].weight, None,
                                        stride=block.downsample[0].stride)
                    identity = F.batch_norm(identity,
                                            block.downsample[1].running_mean,
                                            block.downsample[1].running_var,
                                            block.downsample[1].weight,
                                            block.downsample[1].bias,
                                            training=block.downsample[1].training,
                                            momentum=block.downsample[1].momentum,
                                            eps=block.downsample[1].eps)

                # 将 shortcut 与卷积输出相加后再经过 ReLU
                x = F.relu(out + identity)

        # 全局平均池化、展平，并使用自适应参数的全连接层计算输出
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = F.linear(x, self.adapted_model_para['model.fc.weight'], self.adapted_model_para['model.fc.bias'])
        return x


class VGG16(nn.Module, AdaptedModel):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.model = vgg16(weights=VGG16_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(4096, num_classes)  # 修改最后一个全连接层

        self.adapted_model_para = {name: None for name, val in self.model.named_parameters()}

    def forward(self, x):
        return self.model(x)

    def adapted_forward(self, x):
        # 通过自适应参数调整第一层卷积和BN
        x = F.conv2d(x, weight=self.adapted_model_para['features.0.weight'],
                     bias=self.adapted_model_para['features.0.bias'], stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # 逐层调用自适应特性
        x = self._adapted_features_forward(x, 'features', start=1, end=31)  # VGG16卷积层部分

        # 平均池化后进入分类器
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.linear(x, weight=self.adapted_model_para["classifier.0.weight"],
                     bias=self.adapted_model_para["classifier.0.bias"])
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.linear(x, weight=self.adapted_model_para["classifier.3.weight"],
                     bias=self.adapted_model_para["classifier.3.bias"])
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.linear(x, weight=self.adapted_model_para["classifier.6.weight"],
                     bias=self.adapted_model_para["classifier.6.bias"])
        return x

    def _adapted_features_forward(self, x, features_name, start, end):
        features = getattr(self.model, features_name)
        for i in range(start, end):
            if isinstance(features[i], nn.Conv2d):
                x = F.conv2d(x, weight=self.adapted_model_para[f'{features_name}.{i}.weight'],
                             bias=self.adapted_model_para[f'{features_name}.{i}.bias'], stride=features[i].stride,
                             padding=features[i].padding)
                x = F.relu(x)
            elif isinstance(features[i], nn.MaxPool2d):
                x = F.max_pool2d(x, kernel_size=features[i].kernel_size, stride=features[i].stride,
                                 padding=features[i].padding)
        return x
