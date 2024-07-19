import torch


class MaskedModel(torch.nn.Module):
    def __init__(self, model):
        super(MaskedModel, self).__init__()
        self.model = model
        self.masks = {}

        # 使用您的方法找到最后连续的全连接层名字
        dense_layer_names = self._find_last_consecutive_dense_layers()
        # 为这些全连接层创建掩码
        for name, module in self.model.named_modules():
            if name in dense_layer_names:
                mask = torch.nn.Parameter(torch.ones_like(module.weight.data, requires_grad=True))
                self.register_parameter(f"mask_{name.replace('.', '_')}", mask)
                self.masks[name] = mask

    def forward(self, x):
        original_weights = dict()
        for name, module in self.model.named_modules():
            if name in self.masks:
                original_weights[name] = module.weight.data
                masked_weight = original_weights[name] * torch.sigmoid(self.masks[name])
                module.weight.data = masked_weight
        x = self.model(x)
        for name, module in self.model.named_modules():
            if name in self.masks:
                module.weight.data = original_weights[name]
        return x

    def _find_last_consecutive_dense_layers(self):
        # 从模型的子模块中获取所有层的名称和模块，并反转列表以从后向前遍历
        layers = list(self.model.named_modules())[::-1]

        # 初始化列表以存储全连接层的名称
        dense_layer_names = []

        # 遍历模型的层
        for name, module in layers:
            # 如果遇到全连接层，则记录其名称
            if isinstance(module, torch.nn.Linear) and "dense" in name:
                dense_layer_names.append(name)
            # 如果遇到非全连接层且已有全连接层被记录，停止搜索
            elif "dense" in name:
                continue
            elif "dense" not in name:
                break

        # 返回找到的全连接层名称，顺序反转回正常顺序
        return dense_layer_names[::-1]