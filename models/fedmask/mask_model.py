import torch
import torch.nn.functional as F


class MaskedModel(torch.nn.Module):
    def __init__(self, model):
        super(MaskedModel, self).__init__()
        self.model = model
        self.masks = {}

        dense_layer_names = self._find_last_consecutive_dense_layers()
        for name, module in self.model.named_modules():
            if name in dense_layer_names:
                mask = torch.nn.Parameter(torch.ones_like(module.weight, requires_grad=True))
                self.register_parameter(f"mask_{name.replace('.', '_')}", mask)
                self.masks[name] = mask

    def forward(self, x):
        # 我们需要跟踪模型中当前的子模块路径
        def apply_mask(module, input, output):
            for name, m in self.model.named_modules():
                if m is module:
                    if name in self.masks:
                        mask = torch.sigmoid(self.masks[name])
                        return F.linear(input[0], module.weight * mask, module.bias)
            return output

        # 为所有全连接层添加forward hook
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_path = name.split('.')
                hook = module.register_forward_hook(apply_mask)
                hooks.append(hook)

        # 执行前向传播
        x = self.model(x)

        # 移除所有hooks
        for hook in hooks:
            hook.remove()
        return x

    def parameters(self, recurse: bool = True):
        for name, param in super().named_parameters():
            if 'mask' in name:
                yield param

    def _find_last_consecutive_dense_layers(self):
        layers = list(self.model.named_modules())[::-1]
        dense_layer_names = []

        for name, module in layers:
            if isinstance(module, torch.nn.Linear) and "dense" in name:
                dense_layer_names.append(name)
            elif "dense" in name:
                continue
            elif "dense" not in name:
                break
        return dense_layer_names[::-1]
