from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from minicnn.nn.tensor import Tensor


class Module:
    def __init__(self):
        self.training = True
        self._parameters: OrderedDict[str, Tensor] = OrderedDict()
        self._modules: OrderedDict[str, Module] = OrderedDict()

    def add_parameter(self, name: str, tensor: Tensor):
        self._parameters[name] = tensor
        return tensor

    def add_module(self, name: str, module: 'Module'):
        self._modules[name] = module
        return module

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'{self.__class__.__name__}.forward() is not implemented')

    def parameters(self) -> list[Tensor]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def children(self) -> list['Module']:
        return list(self._modules.values())

    def modules(self) -> list['Module']:
        modules = [self]
        for module in self._modules.values():
            modules.extend(module.modules())
        return modules

    def named_parameters(self, prefix: str = '') -> list[tuple[str, Tensor]]:
        items: list[tuple[str, Tensor]] = []
        for name, tensor in self._parameters.items():
            items.append((f'{prefix}{name}', tensor))
        for name, module in self._modules.items():
            child_prefix = f'{prefix}{name}.'
            items.extend(module.named_parameters(prefix=child_prefix))
        return items

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> dict[str, object]:
        return {name: tensor.data for name, tensor in self.named_parameters()}


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def __iter__(self) -> Iterable[Module]:
        return iter(self._modules.values())

    def __getitem__(self, index: int) -> Module:
        return list(self._modules.values())[index]

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
