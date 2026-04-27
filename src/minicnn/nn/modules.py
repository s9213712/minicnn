from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable, Iterator

import numpy as np

from minicnn.nn.tensor import Tensor


class Module:
    def __init__(self):
        self.training = True
        self._parameters: OrderedDict[str, Tensor] = OrderedDict()
        self._modules: OrderedDict[str, Module] = OrderedDict()
        self._buffers: OrderedDict[str, np.ndarray] = OrderedDict()

    def add_parameter(self, name: str, tensor: Tensor):
        self._parameters[name] = tensor
        return tensor

    def add_module(self, name: str, module: 'Module'):
        self._modules[name] = module
        return module

    def register_buffer(self, name: str, value):
        array = np.asarray(value, dtype=np.float32).copy()
        self._buffers[name] = array
        setattr(self, name, array)
        return array

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

    def buffers(self) -> list[np.ndarray]:
        items = list(self._buffers.values())
        for module in self._modules.values():
            items.extend(module.buffers())
        return items

    def modules(self) -> list['Module']:
        modules = [self]
        for module in self._modules.values():
            modules.extend(module.modules())
        return modules

    def named_modules(
        self,
        prefix: str = '',
        memo: set[int] | None = None,
    ) -> Iterator[tuple[str, 'Module']]:
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, module in self._modules.items():
            child_prefix = f'{prefix}.{name}' if prefix else name
            yield from module.named_modules(prefix=child_prefix, memo=memo)

    def named_parameters(self, prefix: str = '') -> list[tuple[str, Tensor]]:
        items: list[tuple[str, Tensor]] = []
        for name, tensor in self._parameters.items():
            items.append((f'{prefix}{name}', tensor))
        for name, module in self._modules.items():
            child_prefix = f'{prefix}{name}.'
            items.extend(module.named_parameters(prefix=child_prefix))
        return items

    def named_buffers(self, prefix: str = '') -> list[tuple[str, np.ndarray]]:
        items: list[tuple[str, np.ndarray]] = []
        for name, array in self._buffers.items():
            items.append((f'{prefix}{name}', array))
        for name, module in self._modules.items():
            child_prefix = f'{prefix}{name}.'
            items.extend(module.named_buffers(prefix=child_prefix))
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
        state = {name: tensor.data.copy() for name, tensor in self.named_parameters()}
        state.update({name: array.copy() for name, array in self.named_buffers()})
        return state

    def _resolve_target(self, target: str) -> tuple['Module', str]:
        if not target:
            raise KeyError('target must not be empty')
        parts = target.split('.')
        module: Module = self
        for part in parts[:-1]:
            if part not in module._modules:
                raise KeyError(f'unknown module path {target!r}')
            module = module._modules[part]
        return module, parts[-1]

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Copy arrays from state into the model's parameter tensors in-place."""
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        expected_keys = set(params) | set(buffers)
        missing = expected_keys - set(state)
        unexpected = set(state) - expected_keys
        if missing:
            raise KeyError(f'load_state_dict: missing keys: {sorted(missing)}')
        if unexpected:
            raise KeyError(f'load_state_dict: unexpected keys: {sorted(unexpected)}')
        for name, tensor in params.items():
            tensor.data = state[name].copy() if hasattr(state[name], 'copy') else state[name]
        for name in buffers:
            module, leaf = self._resolve_target(name)
            value = np.asarray(state[name], dtype=np.float32).copy()
            module._buffers[leaf] = value
            setattr(module, leaf, value)

    def apply(self, fn: Callable[['Module'], object]) -> 'Module':
        for module in self._modules.values():
            module.apply(fn)
        fn(self)
        return self

    def get_parameter(self, target: str) -> Tensor:
        module, name = self._resolve_target(target)
        if name not in module._parameters:
            raise KeyError(f'get_parameter: unknown parameter {target!r}')
        return module._parameters[name]

    def get_buffer(self, target: str) -> np.ndarray:
        module, name = self._resolve_target(target)
        if name not in module._buffers:
            raise KeyError(f'get_buffer: unknown buffer {target!r}')
        return module._buffers[name]


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self._module_list: list[Module] = []
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def add_module(self, name: str, module: 'Module'):
        replacing = name in self._modules
        added = super().add_module(name, module)
        if replacing:
            self._module_list = list(self._modules.values())
        else:
            self._module_list.append(module)
        return added

    def __iter__(self) -> Iterable[Module]:
        return iter(self._module_list)

    def __len__(self) -> int:
        return len(self._modules)

    def __contains__(self, module: object) -> bool:
        return module in self._module_list

    def __getitem__(self, index: int) -> Module:
        return self._module_list[index]

    def forward(self, x: Tensor) -> Tensor:
        for module in self._module_list:
            x = module(x)
        return x
