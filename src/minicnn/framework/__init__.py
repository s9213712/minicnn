from minicnn.framework.components import register_builtin_components
from minicnn.framework.registry import GLOBAL_REGISTRY

register_builtin_components()

__all__ = ['GLOBAL_REGISTRY', 'register_builtin_components']
