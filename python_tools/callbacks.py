import torch, torch.nn as nn
from typing import Callable, Union, List

from torch.nn.modules.module import Module
from . import Controllable

FLOAT_INPUT_TYPE = Union[None, float, Controllable, torch.Tensor]

class BendingCallback(nn.Module):
    def __init__(self, parameters: List[nn.Parameter] = None, **params) -> None:
        super().__init__()
        self._controllables = nn.ModuleDict()
        self._cache = []
        self._targets = []
        if parameters is not None:
            self._cache.extend([p.data.clone() for p in parameters])
            self._targets.extend(parameters)

    def parse_bending_parameter(self, param):
        if isinstance(param, (int, float)):
            return Controllable(name="prob", value=torch.tensor(float(param)))
        elif isinstance(param, torch.Tensor):
            return Controllable(name="prob", value=param)
        elif isinstance(param, Controllable):
            self.register_controllable(param)
            return param
        else:
            raise TypeError('received invalid prod argument of type %s'%type(param))

    def register_parameters(self, parameters: List[nn.Parameter]):
        if parameters is not None:
            self._cache.extend([p.data.clone() for p in parameters])
            self._targets.extend(parameters)

    def register_controllable(self, control):
        self._controllables[control.name] = control

    def get_controllables(self):
        return dict(self._controllables)

    def update(self, name, value):
        has_set = False
        for k, v in self._controllables.items():
            if v.name == name:
                self._controllables[k].set_value(torch.tensor(value))
                has_set = True
        if not has_set:
            raise RuntimeError('attribute %s not found in Controllable %s'%(name, self))

    def __add__(self, obj):
        if not isinstance(obj, BendingCallback):
            raise TypeError('%s can only be added to BendingCallback instances'%(type(self)))
        return BendingCallbackSequential([self, obj])


class BendingCallbackSequential(BendingCallback):
    def __init__(self, callbacks: List[BendingCallback], parameters: List[nn.Parameter] = None):
        super().__init__(parameters=parameters)
        for c in callbacks:
            if not isinstance(c, BendingCallback):
                raise TypeError('Found %s in callbacks list, only BendingCallbacks required.'%(type(c)))
        new_targets, new_cache, new_controllables = self._check_callback_targets(callbacks)
        self._targets.extend(new_targets)
        self._cache.extend(new_cache)
        self.callbacks = nn.ModuleList(callbacks)
        self._controllables.update(new_controllables)

    def _check_callback_targets(self, callbacks):
        target_list = None
        cache_list = None
        controllables = {}
        for c in callbacks:
            if target_list is None:
                target_list = c._targets
                assert len(c._cache) == len(c._targets), "found inconsistance in callbacks %s ; len(cache) != len(targets)"%c
                cache_list = c._cache
                controllables.update(c._controllables)
            else:
                targets_tmp = c._targets
                if targets_tmp != target_list:
                    raise RuntimeError('Callbacks of BendingCallbackSequential must have similar targets')
                assert len(c._cache) == len(c._targets), "found inconsistance in callbacks %s ; len(cache) != len(targets)"%c
                controllables.update(c._controllables)
        return target_list, cache_list, controllables

    def __repr__(self):
        return str("BendingCallbackSequential("+str(self.callbacks)+")")

    def apply(self):
        for i in range(len(self._targets)):
            current_param = self._cache[i]
            for c in self.callbacks:
                current_param = c(current_param)
            self._targets[i].data = current_param
        
    def forward(self, x: torch.Tensor):
        for c in self.callbacks:
            x = c(x)
        return x

    def __add__(self, obj):
        if isinstance(obj, BendingCallbackSequential):
            return BendingCallbackSequential(list(self.callbacks) + list(obj.callbacks))
        elif isinstance(obj, BendingCallback):
            return BendingCallbackSequential(list(self.callbacks) + [obj])
        

class Mask(BendingCallback):
    def __init__(self, parameters = None, prob: Union[float, torch.Tensor, Controllable] = 1.0, seed: int = 0):
        super().__init__(parameters)
        self.prob = self.parse_bending_parameter(prob)

    def __repr__(self):
        prob = self.prob.get_value()
        return "Mask(prob=%.3f)"%prob

    def apply(self):
        _prob_value: float = float(self.prob.get_value())
        for i in range(len(self._targets)):
            if torch.jit.is_scripting():
                self._targets[i].set_(self._cache[i] * torch.bernoulli(torch.full_like(self._targets[i], _prob_value)))
            else:
                self._targets[i].data = self._cache[i] * torch.bernoulli(torch.full_like(self._targets[i], _prob_value))

    def forward(self, x: torch.Tensor):
        _prob_value: float = float(self.prob.get_value())
        return x * torch.bernoulli(torch.full_like(x, _prob_value))


class Affine(BendingCallback):
    def __init__(self, parameters = None, scale: FLOAT_INPUT_TYPE = 1.0, bias: FLOAT_INPUT_TYPE = 0.0):
        super().__init__(parameters)
        self.scale = self.parse_bending_parameter(scale)
        self.bias = self.parse_bending_parameter(bias)

    def __repr__(self):
        scale = self.scale.get_value()
        bias = self.bias.get_value()
        return "Affine(scale=%.3f, bias=%.3f)"%(scale, bias)

    def apply(self):
        scale = float(self.scale.get_value())
        bias = float(self.bias.get_value())
        for i in range(len(self._targets)):
            if torch.jit.is_scripting():
                self._targets[i].set_(self._cache[i] * scale + bias)
            else:
                self._targets[i].data = self._cache[i] * scale + bias

    def forward(self, x: torch.Tensor):
        scale = float(self.scale.get_value())
        bias = float(self.bias.get_value())
        return x * scale + bias

class Gaussian(BendingCallback):
    def __init__(self, parameters = None, mean: FLOAT_INPUT_TYPE = 0.0, std: FLOAT_INPUT_TYPE = 0.0):
        super().__init__(parameters)
        self.mean = self.parse_bending_parameter(mean)
        self.std = self.parse_bending_parameter(std)
        self.std.min_clamp = 0

    def __repr__(self):
        mean = float(self.mean.get_value())
        std = float(self.std.get_value())
        return "Gaussian(scale=%.3f, bias=%.3f)"%(mean, std)

    def apply(self):
        mean = float(self.mean.get_value())
        std = float(self.std.get_value())
        for i in range(len(self._targets)):
            if torch.jit.is_scripting():
                self._targets[i].set_(self._cache[i] + mean + std * torch.randn_like(self._targets[i]))
            else:
                self._targets[i].data = self._cache[i] + mean + std * torch.randn_like(self._targets[i])

    def forward(self, x: torch.Tensor):
        mean = float(self.mean.get_value())
        std = float(self.std.get_value())
        return x + mean + std * torch.randn_like(x)


class Uniform(BendingCallback):
    def __init__(self, parameters = None, min: FLOAT_INPUT_TYPE = 1.0, max: FLOAT_INPUT_TYPE = 0.0):
        super().__init__(parameters)
        self.min = self.parse_bending_parameter(min)
        self.max = self.parse_bending_parameter(max)

    def __repr__(self):
        min = float(self.min.get_value())
        max = float(self.max.get_value())
        return "Gaussian(scale=%.3f, bias=%.3f)"%(min, max)

    def apply(self):
        min_value = float(self.min.get_value())
        max_value = float(self.max.get_value())
        for i in range(len(self._targets)):
            if torch.jit.is_scripting():
                self._targets[i].set_(self._cache[i] + (max_value - min_value) * torch.rand_like(self._targets[i])) - min_value
            else:
                self._targets[i].data = self._cache[i] + (max_value - min_value) * torch.rand_like(self._targets[i]) - min_value

    def forward(self, x: torch.Tensor):
        min_value = float(self.min.get_value())
        max_value = float(self.max.get_value())
        return x + (max_value - min_value) * torch.rand_like(x) - min_value

# class Scramble(BendingCallback):
#     def __init__(self, parameters=None, seed: int = 0, idx: int = 0):
#         super().__init__(parameters=parameters)
#         self.seed = torch.jit.Attribute(self.parse_bending_parameter(seed), Controllable)
#         self.idx = idx
#         self.register_controllable(self.seed)

#     def __repr__(self):
#         seed = int(self.seed.get_value() if torch.jit.is_scripting() else self.seed.value.get_value())
#         return "Scramble(seed=%d)"%seed
    
#     def apply(self):
#         generator = torch.Generator()
#         seed = int(self.seed.get_value() if torch.jit.is_scripting() else self.seed.value.get_value())
#         generator.manual_seed(seed)
#         for i in range(len(self._targets)):
#             permutation_mask = torch.randperm(self._targets.shape[self.idx], generator=generator)
#             if torch.jit.is_scripting():
#                 self._targets[i].set_(torch.index_select(self._targets[i], self.idx, permutation_mask))
#             else:
#                 self._targets[i].data = torch.index_select(self._targets[i], self.idx, permutation_mask)
        
#     def forward(self, x: torch.Tensor):
#         generator = torch.Generator()
#         seed = int(self.seed.get_value() if torch.jit.is_scripting() else self.seed.value.get_value())
#         generator.manual_seed(seed)
#         permutation_mask = torch.randperm(x.shape[self.idx], generator=generator)
#         return torch.index_select(x, self.idx, permutation_mask)



# TO TEST

# class Spy(object):
#     def __init__(self): 
#         self._cache = []
#         self.callback = None
#         self.current_method = None
#         self.forward = None
#         self.mode = "catch"

#     @property
#     def cache(self):
#         try:
#             return torch.cat(self._cache, 0)
#         except: 
#             return self._cache

#     def catch(self):
#         def spy_callback(x: torch.Tensor):
#             if self.mode == "catch":
#                 self._cache.append(x)
#                 return x
#             elif self.mode == "apply":
#                 return self.callback(x)
#             else:
#                 raise ValueError(f'mode {self.mode} not known')
#         return spy_callback

#     def apply(self, callback: Callable):
#         self.callback = callback(self.cache)
#         self.mode = "apply"

# class MetaSpy():
#     """combines information from different spies"""
#     pass


__all__ = ['Mask', 'Affine', 'Uniform', 'Gaussian']#, 'Scramble']