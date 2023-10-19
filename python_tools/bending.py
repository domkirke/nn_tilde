import re, collections
import torch, torch.nn as nn, torch.fx as fx, nn_tilde
from typing import Dict, Union, List, Tuple, Callable, Any
# from . import activations
from .utils import checklist
from .control import Controllable
from .tracing import hack_graph, get_activations


class BendableModule(nn_tilde.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight_callbacks = nn.ModuleList()
        self.bending_controls = nn.ModuleList()
        self._has_initialized_controllables = False

    ## Bendable attributes handling
    def _get_bending_control(self, name: str) -> torch.Tensor:
        """returns value of a bending control by name"""
        # grrr
        for v in self.bending_controls:
            if v.name == name:
                return v.value.data
        raise RuntimeError()

    def _set_bending_control(self, name: str, value: torch.Tensor) -> None:
        """set a bending control with name and value"""
        for v in self.bending_controls:
            if v.name == name:
                v.set_value(value)
        self._update_weights()

    def _get_bending_controls(self) -> List[torch.Tensor]:
        """returns list of bending controls"""
        return self.bending_controls

    def parse_controllables(self, weight_hacks, act_hacks) -> None:
        bending_controls = []
        all_bending_controls = weight_hacks + act_hacks
        for obj, _ in all_bending_controls:
            bending_controls.extend(list(obj.get_controllables().values()))
        self.bending_controls = torch.nn.ModuleList(set(bending_controls))
        for control in bending_controls:
            self.register_attribute(control.name, float(control.value))
        self._has_initialized_controllables = True

    def hack_module(self, module: nn.Module, act_hacks = [], weight_hacks = [], prefix="", input_shape=None) -> fx.GraphModule:
        if not self._has_initialized_controllables:
            self.parse_controllables(act_hacks=act_hacks, weight_hacks=weight_hacks)
        self.hack_parameters(module, weight_hacks=weight_hacks, prefix=prefix)
        return self.hack_graphs(module, act_hacks=act_hacks, prefix=prefix, input_shape=input_shape)

    def hack_graphs(self, module, act_hacks, prefix="", input_shape = None):
        if input_shape is None:
            assert hasattr(module, "input_shape"), "module of type %s does not have input_shape attribute ; please specify manually"%(type(module))
            input_shape = module.input_shape
        acts = [o[0] for o in get_activations(module=module,func="forward", input_shape=input_shape, print_tabular=False)]
        _new_act_hacks = {}
        for callback, act_names in act_hacks:
            act_names = checklist(act_names)
            for a in acts:
                for act_regexp in act_names:
                    if re.match(act_regexp, f'{prefix}_{a}') is not None:
                        _new_act_hacks[a] = callback#(**kwargs)
        bended_model = hack_graph(_new_act_hacks, method="forward", model=module, verbose=True)
        return bended_model

    def hack_parameters(self, module, weight_hacks, prefix="") -> Dict[str, Tuple[nn.Parameter, nn.Module]]:
        for cb, param_names in weight_hacks:
            param_names = checklist(param_names)
            for k, v in dict(module.named_parameters()).items():
                if prefix is not None:
                    k = f"{prefix}.{k}"
                for name in param_names:
                    if re.match(name, k) is not None:
                        cb.register_parameters([v])
                        if not cb in self.weight_callbacks:
                            self.weight_callbacks.append(cb)
        # self._save_parameters(list(weight_callbacks.keys()), module, prefix=prefix)
        self._update_weights()
    
    def _update_weights(self):
        with torch.no_grad():
            for callback in self.weight_callbacks:
                callback.apply()

    def script(self):
        return torch.jit.script(self) 