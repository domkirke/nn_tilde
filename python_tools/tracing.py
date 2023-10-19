import torch, torch.nn as nn, torch.fx as fx
from collections import OrderedDict
from typing import Union
from . import distributions as dist
from .utils.script import dist_to_tensor
from .utils.misc import _recursive_to
from typing import Any


## Utils

def set_callback(x, env):
    return env[x.name]

class ActivationScrapper(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self.activations = []

    def run_node(self, n: fx.Node) -> Any:
        out = super().run_node(n)
        if n.op == "call_module":
            self.activations.append((n.name, out.shape))
        return out

    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        return self.activations

def _check_shape(shape, fill_value: int = 1):
    shape = list(shape)
    for i, s in enumerate(shape):
        if s is None:
            shape[i] = fill_value
    return tuple(shape)


def get_activations(module: nn.Module, func, input_shape):
    graph = ActDivTracer(func=func).trace(module)
    gm = fx.GraphModule(module, graph)
    scraper = ActivationScrapper(gm)
    input_shape = _check_shape(input_shape)
    inputs = torch.zeros(1, *input_shape)
    outs = scraper.run(inputs)
    return outs

## Tracer

class ActDivTracer(torch.fx.Tracer):
    _dist_count_hash = {}

    def __init__(self, *args, func="forward", **kwargs):
        super(ActDivTracer, self).__init__(*args, **kwargs)
        self.traced_func_name = func 

    def get_dist_count(self, dist_name: str):
        if not dist_name in self._dist_count_hash:
            self._dist_count_hash[dist_name] = 0
        dist_count = int(self._dist_count_hash[dist_name])
        self._dist_count_hash[dist_name] += 1
        return dist_count

    def create_arg(self, a):
        if isinstance(a, dist.Bernoulli):
            a = a.as_tuple()
            a = self.create_proxy("call_function", dist.Bernoulli, args=a, kwargs={}, name=f"_dist_Bernoulli_{self.get_dist_count('Bernoulli')}")
        elif isinstance(a, dist.Normal):
            a = a.as_tuple()
            a = self.create_proxy("call_function", dist.Normal, args=a, kwargs={}, name=f"_dist_Normal_{self.get_dist_count('Normal')}")
        elif isinstance(a, dist.Categorical):
            a = a.as_tuple()
            a = self.create_proxy("call_function", dist.Categorical, args=a, kwargs={}, name=f"_dist_Categorical_{self.get_dist_count('Categorical')}")
        return super(ActDivTracer, self).create_arg(a)

    def create_node(self, kind : str, target, args, kwargs, name = None, type_expr = None, dist_args = {}):
        if kind == "output":
            new_args = []
            for a in args:
                if a.name.startswith('_dist'):
                    a = self.create_node("call_function", dist_to_tensor(a.target), args=(a,), kwargs = dist_args, name=a.name+"_tensor", type_expr=torch.Tensor)
                    type_expr = torch.Tensor
                new_args.append(a)
            return super(ActDivTracer, self).create_node(kind, target, tuple(new_args), kwargs, name=name, type_expr = type_expr)
        else:
            return super(ActDivTracer, self).create_node(kind, target, args, kwargs, name=name, type_expr = type_expr)


class BendedGraphModule(torch.fx.GraphModule):
    def __init__(self, model, input_shape, *args, **kwargs):
        super(BendedGraphModule, self).__init__(model, *args, **kwargs)
        self._input_shape = input_shape
        if hasattr(model, "_versions"):
            self._versions = model._versions.copy()
        else:
            self._versions = None

    def state_dict(self, *args, with_versions=True, **kwargs):
        if with_versions and self._versions is not None:
            state_dict = dict(self._versions)
            state_dict[self._current_version] = super().state_dict(*args, **kwargs)
        else:
            state_dict = super().state_dict(*args, **kwargs)
        return state_dict
    
    def set_version(self, version: str) -> None:
        """Save current statedict in """
        self._current_version = version
        # load request
        if version in self._versions:
            self.load_state_dict(self._versions[version])
        else:
            self._versions[version] = OrderedDict({k: v.clone() for k, v in super().state_dict().items()})
        
    def write_version(self, version: Union[str, None]) -> None:
        if version is None:
            self._versions[version].update(super().state_dict())
            version = version or self._current_version        
        else:
            self._versions[version] = _recursive_to(super().state_dict(), torch.device('cpu'), clone=True)
 
    def input_shape(self, method):
        return self._input_shape


## trace functions

def hack_graph(act_hacks, model, method, verbose: bool = False):
    graph = ActDivTracer(func=method).trace(model)
    new_graph = fx.Graph()
    env = {}
    bended_lookup = {}
    name_hash = {}
    for node in graph.nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        # check arguments to replace by bended node in case
        new_args = list(new_node.args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, torch.fx.Node):
                if arg.name in bended_lookup:
                    new_args[i] = bended_lookup[arg.name]
        new_node.args = tuple(new_args)
        env[node.name] = new_node
        counter = 0
        # add bending layer to graph
        if node.name in act_hacks:
            if verbose:
                print('bending activation %s with function %s...'%(node.name, act_hacks[node.name]))
            # add placeholder
            counter += 1
            # add callback
            bended_node_name = node.name+"_bended"
            hack_obj_name = type(act_hacks[node.name]).__name__
            if not hack_obj_name in name_hash:
                name_hash[hack_obj_name] = 0
                hack_obj_name += "_0"
            else:
                idx = name_hash[hack_obj_name]
                name_hash[hack_obj_name] += 1
                hack_obj_name += f"_{idx}"
            setattr(model, hack_obj_name, act_hacks[node.name])
            bended_node = new_graph.create_node("call_module", hack_obj_name, args=(new_node,), kwargs={}, name=bended_node_name)
            env[bended_node_name] = bended_node 
            bended_lookup[node.name] = bended_node 
    bended_model = BendedGraphModule(model, new_graph)
    unbended_keys = set(act_hacks).difference(bended_lookup.keys())
    if len(unbended_keys) > 0 and verbose:
        print(f'[Warning] Unbended keys : {unbended_keys}')
    return bended_model#, controllable_node_hash