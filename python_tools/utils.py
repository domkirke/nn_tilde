import torch
from active_divergence import distributions as dist

# Probability handling for scripting / tracing

def _bernoulli2tensor(x: dist.Bernoulli) -> torch.Tensor:
    return x.probs

def _normal2tensor(x: dist.Normal, temperature: float = 0.) -> torch.Tensor:
    return x.mean + temperature * x.stddev * torch.randn_like(x.mean)

def _categorical2tensor(x: dist.Normal, return_probs: bool = False, sample: bool = False) -> torch.Tensor:
    if return_probs:
        return x.probs
    else:
        if sample:
            return x.probs.sample()
        else:
            return x.probs.max(1)

def dist_to_tensor(target):
    if target == dist.Bernoulli:
        return _bernoulli2tensor
    elif target == dist.Normal:
        return _normal2tensor
    elif target == dist.Categorical:
        return _categorical2tensor
    else:
        raise NotImplementedError

def checklist(item, n=1, copy=False):
    """Repeat list elemnts
    """
    if not isinstance(item, (list, )):
        if copy:
            item = [copy.deepcopy(item) for _ in range(n)]
        elif isinstance(item, torch.Size):
            item = [i for i in item]
        else:
            item = [item]*n
    return item

def _recursive_to(obj, device, clone=False):
    if isinstance(obj, OrderedDict):
        return OrderedDict({k: _recursive_to(v, device, clone) for k, v in obj.items()})
    if isinstance(obj, dict):
        return {k: _recursive_to(v, device, clone) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_recursive_to(o, device, clone) for o in obj]
    elif torch.is_tensor(obj):
        if clone:
            return obj.to(device=device).clone()
        else:
            return obj.to(device=device)
    else:
        raise TypeError('type %s not handled by _recursive_to'%type(obj))

def checktensor(tensor, dtype=None, allow_0d=True):
    if isinstance(tensor, list):
        return [checktensor(t, dtype=dtype) for t in tensor]
    elif isinstance(tensor, tuple):
        return tuple([checktensor(t, dtype=dtype) for t in tensor])
    elif isinstance(tensor, dict):
        return {k: checktensor(v, dtype=dtype) for k, v in tensor.items()}
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor).to(dtype=dtype)
    elif torch.is_tensor(tensor):
        tensor = tensor.to(dtype=dtype)
        if tensor.ndim == 0 and not allow_0d:
            tensor = torch.Tensor([tensor])
        return tensor
    else:
        if hasattr(tensor, "__iter__"):
            tensor = torch.Tensor(tensor, dtype=dtype)
        else:
            tensor = torch.tensor(tensor, dtype=dtype)
        if tensor.ndim == 0 and not allow_0d:
            tensor = torch.Tensor([tensor])
        return tensor
