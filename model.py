import torch.nn as nn
import torch
import copy
import numpy as np

class CenteredWrapper(nn.Module):
    """
    Wrap any nn.Module so that forward(x) returns model(x) - baseline(x),
    where baseline is a frozen snapshot taken at wrap time.
    """
    def __init__(self, model: nn.Module, baseline_dtype=None):
        super().__init__()
        self.model = model
        self.baseline = copy.deepcopy(model)  # snapshot at wrap time
        for p in self.baseline.parameters():
            p.requires_grad = False
        self.baseline.eval()
        self._baseline_dtype = baseline_dtype
        if baseline_dtype is not None:
            self.baseline.to(dtype=baseline_dtype)

    @torch.no_grad()
    def recenter(self):
        """Reset baseline to current model weights (still frozen)."""
        self.baseline.load_state_dict(self.model.state_dict())
        if self._baseline_dtype is not None:
            self.baseline.to(dtype=self._baseline_dtype)
        self.baseline.eval()

    def forward(self, x):
        y = self.model(x)  # grads here
        with torch.inference_mode():
            y0 = self.baseline(x)  # no grads / low mem
        return y - y0


def centeredMLP(model: nn.Module, baseline_dtype=None) -> nn.Module:
    """
    Usage:
        centerednet = centeredMLP(baseline_net)     # wrap for centering
    """
    return CenteredWrapper(model, baseline_dtype=baseline_dtype)


class MLP(nn.Module):
    def __init__(self, d_in=1, width=4096, depth=2, d_out=1, bias=True, nonlinearity=None, forcezeros=False):
        super().__init__()
        self.d_in, self.width, self.depth, self.d_out = d_in, width, depth, d_out

        self.input_layer = nn.Linear(d_in, width, bias)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width, bias) for _ in range(depth - 1)])
        self.output_layer = nn.Linear(width, d_out, bias)
        if forcezeros:
            with torch.no_grad():
                self.output_layer.weight.zero_()
                if self.output_layer.bias is not None:
                    self.output_layer.bias.zero_()
        self.nonlin = nonlinearity if nonlinearity is not None else nn.ReLU()
        
    def forward(self, x):
        h = self.nonlin(self.input_layer(x))
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
        out = self.output_layer(h)
        return out

    def get_activations(self, x):
        h_acts = []
        h = self.nonlin(self.input_layer(x))
        h_acts.append(h)
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
            h_acts.append(h)
        h_out = self.output_layer(h)
        return h_acts, h_out