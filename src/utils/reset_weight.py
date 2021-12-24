import numpy as np
import torch


def reset_conv_weight(w, index, scale=None):
    with torch.no_grad():
        if scale is None:
            n = w.shape[-2] * w.shape[-1] * w.shape[0]
            scale = np.sqrt(2.0 / n)
        w[index, ...] = torch.randn_like(w[index, ...]) * scale


def reset_by_norm(
    model,
    mode="r",
    value=0.2,
    expected_rate=0.1,
    relative_rate=0.1,
    scale=None,
    name=None,
):
    """
    reset all weights of the model.

    Args.

    model: torch.nn.Module

    mode: str
        reset mode.
        you can choose 'c' (constant value), 'r' (relative value by maximum), or 'e' (expected value).

    value: float
        constant value

    expected_rate: float

    relative_rate: float or str

    scale: str or float

    name: str
        name of parameter reset weight. if None then reset all parameters.
        default None
    """
    assert isinstance(model, torch.nn.Module)

    if mode == "c":
        expected = False
        relative = False
    elif mode == "e":
        expected = True
        relative = False
    elif mode == "r":
        expected = False
        relative = True

    if relative and expected:
        raise ValueError("both relative flag and expected flag can not be true.")

    with torch.no_grad():
        for p_name, param in model.named_parameters():
            if name is not None and name != p_name and "module." + name != p_name:
                continue

            if len(param.shape) == 4:
                w = param.detach().clone()
                norm = w.reshape(len(w), -1).norm(dim=-1)
                if expected:
                    N = param.shape[1] * param.shape[2] * param.shape[3]
                    M = param.shape[0] * param.shape[2] * param.shape[3]
                    value = N / np.sqrt(N + 1) * np.sqrt(2 / M) * expected_rate
                elif relative:
                    out_channel = param.shape[0]
                    if isinstance(relative_rate, float):
                        value = norm.max() * relative_rate
                    elif relative_rate == "out_channel":
                        value = norm.max() / out_channel
                index = norm < value
                if index.sum() > 0:
                    if isinstance(scale, str):
                        N = param.shape[1] * param.shape[2] * param.shape[3]
                        if scale == "max":
                            s = norm.max() / np.sqrt(N)
                            reset_conv_weight(param, index, scale=s)
                        elif scale == "mean":
                            s = norm.mean() / np.sqrt(N)
                            reset_conv_weight(param, index, scale=s)
                    else:
                        reset_conv_weight(param, index, scale=scale)
