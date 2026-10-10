"""Standard-ONNX vectorized LSTM update for one streaming time step."""

import torch
import torch.nn.functional as F


def fuse_lstm_weights(rnn: torch.nn.LSTM) -> tuple[torch.Tensor, torch.Tensor]:
    """``[W_ih | W_hh]`` and ``b_ih + b_hh`` of a one-layer LSTM, for `lstm_step`."""
    weight = torch.cat((rnn.weight_ih_l0, rnn.weight_hh_l0), dim=1)
    return weight.detach().clone(), (rnn.bias_ih_l0 + rnn.bias_hh_l0).detach().clone()


def lstm_step(weight, bias, x, h, c):
    """Advance all frequency positions together; x [F,C], h/c [1,F,H].

    The trained input/hidden projections share one matrix multiplication
    (`fuse_lstm_weights`). The inter path has one layer and one time step, so
    no sequence operator or frequency loop is needed. Gate order is PyTorch's
    i,f,g,o.
    """
    gates = F.linear(torch.cat((x, h[0]), dim=-1), weight, bias)
    i, f, g, o = gates.chunk(4, dim=-1)
    next_c = torch.sigmoid(f) * c[0] + torch.sigmoid(i) * torch.tanh(g)
    next_h = torch.sigmoid(o) * torch.tanh(next_c)
    return next_h, next_h.unsqueeze(0), next_c.unsqueeze(0)
