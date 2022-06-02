import torch.nn as nn

# Aliases.
relu = nn.ReLU
prelu = nn.PReLU
mish = nn.Mish
sigmoid = nn.Sigmoid
tanh = nn.Tanh


def get_activation(name: str) -> nn.Module:
    if name not in ['relu', 'mish', 'prelu', 'sigmoid', 'tanh']:
        raise NameError('Could not interpret activation identifier')

    if isinstance(name, str):
        cls = globals().get(name)
        if cls is None:
            raise ValueError("Could not interpret activation identifier: " + str(name))
        return cls
    else:
        raise ValueError("Could not interpret activation identifier: " + str(name))
