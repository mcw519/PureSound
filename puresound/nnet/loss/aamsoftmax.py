import torch
import torch.nn as nn
import torch.nn.functional as F

TORCH_PI = torch.acos(torch.zeros(1)).item() * 2


class AAMsoftmax(nn.Module):
    """
    Args:
        input_dim: input embedding dimension
        n_class: number of class
        margin: loss margin in AAM softmax
        scale: loss scale in AAM softmax
    """

    def __init__(
        self, input_dim: int, n_class: int, margin: float = 0.2, scale: int = 30
    ) -> None:
        super().__init__()
        self.m = margin
        self.s = scale
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(n_class, input_dim), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = torch.cos(torch.tensor(self.m))
        self.sin_m = torch.sin(torch.tensor(self.m))
        self.th = torch.cos(TORCH_PI - torch.tensor(self.m))
        self.mm = torch.sin(TORCH_PI - torch.tensor(self.m)) * self.m

    def forward(self, x: torch.Tensor, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        loss = self.ce(output, label)

        return loss
