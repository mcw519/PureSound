# Codes come from ESPNET
import math

import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class EcapaBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ChnAttnStatPooling(nn.Module):
    """
    Aggregates frame-level features to single utterance-level feature.

    args:
        input_size: dimensionality of the input frame-level embeddings.
            Determined by encoder hyperparameter.
            For this pooling layer, the output dimensionality will be double of
            the input_size
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_size * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, input_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self._output_size = input_size * 2

    def output_size(self):
        return self._output_size

    def forward(self, x: torch.Tensor):
        t = x.size()[-1]
        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(
                    torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)
                ).repeat(1, 1, t),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))

        x = torch.cat((mu, sg), dim=1)

        return x


class EcapaTdnnExtracotr(nn.Module):
    """
    ECAPA-TDNN, extracts frame-level ECAPA-TDNN embeddings

    Args:
        input_size: input feature dimension.
        embedding_size: ouput speaker embedding dimension.
        model_scale: scale value of the Res2Net architecture.
        ndim: dimensionality of the hidden representation.
        output_size: output embedding dimension.
    """

    def __init__(
        self,
        input_size: int,
        embedding_size: int = 192,
        model_scale: int = 8,
        ndim: int = 1024,
        att_size: int = 1536,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_size, ndim, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(ndim)

        self.layer1 = EcapaBlock(
            ndim, ndim, kernel_size=3, dilation=2, scale=model_scale
        )
        self.layer2 = EcapaBlock(
            ndim, ndim, kernel_size=3, dilation=3, scale=model_scale
        )
        self.layer3 = EcapaBlock(
            ndim, ndim, kernel_size=3, dilation=4, scale=model_scale
        )
        self.layer4 = nn.Conv1d(3 * ndim, att_size, kernel_size=1)

        self.extractor = nn.Sequential(
            ChnAttnStatPooling(input_size=att_size),
            nn.BatchNorm1d(att_size * 2),
            nn.Linear(att_size * 2, embedding_size),
        )

    def forward(self, x: torch.Tensor):
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Input tensor [N, C, T]

        Returns:
            torch.Tensor: Output tensor [N, C, T].
        """
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        x = self.extractor(x)
        return x
