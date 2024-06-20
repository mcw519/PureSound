from typing import Dict

import torch


class Logging:
    def __init__(self) -> None:
        self.bag = {}

    def update(self, outs: Dict):
        for key in sorted(outs.keys()):
            item = outs[key]
            if isinstance(item, torch.Tensor):
                item = item.item()

            if key not in self.bag:
                self.bag[key] = [item]
            else:
                self.bag[key].append(item)

    def clear(self):
        self.bag = {}

    def average(self):
        output = {}
        for key in sorted(self.bag.keys()):
            scores = torch.Tensor(self.bag[key]).mean()
            output[key] = scores

        return output
