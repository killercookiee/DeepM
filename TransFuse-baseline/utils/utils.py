import torch
import numpy as np

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # Ensure val is converted to a Tensor before appending
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, dtype=torch.float32)
        self.losses.append(val)

    def show(self):
        if len(self.losses) == 0:
            return 0.0  # Return 0 if no losses recorded
        # Safely compute mean for the last 'num' elements
        return torch.mean(torch.stack(self.losses[max(len(self.losses) - self.num, 0):])).item()