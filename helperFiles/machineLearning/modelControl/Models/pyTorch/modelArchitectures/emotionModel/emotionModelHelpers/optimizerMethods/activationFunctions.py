import torch.nn as nn
import torch


class PowerSeriesActivation(nn.Module):
    def __init__(self, initial_exponent=2.0, min_exponent=0.1, max_exponent=5.0):
        super(PowerSeriesActivation, self).__init__()
        self.exponent = nn.Parameter(torch.tensor(initial_exponent))
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent

    def forward(self, x):
        # Apply constraints to the exponent
        constrained_exponent = torch.clamp(self.exponent, self.min_exponent, self.max_exponent)

        # Apply the power series activation
        return torch.pow(x, constrained_exponent)


class LearnableTanhshrink(nn.Module):
    def __init__(self, initial_scale=1.0):
        super(LearnableTanhshrink, self).__init__()
        # Initialize the learnable scale parameter
        self.nonLinearScale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, x):
        # Apply the tanh function to constrain the scale parameter between -1 and 1
        constrainedScale = torch.tanh(self.nonLinearScale)

        # Apply the Tanhshrink activation function with the learnable scale
        return x - constrainedScale * torch.tanh(x)
