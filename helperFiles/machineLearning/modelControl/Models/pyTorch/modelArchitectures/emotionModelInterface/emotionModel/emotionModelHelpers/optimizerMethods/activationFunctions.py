import torch.nn as nn
import torch


class switchActivation(nn.Module):
    def __init__(self, activationFunction, switchState=True):
        super(switchActivation, self).__init__()
        self.activationFunction = activationFunction
        self.switchState = switchState

    def forward(self, x):
        if self.switchState:
            return self.activationFunction(x)
        else:
            return x

class sinh(nn.Module):
    def __init__(self, clampCoeff=[-0.2, 0.2]):
        super(sinh, self).__init__()
        # Initialize coefficients with a starting value.
        self.coefficients = nn.Parameter(torch.tensor([0.3]))
        self.clampCoeff = clampCoeff

    def forward(self, x):
        # Update the coefficient clamp.
        self.coefficients = self.coefficients.clamp(min=self.clampCoeff[0], max=self.clampCoeff[1])

        return torch.sinh(self.coefficients*x)

class powerSeriesActivation(nn.Module):
    def __init__(self, numCoeffs=3, stabilityConstant=3.0, maxGrad=1, seriesType='full'):
        super(powerSeriesActivation, self).__init__()
        self.stabilityConstant = nn.Parameter(torch.tensor(stabilityConstant))
        self.coefficients = nn.Parameter(torch.ones(numCoeffs))
        self.seriesType = seriesType
        self.maxGrad = maxGrad

        # Register the hook with the coefficients
        self.stabilityConstant.register_hook(self.stabilityGradientHook)
        self.coefficients.register_hook(self.coeffGradientHook)

    def coeffGradientHook(self, grad):
        return grad.clamp(min=-self.maxGrad, max=self.maxGrad)

    def stabilityGradientHook(self, grad):
        # Clamp the gradients to be within the range [-self.stabilityConstant, self.stabilityConstant]
        return grad.clamp(min=-self.maxGrad, max=self.maxGrad)

    def forward(self, x):
        output = 0

        for coeffInd in range(len(self.coefficients)):
            functionPower = coeffInd + 1  # Skip the bias term.

            if self.seriesType == 'full':
                functionPower = functionPower  # Full series: f(x) = a_0*x + a_1*x^2 + ... + a_n*x^n
            elif self.seriesType == 'even':
                functionPower = 2*functionPower  # Even series: f(x) = a_0*x^2 + a_1*x^4 + ... + a_n*x^(2n)
            elif self.seriesType == 'odd':
                functionPower = 2*functionPower - 1  # Odd series: f(x) = a_0*x + a_1*x^3 + ... + a_n*x^(2n+1)
            else:
                raise NotImplementedError

            # Adjust the output.
            output += torch.exp(self.stabilityConstant) * torch.pow(x, functionPower) * self.coefficients[coeffInd]

        return output

class powerActivation(nn.Module):
    def __init__(self, initial_exponent=2.0, min_exponent=0.1, max_exponent=5.0):
        super(powerActivation, self).__init__()
        self.exponent = nn.Parameter(torch.tensor(initial_exponent))
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent

    def forward(self, x):
        # Apply constraints to the exponent
        constrained_exponent = torch.clamp(self.exponent, self.min_exponent, self.max_exponent)

        # Apply the power series activation
        return torch.pow(x, constrained_exponent)


class learnableTanhshrink(nn.Module):
    def __init__(self, initial_scale=1.0):
        super(learnableTanhshrink, self).__init__()
        # Initialize the learnable scale parameter
        self.nonLinearScale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, x):
        # Apply the tanh function to constrain the scale parameter between -1 and 1
        constrainedScale = torch.tanh(self.nonLinearScale)

        # Apply the Tanhshrink activation function with the learnable scale
        return x - constrainedScale * torch.tanh(x)
