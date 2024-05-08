import torch
import torch.optim as optim


class CustomLRScheduler:
    def __init__(self, optimizer, baseLRs, minLRs, maxLRs, numModels, numLossesTrack, scaleUp=2.0, scaleDown=0.5, invertLRAdjustment=False):
        # General parameters
        self.numLossesTrack = numLossesTrack
        self._optimizer = optimizer
        self.numModels = numModels
        self.numEpochs = 0
        self.minLRs = minLRs
        self.maxLRs = maxLRs
        self.lrs = baseLRs
        self.scaleUp = scaleUp
        self.scaleDown = scaleDown
        self.invertLRAdjustment = invertLRAdjustment  # Flag to toggle the behavior of learning rate adjustment

        # Keep track of the learning rate history
        self.lossHistory = []
        self.lrs = baseLRs



        # For each trainable parameter in the model.
        for layerParams in model.parameters():
            # Calculate the L2 norm. THIS IS NOT SN, except for the 1D case.
            paramNorm = torch.norm(layerParams, p='fro').item()

            # Constrain the spectral norm.
            if maxNorm < paramNorm != 0:
                layerParams.data = layerParams * (maxNorm / paramNorm)

    def add_loss(self, modelIndex, loss):
        # Ensure the model index is valid
        if not (0 <= modelIndex < self.numModels):
            raise ValueError("Invalid model index")

        # Append the new loss value
        self.lossHistory[modelIndex].append(loss)

        # Check if the loss history exceeds the maximum length
        if self.numLossesTrack < len(self.lossHistory[modelIndex]):
            # Remove the oldest loss value
            self.lossHistory[modelIndex].pop(0)

    def resetLossHistory(self):
        self.lossHistory = [[] for _ in range(self.numModels)]

    def step_and_update_lr(self):
        """ Step with the inner optimizer """
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """ Zero out the gradients with the inner optimizer """
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        """ Learning rate scheduling per step based on average loss history """
        self.numEpochs += 1

        for modelInd in range(self.numModels):
            if len(self.lossHistory) == 0:
                continue  # Skip if no loss history

            avg_loss = sum(self.lossHistory) / len(self.lossHistory)
            lr_adjustment = self.scaleDown  # Default to scale down

            if (avg_loss < 0.1 and not self.invertLRAdjustment) or (avg_loss > 0.1 and self.invertLRAdjustment):
                lr_adjustment = self.scaleUp  # Scale up condition

            # Update learning rate within bounds
            new_lr = min(max(self.minLRs[modelInd], self.lrs[modelInd] * lr_adjustment), self.maxLRs[modelInd])
            self.lrs[modelInd] = new_lr

            # Apply the new learning rate
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = new_lr
