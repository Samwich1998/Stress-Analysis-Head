# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# REVIEW IF YOU ARE USING. 
def weightLoss(allLosses, class_weights, targets):
    # Ensure proper data format.
    targets = targets.int()  # Targets should be class indices.
    assert len(allLosses.shape) == 1, f"We should only have one dimension in the loss: {allLosses.shape}"
    assert len(targets.shape) == 1, f"We should only have one dimension in the targets: {targets.shape}"

    normalizationFactor = 0
    # Apply normalization correction for weighting
    for classInd in range(len(class_weights)):
        classWeight = class_weights[classInd]

        # Find the predictions for this class.
        classMask = targets == classInd
        classCounts = classMask.sum()

        # If the class is present.
        if classCounts != 0:
            # This will average each class index's loss (once we sum at the end)
            allLosses[classMask] *= classWeight / classCounts
            normalizationFactor += classWeight

    # Perform a weighted sum for a given class index.
    finalLoss = allLosses.sum() / normalizationFactor

    return finalLoss


class pytorchLossMethods:

    def __init__(self, lossType, class_weights=None):
        # Normalize the class weights if provided
        if class_weights is not None:
            class_weights = class_weights / class_weights[class_weights != torch.inf].sum()

        # --------------- Compile Classification Loss Methods -------------- #

        # Negative Log Likelihood: It aims to maximize the log-likelihood (log probability) of the true class.
        #               The true labels should be class indices OR 1-hot encoded.
        #               Must apply log_softmax at last layer (this is the Log part).
        #               Can handle class imbalances through class_weights.
        if lossType == "NLLLoss":
            self.loss_fn = torch.nn.NLLLoss(weight=class_weights, reduction="none")

        # Cross Entropy: Compares true vs. predicted probability distributions (similar to NLL loss).
        #               The true labels should be class indices OR 1-hot encoded.
        #               Should NOT apply softmax at last layer (done internally in pytorch).
        #               Can handle class imbalances through class_weights.
        #               Similar to NLLLoss if you apply log_softmax.
        elif lossType == "CrossEntropyLoss":
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none', label_smoothing=0.0)

        # KL Divergence: Compares how different probability distributions are (NOT symmetric). 
        #               The true labels should be class probability distributions (not log-probabilities, see log_target)
        #               Must apply log_softmax to the predicted distribution (at the last layer) before calculating the loss.
        elif lossType == "KLDivLoss":
            self.loss_fn = torch.nn.KLDivLoss(reduction='none', log_target=False)

        elif lossType == "BCEWithLogitsLoss":
            self.loss_fn = nn.BCEWithLogitsLoss(weight=class_weights, reduction='mean', pos_weight=None)

        # ----------------- Compile Regression Loss Methods ---------------- #

        # Mean Squared Error: Minimizes the squared difference between predicted and true values.
        elif lossType == "MeanSquaredError":
            self.loss_fn = torch.nn.MSELoss(reduction='none')

        # Mean Absolute Error: Minimizes the absolute difference between predicted and true values.
        elif lossType == "MeanAbsoluteError":
            self.loss_fn = nn.L1Loss()

        # Huber: A combination of L1 and MSE loss, controlled by the parameter delta.
        #               It behaves like L1 loss when the difference is large and like MSE loss when the difference is small.
        #               When delta is set to 1, Huber loss is equivalent to SmoothL1Loss.
        elif lossType == "Huber":
            self.loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)

        # Smooth L1: Very similar as Huber loss, it is used for regression tasks.
        #               It's a combination of L1 loss and L2 loss and is less sensitive to outliers.
        #               The beta parameter controls the transition point between L1 and L2 loss.
        elif lossType == "SmoothL1Loss":
            self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)

        # Poisson-Negative Log Likelihood: positive target values
        elif lossType == "PoissonNLLLoss":
            self.loss_fn = nn.PoissonNLLLoss()

        # ------------------- Compile Custom Loss Methods ------------------ #

        elif lossType == "FocalLoss":
            self.loss_fn = FocalLoss(gamma=0.5, alpha=None, size_average=True)
        else:
            self.loss_fn = customLossMethods(lossType, class_weights)

        # ------------------------------------------------------------------ #


class customLossMethods(nn.Module):
    def __init__(self, lossType, class_weights=None):
        super(customLossMethods, self).__init__()
        self.lossType = lossType

        if class_weights is not None:
            # Normalize the class weights
            self.class_weights = class_weights
            self.class_weights[self.class_weights == torch.inf] = 0
            self.class_weights /= self.class_weights.sum()
            # Save the normalization constants.
            self.numClassesNormalized = (class_weights != torch.inf).sum()

        # --------------- Compile Classification Loss Methods -------------- #

        if lossType == "diceLoss":
            self.loss_fn = self.diceLoss
        elif lossType == "weightedKLDiv":
            self.loss_fn = self.weightedKLDiv

        # ----------------- Compile Regression Loss Methods ---------------- #

        # Chose the loss method
        elif lossType == "R2":
            self.loss_fn = self.R2
        elif lossType == "pearson":
            self.loss_fn = self.pearsonLoss
        elif lossType == "LogCoshLoss":
            self.loss_fn = self.LogCoshLoss
        elif lossType == "weightedMSE":
            self.loss_fn = self.weightedMSE

        # ------------------------------------------------------------------ #

        else:
            assert False, f"Unknown loss requested: {lossType}"

    def forward(self, predictedVals, targetVals, class_weights=None, targets=None):
        if self.lossType == "weightedKLDiv":
            lossValue = self.loss_fn(predictedVals, targetVals, targets, class_weights)
        else:
            # Calculate the loss of the prediction and ensure a smooth value.
            lossValue = self.loss_fn(predictedVals, targetVals, class_weights)
        assert not lossValue.isnan().any().item() and not lossValue.isinf().any().item(), print(predictedVals, targetVals, lossValue)

        return lossValue

    @staticmethod
    def R2(predictedVals, trueLabels):
        # Calculate the components of R2.
        sumOfSquares = torch.square(trueLabels - predictedVals).sum(axis=0)
        trueVariance = torch.square(trueLabels - trueLabels.mean(axis=0)).sum(axis=0)
        # Calculate and return the R2 loss
        loss = 1 - sumOfSquares / trueVariance
        return loss

    @staticmethod
    def pearsonLoss(predictedVals, trueVals):
        # Calculate the difference between the means of both points.
        predictedMeanDiff = predictedVals - predictedVals.mean(axis=0)
        trueMeanDiff = trueVals - trueVals.mean(axis=0)

        # Calculate the pearson correlation terms.
        covariance = (predictedMeanDiff * trueMeanDiff).sum(axis=0)
        predictedSTD = predictedMeanDiff.norm()
        trueSTD = trueMeanDiff.norm()

        # If all true values are the same, there is no score.
        if trueSTD == 0:
            print("There is no variation of true values")
            return 1
        # If all predicted values are the same, there is no score.
        if predictedSTD == 0:
            print("There is no variation of predicted values")
            return 1

        # Calculate the pearson correlation.
        pearsonCorr = covariance / (predictedSTD * trueSTD)

        # Convert correlation to a loss value (higher correlation indicates lower loss)
        lossValue = 1 - pearsonCorr.abs()
        return lossValue

    def diceLoss(self, predictedDist, trueDist, smooth=1e-10):
        # Calculate the dice loss
        intersection = (predictedDist * trueDist).sum(dim=1)  # Intersection of probability distribution
        union = (predictedDist + trueDist).sum(dim=1)  # Total squared area
        # Minimize the loss instead of maximizing.
        diceLosses = 1 - (2 * intersection + smooth) / (union + smooth)  # Add epsilon for numerical stability

        # If class weights are provided.
        diceLoss = self.weightLoss_DEPRECATED(diceLosses, trueDist)

        return diceLoss

    def weightedKLDiv(self, predictedDist, trueDist, targets, class_weights=None):
        """
        predictedDist is log-prob and trueDist is prob
        targets : An array of targeted values. I currently am expecting a 1D array of integers.
        class_weights : An array of class weights. I currently am expecting a 1D array of length numClasses.
        """
        # Preprocess the distributions.
        trueDist = torch.clamp(trueDist, min=1e-20, max=1.0)  # Avoid zero probabilities

        # Calculate the KL Loss
        klLosses = (trueDist * (trueDist.log() - predictedDist)).sum(dim=1)

        # Apply normalization correction for weighting
        klLoss = self.weightLoss_DEPRECATED(klLosses, class_weights, targets)

        return klLoss

    def weightedMSE(self, predictions, targets, class_weights):
        """
        predictions : An array of predicted values. I currently am expecting a 1D array of floats.
        targets : An array of targeted values. I currently am expecting a 1D array of integers.
        class_weights : An array of class weights. I currently am expecting a 1D array of length numClasses.
        """
        # Assert the integrity of the expected data.
        assert predictions.shape == targets.shape, "Shapes of predictions and targets must match"

        # Calculate the weighted squared error
        squared_errors = (predictions - targets) ** 2

        # Apply normalization correction for weighting
        mseLoss = self.weightLoss_DEPRECATED(squared_errors, class_weights, targets)

        return mseLoss

    def variationOfDiff(self, predictedDist, targets, class_weights):
        """
        predictions : An array of predicted values. I currently am expecting a 1D array of floats.
        targets : An array of targeted values. I currently am expecting a 1D array of integers.
        class_weights : An array of class weights. I currently am expecting a 1D array of length numClasses.
        """
        # Assert the integrity of the expected data.
        # Assert error.shape == targets.shape, "Shapes of predictions and targets must match"

        # Calculate the weighted squared error
        smoothnessLoss = predictedDist.diff(n=1, dim=-1).var(dim=-1)

        # Apply normalization correction for weighting
        mseLoss = self.weightLoss_DEPRECATED(smoothnessLoss, class_weights, targets)

        return mseLoss

    # REVIEW IF YOU ARE USING. 
    @staticmethod
    def weightLoss_DEPRECATED(allLosses, class_weights, targets):
        # Assign class weights to all the targets.
        class_weights /= class_weights.sum()
        targetWeights = class_weights[targets.int()]
        # Weight the loss values.
        weightedLosses = allLosses * targetWeights

        # Apply normalization correction for weighting
        for classInd in range(len(class_weights)):
            classMask = targets == classInd
            classCounts = classMask.sum()

            if classCounts != 0:
                # This will average each class index's loss (once we sum at the end)
                weightedLosses[classMask] /= classCounts

        # Perform a weighted sum for a given class index.
        finalLoss = weightedLosses.sum()

        return finalLoss

    @staticmethod
    def LogCoshLoss(predictedVals, targetVals):
        # Compute the element-wise log cosh loss
        loss = (predictedVals - targetVals).cosh().log().mean()
        # from torchmetrics.regression import LogCoshError
        # LogCoshError()(preds, target)

        return loss


class FocalLoss(nn.Module):
    """ 
    Implemented from: https://github.com/clcarwin/focal_loss_pytorch/tree/master 
    gamma = 0 makes it the same as cross-entropy loss
    """

    def __init__(self, gamma=0.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, predictedVals, targetVals):
        if predictedVals.dim() > 2:
            predictedVals = predictedVals.view(predictedVals.size(0), predictedVals.size(1), -1)  # N,C,H,W => N,C,H*W
            predictedVals = predictedVals.transpose(1, 2)  # N,C,H*W => N,H*W,C
            predictedVals = predictedVals.contiguous().view(-1, predictedVals.size(2))  # N,H*W,C => N*H*W,C
        targetVals = targetVals.view(-1, 1)

        logpt = F.log_softmax(predictedVals)
        logpt = logpt.gather(1, targetVals)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != predictedVals.data.type():
                self.alpha = self.alpha.type_as(predictedVals.data)
            at = self.alpha.gather(0, targetVals.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
