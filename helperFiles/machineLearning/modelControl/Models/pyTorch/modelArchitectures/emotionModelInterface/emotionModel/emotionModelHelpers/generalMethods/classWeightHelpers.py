import torch
import torch.nn.functional as F

# Import helper classes.
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods


class classWeightHelpers:

    @staticmethod
    def oneHotEncoding(classLabels, num_classes):
        return F.one_hot(classLabels.long(), num_classes=num_classes).float()

    @staticmethod
    def gausEncoding(classLabels, num_classes, array_length, gaus_stds=0.3, relative_amplitudes=1):
        finalLabels = torch.zeros((len(classLabels), array_length), device=classLabels.device)

        # For each label requested.
        for finalLabelInd in range(len(classLabels)):
            classInd = classLabels[finalLabelInd]

            # Create a distribution around that label's index.
            finalLabels[finalLabelInd] = generalMethods.create_gaussian_array(
                array_length=array_length,
                num_classes=num_classes,
                mean_gaus_indices=classInd,
                relative_amplitudes=relative_amplitudes,
                gaus_stds=gaus_stds,
                device=classLabels.device,
            )
        return finalLabels

    @staticmethod
    def extractClassIndex(predictionArray, num_classes, axisDimension, returnIndex=True):
        # If we have a torch tensor, convert to numpy.
        # if isinstance(predictionArray, torch.Tensor):
        #     predictionArray = predictionArray.detach()

        # Calculate the class index.
        array_length = predictionArray.shape[axisDimension]
        maxProbIndexes = predictionArray.argmax(dim=axisDimension)
        classFloat = (maxProbIndexes * num_classes / array_length - 0.5)

        if returnIndex:
            return classFloat.round().astype(int)

        return classFloat
