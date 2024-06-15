# General
import torch

# Import machine learning files
from .waveletNeuralOperatorWeights import waveletNeuralOperatorWeights


class waveletNeuralOperatorLayer(waveletNeuralOperatorWeights):

    def __init__(self, numInputSignals, numOutputSignals, sequenceBounds, numDecompositions=2, waveletType='db3', mode='zero', addBiasTerm=False, smoothingKernelSize=0, activationMethod="none",
                 encodeLowFrequencyProtocol=0, encodeHighFrequencyProtocol=0, useConvolutionFlag=True, independentChannels=False, skipConnectionProtocol='CNN'):
        super(waveletNeuralOperatorLayer, self).__init__(numInputSignals, numOutputSignals, sequenceBounds, numDecompositions, waveletType, mode, addBiasTerm, smoothingKernelSize, activationMethod,
                                                         encodeLowFrequencyProtocol, encodeHighFrequencyProtocol, useConvolutionFlag, independentChannels, skipConnectionProtocol)

    def forward(self, inputData, extraSkipConnection=0, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Apply the wavelet neural operator and the skip connection.
        neuralOperatorOutput = self.waveletNeuralOperator(inputData, lowFrequencyTerms, highFrequencyTerms)
        neuralOperatorOutput = neuralOperatorOutput + self.skipConnectionModel(inputData) + extraSkipConnection
        # neuralOperatorOutput dimension: batchSize, numOutputSignals, sequenceLength

        # Apply the activation function.
        neuralOperatorOutput = self.activationFunction(neuralOperatorOutput)
        # neuralOperatorOutput dimension: batchSize, numOutputSignals, sequenceLength

        return neuralOperatorOutput

    def waveletNeuralOperator(self, inputData, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Extract the input data dimensions.
        batchSize, numInputSignals, sequenceLength = inputData.size()

        # Pad the data to the maximum sequence length.
        inputData = torch.nn.functional.pad(inputData, pad=(self.sequenceBounds[1] - sequenceLength, 0), mode='constant', value=0)
        # inputData dimension: batchSize, numLiftedChannels, maxSequenceLength

        # Perform wavelet decomposition.
        lowFrequency, highFrequencies = self.dwt(inputData)  # Note: each channel is treated independently here.
        # highFrequencies[decompositionLayer] dimension: batchSize, numLiftedChannels, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numLiftedChannels, lowFrequencyShape

        # Mix each frequency decomposition, separating high and low frequencies.
        lowFrequency, highFrequencies = self.mixSeperatedFrequencyComponents(lowFrequency, highFrequencies, lowFrequencyTerms, highFrequencyTerms)
        # highFrequencies[highFrequencyInd] dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        # Mix all the frequency terms into one set of frequencies.
        lowFrequency, highFrequencies = self.mixAllFrequencyComponents(lowFrequency, highFrequencies)
        # highFrequencies[highFrequencyInd] dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        # Perform wavelet reconstruction.
        reconstructedData = self.idwt((lowFrequency, highFrequencies))
        # reconstructedData dimension: batchSize, numOutputSignals, sequenceLength

        # Add smoothing to the signals if necessary.
        if self.smoothingKernelSize:
            reconstructedData = self.applySmoothing(reconstructedData, self.smoothingKernel)
            # reconstructedData dimension: batchSize, numOutputSignals, sequenceLength

        # Remove the padding.
        reconstructedData = reconstructedData[:, :, -sequenceLength:]
        # reconstructedData dimension: batchSize, numOutputSignals, sequenceLength

        if self.addBiasTerm:
            # Add the bias terms.
            reconstructedData = reconstructedData + self.operatorBiases
            # outputData dimension: batchSize, numOutputSignals, sequenceLength

        return reconstructedData

    def mixSeperatedFrequencyComponents(self, lowFrequency, highFrequencies, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Set up the equation to apply the weights.
        equationString = 'oin,bin->bon'  # The equation to apply the weights.
        # b = batchSize, i = numLiftedChannels, o = numOutputSignals, n = signalDimension
        # 'oin,bin->bon' = weights.size(), frequencies.size() -> frequencies.size()

        if self.encodeHighFrequencies or highFrequencyTerms is not None:
            # For each set of high-frequency coefficients.
            for highFrequencyInd in range(len(highFrequencies)):
                # Learn a new set of wavelet coefficients to transform the data.
                highFrequencies[highFrequencyInd] = self.applyEncoding(equationString, highFrequencies[highFrequencyInd], self.highFrequenciesWeights[highFrequencyInd], highFrequencyTerms)
                # highFrequencies[highFrequencyInd] dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]

        if self.encodeLowFrequency or lowFrequencyTerms is not None:
            # Learn a new set of wavelet coefficients to transform the data.
            lowFrequency = self.applyEncoding(equationString, lowFrequency, self.lowFrequencyWeights, lowFrequencyTerms)
            # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        return lowFrequency, highFrequencies

    def mixAllFrequencyComponents(self, lowFrequency, highFrequencies):
        # Initialize the relevant parameters.
        lowFrequencyHolder = lowFrequency
        equationString = 'fic,bci->bcf'  # The equation to apply the weights.
        # b = batchSize, c = numOutputSignals, i = initialFrequencyDim, f = finalFrequencyDim
        # 'oif,boi->bof' = weights.size(), frequencies.size() -> frequencies.size()

        if self.encodeLowFrequencyFull:
            # Mix all the frequency terms into one set of frequencies.
            lowFrequencyHolder = torch.cat(tensors=(lowFrequencyHolder, *highFrequencies), dim=2)
            # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape + sum(highFrequenciesShapes)

            # Learn a new set of wavelet coefficients to transform the data.
            lowFrequencyHolder = self.applyEncoding(equationString, lowFrequencyHolder, self.fullLowFrequencyWeights, frequencyTerms=None)
            # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        if self.encodeHighFrequencyFull:
            # For each set of high-frequency coefficients.
            for highFrequencyInd in range(len(highFrequencies)):
                # Mix all the frequency terms into one set of frequencies.
                highFrequenciesComponent = torch.cat(tensors=(lowFrequency, highFrequencies[highFrequencyInd]), dim=2)
                # highFrequenciesComponent dimension: batchSize, numOutputSignals, lowFrequencyShape + highFrequenciesShapes[highFrequencyInd]

                # Learn a new set of wavelet coefficients to transform the data.
                highFrequencies[highFrequencyInd] = self.applyEncoding(equationString, highFrequenciesComponent, self.fullHighFrequencyWeights[highFrequencyInd], frequencyTerms=None)
                # lowFrequency dimension: batchSize, numOutputSignals, highFrequenciesShapes[highFrequencyInd]

        if self.encodeLowFrequencyFull:
            lowFrequency = lowFrequencyHolder

        return lowFrequency, highFrequencies

    def applyEncoding(self, equationString, frequencies, weights, frequencyTerms=None):
        if frequencyTerms is not None:
            # Apply the learned wavelet coefficients.
            frequencies = frequencies + frequencyTerms
            # frequencies dimension: batchSize, numLiftedChannels, frequencyDimension

        if weights is not None:
            if self.independentChannels or self.useConvolutionFlag:
                frequencies = weights(frequencies)  # Learn a new set of wavelet coefficients to transform the data.
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension
            else:
                # Learn a new set of wavelet coefficients to transform the data.
                frequencies = torch.einsum(equationString, weights, frequencies)
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

        return frequencies
