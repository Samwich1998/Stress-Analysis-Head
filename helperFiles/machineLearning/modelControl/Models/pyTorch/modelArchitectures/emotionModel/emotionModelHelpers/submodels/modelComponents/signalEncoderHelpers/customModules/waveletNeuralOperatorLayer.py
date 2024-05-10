# General
import torch

# Import machine learning files
from .waveletNeuralHelpers import waveletNeuralHelpers


class waveletNeuralOperatorLayer(waveletNeuralHelpers):

    def __init__(self, numInputSignals, numOutputSignals, sequenceBounds, numDecompositions=2, wavelet='db3', mode='zero', numLayers=1, encodeLowFrequencyProtocol=0, encodeHighFrequencyProtocol=0, skipConnectionProtocol='CNN'):
        super(waveletNeuralOperatorLayer, self).__init__(numInputSignals, numOutputSignals, sequenceBounds, numDecompositions, wavelet, mode, numLayers, encodeLowFrequencyProtocol, encodeHighFrequencyProtocol, skipConnectionProtocol)

    def forward(self, inputData, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Apply the wavelet neural operator and the skip connection.
        neuralOperatorOutput = self.waveletNeuralOperator(inputData, lowFrequencyTerms, highFrequencyTerms)
        neuralOperatorOutput = neuralOperatorOutput + self.skipConnectionModel(inputData)
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
        # inputData dimension: batchSize, numInputSignals, maxSequenceLength

        # Perform wavelet decomposition.
        lowFrequency, highFrequencies = self.dwt(inputData)  # Note: each channel is treated independently here.
        # highFrequencies[decompositionLayer] dimension: batchSize, numInputSignals, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numInputSignals, lowFrequencyShape

        # Apply the activation function if we already applied a linear transformation.
        lowFrequency = self.activationFunction(lowFrequency)
        for highFrequencyInd in range(len(highFrequencies)):
            highFrequencies[highFrequencyInd] = self.activationFunction(highFrequencies[highFrequencyInd])

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

        # Remove the padding.
        reconstructedData = reconstructedData[:, :, -sequenceLength:]
        # reconstructedData dimension: batchSize, numOutputSignals, sequenceLength

        # Add the bias terms.
        reconstructedData = reconstructedData + self.operatorBiases
        reconstructedData = self.activationFunction(reconstructedData)
        # outputData dimension: batchSize, numOutputSignals, sequenceLength

        return reconstructedData

    def mixSeperatedFrequencyComponents(self, lowFrequency, highFrequencies, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Set up the equation to apply the weights.
        equationString = 'oin,bin->bon'  # The equation to apply the weights.
        # b = batchSize, i = numInputSignals, o = numOutputSignals, n = signalDimension
        # 'oin,bin->bon' = weights.size(), frequencies.size() -> frequencies.size()

        if self.encodeHighFrequencies:
            # For each set of high-frequency coefficients.
            for highFrequencyInd in range(len(highFrequencies)):
                # Learn a new set of wavelet coefficients to transform the data.
                highFrequencies[highFrequencyInd] = self.applyEncoding(equationString, highFrequencies[highFrequencyInd], self.highFrequenciesWeights[highFrequencyInd], highFrequencyTerms)
                # highFrequencies[highFrequencyInd] dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]

        if self.encodeLowFrequency:
            # Learn a new set of wavelet coefficients to transform the data.
            lowFrequency = self.applyEncoding(equationString, lowFrequency, self.lowFrequencyWeights, lowFrequencyTerms)
            # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        return lowFrequency, highFrequencies

    def mixAllFrequencyComponents(self, lowFrequency, highFrequencies):
        # Initialize the relevant parameters.
        lowFrequencyHolder = lowFrequency
        equationString = 'oif,boi->bof'  # The equation to apply the weights.
        # b = batchSize, o = numOutputSignals, i = initialFrequencyDim, f = finalFrequencyDim
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
                # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        if self.encodeLowFrequencyFull:
            lowFrequency = lowFrequencyHolder

        return lowFrequency, highFrequencies

    def applyEncoding(self, equationString, frequencies, weights, frequencyTerms=None):
        if frequencyTerms is not None:
            # Apply the learned wavelet coefficients.
            frequencies = frequencies + frequencyTerms
            # frequencies dimension: batchSize, numInputSignals, frequencyDimension

        for layerInd in range(self.numLayers):
            # # Apply the activation function if we already applied a linear transformation.
            # if layerInd != 0: frequencies = self.activationFunction(frequencies)
            # # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

            # Learn a new set of wavelet coefficients to transform the data.
            frequencies = torch.einsum(equationString, weights[layerInd], frequencies)
            # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

            # Apply the activation function if we already applied a linear transformation.
            frequencies = self.activationFunction(frequencies)
            # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

        return frequencies
