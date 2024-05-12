#!/bin/bash

# Define the source base directory
SOURCE_BASE="helperFiles/machineLearning"

# Define the destination directory
DESTINATION="./../_mainBranchCopies/"

# Copy each file to the destination directory
cp "${SOURCE_BASE}/machineLearningInterface.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHead.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/emotionDataInterface.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/generalMethods/modelHelpers.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/generalMethods/weightInitialization.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/lossInformation/lossCalculations.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/lossInformation/organizeTrainingLosses.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/modelVisualizations/_signalEncoderVisualizations.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/modelVisualizations/modelVisualizations.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/helperModules/trainingSignalEncoder.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/generalSignalEncoder.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/modelHelpers/abnormalConvolutions.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/modelHelpers/convolutionalHelpers.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/signalEncoderHelpers/changeVariance.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/signalEncoderHelpers/channelEncoding.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/signalEncoderHelpers/channelPositionalEncoding.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/signalEncoderHelpers/customModules/waveletNeuralHelpers.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/signalEncoderHelpers/customModules/waveletNeuralOperatorLayer.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/signalEncoderHelpers/denoiser.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/signalEncoderHelpers/signalEncoderHelpers.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/modelComponents/signalEncoderHelpers/signalEncoderModules.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionModelHelpers/submodels/signalEncoderModel.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/Models/pyTorch/modelArchitectures/emotionModel/emotionPipeline.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/modelControl/_finalModels/_storeModels.py" "${DESTINATION}/"
cp "${SOURCE_BASE}/dataInterface/compileModelData.py" "${DESTINATION}/"
cp "./metaTrainingControl.py" "${DESTINATION}/"

echo "All files have been copied successfully."
