# Import files for machine learning
from .emotionModel.emotionModelHelpers.lossInformation.organizeTrainingLosses import organizeTrainingLosses
from .emotionModel.emotionModelHelpers.modelVisualizations.modelVisualizations import modelVisualizations
from .emotionModel.emotionModelHelpers.generalMethods.weightInitialization import weightInitialization
from .emotionModel.emotionModelHelpers.optimizerMethods.optimizerMethods import optimizerMethods
from .emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from .emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from .emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from .emotionModel.emotionModelHelpers.modelParameters import modelParameters
from .emotionModel.emotionModelHead import emotionModelHead
from ...Helpers.modelMigration import modelMigration


class emotionPipelineHelpers:

    def __init__(self, accelerator, modelID, datasetName, modelName, allEmotionClasses, sequenceLength, maxNumSignals,
                 numSubjectIdentifiers, demographicLength, numSubjects, userInputParams, emotionNames,
                 activityNames, featureNames, submodel, useFinalParams, debuggingResults=False):
        # General parameters.
        self.numSubjectIdentifiers = numSubjectIdentifiers  # The number of subject identifiers to consider. Dim: [numSubjectIdentifiers]
        self.demographicLength = demographicLength  # The amount of demographic information provided to the model (age, weight, etc.). Dim: [numDemographics]
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.sequenceLength = sequenceLength  # The length of each incoming signal. Type: int
        self.useFinalParams = useFinalParams  # Whether to use the HPC parameters.
        self.accelerator = accelerator  # Hugging face interface to speed up the training process.
        self.modelName = modelName  # The unique name of the model to initialize.
        self.modelID = modelID  # A unique integer identifier for this model.

        # Pre-initialize later parameters.
        self.optimizer = None
        self.scheduler = None
        self.model = None

        # Dataset-specific parameters.
        self.allEmotionClasses = allEmotionClasses  # The number of classes (intensity levels) within each emotion to predict. Dim: [numEmotions]
        self.activityLabelInd = len(emotionNames)  # The index of the activity label in the label array.
        self.numActivities = len(activityNames)  # The number of activities we are predicting. Type: int
        self.numEmotions = len(emotionNames)  # The number of emotions we are predicting. Type: int
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.maxNumSignals = maxNumSignals  # The maximum number of signals to consider. Type: int
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.featureNames = featureNames  # The names of each signal in the model. Dim: numSignals
        self.datasetName = datasetName  # The name of the specific dataset being used in this model (case, wesad, etc.)

        # Store model parameters.
        signalMinMaxScale = modelParameters.getSignalMinMaxScale()

        # Initialize the emotion model.
        if modelName == "emotionModel":
            self.model = emotionModelHead(submodel, accelerator, sequenceLength, signalMinMaxScale, maxNumSignals, numSubjectIdentifiers, demographicLength, userInputParams,
                                          emotionNames, activityNames, featureNames, numSubjects, datasetName, useFinalParams, debuggingResults)
        # Assert that the model has been initialized.
        assert hasattr(self, 'model'), f"Unknown Model Type Requested: {modelName}"

        # Extract relevant properties from the model.
        self.generalTimeWindow = self.model.timeWindows[-1]  # The default time window to use for training and testing.

        # Initialize helper classes.
        self.organizeLossInfo = organizeTrainingLosses(accelerator, self.model, allEmotionClasses, self.activityLabelInd, self.generalTimeWindow, self.useFinalParams)
        self.modelVisualization = modelVisualizations(accelerator, self.generalTimeWindow, modelSubfolder="trainingFigures/")
        self.modelParameters = modelParameters(userInputParams=userInputParams, accelerator=accelerator)
        self.optimizerMethods = optimizerMethods(userInputParams, useFinalParams)
        self.weightInitialization = weightInitialization()
        self.modelMigration = modelMigration(accelerator)
        self.dataInterface = emotionDataInterface()
        self.generalMethods = generalMethods()
        self.modelHelpers = modelHelpers()

        if submodel == "emotionPrediction":
            # Finalize model setup.
            self.model.sharedEmotionModel.lastActivityLayer = self.modelHelpers.getLastActivationLayer(self.organizeLossInfo.activityClass_lossType, predictingProb=True)  # Apply activation on the last layer: 'softmax', 'logsoftmax', or None.
            self.model.sharedEmotionModel.lastEmotionLayer = self.modelHelpers.getLastActivationLayer(self.organizeLossInfo.emotionDist_lossType, predictingProb=True)  # Apply activation on the last layer: 'softmax', 'logsoftmax', or None.
        else:
            self.generalTimeWindowInd = self.model.timeWindows.index(self.generalTimeWindow)

        # Assert data integrity of the inputs.
        assert len(self.emotionNames) == len(self.allEmotionClasses), f"Found {len(self.emotionNames)} emotions with {len(self.allEmotionClasses)} classes specified."
        assert len(self.activityNames) == self.numActivities, f"Found {len(self.activityNames)} activities with {self.numActivities} classes specified."

    def compileOptimizer(self, submodel):
        # Get the models, while considering whether they are distributed or not.
        trainingInformation, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel = self.getDistributedModels(model=None, submodel=None)

        # Initialize the optimizer and scheduler.
        self.optimizer, self.scheduler = self.optimizerMethods.addOptimizer(submodel, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel)

    def acceleratorInterface(self, dataLoader=None):
        if dataLoader is None:
            self.optimizer, self.scheduler, self.model = self.accelerator.prepare(self.optimizer, self.scheduler, self.model)
        else:
            self.optimizer, self.scheduler, self.model, dataLoader = self.accelerator.prepare(self.optimizer, self.scheduler, self.model, dataLoader)

        return dataLoader

    def prepareInformation(self, dataLoader):
        # Load in all the data and labels for final predictions.
        allData, allLabels, allTrainingMasks, allTestingMasks = dataLoader.dataset.getAll()
        allSignalData, allDemographicData, allSubjectIdentifiers = self.dataInterface.separateData(allData, self.sequenceLength, self.numSubjectIdentifiers, self.demographicLength)
        reconstructionIndex = self.dataInterface.getReconstructionIndex(allTrainingMasks)
        assert reconstructionIndex is not None

        # Assert the integrity of the dataloader.
        assert allLabels.shape[1] == self.numEmotions + 1, f"Found {allLabels.shape[1]} labels, but expected {self.numEmotions} emotions + 1 activity label."
        assert allLabels.shape == allTrainingMasks.shape, "We should specify the training indices for each label"
        assert allLabels.shape == allTestingMasks.shape, "We should specify the testing indices for each label"

        return allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allDemographicData, allSubjectIdentifiers, reconstructionIndex

    # ------------------------------------------------------------------ #

    def getTrainingEpoch(self, submodel):
        # Get the models, while considering whether they are distributed or not.
        trainingInformation, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel = self.getDistributedModels()

        if submodel == "signalEncoder":
            return max(0, len(signalEncoderModel.trainingLosses_timeReconstructionAnalysis[self.generalTimeWindowInd]) - 1)
        elif submodel == "autoencoder":
            return max(0, len(autoencoderModel.trainingLosses_timeReconstructionAnalysis[self.generalTimeWindowInd]) - 1)
        elif submodel == "emotionPrediction":
            return max(0, len(specificEmotionModel.trainingLosses_signalReconstruction) - 1)
        else:
            raise Exception()

    def setupTraining(self, submodel):
        # Do not train the model at all.
        self.setupTrainingFlags(self.model, trainingFlag=False)

        # Get the models, while considering whether they are distributed or not.
        trainingInformation, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel = self.getDistributedModels()

        # Label the model we are training.
        if submodel == "signalEncoder":
            self.setupTrainingFlags(signalEncoderModel, trainingFlag=True)
        elif submodel == "autoencoder":
            self.setupTrainingFlags(autoencoderModel, trainingFlag=True)
        elif submodel == "emotionPrediction":
            self.setupTrainingFlags(signalMappingModel, trainingFlag=True)
            self.setupTrainingFlags(specificEmotionModel, trainingFlag=True)
            self.setupTrainingFlags(sharedEmotionModel, trainingFlag=True)
        else:
            assert False, "No model initialized"

    @staticmethod
    def setupTrainingFlags(model, trainingFlag):
        # Set the model to training mode.
        if trainingFlag:
            model.train()
            # Or evaluation mode.
        else:
            model.eval()

        # Recursively set the mode for all submodules
        for submodule in model.modules():  # This ensures all submodules are included
            if trainingFlag:
                submodule.train()
            else:
                submodule.eval()

        # For each model parameter.
        for param in model.parameters():
            # Change the gradient tracking status
            param.requires_grad = trainingFlag

    # ------------------------------------------------------------------ #

    def getDistributedModel(self):
        # Check if the model is wrapped with DDP
        if hasattr(self.model, 'module'):
            return self.model.module
        else:
            return self.model

    def getDistributedModels(self, model=None, submodel=None):
        if model is None:
            model = self.getDistributedModel()
        # Get the specific models.
        trainingInformation = model.trainingInformation
        signalEncoderModel = model.signalEncoderModel
        autoencoderModel = model.autoencoderModel
        signalMappingModel = model.signalMappingModel
        sharedEmotionModel = model.sharedEmotionModel
        specificEmotionModel = model.specificEmotionModel

        if submodel == "trainingInformation":
            return trainingInformation
        elif submodel == "signalEncoder":
            return signalEncoderModel
        elif submodel == "autoencoder":
            return autoencoderModel
        elif submodel == "emotionPrediction":
            return signalMappingModel, sharedEmotionModel, specificEmotionModel
        elif submodel is None:
            return trainingInformation, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel
        else:
            assert False, "No model initialized"
