import os
import sklearn
import itertools
import matplotlib.pyplot as plt

# Supress tensorflow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Tensorflow and keras modules
import keras
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

# Import files
from .._globalModel import globalModel # Global model class
from ..generalModels.modelHelpers.Metrics.evaluationMetrics import evaluationMetrics

# ----------------------------------------------------------------------------#
# ----------------------------- Neural Network ------------------------------ #

class Helpers:
    def __init__(self, name, dataDimension, numClasses = 6, optimizer=None, lossFuncs=None, metrics=None):
        self.name = name
        self.dataDimension = dataDimension
        self.numClasses = numClasses
        if optimizer:
            self.optimizers = list(optimizer)
        else:
            self.optimizers = [
                tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'),
                tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad'),
                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam'),
                tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax'),
               # tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
               # tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop'),
               # tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),
               # tf.keras.optimizers.Ftrl(learning_rate=0.001, learning_rate_power=-0.5,
               #        initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0,
               #         name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)
                ]
        if lossFuncs:
            self.loss = list(lossFuncs)
        else:
            self.loss = [
                tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO, name='binary_crossentropy'),
                tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO, name='categorical_crossentropy'),
                tf.keras.losses.CategoricalHinge(reduction=losses_utils.ReductionV2.AUTO, name='categorical_hinge'),
                tf.keras.losses.CosineSimilarity(axis=-1, reduction=losses_utils.ReductionV2.AUTO, name='cosine_similarity'),
                tf.keras.losses.Hinge(reduction=losses_utils.ReductionV2.AUTO, name='hinge'),
                tf.keras.losses.Huber(delta=1.0, reduction=losses_utils.ReductionV2.AUTO, name='huber_loss'),
                tf.keras.losses.KLDivergence(reduction=losses_utils.ReductionV2.AUTO, name='kl_divergence'),
                tf.keras.losses.LogCosh(reduction=losses_utils.ReductionV2.AUTO, name='log_cosh'),
                tf.keras.losses.Loss(reduction=losses_utils.ReductionV2.AUTO, name=None),
                tf.keras.losses.MeanAbsoluteError(reduction=losses_utils.ReductionV2.AUTO, name='mean_absolute_error'),
                tf.keras.losses.MeanAbsolutePercentageError(reduction=losses_utils.ReductionV2.AUTO, name='mean_absolute_percentage_error'),
                tf.keras.losses.MeanSquaredError(reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_error'),
                tf.keras.losses.MeanSquaredLogarithmicError(reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_logarithmic_error'),
                tf.keras.losses.Poisson(reduction=losses_utils.ReductionV2.AUTO, name='poisson'),
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.AUTO, name='sparse_categorical_crossentropy'),
                tf.keras.losses.SquaredHinge(reduction=losses_utils.ReductionV2.AUTO, name='squared_hinge'),
                ]
        if metrics:
            self.metrics = list(metrics)
        else:
            self.metrics = [
                tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', name=None, dtype=None, thresholds=None, multi_label=False, label_weights=None),
                tf.keras.metrics.Accuracy(name='accuracy', dtype=None),
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5),
                tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy', dtype=None, from_logits=False, label_smoothing=0),
                tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None),
                tf.keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy', dtype=None, from_logits=False, label_smoothing=0),
                tf.keras.metrics.CategoricalHinge(name='categorical_hinge', dtype=None),
                tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1),
                tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None),
                tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None),
                tf.keras.metrics.Hinge(name='hinge', dtype=None),
                tf.keras.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=None),
                tf.keras.metrics.LogCoshError(name='logcosh', dtype=None),
                tf.keras.metrics.Mean(name='mean', dtype=None),
                tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error', dtype=None),
                tf.keras.metrics.MeanAbsolutePercentageError(name='mean_absolute_percentage_error', dtype=None),
                tf.keras.metrics.MeanIoU(num_classes=numClasses, name=None, dtype=None),
                tf.keras.metrics.MeanRelativeError(normalizer=[1]*dataDimension, name=None, dtype=None),
                tf.keras.metrics.MeanSquaredError(name='mean_squared_error', dtype=None),
                tf.keras.metrics.MeanSquaredLogarithmicError(name='mean_squared_logarithmic_error', dtype=None),
                tf.keras.metrics.MeanTensor(name='mean_tensor', dtype=None),
                tf.keras.metrics.Poisson(name='poisson', dtype=None),
                tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
                tf.keras.metrics.PrecisionAtRecall(recall=0.5, num_thresholds=200, name=None, dtype=None),
                tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
                tf.keras.metrics.RecallAtPrecision(precision=0.8, num_thresholds=200, name=None, dtype=None),
                tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error', dtype=None),
                tf.keras.metrics.SensitivityAtSpecificity(specificity=0.5, num_thresholds=200, name=None, dtype=None),
                tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype=None),
                tf.keras.metrics.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy', dtype=None, from_logits=False, axis=-1),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='sparse_top_k_categorical_accuracy', dtype=None),
                tf.keras.metrics.SpecificityAtSensitivity(sensitivity=0.5, num_thresholds=200, name=None, dtype=None),
                tf.keras.metrics.SquaredHinge(name='squared_hinge', dtype=None),
                tf.keras.metrics.Sum(name='sum', dtype=None),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_k_categorical_accuracy', dtype=None),
                tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None),
                tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None),
                ]
        
    def neuralPermutations(self):
        neuralOptimizerList = []
        for opt in self.optimizers:
            for loss in self.loss:
                for metric in self.metrics:
                    neuralOptimizerList.append(neuralNetwork(self.name, self.dataDimension, opt, loss, metric))
        return neuralOptimizerList
    
    def permuteMetrics(self, opt, loss):
        neuralOptimizerList = []
        for metric in itertools.permutations(self.metrics, 2):
            neuralOptimizerList.append(neuralNetwork(self.name, self.dataDimension, opt, loss, list(metric)))
        return neuralOptimizerList


    
# class mse_Margin(tf.keras.losses.Loss):
#     def __init__(self, margin):
#         super().__init__()
#         self.relu = tf.keras.layers.ReLU(threshold=margin**3)
        
#     def call(self, y_true, y_pred):
#         squared_difference = tf.square(y_true - y_pred)
#         squared_difference_Threshold = self.relu(squared_difference)
#         mse = tf.reduce_mean(squared_difference_Threshold, axis=-1)
#         return mse
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
    
class neuralNetwork(globalModel):
    """
    Define a Neural Network Class
    """
    def __init__(self, modelPath, modelType, allFeatureNames, overwriteModel):
        """
        Input:
            name: The Name of the Neural Network to Save/Load
        Output: None
        Save: model, name
        """
        # Initialize common model class
        super().__init__(modelPath, modelType, allFeatureNames, overwriteModel)
        
        self.metricsClass = evaluationMetrics() # Define evaluation metrics to score the states.
        
        # Define Model Parameters
        self.history = None

    def _loadModel(self):
        # Save the model
        self.model = keras.models.load_model(self.modelPath, compile=False)
    
    def createModel(self, numInputFeature = 32, numOutputFeatures = 1, opt=None, loss=None, metric=None):
        """
        Parameters
        ----------
        dataDim : The dimension of 1 data point (# of columns in data)
        opt : Neural Network Optimizer
        loss : Neural Network Loss Function
        metric : Neurala Network Metric to Score Accuracy
        """
        # Define a TensorFlow Neural Network using Keras
            # Sequential: Input the List of Hidden Layers into the Network (in order)
            # Dense: Adds a layer of neurons
                # (unit = # neurons in layer, activation function, *if first layer* shape of input data)
            # Input_shape: The dimension of 1 Data Point (# of rows in one column)
        self.model = tf.keras.Sequential()
        
        # Model Layers
        # self.model.add(tf.keras.layers.Dense(units = 3*numInputFeature, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0))),
        # self.model.add(tf.keras.layers.Dense(units = 2*numInputFeature, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))),
        # self.model.add(tf.keras.layers.Dense(units = 28, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04))),
        # self.model.add(tf.keras.layers.Dense(units = 1, activation='linear'))
        
        self.model.add(tf.keras.layers.Dense(units = 3*numInputFeature, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units = 2*numInputFeature, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units = numOutputFeatures, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units = 1, activation='linear'))
        

        # self.model.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        # self.model.add(tf.keras.layers.BatchNormalization())
        
        # self.model.add(tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        # self.model.add(tf.keras.layers.BatchNormalization())
        
        # self.model.add(tf.keras.layers.Dense(units=numOutputFeatures, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        
        # self.model.add(tf.keras.layers.Dense(units=1, activation='linear'))
        
        # Define the Loss Function and Optimizer for the Model
            # Compile: Initializing the optimizer and the loss in the Neural Network
            # Optimizer: The method used to change the Weights in the Network
            # Loss: The Function used to estimate how bad our weights are
        if opt == None: opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if loss == None: loss = tf.keras.losses.MeanSquaredError() # mse_Margin(0.07928499)
        if metric == None: metric = ['accuracy', 'mae']
        
        # Compile the Model
        self.model.compile(optimizer = opt, loss = loss, metrics = list([metric]))
        # print("NN Model Created")
        
    def resetModel(self):
        self.createModel()
    
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames, epochs = 250, seeTrainingSteps = False, returnScore = False):
        # self.createModel(len(Training_Data[0]), opt=None, loss=None, metric=None)
        assert len(featureNames) == len(Training_Data[0]), print(len(featureNames), len(Training_Data[0]))
        
        # For mini-batch gradient decent we want it small (not full batch) to better generalize data
        max_batch_size = 33  # Keep Batch sizes relatively small (no more than 64 or 128)
        mini_batch_gd = min(len(Training_Data)//4, max_batch_size)
        mini_batch_gd = max(1, mini_batch_gd)  # For really small data samples at least take 1 data point
        # For every Epoch (loop), run the Neural Network by:
            # With uninitialized weights, bring data4//4 through network
            # Calculate the loss based on the data
            # Perform optimizer to update the weights
        self.history = self.model.fit(Training_Data, Training_Labels, validation_split=0.05, epochs=int(epochs), shuffle=True, batch_size = int(mini_batch_gd), verbose = 0)
        
        # Save the feature names we trained on
        self.finalFeatureNames = featureNames
        
        if not returnScore:
            return None
        
        # Score the Model
        return self.scoreModel(Testing_Data, Testing_Labels, mini_batch_gd, seeTrainingSteps)
    
    def specificTraining(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, num_epochs = 1):
        batch_size = len(Training_Data) #32
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        total_batches = len(Training_Data) // batch_size
        
        # Iterate through the epochs
        for epoch in range(num_epochs):
            # Shuffle the data at the beginning of each epoch
            X_train_shuffled, y_train_shuffled = sklearn.utils.shuffle(Training_Data, Training_Labels)
            
            # Iterate through the batches
            for batch in range(total_batches):
                # Extract the current batch
                X_batch = X_train_shuffled[batch*batch_size : (batch+1)*batch_size]
                y_batch = y_train_shuffled[batch*batch_size : (batch+1)*batch_size]
                
                # Perform forward propagation
                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred = self.model(X_batch, training=True)  # Assuming self.model is your initialized model
                    
                    # Calculate the loss
                    loss_value = loss_fn(y_batch, y_pred)
                
                # Perform backward propagation
                gradients = tape.gradient(loss_value, self.model.trainable_variables)
                
                # Weight updates
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def scoreModel(self, Testing_Data, Testing_Labels, mini_batch_gd, seeTrainingSteps, plotTraining = False):
        # Score the Model
        results = self.model.evaluate(Testing_Data, Testing_Labels, batch_size=mini_batch_gd, verbose = seeTrainingSteps)
        score = results[0]; accuracy = results[1]; 
        R2 = sklearn.metrics.r2_score(Testing_Labels, self.predict(Testing_Data))
        if plotTraining:
            self.plotStats()
        # print('Test score:', score)
        # print('Test accuracy:', accuracy)
        print(R2)
        return R2
    
    def predict(self, newFeatures):
        # Predict label based on new Data
        return self.model.predict(newFeatures, verbose = 0).reshape(1,-1)[0]
    
    def _saveModel(self, standardizationInfo = [[], [], None, []]):
        # Save the model
        self.model.save(self.modelPath)  # creates a HDF5 file 'my_model.h5'    
    
    def plotStats(self):
        # plot loss during training
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='val')
        plt.legend()
        # plot accuracy during training
        #plt.subplot(212)
        #plt.title('Accuracy')
        #plt.plot(history.history['accuracy'], label='train')
        #plt.plot(history.history['val_accuracy'], label='test')
        #plt.legend()
        plt.show()
    



