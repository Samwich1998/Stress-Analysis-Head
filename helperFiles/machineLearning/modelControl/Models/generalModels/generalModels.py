
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import sys
import joblib
# Machine learning modules
import sklearn
import xgboost
import lightgbm

# Import files
from .._globalModel import globalModel

# -------------------------------------------------------------------------- #
# ------------------------ Simple Regression Modules ----------------------- #

class generalModel(globalModel):
    
    def __init__(self, modelPath, modelType, allFeatureNames, overwriteModel):
        # Initialize common model class
        super().__init__(modelPath, modelType, allFeatureNames, overwriteModel)
            
    def createModel(self):
        
        # ------------------------------------------------------------------ #
        # ----------------------- Regression Modules ----------------------- #
        
        # Linear regression
        if self.modelType == "linReg":
            self.model = sklearn.linear_model.LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
        
        # Logistic regression
        elif self.modelType == "logReg":
            self.model = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                            fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
                            max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
        
        # Ridge regression
        elif self.modelType == "ridgeReg":
            self.model = sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto',
                            positive=False, random_state=None)
        
        # Elastic net
        elif self.modelType == "elasticNet":
            self.model = sklearn.linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, precompute=False,
                            max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, selection='cyclic')
            
        # Support vector regression: SVR_linear, SVR_poly, SVR_rbf, SVR_sigmoid, SVR_precomputed
        elif self.modelType.split('_')[0] == "SVR":
            # C: penatly term. Low C means that we ignore outliers. High C means that we fit perfectly. 1/C = Regularization
            # epsilon: the area around the hyperplane where we will ignore error.
            self.model = sklearn.svm.SVR(kernel=self.modelType.split('_')[1], degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, 
                                         epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
       
        # ------------------------------------------------------------------ #
        # --------------------- Classification Modules --------------------- #
        
        # Support vector classification
        elif self.modelType.split('_')[0] == "SVC":
            if len(self.modelType.split('_')) == 3:
                self.model = sklearn.svm.SVC(kernel=self.modelType.split('_')[1], C=1.0, degree=self.modelType.split('_')[2], gamma='scale', coef0=0.0, shrinking=True, 
                                 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
                                 decision_function_shape='ovr', break_ties=False, random_state=None)
        
            else:
                self.model = sklearn.svm.SVC(kernel=self.modelType.split('_')[1], C=1.0, degree=3, gamma='scale', coef0=0.0, shrinking=True, 
                                 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
                                 decision_function_shape='ovr', break_ties=False, random_state=None)
        
        # K-nearest neighbors
        elif self.modelType == "KNN":
            if len(self.modelType.split('_')) == 2:
                self.model = sklearn.neighbors.KNeighborsClassifier(n_neighbors = self.modelType.split('_')[1], weights = 'uniform', algorithm = 'auto', 
                                leaf_size = 30, p = 2, metric = 'minkowski', metric_params = None, n_jobs = None)
            elif len(self.modelType.split('_')) == 3:
                 self.model = sklearn.neighbors.KNeighborsClassifier(n_neighbors = self.modelType.split('_')[1], weights = 'uniform', algorithm = 'auto', 
                                 leaf_size = self.modelType.split('_')[2], p = 2, metric = 'minkowski', metric_params = None, n_jobs = None)
            else:
                self.model = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', algorithm = 'auto', 
                            leaf_size = 30, p = 2, metric = 'minkowski', metric_params = None, n_jobs = None)
        
        # ------------------------------------------------------------------ #
        # -------------------------- Tree Modules -------------------------- #
        
        # Random forest
        elif self.modelType == "RF":
            self.model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                        max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, 
                        random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
        
        # Adaboosting
        elif self.modelType == "ADA":
            self.model = sklearn.ensemble.AdaBoostClassifier(estimator=None, n_estimators=100, learning_rate=1, 
                                    algorithm='SAMME.R', random_state=None)
        
        # ------------------------------------------------------------------ #
        # ------------------------- XGBoost Modules ------------------------ #
        
        # XGBoost classification
        elif self.modelType == "XGB":
            self.model = xgboost.XGBClassifier()
            
        # XGBoost regression
        elif self.modelType == "XGB_Reg":
            self.model = xgboost.XGBRegressor(n_estimators=1000, max_leaves=0, max_depth=7, eta=0.1, subsample=0.7, 
                                              colsample_bytree=0.8, verbosity  = 0)
        
        elif self.modelType == "lightGBM_Reg":
            self.model = lightgbm.LGBMRegressor(boosting_type='gbdt', num_leaves=100, max_depth=-1, learning_rate=0.1, 
                        n_estimators=200, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, 
                        min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, 
                        reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=None, importance_type='split')
            
        # ------------------------------------------------------------------ #
        
        else:
            sys.exit("Looked in the general models. Found no model called '" + self.modelType + "'")
            
    def resetModel(self):
        self.createModel()
            
    def _loadModel(self):
        with open(self.modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        
    def _saveModel(self):
        # Save the model
        joblib.dump(self.model, self.modelPath)
        
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames, imbalancedData = False):  
        assert len(featureNames) == len(Training_Data[0])
        
        self.resetModel()

        # Train the Model
        if self.modelType.split('_')[0] == "XGB":
            # Create validation set
            Training_Data, Validation_Data, Training_Labels, Validation_Labels = sklearn.model_selection.train_test_split(
                           Training_Data, Training_Labels, test_size=0.1, shuffle= True, stratify=None)
            self.model.fit(Training_Data, Training_Labels, eval_set=[(Validation_Data, Validation_Labels)], sample_weight=None, \
                           base_margin=None, eval_metric=None, early_stopping_rounds=None, verbose=False, \
                           xgb_model=None, sample_weight_eval_set=None, base_margin_eval_set=None, feature_weights=None, callbacks=None) 
        else: 
             self.model.fit(Training_Data, Training_Labels)
        
        # Save the feature names we trained on
        self.finalFeatureNames = featureNames

        return self.scoreModel(Testing_Data, Testing_Labels, imbalancedData = imbalancedData)
        

        # Return the model score
        return self.scoreModel(Testing_Data, Testing_Labels, imbalancedData = imbalancedData)
    
    def scoreModel(self, signalData, signalLabels, imbalancedData = False):
        if imbalancedData:
            pred = self.predict(signalData)
            return sklearn.metrics.balanced_accuracy_score(signalLabels, pred)
        return self.model.score(signalData, signalLabels)
    
    def predict(self, newFeatures):
        # Predict label based on new Data
        return self.model.predict(newFeatures).reshape(1,-1)[0]
    
# ---------------------------------------------------------------------------#
