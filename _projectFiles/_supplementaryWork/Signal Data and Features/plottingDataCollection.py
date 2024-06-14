
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import numpy as np
from natsort import natsorted

# Import interfaces for reading/writing data
sys.path.append(os.path.dirname(__file__) + "/../../Helper Files/Data Aquisition and Analysis/Excel Processing/") 
import extractDataProtocols  # Functions to Save/Read in Data from Excel
# Import data aquisition and analysis files
sys.path.append(os.path.dirname(__file__) + "/../../Helper Files/Data Aquisition and Analysis/") 
import streamingProtocols      # Functions to Handle Data from Arduino

# Import files for extracting feature names
sys.path.append(os.path.dirname(__file__) + "/../../Helper Files/Machine Learning/Feature Analysis/_compiledFeatureNames/") 
import _compileFeatureNames  # Functions to extract feature names

# Import plotting files
sys.path.append(os.path.dirname(__file__) + "/Helper Methods/") 
import signalPlotting  
import labelPlotting
import varyWindows

if __name__ == "__main__":

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #

    # Specify biomarker information.
    streamingOrder = ["eog", "eeg", "eda", "temp"]  # A List Representing the Order of the Sensors being Streamed in.
    extractFeaturesFrom = ["eog", "eeg", "eda", "temp"] # A list with all the biomarkers from streamingOrder for feature extraction
    allAverageIntervals = [120, 75, 90, 45] # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60
    allAverageIntervals = [60, 30, 30, 30] # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60
        
    # Compile feature names
    featureNames, biomarkerFeatureNames, biomarkerFeatureOrder = _compileFeatureNames.compileFeatureNames().extractFeatureNames(extractFeaturesFrom)
    
    featureAverageWindows = []
    # Compile feature average windows.
    for biomarker in biomarkerFeatureOrder:
        featureAverageWindows.append(allAverageIntervals[streamingOrder.index(biomarker)])
    # sys.exit()
        
    analyzeAllData = True
    if analyzeAllData:
        dataDirectory = './../../Data/allSensors_finalDataset/'
        testDataExcelFiles = []
        
        for fileName in natsorted(os.listdir(dataDirectory)):
            if fileName.startswith("2023"):
                testDataExcelFiles.append(dataDirectory + fileName)
    else:
        # Specify the input file to analyze
        testDataExcelFiles = [            
            './../../Data/allSensors_finalDataset/2023-01-30 CPT Trial Daniel.xlsx',
            './../../Data/allSensors_finalDataset/2023-01-19 Music Trial Juliane.xlsx',
            './../../Data/allSensors_finalDataset/2023-02-23 Exercise Trial Hyunah.xlsx',
            './../../Data/allSensors_finalDataset/2023-04-21 VR Trial Karteek.xlsx',  
        ]
        
        testDataExcelFiles = [            
            './../../Data/allSensors_finalDataset/2023-01-30 CPT Trial Daniel.xlsx',
            './../../Data/allSensors_finalDataset/2023-02-01 CPT Trial Ben.xlsx',
            './../../Data/allSensors_finalDataset/2023-02-01 CPT Trial Jose.xlsx',
            './../../Data/allSensors_finalDataset/2023-02-02 CPT Trial Soyoung.xlsx',  
        ]
    
    sys.exit()

    # ---------------------------------------------------------------------- #
    # ---------------------- Read and Analyze the Data --------------------- #

    readDatas = []; compiledRawDatas = []
    for testDataExcelFile in testDataExcelFiles:
        # Initialize instance to analyze the data
        readData = streamingProtocols.streamingProtocols(None, [], None, 2048576, 1048100, streamingOrder, [], featureAverageWindows, False)
        
        # Collect the Data from Excel
        compiledRawData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions = \
                                  extractDataProtocols.extractData().getData(testDataExcelFile, numberOfChannels = len(streamingOrder), testSheetNum = 0)
        # Analyze the Data using the Correct Protocol
        readData.streamExcelData(compiledRawData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, 
                                  surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, testDataExcelFile)

        # Store the information
        readDatas.append(readData)
        compiledRawDatas.append(compiledRawData)
    sys.exit()
    
    # ---------------------------------------------------------------------- #
    # ------------------- Plotting the data and Features ------------------- #
    
    # Initialize plotting class
    plottingLabelClass = labelPlotting.plotData()
    plottingSignalClass = signalPlotting.plotData()
    plottingFeatureWindows = varyWindows.varyWindows()
    
    # Create and save the plots.
    # plottingSignalClass.plotFigures(readDatas, testDataExcelFiles, featureNames, biomarkerFeatureNames)
    plottingLabelClass.plotSurveyInfo(readDatas, surveyQuestions)
    
    
    # # featureTimeWindows = np.arange(4.5, 60, 0.01)  # Cant go lower than 4 (not equal to 4).
    # featureTimeWindows = np.arange(4.5, 20, 0.01)  # Cant go lower than 4 (not equal to 4).
    # plottingFeatureWindows.varyAnalysisParam(readDatas, compiledRawDatas, testDataExcelFiles, featureTimeWindows, 
    #                                           featureAverageWindows, biomarkerFeatureOrder, biomarkerFeatureNames)
    
    



