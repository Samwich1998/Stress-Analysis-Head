
# General Modules
import os
import time
import collections
# Data interface modules
from openpyxl import load_workbook, Workbook

# Import file for handling excel format
from .excelFormatting import handlingExcelFormat


class saveExcelData(handlingExcelFormat):
    
    def __init__(self):
        super().__init__()
    
    def getExcelDocument(self, excelFile, overwriteSave = False):
        # If the excel file you are saving already exists.
        if os.path.isfile(excelFile):
            # If You Want to Overwrite the Excel.
            if overwriteSave:
                print("\t\tDeleting Old Excel Workbook")
                os.remove(excelFile) 
            else:
                print("\t\tNot overwriting the file ... but your file already exists??")
            
        # If the File is Not Present: Create The Excel File
        if not os.path.isfile(excelFile):
            print("\t\tCreating New Excel Workbook")
            # Make Excel WorkBook
            WB = Workbook()
            worksheet = WB.active 
            worksheet.title = self.emptySheetName
        else:
            print("\t\tExcel File Already Exists. Adding New Sheet to File")
            WB = load_workbook(excelFile, read_only=False)
            worksheet = WB.create_sheet(self.emptySheetName)
        return WB, worksheet

    def addSubjectInfo(self, WB, worksheet, subjectInformationAnswers, subjectInformationQuestions):
        # Assert that the data is in the correct configuration
        assert len(subjectInformationAnswers) == len(subjectInformationQuestions), \
            f"{len(subjectInformationAnswers)} {subjectInformationAnswers} \n {len(subjectInformationQuestions)} {subjectInformationQuestions}"
        
        # Get the information ready for the file
        header = ["Background Questions", "Answers"]
        subjectInformationPointer = 0
                
        # Loop through/save all the data in batches of maxAddToExcelSheet.
        for firstIndexInFile in range(0, len(subjectInformationQuestions), self.maxAddToExcelSheet):
            # Add the information to the page
            worksheet.title = self.subjectInfo_SheetName
            worksheet.append(header)  # Add the header labels to this specific file.
            
            # Add the info to the first page
            while subjectInformationPointer != len(subjectInformationQuestions):
                # Add the data row to the worksheet
                row = [subjectInformationQuestions[subjectInformationPointer], subjectInformationAnswers[subjectInformationPointer]]
                worksheet.append(row)
                
                subjectInformationPointer += 1
                # Move onto next excel sheet if too much data
                if int(subjectInformationPointer/(firstIndexInFile+1)) == self.maxAddToExcelSheet:
                    break

            # Finalize document
            worksheet = self.addExcelAesthetics(worksheet) # Add Excel Aesthetics
            worksheet = WB.create_sheet(self.emptySheetName) # Add Sheet
        # Remove empty page
        WB.remove(worksheet)
    
    def addExperimentInfo(self, WB, worksheet, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions):
        # Assert that the data is in the correct configuration
        assert len(experimentTimes) == len(experimentNames)
        assert len(surveyAnswerTimes) == len(surveyAnswersList)
        # Set pointer
        experimentInfoPointer = 0; featureInfoPointer = 0
    
        # Get the Header for the experiment and survey
        header = ["Start Experiment (Seconds)", "End Experiment (Seconds)", "Experiment Label"]
        header.append("Feature Collection (Seconds)")
        header.extend(surveyQuestions)
                
        # Loop through/save all the data in batches of maxAddToExcelSheet.
        for firstIndexInFile in range(0, max(len(experimentTimes), len(surveyAnswerTimes)), self.maxAddToExcelSheet):
            # Add the information to the page
            worksheet.title = self.experimentalInfo_SheetName
            worksheet.append(header)  # Add the header labels to this specific file.
            
            # Add the info to the first page
            while experimentInfoPointer != len(experimentTimes) or featureInfoPointer != len(surveyAnswerTimes):
                row = []
                
                # Add experimental information
                if experimentInfoPointer != len(experimentTimes):
                    row.extend(experimentTimes[experimentInfoPointer])
                    row.append(experimentNames[experimentInfoPointer])
                    experimentInfoPointer += 1
                else:
                    row.extend([None]*3)
                # Add feature information
                if featureInfoPointer != len(surveyAnswerTimes):
                    row.append(surveyAnswerTimes[featureInfoPointer])
                    row.extend(surveyAnswersList[featureInfoPointer])
                    featureInfoPointer += 1
                
                # Add the data row to the worksheet
                worksheet.append(row)
                # Move onto the next Excel sheet if too much data
                if int(experimentInfoPointer/(firstIndexInFile+1)) == self.maxAddToExcelSheet or int(featureInfoPointer/(firstIndexInFile+1)) == self.maxAddToExcelSheet:
                    break

            # Finalize document
            self.addExcelAesthetics(worksheet) # Add Excel Aesthetics
            worksheet = WB.create_sheet(self.emptySheetName) # Add Sheet
        # Remove empty page
        WB.remove(worksheet)
    
    def saveData(self, timePoints, signalData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, 
                 surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, dataHeaders, saveExcelPath, overwriteSave = False):
        print("\n\tSaving raw signals")
        # ------------------------------------------------------------------ #
        # -------------------- Setup the excel document -------------------- #
        # Create the path to save the excel file.
        os.makedirs(os.path.dirname(saveExcelPath), exist_ok=True) # Create Output File Directory to Save Data: If None Exists
        
        # Get the excel document.
        WB, worksheet = self.getExcelDocument(saveExcelPath, overwriteSave)

        # ------------------------------------------------------------------ #
        # -------------- Add experimental/subject information -------------- #
        # Add subject information
        self.addSubjectInfo(WB, worksheet, subjectInformationAnswers, subjectInformationQuestions)
        worksheet = WB.create_sheet(self.emptySheetName) # Add 
        # Add experimental information
        self.addExperimentInfo(WB, worksheet, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions)
        worksheet = WB.create_sheet(self.emptySheetName) # Add Sheet
        
        # ------------------------------------------------------------------ #
        # ---------------------- Add data to document ---------------------- #     
        # Get the Header for the Data
        header = ["Time (Seconds)"]
        header.extend([dataHeader.upper() + " Raw Data" for dataHeader in dataHeaders])
        
        # Loop through/save all the data in batches of maxAddToExcelSheet.
        for firstIndexInFile in range(0, len(timePoints), self.maxAddToExcelSheet):
            startTimer = time.time()
            # Add the information to the page
            worksheet.title = self.rawSignals_Sheetname
            worksheet.append(header)  # Add the header labels to this specific file.
                        
            # Loop through all data to be saved within this sheet in the excel file.
            for dataInd in range(firstIndexInFile, min(firstIndexInFile+self.maxAddToExcelSheet, len(timePoints))):
                # Organize all the data
                row = [timePoints[dataInd]]
                row.extend([dataCol[dataInd] for dataCol in signalData])
                
                # Add the row to the worksheet
                worksheet.append(row)
    
            # Finalize document
            worksheet = self.addExcelAesthetics(worksheet) # Add Excel Aesthetics
            worksheet = WB.create_sheet(self.emptySheetName) # Add Sheet
            
            # If I need to use another sheet
            if firstIndexInFile + self.maxAddToExcelSheet < len(timePoints):
                # Keep track of how long it is taking.
                endTimer = time.time()
                numberOfSheetsLeft = 1+(len(timePoints) - firstIndexInFile - self.maxAddToExcelSheet)//self.maxAddToExcelSheet
                timeRemaining = (endTimer - startTimer)*numberOfSheetsLeft
                print("\tEstimated Time Remaining " + str(timeRemaining) + " seconds; Excel Sheets Left to Add: " + str(numberOfSheetsLeft))
        # Remove empty page
        if worksheet.title == self.emptySheetName:
            WB.remove(worksheet)
        
        # ------------------------------------------------------------------ #
        # ------------------------ Save the document ----------------------- #  
        # Save as New Excel File
        WB.save(saveExcelPath)
        WB.close()
            
    def saveRawFeatures(self, rawFeatureTimesHolder, rawFeatureHolder, indivisualFeatureNames, biomarkerOrder, experimentTimes, experimentNames,
                     surveyAnswerTimes, surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, excelFilename, overwriteSave = True): 
        print("\n\tSaving raw features")
        # ------------------------------------------------------------------ #
        # -------------------- Setup the excel document -------------------- #
        # Organize variables
        baseFilename = os.path.basename(excelFilename).split('.')[0]
        excelPath = excelFilename.split(baseFilename)[0]
        
        # Create the path to save the excel file.
        saveDataFolder = excelPath + self.saveFeatureFolder
        os.makedirs(saveDataFolder, exist_ok=True) # Create Output File Directory to Save Data: If None Exists
        # Specify the name of the file to save
        saveExcelName = baseFilename + self.saveFeatureFile_Appended
        excelFile = saveDataFolder + saveExcelName
        
        # Get the excel document.
        WB, worksheet = self.getExcelDocument(excelFile, overwriteSave)
        
        # ------------------------------------------------------------------ #
        # -------------- Add experimental/subject information -------------- #
        # Add subject information
        self.addSubjectInfo(WB, worksheet, subjectInformationAnswers, subjectInformationQuestions)
        worksheet = WB.create_sheet(self.emptySheetName) # Add 
        # Add experimental information
        self.addExperimentInfo(WB, worksheet, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions)
        worksheet = WB.create_sheet(self.emptySheetName) # Add Sheet

        # ------------------------------------------------------------------ #
        # ---------------------- Add data to document ---------------------- #  
        counter = collections.Counter()
        biomarkerChannelIndices = [counter.update({item: 1}) or counter[item] - 1 for item in biomarkerOrder]
        
        # Indivisually add features from each sensor to the excel file.
        for biomarkerInd in range(len(biomarkerOrder)):
            featureNames = indivisualFeatureNames[biomarkerInd]
            channelIndex = biomarkerChannelIndices[biomarkerInd]
            # Extract raw features
            featureTimes = rawFeatureTimesHolder[biomarkerInd]
            rawFeatures = rawFeatureHolder[biomarkerInd] 
            
            # Create the header bar
            header = ["Time (Seconds)"]
            header.extend(featureNames)
        
            # Loop through/save all the data in batches of maxAddToExcelSheet.
            for firstIndexInFile in range(0, len(featureTimes), self.maxAddToExcelSheet):
                # Add the information to the page
                worksheet.title = biomarkerOrder[biomarkerInd].upper() + f" CH{channelIndex}" + self.rawFeatures_AppendedSheetName # Add the sheet name to the file
                worksheet.append(header)  # Add the header labels to this specific file.
                            
                # Loop through all data to be saved within this sheet in the excel file.
                for dataInd in range(firstIndexInFile, min(firstIndexInFile+self.maxAddToExcelSheet, len(featureTimes))):
                    # Organize all the data
                    row = [featureTimes[dataInd]]
                    row.extend(rawFeatures[dataInd])
                                    
                    # Add the row to the worksheet
                    worksheet.append(row)
        
                # Finalize document
                worksheet = self.addExcelAesthetics(worksheet) # Add Excel Aesthetics
                worksheet = WB.create_sheet(self.emptySheetName) # Add Sheet
        # Remove empty page
        if worksheet.title == self.emptySheetName:
            WB.remove(worksheet)
            
        # ------------------------------------------------------------------ #
        # ------------------------ Save the document ----------------------- #  
        # Save as New Excel File
        WB.save(excelFile)
        WB.close()
        
    def saveFeatureComparison(self, dataMatrix, rowHeaders, colHeaders, saveDataFolder, saveExcelName, sheetName = "Feature Comparison", saveFirstSheet = False):
        print("Saving the Data")
        # Create Output File Directory to Save Data: If Not Already Created
        os.makedirs(saveDataFolder, exist_ok=True)
        
        # Create Path to Save the Excel File
        excelFile = saveDataFolder + saveExcelName
        print(excelFile)
        
        # If the File is Not Present: Create it
        if not os.path.isfile(excelFile):
            # Make Excel WorkBook
            WB = Workbook()
            WB_worksheet = WB.active 
            WB_worksheet.title = sheetName
        else:
            print("Excel File Already Exists. Adding New Sheet to File")
            WB = load_workbook(excelFile)
            WB_worksheet = WB.create_sheet(sheetName)
        

        maxAddToExcelSheet = 1048500  # Max Rows in a Worksheet
        # Save Data to Worksheet
        for firstIndexInList in range(0, len(dataMatrix), maxAddToExcelSheet):
            # Label First Row
            WB_worksheet.append(colHeaders)
            
            # Add data to the Worksheet
            for rowInd in range(firstIndexInList, min(firstIndexInList+maxAddToExcelSheet, len(dataMatrix))):
                dataRow = []
                
                if rowInd < len(rowHeaders):
                    rowHeader = rowHeaders[rowInd]
                    dataRow.append(rowHeader)
                
                dataRow.extend(dataMatrix[rowInd])
                dataRow[0] = float(dataRow[0])
                dataRow[1] = float(dataRow[1])
                # dataRow[3] = float(dataRow[3])
                # Write the Data to Excel
                WB_worksheet.append(dataRow)
        
            # Add Excel Aesthetics
            WB_worksheet = self.addExcelAesthetics(WB_worksheet)  
            # Add Sheet
            WB_worksheet = WB.create_sheet(sheetName)
            
            if saveFirstSheet:
                break

        WB.remove(WB_worksheet)
        # Save as New Excel File
        WB.save(excelFile)
        WB.close()

