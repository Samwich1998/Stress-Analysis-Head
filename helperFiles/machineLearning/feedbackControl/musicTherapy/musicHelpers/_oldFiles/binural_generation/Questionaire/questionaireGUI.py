
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General modules
import sys
import itertools
# Modules for storing questionaire
import json
# GUI Modules
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QSize
import subprocess


# Colors
    # Yellow: #fece53
    # Red: #fc7276
    # Green: #7cdf91
    # Blue: #00B2FF 
# -------------------------------------------------------------------------- #
# --------------------------- Stress Questionaire -------------------------- #

def runBinauralTherapy():
    # Path to your binauralTherapy.py script
    script_path = './../binauralTherapy.py'
    
    # Running the script
    subprocess.run(['python3', script_path])


class stressQuestionaireGUI(QtWidgets.QMainWindow):

    def __init__(self, readData = None, folderPath = "./"):
        # Initialize the GUI application
        self.guiApp = QtWidgets.QApplication(sys.argv)  # Create a GUI, Parent Object
        super().__init__()  # Initialize QMainWindow functions.

        # Initialize survey parameters.
        self.surveyTitles = []
        self.surveyQuestions = []       # A map of the survey type to a list of survey questions (in order) to ask the user.
        self.surveyInstructions = []    # A map of the survey type to a description of the survey rules.
        self.surveyAnswerChoices = []   # A map of the survey type to a list of answer choices (in order) for each question. Each question has the SAME answer choices.
        # Add link to the streaming class
        self.readData = readData        # Instance of the data steaming class
        
        # Add a page/window to the GUI
        self.guiWindow = QtWidgets.QWidget()
        # Add a layout to the window
        self.guiLayout = QtWidgets.QGridLayout() # The layout that contains the text and buttons.
        self.guiLayout.setSpacing(0)
        self.guiWindow.setLayout(self.guiLayout)
        # Add scrollbar to the window
        self.scroll = QtWidgets.QScrollArea()    # Scroll Area which contains the widgets, set as the centralWidget
        self.scroll.setWidget(self.guiWindow)

        # Specify the GUI dimensions.
        upperLeftXPos, upperLeftYPos = 50, 200   # The coordinates of the GUI from the top left hand corner.
        self.width, self.height = 1100, 700      # Specify the length, height of the GUI Window from upperLeftXPos, upperLeftYPos
        # Set window information
        self.setWindowTitle("Stress Questionaire App")
        self.setGeometry(upperLeftXPos, upperLeftYPos, self.width, self.height)
        self.setStyleSheet("background-color : white")

        # Set scrollbar properties.
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        #self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.setCentralWidget(self.scroll)
        self.scroll.verticalScrollBar().setStyleSheet('background: grey; margin-left: 1rem;')

        # Get the questioniare data: question and answers
        self.readQuestionaireData(folderPath + "PANAS_Questions.json", folderPath + "I-STAI-Y1_Questions.json")
        self.surveyDemographicOptions(folderPath + "Subject_Demographics.json")
        
        # Initialize user-interaction parameters
        self.resetSurveyInformation()

        # Start the GUI
        self.show() # Display the GUI on the screen (pop-up)
        self.guiApp.exec() # Run the GUI continuosly
        
    # ---------------------------------------------------------------------- #
    # ------------------ Initialize the Survey Parameters ------------------ #
    
    def resetSurveyInformation(self):
        # Reset the survey parameters
        self.startSurveyTime = None      # The start time of the survey (when the user presses the start button).
        self.currentExperimentName = ""  # The name of the current experiment running
        
        # Display the start screen
        self.displayHomeScreen()

    def readQuestionaireData(self, panasQuestionaireFile, staiQuestionaireFile):
        # Get the data from the json file
        with open(panasQuestionaireFile) as questionaireFile:
            panasInfo = json.load(questionaireFile)
        with open(staiQuestionaireFile) as questionaireFile:
            staiInfo = json.load(questionaireFile)

        self.surveyTitles.append("PANAS")
        # Extract the questions, answerChoices, and surveyInstructions
        self.surveyQuestions.append(panasInfo['questions'])
        self.surveyAnswerChoices.append(panasInfo['answerChoices'])
        self.surveyInstructions.append(panasInfo['surveyInstructions'][0])
        
        self.surveyTitles.append("STAI")
        # Extract the questions, answerChoices, and surveyInstructions
        self.surveyQuestions.append(staiInfo['questions'])
        self.surveyAnswerChoices.append(staiInfo['answerChoices'])
        self.surveyInstructions.append(staiInfo['surveyInstructions'][0])
        
        # Save the questions asked during this survey if streaming in data
        if self.readData != None:
            self.readData.surveyQuestions = list(itertools.chain.from_iterable(self.surveyQuestions))
    
    def surveyDemographicOptions(self, subjectInfoFile):
        # Get the data from the json file
        with open(subjectInfoFile) as questionaireFile:
            data = json.load(questionaireFile)

        # Extract the demographic information.
        self.subjectInformationQuestions = list(data['surveyQuestions'].keys())
        
        self.subjectInformationAnswerFormats = []
        self.subjectInformationAnswerChoices = []
        # Loop
        # Each question has its own answer choices, and can have multiple answers.
        # Ex. question = height, answers = feet, inches
        for question in self.subjectInformationQuestions:
            answerFormats = list(data['surveyQuestions'][question].keys())
            subjectInformationAnswerFormatsQuestion = []
            subjectInformationAnswerChoicesQuestion = []
            
            for answerFormat in answerFormats:
                for answerChoiceIndex in range(len(data['surveyQuestions'][question][answerFormat])):
                    subjectInformationAnswerFormatsQuestion.append(answerFormats)
                    
                    subjectInformationAnswerChoicesQuestion.append(data['surveyQuestions'][question][answerFormat][answerChoiceIndex])
            # append question's answer formatting and choices to survey's arrays,
            # self.subjectInformationAnswerFormats and self.subjectInformationAnswerChoices
            self.subjectInformationAnswerFormats.append(subjectInformationAnswerFormatsQuestion)
            self.subjectInformationAnswerChoices.append(subjectInformationAnswerChoicesQuestion)
            
    # ---------------------------------------------------------------------- #
    # -------------- Backend Intergration with Streaming Data -------------- #
    
    def _recordNewExperiment(self, experimentName = ""):
        # Save the experiment name
        self.currentExperimentName = experimentName
        name = experimentName

        # Record that an experiment has begun.
        if self.readData != None:
            currentTime = self.readData.getCurrentTime()
            # Track the experiment information.
            self.readData.experimentTimes.append([currentTime, None])
            self.readData.experimentNames.append(self.currentExperimentName)
            # self._resumeDataStreaming()
        else:
            print("\nNew Experiment Recorded:", self.currentExperimentName)
        
        # Display the experiment screen.
        self.displayExperimentScreen()
    
    def _pauseDataStreaming(self):
        if self.readData != None:
            self.readData.storeIncomingData = False

    def _resumeDataStreaming(self):
        if self.readData != None:
            self.readData.storeIncomingData = True
        
    def _changeExperimentName(self, newExperimentalName = ""):
        # Change the saved experimental name
        self.currentExperimentName = newExperimentalName
        if self.readData != None:
            self.readData.experimentNames[-1] = newExperimentalName

        # Display the experiment screen.
        self.displayExperimentScreen()
    
    def _recordEndExperiment(self):
        # Collect data to be saved
        data_to_save = {
            'experimentName': self.currentExperimentName,
            'surveyResults': self.readData.surveyAnswersList if self.readData else [],
            'subjectInformation': self.readData.subjectInformationAnswers if self.readData else []
        }
    
        # Specify the filename, you can include a timestamp or experiment name to make it unique
        filename = f'{self.currentExperimentName}_results.json'
    
        # Save data to JSON file
        with open(filename, 'w') as file:
            json.dump(data_to_save, file, indent=4)
    
        # Original end experiment actions here (if any)
        print("Experiment Finished")
    
        # Go back to the start screen or close the application as needed
        self.resetSurveyInformation()

        ## Record that the experiment has ended.
        #if self.readData != None:
        #    currentTime = self.readData.getCurrentTime()
        #    # Track the experiment information
        #    self.readData.experimentTimes[-1][1] = currentTime
        #    # self._pauseDataStreaming()
        #else:
        #    print("Experiment Finised")
        #    
        ## Go back to the start screen
        #self.resetSurveyInformation()
            
    def _recordStartSurvey(self):
        # If you are streaming in data.
        if self.readData != None:
            # Retrieve the current time.
            self.startSurveyTime = self.readData.getCurrentTime()
        else:
            print("The Survey has Begun!")
        
        # Begin the survey.
        self.displaySurvey(surveyInd = 0)
    
    def _recordSurveyResults(self, surveyAnswers, surveyInd):
        #Results = []
        # Assert validity of the incoming data
        assert len(self.surveyQuestions[surveyInd]) == len(surveyAnswers)

        # If you are streaming in data, send the data to the streaming class.
        if self.readData != None:
            if surveyInd == 0:
                self.readData.surveyAnswerTimes.append(self.startSurveyTime)
                self.readData.surveyAnswersList.append(surveyAnswers)
            else:
                self.readData.surveyAnswersList[-1].extend(surveyAnswers)
        # Print the answers if testing.
        else:
            print("\nAnswers selected:")
            for questionsInd in range(len(self.surveyQuestions[surveyInd])):
               responce = self.surveyQuestions[surveyInd][questionsInd] + ":" + self.surveyAnswerChoices[surveyInd][surveyAnswers[questionsInd] - 1] 
               print("\t", responce)
               Results.append(responce)
        
        # Reset the data
        surveyInd += 1
        self.startSurveyTime = None
        
        # Move onto the next page
        if surveyInd < len(self.surveyQuestions):
            # Move through the surveys
            self.displaySurvey(surveyInd)
        else:
            # Finished surveys, move to experiment screen
            self.displayExperimentScreen()

        return Results

    
    def informBadInfoSurvey(self):
        # informs that survey information is incomplete
        print("Bad Subject Information - Resetting Subject Info Survey")
        
    def _recordSubjectInfo(self, answerBoxes, metricChecks = {}):
        subjectInformationAnswers = []
        heightInfo = []
        
        tempQuestionIndex = 0
        tempAnswerIndex = 0
        # formats answers to question based on answerbox type
        # loop through each answerbox
        for answerBoxIndex in range(len(answerBoxes)):
            if (len(self.subjectInformationAnswerFormats[tempQuestionIndex]) == tempAnswerIndex):
                tempAnswerIndex = 0
                tempQuestionIndex += 1
            tempAnswerIndex += 1
            if answerBoxes[answerBoxIndex][1] == "Dropdown":
                subjectInformationAnswers.append(answerBoxes[answerBoxIndex][0].currentText())
            elif answerBoxes[answerBoxIndex][1] == "Textbox":
                if answerBoxes[answerBoxIndex][0].text():
                    # checks for height information and stores as separate answer
                    if "Height" in self.subjectInformationQuestions[tempQuestionIndex]:
                        heightInfo.append(int(answerBoxes[answerBoxIndex][0].text()))
                    elif "Weight" in self.subjectInformationQuestions[tempQuestionIndex] and metricChecks["Weight"].isChecked():
                            convertedWeight = float(answerBoxes[answerBoxIndex][0].text()) * 2.205
                            subjectInformationAnswers.append(convertedWeight)
                    else:
                        subjectInformationAnswers.append(answerBoxes[answerBoxIndex][0].text())
                
                else:
                    # checks that there is an inputted answer to the answerbox
                    if "Height" in self.subjectInformationQuestions[tempQuestionIndex]:
                        heightInfo.append(0)
                    else:
                        self.informBadInfoSurvey()
                        return
            else:
                # answerbox is not of Dropdown or textbox format, which is invalid
                self.informBadInfoSurvey()
                return
        
        # reformats height information to fit the subject answer format.
        # height has two separate answerboxes and needs to be a singular integer and in meters
        if len(heightInfo) == 2:
            if metricChecks["Height"].isChecked():
                print("here")
                convertedHeight = heightInfo[0] / 100
                subjectInformationAnswers.append(convertedHeight)
            else:
                subjectInformationAnswers.append(0.0254 * ((heightInfo[0] * 12) + heightInfo[1]))
                
                
        # if foundGoodData:
            # If you are streaming in data, send the data to the streaming class.
        if self.readData != None:
            self.readData.subjectInformationAnswers = subjectInformationAnswers
            self.readData.subjectInformationQuestions = self.subjectInformationQuestions
        # Print the answers if testing.
        else:
            print("Info Recorded:")
            if (len(self.subjectInformationQuestions) != len(subjectInformationAnswers)):
                self.informBadInfoSurvey()
                return
            else:
                for questionsInd in range(len(self.subjectInformationQuestions)):
                    print("\t", self.subjectInformationQuestions[questionsInd] + ":", subjectInformationAnswers[questionsInd])
                    Info.append(self.subjectInformationQuestions[questionsInd] + ":" + str(subjectInformationAnswers[questionsInd]))           
        self.displayHomeScreen()
    
    def _finishedRun(self):
        ## Prepare data to save, assuming data is in a format that can be directly written to a JSON file
        #data_to_save = {
        #    'experimentName': self.currentExperimentName,
        #    'surveyResults': self.readData.surveyAnswersList if self.readData else [],
        #    'subjectInformation': self.readData.subjectInformationAnswers if self.readData else []
        #}

        ## Specify the filename
        #filename = 'experiment_results.json'
    
        ## Write the data to a JSON file
        #with open(filename, 'w') as f:
        #    json.dump(data_to_save, f, indent=4)
    
        ## Close the GUI App
        #self.close()
    
        ## Additional cleanup if necessary
        #if self.readData != None:
        #    self.readData.stopTimeStreaming = 0
        #    self._pauseDataStreaming()

        # Close the GUI App
        self.close()
        # Close the arduino streaming
        if self.readData != None:
            self.readData.stopTimeStreaming = 0
            self._pauseDataStreaming()
    
    # ---------------------------------------------------------------------- #
    # ------------------------ Creating GUI Buttons ------------------------ #
    
    def setButtonAesthetics(self, widget, backgroundColor = "black", textColor = "white", fontFamily = "Arial", fontSize = 15):
        """
        This method sets the style sheet of a given widget, using either default
        stylistic choices, or user inputted choices.
        """
        widget.setAutoFillBackground(True)
        widget.setStyleSheet(
            "background-color: " + backgroundColor + "; " + \
            "color: " + textColor + "; " + \
            "border: 0.5px solid black; font-weight: 5rem; padding: 3rem 5rem; text-align: center; width: 100%; margin: 1rem 2rem;" \
        )

        # Add font information
        widget.setFont(QFont(fontFamily, fontSize))
    
    def setDimensions(self, element, minWidth, maxWidth, minHeight, maxHeight):
        """
        This method sets the minimum and maximum dimensions of a given widget.
        """
        # Add dynamic sizing
        if minHeight != None and minWidth != None:
            element.setMinimumSize(QSize(minWidth, minHeight));
        if maxWidth != None and maxHeight != None:
            element.setMaximumSize(QSize(maxWidth, maxHeight));
    
    def addTitleToGui(self, titleText, fontSize = 50, backgroundColor = "white"):
        """
        This method creates a title for the window with default
        styling given the title text, and adds it to the window GUI layout.
        """
        # Create the title text
        titleLabel = QtWidgets.QLabel(titleText)
        
        # Add styling to the title
        titleLabel.setAutoFillBackground(True)
        titleLabel.setFont(QFont("Arial", fontSize)) # Specify the font
        titleLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter) # Specify the alignment
        titleLabel.setStyleSheet("background-color: " + backgroundColor + "; " + \
                                 "color: black; text-align: center; margin: 1rem auto;")
        
        # Add the title to the GUI
        self.guiLayout.addWidget(titleLabel, 0, 0, 1, 6)
    
    def subjectInfoDropdown(self, answerChoices, startRow, startColumn):
        """
        This method creates and returns a dropdown menu Widget with default
        styling given position and answer choices.
        
        Adds dropdown to the window GUI layout and returns a QComboBox Widget
        """
        # Create the dropdown menu
        dropdownMenu = QtWidgets.QComboBox()
        dropdownMenu.addItems(answerChoices)
        # Style the dropdown menu
        dropdownMenu.setStyleSheet("border: 1px solid black; selection-background-color: lightGray; selection-color: black; color: black; font-weight: 3rem;")
        self.setDimensions(dropdownMenu, minWidth=300, maxWidth=300, minHeight=50, maxHeight=50)
        # idk what this does??
        listView = QtWidgets.QListView()
        listView.setWordWrap(True)
        dropdownMenu.setView(listView)
        # Add the dropdown menu to the GUI
        self.guiLayout.addWidget(dropdownMenu, startRow, startColumn, 1, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        
        return dropdownMenu
    
    def inputText(self, startRow, startColumn, rows, columns, placeholderText = "Enter Text Here", onlyInt = False):
        """
        This method creates and returns a input textbox Widget with default
        styling given position and dimensions.
        
        If onlyInt is true, the widget will only accept integer values.
        
        Adds textbox to the window GUI layout and returns a QLineEdit Widget
        """
        currInput = QtWidgets.QLineEdit()
        currInput.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDimensions(currInput, minWidth=300, maxWidth=400, minHeight=50, maxHeight=50)
        
        if onlyInt:
            # sets textbox to only accept integers between 0 and 1000.
            validator = QtGui.QIntValidator(0, 1000)
            currInput.setValidator(validator)
        
        # sets placeholder (default) text and adds stylistic choice
        currInput.setPlaceholderText(placeholderText)
        currInput.setStyleSheet("color: black; margin: 1rem 2rem; border: 0.5px solid black; padding: 1rem 3rem;")
        # Add input textbox to the GUI
        self.guiLayout.addWidget(currInput, startRow, startColumn, rows, columns, alignment=Qt.AlignmentFlag.AlignCenter)

        return currInput
        
    
    # ---------------------------------------------------------------------- #
    # ------------------------ Creating GUI Windows ------------------------ #
    
    def displayHomeScreen(self):
        """
        This method calls sets up and adjusts the layout to set up the home
        screen page, which consists of a textbox to input experiment name and
        buttons to direct to experiment and subject info pages.
        """
        # Start off fresh by clearning the GUI elements.
        self.clearScreen()
        self.guiLayout.setSpacing(5)
        self.guiWindow.setLayout(self.guiLayout)
        
        # Add the title to the GUI
        self.addTitleToGui("Waiting for Experiment to Start", fontSize = 50)
        
        # Creates box for experiment buttons
        itemBox = QtWidgets.QLabel()
        itemBox.setStyleSheet("padding: 10px 5px; background-color: #dbdbdb; width: 100%; margin: 10px 5px; border: 2px solid black;")
        self.guiLayout.addWidget(itemBox, 2, 0, 6, 6)
        
        # Create textbox for user to label experiment
        experimentInput_Object = self.inputText(3, 3, 2, 1, "Enter Experiment Name:")
        experimentInput_Object.setStyleSheet("color: black;")
        if self.currentExperimentName != "":
            experimentInput_Object.setText(self.currentExperimentName)
            
        """
        displays the three buttons on the screen. The first one is
        the start experiment button which records the start time and moves to
        the experiment screen. The second oneis the add subject information 
        button which moves to the subject demographics survey screen. The third
        is the quit button, which exits the GUI program.
        """
        
        
        # Create the start button to begin an experiment.
        startExperimentButton = QtWidgets.QPushButton("Start Experiment")  
        startExperimentButton.clicked.connect(runBinauralTherapy)
        startExperimentButton.clicked.connect(lambda: self._recordNewExperiment(experimentInput_Object.text()))
        # Add button aesthetics: colors and text styles.
        self.setDimensions(startExperimentButton, minWidth=150, maxWidth=200, minHeight=50, maxHeight=50)
        self.setButtonAesthetics(startExperimentButton,backgroundColor = "#7cdf91", textColor = "#000000", fontFamily = "Arial", fontSize = 15)
        # Add the quit button to the layout.
        self.guiLayout.addWidget(startExperimentButton, 3, 2, 2, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        # Create the start button to begin an experiment.
        addSubjectInfoButton = QtWidgets.QPushButton("Add Subject Information")
        addSubjectInfoButton.clicked.connect(lambda: self.displaySubjectInfoScreen(experimentInput_Object.text()))
        # Add button aesthetics: colors and text styles.
        self.setDimensions(addSubjectInfoButton, minWidth=200, maxWidth=200, minHeight=50, maxHeight=50)
        self.setButtonAesthetics(addSubjectInfoButton, backgroundColor = "#00B2FF", textColor = "#000000", fontFamily = "Arial", fontSize = 15)
        # Add the quit button to the layout.
        self.guiLayout.addWidget(addSubjectInfoButton, 5, 0, 1, 6, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        # Create a quit button to exit the GUI.
        quitSurveyButton = QtWidgets.QPushButton("Quit")    
        quitSurveyButton.clicked.connect(self._finishedRun)
        # Add button aesthetics: colors and text styles.
        self.setDimensions(quitSurveyButton, minWidth=200, maxWidth=200, minHeight=50, maxHeight=50)
        self.setButtonAesthetics(quitSurveyButton, backgroundColor = "#fc7276", textColor = "#000000", fontFamily = "Arial", fontSize = 15)
        # Add the quit button
        self.guiLayout.addWidget(quitSurveyButton, 9, 0, 1, 6, alignment=Qt.AlignmentFlag.AlignHCenter) # row, column, rowspan, columnspan
    
    def metricCheckbox_Height(self, checkbox, box1, box2, placeholder):
        if checkbox.isChecked():
            box1.setPlaceholderText("Height (cm)")
            box2.setText("0")
        else:
            box1.setPlaceholderText(placeholder)
            box2.setText("")
    def metricCheckbox_Weight(self, checkbox, box, placeholder):
        if checkbox.isChecked():
            box.setPlaceholderText("Weight (kg)")
        else:
            box.setPlaceholderText(placeholder)
        
    
    def displaySubjectInfoScreen(self, experimentName):
        """
        This method calls sets up and adjusts the layout to set up the subject
        information page, which consists of dropdown and textbox questions. Also
        sets up logic for helper methods to read survey data.
        """
        self.currentExperimentName = experimentName
        # Start off fresh by clearning the GUI elements.
        self.clearScreen()
        self.guiLayout.setSpacing(3)
        self.guiWindow.setLayout(self.guiLayout)
        
        # Add the title to the GUI
        self.addTitleToGui("Subject Information", fontSize=50)
        rowIndex = 0
        
        answerBoxes = []
        metricChecks = {}
        # For each background question
        for questionInd in range(len(self.subjectInformationQuestions)):
            # Extract the question information
            question = self.subjectInformationQuestions[questionInd]
            answerTypes = self.subjectInformationAnswerFormats[questionInd]
            answerChoices = self.subjectInformationAnswerChoices[questionInd]
            rowIndex += 1
            
            # Add the question to the GUI
            titleLabel = QtWidgets.QLabel(question)
            titleLabel.setFont(QFont("Arial", 20)) # Specify the font
            titleLabel.setStyleSheet("color: black; margin: 1rem auto;")
            self.guiLayout.addWidget(titleLabel, rowIndex, 1, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter) # row, column, rowspan, columnspan
            
            for answerTypeInd in range(len(answerTypes)):
                if "Dropdown" in answerTypes[answerTypeInd]:
                    # Add the answer choices
                    answerDropdown = self.subjectInfoDropdown(answerChoices[answerTypeInd], startRow = rowIndex, startColumn = 2)
                    answerBoxes.append([answerDropdown, "Dropdown"])
                    rowIndex += 1
                elif "Textbox" in answerTypes[answerTypeInd]:
                    answerText = self.inputText(rowIndex, 2, 1, 3, answerChoices[answerTypeInd][0])
                    # self.guiLayout.addWidget(answerText, 2 + (questionInd * 2), int(6/len(answerTypes)) * answerTypeInd, 1, int(6/len(answerTypes)), alignment=Qt.AlignmentFlag.AlignHCenter)
                    answerBoxes.append([answerText, "Textbox"])
                    rowIndex += 1
                elif "Textbox_Int" in answerTypes[answerTypeInd]:
                    answerText = self.inputText(rowIndex, 2, 1, 3, answerChoices[answerTypeInd][0], onlyInt = True)
                    # self.guiLayout.addWidget(answerText, 2 + (questionInd * 2), int(6/len(answerTypes)) * answerTypeInd, 1, int(6/len(answerTypes)), alignment=Qt.AlignmentFlag.AlignHCenter)
                    answerBoxes.append([answerText, "Textbox"])
                    rowIndex += 1
                    
                if "Weight (lbs)" in answerChoices[answerTypeInd] or "Height (ft)" in answerChoices[answerTypeInd]:
                    dict_key = (answerChoices[answerTypeInd][0]).split(" ")[0]
                    if dict_key == "Weight":
                        metricButton = QtWidgets.QCheckBox("Metric (kg)")
                        WeightInd = questionInd
                        metricChecks[dict_key] = metricButton
                        metricButton.stateChanged.connect(lambda: self.metricCheckbox_Weight(metricChecks["Weight"], answerBoxes[WeightInd][0], self.subjectInformationAnswerChoices[WeightInd][0][0]))
                    elif dict_key == "Height":
                        metricButton = QtWidgets.QCheckBox("Metric (cm)")
                        HeightInd = questionInd
                        metricChecks[dict_key] = metricButton
                        metricButton.stateChanged.connect(lambda: self.metricCheckbox_Height(metricChecks["Height"], answerBoxes[HeightInd][0], answerBoxes[HeightInd + 1][0], self.subjectInformationAnswerChoices[HeightInd][0][0]))

                    else:
                        metricButton = QtWidgets.QCheckBox("Metric")
                        metricChecks[dict_key] = metricButton
                    metricButton.setStyleSheet("color: black; margin: 1rem auto;")
                    self.guiLayout.addWidget(metricButton, rowIndex - 1, 4, 1, 1)
                    
            blankSpace = QtWidgets.QLabel("")
            self.guiLayout.addWidget(blankSpace, rowIndex, 0) # row, column, rowspan, columnspan
            rowIndex += 1
            
        """
        displays the two buttons on the screen. The first one is
        the back button which moves back to the experiment screen without
        saving the data. The second one is the save button which moves back to
        the experiment screen after saving the survey data.
        """
        
        # Create a quit button to exit the GUI.
        backInfoButton = QtWidgets.QPushButton("Back")    
        backInfoButton.clicked.connect(self.displayHomeScreen)
        # Add button aesthetics: colors and text styles.
        self.setDimensions(backInfoButton, minWidth=200, maxWidth=200, minHeight=50, maxHeight=50)
        self.setButtonAesthetics(backInfoButton, backgroundColor = "#fc7276", textColor = "#000000", fontFamily = "Arial", fontSize = 15)
        # Add the quit button
        self.guiLayout.addWidget(backInfoButton, rowIndex, 2, 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter) # row 0, column 1
        
        # Create a quit button to exit the GUI.
        saveInfoButton = QtWidgets.QPushButton("Save")    
        
        saveInfoButton.clicked.connect(lambda: self._recordSubjectInfo(answerBoxes, metricChecks))
        # Add button aesthetics: colors and text styles.
        self.setDimensions(saveInfoButton, minWidth=200, maxWidth=200, minHeight=50, maxHeight=50)
        self.setButtonAesthetics(saveInfoButton, backgroundColor = "#7cdf91", textColor = "#000000", fontFamily = "Arial", fontSize = 15)
        # Add the quit button
        self.guiLayout.addWidget(saveInfoButton, rowIndex, 3, 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter) # row 0, column 1
        
    
    def displayExperimentScreen(self):
        """
        This method calls helper methods and adjusts the layout to set up
        the experiment page.
        """
        # Start off fresh by clearning the GUI elements.
        self.clearScreen()
        self.guiLayout.setSpacing(5)
        self.guiWindow.setLayout(self.guiLayout)
        
        # Add the title to the GUI
        self.addTitleToGui("Experiment Started", fontSize=50)
        
        # Creates box for experiment buttons
        itemBox = QtWidgets.QLabel()
        itemBox.setStyleSheet("padding: 10px 5px; background-color: #dbdbdb; width: 100%; margin: 10px 5px; border: 2px solid black;")
        self.guiLayout.addWidget(itemBox, 2, 0, 6, 6)
        
        # Add experiment name to the GUI
        if self.currentExperimentName != "":
            nameLabel = QtWidgets.QLabel("Experiment Name: " + self.currentExperimentName)
            nameLabel.setAutoFillBackground(True)
            nameLabel.setFont(QFont("Arial", 20)) # Specify the font
            self.setDimensions(nameLabel, minWidth=None, maxWidth=None, minHeight=50, maxHeight=50)
            nameLabel.setAlignment(Qt.AlignmentFlag.AlignCenter) # Specify the alignment
            nameLabel.setStyleSheet("color: black; text-align: center; margin: 1rem 0rem 1rem auto; padding: 5px 10px;")
            # Add the experiment name to the GUI
            self.guiLayout.addWidget(nameLabel, 4, 1, 1, 4)
            
            changeExperimentNameButton = QtWidgets.QPushButton(" X ")
            changeExperimentNameButton.clicked.connect(lambda: self._changeExperimentName(""))
            # Add button aesthetics: colors and text styles.
            changeExperimentNameButton.setStyleSheet("margin: 1rem auto 1rem 0rem; padding: 5px 10px;")
            self.setDimensions(changeExperimentNameButton, minWidth=20, maxWidth=20, minHeight=15, maxHeight=15)
            self.setButtonAesthetics(changeExperimentNameButton, backgroundColor = "#fc7276", textColor = "black", fontFamily = "Arial", fontSize = 5)
            # Add the start survey button button to the layout.
            self.guiLayout.addWidget(changeExperimentNameButton, 5, 3, 1, 1) # row, column, rowspan, columnspan
        else:
            changeExperimentNameButton = QtWidgets.QPushButton("Save Name")
            changeExperimentNameButton.clicked.connect(lambda: self._changeExperimentName(experimentInput_Object.text()))
            # Add button aesthetics: colors and text styles.
            changeExperimentNameButton.setStyleSheet("margin: 1rem auto 1rem 0rem; float: right;")
            self.setDimensions(changeExperimentNameButton, minWidth=100, maxWidth=120, minHeight=50, maxHeight=50)
            self.setButtonAesthetics(changeExperimentNameButton, backgroundColor = "#00B2FF", textColor = "black", fontFamily = "Arial", fontSize = 15)
            # Add the start survey button button to the layout.
            self.guiLayout.addWidget(changeExperimentNameButton, 4, 2, 1, 1) # row, column, rowspan, columnspan
            
            # Create textbox for user to label experiment
            experimentInput_Object = self.inputText(4, 2, 1, 3, "Enter Experiment Name")
            experimentInput_Object.setStyleSheet("color: black; margin: 1rem 0rem 1rem auto; border: 1px solid black; float: left;")
            if self.currentExperimentName != "":
                experimentInput_Object.setText(self.currentExperimentName)
                
        """
        displays the two buttons on the screen. The first one is
        the end experiment button which moves back to the home screen after
        recording the time. The second one is the survey button begins the survey
        and displays the survey info screen.
        """
        
        # Create the end survey button to display the survey.
        endExperimentButton = QtWidgets.QPushButton("End Experiment")
        endExperimentButton.clicked.connect(self._recordEndExperiment)
        # Add button aesthetics: colors and text styles.
        self.setDimensions(endExperimentButton, minWidth=200, maxWidth=200, minHeight=50, maxHeight=50)
        self.setButtonAesthetics(endExperimentButton, backgroundColor = "#fc7276", textColor = "black", fontFamily = "Arial", fontSize = 15)
        # Add the start survey button button to the layout.
        self.guiLayout.addWidget(endExperimentButton, 9, 2, 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        # Create the start survey button to display the survey.
        startSurveyButton = QtWidgets.QPushButton("Survey")
        startSurveyButton.clicked.connect(self._recordStartSurvey)
        # Add button aesthetics: colors and text styles.
        self.setDimensions(startSurveyButton, minWidth=200, maxWidth=200, minHeight=50, maxHeight=50)
        self.setButtonAesthetics(startSurveyButton,backgroundColor = "#7cdf91", textColor = "black", fontFamily = "Arial", fontSize = 15)
        # Add the start survey button button to the layout.
        self.guiLayout.addWidget(startSurveyButton, 9, 3, 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
    
        
    def displaySurvey(self, surveyInd = 0):
        """
        This method calls helper methods and adjusts the layout to set up
        the survey page.
        """
        # Start off fresh by clearning the GUI elements.
        self.clearScreen()
        self.guiLayout.setSpacing(0)
        self.guiWindow.setLayout(self.guiLayout)
        
        # Create and add GUI elements to the window.
        self.addSurveyTitle(surveyInd = surveyInd)              # Add the title to the GUI.
        startRow, questionObjects = self.displaySurveyQuestions(7, surveyInd)   # Add the question and answers to the GUI.
        
        blankSpace = QtWidgets.QLabel(" ")
        self.guiLayout.addWidget(blankSpace, startRow, 0, 2, 6)
        startRow += 2
        
        """
        displays the two buttons on the screen. The first one is
        the back button which moves back to the experiment screen without
        saving the data. The second one is the save button which moves back to
        the experiment screen after saving the survey data.
        """
        
        # Create back button
        backSurveyButton = QtWidgets.QPushButton("Back")  
        self.setButtonAesthetics(backSurveyButton, backgroundColor = "#fc7276", textColor = "black", fontFamily = "Arial", fontSize = 15)
        self.setDimensions(backSurveyButton, minWidth=75, maxWidth=75, minHeight=30, maxHeight=30)
        self.guiLayout.addWidget(backSurveyButton, startRow, 2, alignment = Qt.AlignmentFlag.AlignHCenter)
        backSurveyButton.clicked.connect(self.displayExperimentScreen) # Connects back button to experiment screen
        # Create save button
        saveSurveyButton = QtWidgets.QPushButton("Next")
        self.setButtonAesthetics(saveSurveyButton, backgroundColor = "#7cdf91", textColor = "black", fontFamily = "Arial", fontSize = 15)
        self.setDimensions(saveSurveyButton, minWidth=75, maxWidth=75, minHeight=30, maxHeight=30)
        self.guiLayout.addWidget(saveSurveyButton, startRow, 3, alignment = Qt.AlignmentFlag.AlignHCenter)
        saveSurveyButton.clicked.connect(lambda: self.saveSurveyResults(questionObjects, surveyInd)) # Connects save button to saving the survey data

        
    def clearScreen(self):
        """
        This method removes all widgets from the screen.
        """
        for i in reversed(range(self.guiLayout.count())): 
            self.guiLayout.itemAt(i).widget().setParent(None)

    # -------------------------- Survey Helpers  ---------------------------- #
    def addSurveyTitle(self, surveyInd):
        """
        This method adds the title and instructions for the survey page.
        """
        
        self.addTitleToGui(self.surveyTitles[surveyInd], backgroundColor = "lightgray")
        
        # Create the survey instructions
        surveyInstructionsLabel = QtWidgets.QLabel(self.surveyInstructions[surveyInd])
        # Style the survey instructions
        surveyInstructionsLabel.setWordWrap(True)
        surveyInstructionsLabel.setAlignment(Qt.AlignmentFlag.AlignCenter) # Specify the alignment
        self.setButtonAesthetics(surveyInstructionsLabel, backgroundColor = "white", textColor = "black", fontFamily = "Arial", fontSize = 20)
        # Add the survey instructions
        self.guiLayout.addWidget(surveyInstructionsLabel, 1, 0, 4, 6)
    
    def displaySurveyQuestions(self, startRow, surveyInd):
        """
        This method sets up a question row for every question specified in the
        survey Instructions. A question row consists of a label and a row of
        5 buttons in a radio button group.
        """
        blankSpace = QtWidgets.QLabel(" ")
        self.guiLayout.addWidget(blankSpace, startRow, 0, 1, 6)
        startRow += 1
        
        questionObjects = []
        # For each answer choice option
        for answerChoiceInd in range(len(self.surveyAnswerChoices[surveyInd])):
            # Add the answer choice text.
             answerChoiceLabel = QtWidgets.QLabel(self.surveyAnswerChoices[surveyInd][answerChoiceInd])
             answerChoiceLabel.setStyleSheet("color: black; text-align: center; width: 100%; margin: 1rem auto; font-size: 14px; font-weight: 500")
             self.guiLayout.addWidget(answerChoiceLabel, startRow, answerChoiceInd + 1, alignment = Qt.AlignmentFlag.AlignHCenter)
        startRow += 1
        
        # Add the survey questions to the page
        for currentQuestionInd in range(len(self.surveyQuestions[surveyInd])):
            # Highlight every other row
            highlightRow = currentQuestionInd % 2 == 0
            currentRowBar = QtWidgets.QLabel()
            if highlightRow: currentRowBar.setStyleSheet("padding: 5px; background-color: lightGray; width: 100%")
            self.guiLayout.addWidget(currentRowBar, startRow, 0, 1, 6)
            
            # Add the survey question
            surveyQuestionLabel = QtWidgets.QLabel(self.surveyQuestions[surveyInd][currentQuestionInd])
            styleSheet = "color: black; width: 100%; padding: 10px 5px; font-size: 14px; font-weight: 650;"
            if highlightRow: styleSheet += " background-color: lightGray;"
            surveyQuestionLabel.setStyleSheet(styleSheet)
            self.guiLayout.addWidget(surveyQuestionLabel, startRow, 0, 1, 1)
            
            # Add the answer choices
            questionObjects.append(self.display1RowSurveyButtons(startRow, highlightRow, surveyInd))
            # Spacing between rows
            startRow += 3
        
        return startRow, questionObjects

    def display1RowSurveyButtons(self, startRow, highlightRow, surveyInd):
        """
        This method displays all buttons in a row on the screen for a specific question.
        """
        self.radioButtons = []
        # create new button group row, which sets up constraint that only one
        # button can be selected per group
        buttonGroupRow = QtWidgets.QButtonGroup(self.guiWindow)
        curr_answerChoices = None
        curr_answerChoices = self.surveyAnswerChoices[surveyInd]

        # loop through answer choices, creating a new button
        for answerChoiceInd in range(len(curr_answerChoices)):
            newButton = QtWidgets.QRadioButton()
            # Add styling to the button 
            buttonStyleSheet = ""
            # every other row is light grey
            if highlightRow: 
                buttonStyleSheet = "background-color: lightGray; " + buttonStyleSheet
            newButton.setStyleSheet(buttonStyleSheet)
            # Add the button to the page
            self.guiLayout.addWidget(newButton, startRow, answerChoiceInd + 1, alignment = Qt.AlignmentFlag.AlignHCenter)
            
            # Keep track of the buttons.
            self.radioButtons.append(newButton)
            buttonGroupRow.addButton(newButton) # Specify that this button is connect to the other ones on the row.
            
        return self.radioButtons

    def saveSurveyResults(self, questionObjects, surveyInd):
        """
        This method reads the answer to each survey question as an array.
        """
        # Survey parameters
        surveyAnswers = [] # Hold survey answers as integers in question order
        
        # For each PANAS survey question
        for question in questionObjects:
            questionAnswered = False
            
            # For each answer choice in the question
            for answerInd in range(len(question)):
                # If an answer was given
                if question[answerInd].isChecked():
                    # Record the answer for the question
                    surveyAnswers.append(answerInd + 1)
                    questionAnswered = True
                    break
            
            # If the subject did not answer the question
            if not questionAnswered:
                # Alert the subject that they MUST fully complete the survey
                alertSurveyIncomplete = QtWidgets.QLabel("Please answer all statements")
                # Add styling to the alert
                alertSurveyIncomplete.setAlignment(Qt.AlignmentFlag.AlignHCenter) # Specify the alignment
                self.setButtonAesthetics(alertSurveyIncomplete, backgroundColor = "#b51b43", textColor = "white", fontFamily = "Arial", fontSize = 30)
                # Add the alert to the GUI
                self.guiLayout.addWidget(alertSurveyIncomplete, 9, 0, 1, 6)
                return
        
        # When done, save the completed survey and return to the experiment screen.
        Results = self._recordSurveyResults(surveyAnswers, surveyInd)


# -------------------------------------------------------------------------- #
# ------------------------- Run GUI From this File ------------------------- #

if __name__ == "__main__":
    Results = []
    Info = []
    name = ''

    # Input parameters
    readData = None
    # Open the questionaire GUI.
    guiObject = stressQuestionaireGUI(readData, folderPath = "./")

    print(name)
    print(Info)
    print(Results)    


    # BONUS POINTS: Have the 'THANK YOU :)' print out while the GUI is still running.
    # import threading
    # threading.Thread(target = stressQuestionaireGUI, args = (readData, stressQuestionaireFile), daemon=False).start()
    # print("THANK YOU :)")
    # while True:
    #     infiniteLoop = "Yes"
    
    
    
