
  
# Basic Modules
import numpy as np
# Graphing Modules
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow

  
class fastPlot(QMainWindow):
  
    def __init__(self, numChannels = 2, title = "Insert Title"):
        super().__init__()
        
        
        
        # Information from Analysis
        self.numChannels = numChannels
        
        # Setting Up Gui
        self.App = QtWidgets.QApplication([])
        
        
        # Setting Up the Plot Frame
        self.window = pg.GraphicsLayoutWidget() # Creating Plotting Window
        self.setWindowTitle(title)  # Set Title
        self.setGeometry(100, 100, 600, 500) # Set Geometry
        self.setWindowIcon(QtGui.QIcon("skin.png")) # setting icon to the window
        self.window.setLayout(QtWidgets.QGridLayout()) # setting this layout to the widget
        self.setCentralWidget(self.window)  # setting this widget as central widget of the main widow          
            
        # Create Plots
        self.UiComponents()
  
        # Showing All Widgets
        self.show()
  
    # method for components
    def UiComponents(self):
        
        # Create Plot Holders
        self.trailingAveragePlots = []
        self.bioelectricDataPlots = []
        self.filteredBioelectricDataPlots = []
        # Add Plots to the Holders
        for channelIndex in range(self.numChannels):
            # Generate Subplots
            filteredBioelectricDataSubplots = self.window.addPlot(title = "Filtered Bioelectric Signal in Channel " + str(channelIndex + 1))
            bioelectricDataSubplots = self.window.addPlot(title = "Bioelectric Signal in Channel " + str(channelIndex + 1))
            
            # Add Plots to Graph            
            trailingAverageCurve = filteredBioelectricDataSubplots.plot([], [], pen ='b', symbolBrush = 2, name ='trailingAverageCurve')
            bioelectricDataCurve = bioelectricDataSubplots.plot([], [], pen ='r', symbolBrush = 2, name ='bioelectricDataCurve')
            filteredDataCurve = filteredBioelectricDataSubplots.plot([], [], pen ='r', symbolBrush = 2, name ='filteredDataCurve')
            
            # Set Labels
            filteredBioelectricDataSubplots.setLabels(bottom='Time (Seconds)', left='Filtered Signal (Volts)')   
            bioelectricDataSubplots.setLabels(bottom='Time (Seconds)', left='Filtered Signal (Volts)')
            
            # Keep Track of Plots
            self.trailingAveragePlots.append(trailingAverageCurve)
            self.bioelectricDataPlots.append(bioelectricDataCurve)
            self.filteredBioelectricDataPlots.append(filteredDataCurve)


