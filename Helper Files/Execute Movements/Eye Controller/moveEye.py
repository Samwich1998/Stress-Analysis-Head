# Import Modules
import time
import sys
import linecache
import functools
# Import innfosControlFunctions.py (Must be in Same Folder!)
import pyautogui
# import Read Arduino
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
sys.path.append('../Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import readDataArduino as readArduinoClass   # Functions to Read in Data from Arduino

# --------------------------------------------------------------------------- #
# --------------------------- User Can Edit --------------------------------- #

class moveEye():
    def __init__(self, guiApp = None):        
        self.guiApp = None
        self.restPos = (0,0)
    
    def moveMouseTo(self, xLoc, yLoc):
        pyautogui.moveTo(xLoc, yLoc)
    
    def moveMouseRelativeTo(self, xLoc, yLoc):
        pyautogui.moveRel(xLoc, yLoc)
    
    def dragMouseTo(self, xLoc, yLoc):
        pyautogui.dragTo(xLoc, yLoc)
    
    def dragMouseRelativeTo(self, xLoc, yLoc):
        pyautogui.dragRel(xLoc, yLoc)
        
    def getCurrentMousePos(self):
        return pyautogui.position()
    
    def getWindowLocBounds(self):
        return pyautogui.size()
    
    def moveToRest(self):
        self.moveMouseTo(self.restPos[0], self.restPos[1])
    
    def setRestPos(self, restPos):
        self.restPos = restPos
    
    def mouseDown(self, x = None, y = None):
        if x and y:
            pyautogui.mouseDown(x=x, y=y)
        else:
            pyautogui.mouseDown()
    
    def mouseUp(self):
        pyautogui.mouseUp()
        
        
    def moveRight(self):
        self.mouseDown(x=740, y=348)
        time.sleep(1)
        self.mouseUp()
    
    def moveLeft(self):
        self.mouseDown(x=111, y=402)
        time.sleep(1)
        self.mouseUp()
    
    def moveUp(self):
        self.mouseDown(x=376, y=96)
        time.sleep(1)
        self.mouseUp()
    
    def moveDown(self):
        self.mouseDown(x=423, y=572)
        time.sleep(1)
        self.mouseUp()


# --------------------------------------------------------------------------- #
# ------------------------- Defined Program --------------------------------- #

if __name__ == "__main__":
    # General Data Collection Information (You Will Likely Not Edit These)
    eyeController = moveEye()

    
