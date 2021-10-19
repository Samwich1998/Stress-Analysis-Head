
# Virtual Reality Modules
import viz
#import vizact


class virtualWorld:
    
    def __init__(self, virtualFile = 'piazza.osgb'):
        # Enable full screen anti-aliasing (FSAA) to smooth edges
        viz.setMultiSample(4)
        
        # Initialize the Reality
        viz.go() # Initialize Empty Screen
        self.myWorld = viz.addChild(virtualFile) # Add PreDetermined Image to the Screen
        
        # Set VR General Parameters
        viz.MainWindow.fov(60) # Increase the Field of View of the User; Input Into Degrees; 40 Degrees = Default
        viz.MainView.getHeadLight().disable()
        viz.collision(viz.ON)

class controlReality(virtualWorld):
    def __init__(self, virtualFile):
        # Setup Virtual Reality GUI
        super().__init__(virtualFile)
        
        # Set Yaw, Pitch, and Roll Parameters
        self.getYawPitchRoll()
    
    def getYawPitchRoll(self):
        # Set Yaw, Pitch, and Roll Parameters
        self.yaw, self.pitch, self.roll = viz.MainView.getEuler()


class gazeControl(controlReality):
    def __init__(self, virtualFile): 
        super().__init__(virtualFile)
        
    def setGaze(self, channelAngles = []):
        print(channelAngles)
        viz.MainView.setEuler([channelAngles[0], channelAngles[1], self.roll])
    
    def moveLeft(self):
        self.yaw -= 10
        viz.MainView.setEuler([self.yaw, self.pitch, self.roll])
        
        #spinLeft = vizact.spin(0,1,0,90,1)
        #self.myWorld.addAction(spinLeft)
    
    def moveRight(self):
        self.yaw += 20
        viz.MainView.setEuler([self.yaw, self.pitch, self.roll])
    
    def moveUp(self):
        self.pitch += 30
        viz.MainView.setEuler([self.yaw, self.pitch, self.roll])
    
    def moveDown(self):
        self.pitch -= 40
        viz.MainView.setEuler([self.yaw, self.pitch, self.roll])
