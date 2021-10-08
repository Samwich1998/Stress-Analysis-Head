
# Virtual Reality Modules
import viz
import vizact
import time

class controlReality:
	def __init__(self, viz, myWorld):
		# Virtual Reality GUI
		self.viz = viz
		self.myWorld = myWorld
		
		# Set Yaw, Pitch, and Roll Parameters
		self.getYawPitchRoll()
	
	def getYawPitchRoll(self):
		# Set Yaw, Pitch, and Roll Parameters
		self.yaw, self.pitch, self.roll = viz.MainView.getEuler()

class gazeControl(controlReality):
	def __init__(self, viz, myWorld): 
		super().__init__(viz, myWorld)
	
	def moveLeft(self):
		self.yaw -= 10
		self.viz.MainView.setEuler([self.yaw, self.pitch, self.roll])
		
		spinLeft = vizact.spin(0,1,0,90,1)
		self.myWorld.addAction(spinLeft)
	
	def moveRight(self):
		self.yaw += 20
		self.viz.MainView.setEuler([self.yaw, self.pitch, self.roll])
	
	def moveUp(self):
		self.pitch += 30
		self.viz.MainView.setEuler([self.yaw, self.pitch, self.roll])
	
	def moveDown(self):
		self.pitch -= 40
		self.viz.MainView.setEuler([self.yaw, self.pitch, self.roll])
		
		
#gazeControl = gazeControl(viz)
#gazeControl.moveLeft()
#gazeControl.moveRight()
#gazeControl.moveUp()
#gazeControl.moveDown()