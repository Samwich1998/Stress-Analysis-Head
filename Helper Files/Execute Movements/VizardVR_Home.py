

# Virtual Reality Modules
import viz
import vizact
# Import Files
import virtualRealityControl as virtualRealityControl  # File to Control Virtual Reality

# Enable full screen anti-aliasing (FSAA) to smooth edges
viz.setMultiSample(4)

# Initialize the Reality
viz.go() # Initialize Empty Screen
myWorld = viz.addChild('piazza.osgb') # Add PreDetermined Image to the Screen

#Increase the Field of View of the User
viz.MainWindow.fov(60) # Input Into Degrees; 40 Degrees = Default

# Test Gaze Control
gazeControl = virtualRealityControl.gazeControl(viz, myWorld)

viz.MainView.getHeadLight().disable()

vizact.spin(0,1,0,90,1)