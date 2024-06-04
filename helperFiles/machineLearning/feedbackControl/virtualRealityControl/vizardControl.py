import time
import viz
import oculus
import vizact
import vizconnect


class virtualWorld:

    def __init__(self, virtualFileInd=0, usingHeadset=False):
        # Parameters of the virtual world
        self.virtualDisplay = None  # The parent node of the scene. This should represent the current virtual environment.
        self.virtualEnvironments = ["piazza.osgb", "gallery.osgb", "pit.osgb", "maze.osgb", "dojo.osgb"]

        # If the subject is wearing a headset.
        if usingHeadset:
            vizconnect.go('vizConnect_oculus.py')
            # Link the headset to vizard
            headMountedDisplay = oculus.Rift(2)  # Create an instance of the Oculus Quest 2 headset.
            self.connectHeadset(headMountedDisplay, controllerType=oculus.Rift)

        else:
            # Create a new VR window.
            viz.go(mode=viz.HMD)  # Launch the VR window. Begin rendering the 3D scene, processing input events, and updating the state of the VR headset and motion controllers.

        # Set the parameters for VR rendering.
        viz.setMultiSample(4)  # This sets the level of multisampling for antialiasing. A higher value results in a smoother image, but also increases processing time and decreases performance.
        viz.MainWindow.fov(60)  # This sets the viewing frustum angle, which determines the portion of the 3D scene that is visible from the perspective of the camera. A larger value makes the screen appear zoomed out.
        viz.MainView.collision(viz.ON)  # This enables collision detection between objects in the scene, which can impact performance.
        viz.window.setFullscreen(False)  # This sets whether the VR window should be fullscreen on the laptop display.
        viz.window.setFullscreenMonitor(0)
        viz.MainView.getHeadLight().disable()  # This disables the headlight, which is a special type of light that is automatically positioned and oriented based on the viewer's perspective.

    def connectHeadset(self, headMountedDisplay, controllerType):
        # Link the headset to Vizard. NOTE: I should link AFTER I call viz.go() or you may link to the wrong window.
        viz.link(headMountedDisplay.getSensor(), viz.MainView)  # Link the headset to Vizard's main view.
        viz.setOption('viz.mode', controllerType)  # Set the default output device.


class controlReality(virtualWorld):

    def __init__(self, virtualFileInd=0, usingHeadset=False):
        # Setup vizard parameters
        super().__init__(virtualFileInd, usingHeadset)

        # Orient the user in the environment.
        self.setPosition(x=0, y=1.7, z=0)  # Set the horizontal, vertical, and depth (x,y,z) coordinates of the user.

        # Display the virtual environment.
        self.displayEnvironment(virtualFileInd=0)

    def displayEnvironment(self, virtualFileInd):
        # Remove the old environment if present.
        if self.virtualDisplay is not None:
            self.virtualDisplay.remove()

        # Display the current environment.
        self.virtualDisplay = viz.add(self.virtualEnvironments[virtualFileInd])
        self.virtualDisplay.visible(viz.ON)

        # Update the scene immediately.
        # viz.update(viz.SCREEN)

    @staticmethod
    def setPosition(x=0, y=1.8, z=0):
        # Position the viewer in the virtual environment.
        viz.MainView.setPosition((x, y, z))


class gazeControl(controlReality):

    def __init__(self, virtualFileInd=0, usingHeadset=False):
        # Setup vizard parameters
        super().__init__(virtualFileInd, usingHeadset)

        self.yaw, self.pitch, self.roll = self.getYawPitchRoll()  # Set yaw, pitch, and roll parameters.

    def setGaze(self, channelAngles=()):
        print(channelAngles)
        viz.MainView.setEuler([channelAngles[0], channelAngles[1], self.roll])

    @staticmethod
    def getYawPitchRoll():
        # Get yaw, pitch, and roll parameters.
        yaw, pitch, roll = viz.MainView.getEuler()
        return yaw, pitch, roll

    def moveLeft(self):
        self.yaw -= 10
        viz.MainView.setEuler([self.yaw, self.pitch, self.roll])
        # spinLeft = vizact.spin(0,1,0,90,1)
        # self.myWorld.addAction(spinLeft)

    def moveRight(self):
        self.yaw += 20
        viz.MainView.setEuler([self.yaw, self.pitch, self.roll])

    def moveUp(self):
        self.pitch += 30
        viz.MainView.setEuler([self.yaw, self.pitch, self.roll])

    def moveDown(self):
        self.pitch -= 40
        viz.MainView.setEuler([self.yaw, self.pitch, self.roll])

    # ---------------------------------------------------------------------- #


if __name__ == "__main__":

    # Specify the VR File and Create the VR World
    controlVR = gazeControl(virtualFileInd=1, usingHeadset=False)

    vizact.onkeydown('0', controlVR.displayEnvironment, 0)
    vizact.onkeydown('1', controlVR.displayEnvironment, 1)
    vizact.onkeydown('2', controlVR.displayEnvironment, 2)
    vizact.onkeydown('3', controlVR.displayEnvironment, 3)
    vizact.onkeydown('4', controlVR.displayEnvironment, 4)

    controlVR.displayEnvironment(0)
    controlVR.displayEnvironment(2)

    for _ in range(100):
        controlVR.moveUp()
        time.sleep(1)
        controlVR.moveDown()
