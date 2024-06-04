import viz
import vizconnect

#################################
# Parent configuration, if any
#################################

def getParentConfiguration():
	#VC: set the parent configuration
	_parent = ''
	
	#VC: return the parent configuration
	return _parent


#################################
# Pre viz.go() Code
#################################

def preVizGo():
	return True


#################################
# Pre-initialization Code
#################################

def preInit():
	"""Add any code here which should be called after viz.go but before any initializations happen.
	Returned values can be obtained by calling getPreInitResult for this file's vizconnect.Configuration instance."""
	return None


#################################
# Group Code
#################################

def initGroups(initFlag=vizconnect.INIT_INDEPENDENT, initList=None):
	#VC: place any general initialization code here
	rawGroup = vizconnect.getRawGroupDict()

	#VC: initialize a new group
	_name = 'group'
	if vizconnect.isPendingInit('group', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: create the raw object
			rawGroup[_name] = viz.addGroup()
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addGroup(rawGroup[_name], _name, make='Virtual', model='Group')
	
		#VC: set the parent of the node
		if initFlag&vizconnect.INIT_PARENTS:
			vizconnect.getGroup(_name).setParent(vizconnect.getAvatar('mark').getAttachmentPoint('r_hand'))

	#VC: initialize a new group
	_name = 'transportGroup'
	if vizconnect.isPendingInit('group', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: create the raw object
			rawGroup[_name] = viz.addGroup()
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addGroup(rawGroup[_name], _name, make='Virtual', model='Group')

	#VC: return values can be modified here
	return None


#################################
# Display Code
#################################

def initDisplays(initFlag=vizconnect.INIT_INDEPENDENT, initList=None):
	#VC: place any general initialization code here
	rawDisplay = vizconnect.getRawDisplayDict()

	#VC: initialize a new display
	_name = 'rift'
	if vizconnect.isPendingInit('display', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: set the window for the display
			_window = viz.MainWindow
			
			#VC: set some parameters
			autoDetectMonitor = True
			
			#VC: create the raw object
			import oculus
			try:
				display = oculus.Rift(window=_window, autoDetectMonitor=autoDetectMonitor)
				display.setMonoMirror(True)
				_window.displayNode = display
			except AttributeError:
				_window.displayNode = None
			rawDisplay[_name] = _window
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addDisplay(rawDisplay[_name], _name, make='Oculus VR', model='Rift')
	
		#VC: set the parent of the node
		if initFlag&vizconnect.INIT_PARENTS:
			vizconnect.getDisplay(_name).setParent(vizconnect.getAvatar('mark').getAttachmentPoint('head'))

	#VC: return values can be modified here
	return None


#################################
# Tracker Code
#################################

def initTrackers(initFlag=vizconnect.INIT_INDEPENDENT, initList=None):
	#VC: place any general initialization code here
	rawTracker = vizconnect.getRawTrackerDict()

	#VC: initialize a new tracker
	_name = 'l_hand_tracker'
	if vizconnect.isPendingInit('tracker', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: set some parameters
			hmdIndex = 0
			left = True
			
			#VC: create the raw object
			import oculus
			try:
				hmd = oculus.HMD(sensor=oculus.getSensors()[hmdIndex])
				if left:
					tracker = oculus.HMD().getLeftTouchController()
				else:
					tracker = oculus.HMD().getRightTouchController()
				tracker._isValid = True
			except IndexError:
				viz.logWarn("** WARNING: Not able to connect to an oculus touch controller")
				tracker = viz.addGroup()
				tracker.invalidTracker = True
			rawTracker[_name] = tracker
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addTracker(rawTracker[_name], _name, make='Oculus VR', model='Touch Controller')
	
		#VC: init the offsets
		if initFlag&vizconnect.INIT_OFFSETS:
			_link = vizconnect.getTracker(_name).getLink()
			#VC: clear link offsets
			_link.reset(viz.RESET_OPERATORS)
			
			#VC: apply offsets
			_link.postTrans([0, 1.8, 0])

	#VC: initialize a new tracker
	_name = 'r_hand_tracker'
	if vizconnect.isPendingInit('tracker', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: set some parameters
			hmdIndex = 0
			left = False
			
			#VC: create the raw object
			import oculus
			try:
				hmd = oculus.HMD(sensor=oculus.getSensors()[hmdIndex])
				if left:
					tracker = oculus.HMD().getLeftTouchController()
				else:
					tracker = oculus.HMD().getRightTouchController()
				tracker._isValid = True
			except IndexError:
				viz.logWarn("** WARNING: Not able to connect to an oculus touch controller")
				tracker = viz.addGroup()
				tracker.invalidTracker = True
			rawTracker[_name] = tracker
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addTracker(rawTracker[_name], _name, make='Oculus VR', model='Touch Controller')
	
		#VC: init the offsets
		if initFlag&vizconnect.INIT_OFFSETS:
			_link = vizconnect.getTracker(_name).getLink()
			#VC: clear link offsets
			_link.reset(viz.RESET_OPERATORS)
			
			#VC: apply offsets
			_link.postTrans([0, 1.8, 0])

	#VC: initialize a new tracker
	_name = 'head_tracker'
	if vizconnect.isPendingInit('tracker', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: set some parameters
			index = 0
			
			#VC: create the raw object
			import oculus
			sensorList = oculus.getSensors()
			if index < len(sensorList):
				orientationTracker = sensorList[index]
			else:
				viz.logWarn("** WARNING: Oculus VR Rift Orientation Tracker not present.")
				orientationTracker = viz.addGroup()
				orientationTracker.invalidTracker = True
			rawTracker[_name] = orientationTracker
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addTracker(rawTracker[_name], _name, make='Oculus VR', model='Rift Orientation Tracker')
	
		#VC: init the offsets
		if initFlag&vizconnect.INIT_OFFSETS:
			_link = vizconnect.getTracker(_name).getLink()
			#VC: clear link offsets
			_link.reset(viz.RESET_OPERATORS)
			
			#VC: apply offsets
			_link.postTrans([0, 1.8, 0])

	#VC: return values can be modified here
	return None


#################################
# Input Code
#################################

def initInputs(initFlag=vizconnect.INIT_INDEPENDENT, initList=None):
	#VC: place any general initialization code here
	rawInput = vizconnect.getRawInputDict()

	#VC: initialize a new input
	_name = 'left_touch_controller'
	if vizconnect.isPendingInit('input', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: set some parameters
			hmdIndex = 0
			left = True
			
			#VC: create the raw object
			import oculus
			try:
				hmd = oculus.HMD(sensor=oculus.getSensors()[hmdIndex])
				if left:
					input = oculus.HMD().getLeftTouchController()
				else:
					input = oculus.HMD().getRightTouchController()
				input._isValid = True
			except IndexError:
				viz.logWarn("** WARNING: Not able to connect to an oculus touch controller")
				input = viz.VizExtensionSensor(-1)
				input.isButtonDown = lambda e: False
				input.getTrackpad = lambda: [0,0]
				input._isValid = False
			rawInput[_name] = input
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addInput(rawInput[_name], _name, make='Oculus VR', model='Touch Controller')

	#VC: initialize a new input
	_name = 'right_touch_controller'
	if vizconnect.isPendingInit('input', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: set some parameters
			hmdIndex = 0
			left = False
			
			#VC: create the raw object
			import oculus
			try:
				hmd = oculus.HMD(sensor=oculus.getSensors()[hmdIndex])
				if left:
					input = oculus.HMD().getLeftTouchController()
				else:
					input = oculus.HMD().getRightTouchController()
				input._isValid = True
			except IndexError:
				viz.logWarn("** WARNING: Not able to connect to an oculus touch controller")
				input = viz.VizExtensionSensor(-1)
				input.isButtonDown = lambda e: False
				input.getTrackpad = lambda: [0,0]
				input._isValid = False
			rawInput[_name] = input
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addInput(rawInput[_name], _name, make='Oculus VR', model='Touch Controller')

	#VC: return values can be modified here
	return None


#################################
# Event Code
#################################

def initEvents(initFlag=vizconnect.INIT_INDEPENDENT, initList=None):
	#VC: place any general initialization code here
	rawEvent = vizconnect.getRawEventDict()

	#VC: return values can be modified here
	return None


#################################
# Transport Code
#################################

def initTransports(initFlag=vizconnect.INIT_INDEPENDENT, initList=None):
	#VC: place any general initialization code here
	rawTransport = vizconnect.getRawTransportDict()

	#VC: initialize a new transport
	_name = 'main_transport'
	if vizconnect.isPendingInit('transport', _name, initFlag, initList):
		#VC: request that any dependencies be created
		if initFlag&vizconnect.INIT_INDEPENDENT:
			initGroups(vizconnect.INIT_INDEPENDENT, ['group'])
			initGroups(vizconnect.INIT_INDEPENDENT, ['transportGroup'])
	
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: set some parameters
			arcSourceNode = vizconnect.getGroup('group').getNode3d()
			arcRange = 20
			transportationGroup = vizconnect.getGroup('transportGroup').getNode3d()
			
			#VC: create the raw object
			from transportation import arc_teleport
			rawTransport[_name] = arc_teleport.ArcTeleport(arcSourceNode=arcSourceNode
											,arcRange=arcRange
											,node=transportationGroup)
	
		#VC: init the mappings for the raw object
		if initFlag&vizconnect.INIT_MAPPINGS:
			#VC: per frame mappings
			if initFlag&vizconnect.INIT_MAPPINGS_PER_FRAME:
				#VC: get the raw input dict so we have access to signals
				import vizact
				rawInput = vizconnect.getConfiguration().getRawDict('input')
				#VC: set the update function which checks for input signals
				def update(transport):
					if rawInput['right_touch_controller'].isButtonDown(1):# make=Oculus VR, model=Touch Controller, name=right_touch_controller, signal=Button B
						transport.start()
					if not rawInput['right_touch_controller'].isButtonDown(1):# make=Oculus VR, model=Touch Controller, name=right_touch_controller, signal=Button B
						transport.stop()
				rawTransport[_name].setUpdateFunction(update)
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addTransport(rawTransport[_name], _name, make='Virtual', model='ArcTeleport')

	#VC: initialize a new transport
	_name = 'wandmagiccarpet'
	if vizconnect.isPendingInit('transport', _name, initFlag, initList):
		#VC: request that any dependencies be created
		if initFlag&vizconnect.INIT_INDEPENDENT:
			initTrackers(vizconnect.INIT_INDEPENDENT, ['head_tracker'])
			initGroups(vizconnect.INIT_INDEPENDENT, ['transportGroup'])
	
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: set some parameters
			orientationTracker = vizconnect.getTracker('head_tracker').getNode3d()
			debug = False
			acceleration = 2
			maxSpeed = 2
			rotationAcceleration = 60
			maxRotationSpeed = 65
			autoBreakingDragCoef = 0.1
			dragCoef = 0.0001
			rotationAutoBreakingDragCoef = 0.2
			rotationDragCoef = 0.0001
			usingPhysics = False
			parentedTracker = False
			transportationGroup = vizconnect.getGroup('transportGroup').getNode3d()
			
			#VC: create the raw object
			from transportation import wand_magic_carpet
			rawTransport[_name] = wand_magic_carpet.WandMagicCarpet(	orientationTracker=orientationTracker,
																					debug=debug,
																					acceleration=acceleration,
																					maxSpeed=maxSpeed,
																					rotationAcceleration=rotationAcceleration,
																					maxRotationSpeed=maxRotationSpeed,
																					autoBreakingDragCoef=autoBreakingDragCoef,
																					dragCoef=dragCoef,
																					rotationAutoBreakingDragCoef=rotationAutoBreakingDragCoef,
																					rotationDragCoef=rotationDragCoef,
																					usingPhysics=usingPhysics,
																					parentedTracker=parentedTracker,
																					node=transportationGroup)
	
		#VC: init the mappings for the raw object
		if initFlag&vizconnect.INIT_MAPPINGS:
			#VC: per frame mappings
			if initFlag&vizconnect.INIT_MAPPINGS_PER_FRAME:
				#VC: get the raw input dict so we have access to signals
				import vizact
				rawInput = vizconnect.getConfiguration().getRawDict('input')
				#VC: set the update function which checks for input signals
				def update(transport):
					if rawInput['left_touch_controller'].getStick()[1] > 0.01:# make=Oculus VR, model=Touch Controller, name=left_touch_controller, signal=Stick Up
						transport.moveForward(mag=abs(rawInput['left_touch_controller'].getStick()[1]))
					if rawInput['left_touch_controller'].getStick()[1] < -0.01:# make=Oculus VR, model=Touch Controller, name=left_touch_controller, signal=Stick Down
						transport.moveBackward(mag=abs(rawInput['left_touch_controller'].getStick()[1]))
					if rawInput['left_touch_controller'].getStick()[0] < -0.01:# make=Oculus VR, model=Touch Controller, name=left_touch_controller, signal=Stick Left
						transport.moveLeft(mag=abs(rawInput['left_touch_controller'].getStick()[0]))
					if rawInput['left_touch_controller'].getStick()[0] > 0.01:# make=Oculus VR, model=Touch Controller, name=left_touch_controller, signal=Stick Right
						transport.moveRight(mag=abs(rawInput['left_touch_controller'].getStick()[0]))
				rawTransport[_name].setUpdateFunction(update)
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addTransport(rawTransport[_name], _name, make='Virtual', model='WandMagicCarpet')

	#VC: return values can be modified here
	return None


#################################
# Tool Code
#################################

def initTools(initFlag=vizconnect.INIT_INDEPENDENT, initList=None):
	#VC: place any general initialization code here
	rawTool = vizconnect.getRawToolDict()

	#VC: initialize a new tool
	_name = 'l_proxy'
	if vizconnect.isPendingInit('tool', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: create the raw object
			from tools import proxy
			rawTool[_name] = proxy.Proxy()
	
		#VC: init the mappings for the raw object
		if initFlag&vizconnect.INIT_MAPPINGS:
			#VC: per frame mappings
			if initFlag&vizconnect.INIT_MAPPINGS_PER_FRAME:
				#VC: get the raw input dict so we have access to signals
				import vizact
				rawInput = vizconnect.getConfiguration().getRawDict('input')
				#VC: set the update function which checks for input signals
				def update(tool):
					if rawInput['left_touch_controller'].isButtonDown(10):# make=Oculus VR, model=Touch Controller, name=left_touch_controller, signal=Button Index
						tool.action1()
				rawTool[_name].setUpdateFunction(update)
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addTool(rawTool[_name], _name, make='Virtual', model='Proxy')
	
		#VC: set the parent of the node
		if initFlag&vizconnect.INIT_PARENTS:
			vizconnect.getTool(_name).setParent(vizconnect.getAvatar('mark').getAttachmentPoint('l_hand'))

	#VC: initialize a new tool
	_name = 'proxy'
	if vizconnect.isPendingInit('tool', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: create the raw object
			from tools import proxy
			rawTool[_name] = proxy.Proxy()
	
		#VC: init the mappings for the raw object
		if initFlag&vizconnect.INIT_MAPPINGS:
			#VC: per frame mappings
			if initFlag&vizconnect.INIT_MAPPINGS_PER_FRAME:
				#VC: get the raw input dict so we have access to signals
				import vizact
				rawInput = vizconnect.getConfiguration().getRawDict('input')
				#VC: set the update function which checks for input signals
				def update(tool):
					if rawInput['right_touch_controller'].isButtonDown(10):# make=Oculus VR, model=Touch Controller, name=right_touch_controller, signal=Button Index
						tool.action1()
				rawTool[_name].setUpdateFunction(update)
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addTool(rawTool[_name], _name, make='Virtual', model='Proxy')
	
		#VC: set the parent of the node
		if initFlag&vizconnect.INIT_PARENTS:
			vizconnect.getTool(_name).setParent(vizconnect.getAvatar('mark').getAttachmentPoint('r_hand'))

	#VC: return values can be modified here
	return None


#################################
# Avatar Code
#################################

def initAvatars(initFlag=vizconnect.INIT_INDEPENDENT, initList=None):
	#VC: place any general initialization code here
	rawAvatar = vizconnect.getRawAvatarDict()

	#VC: initialize a new avatar
	_name = 'mark'
	if vizconnect.isPendingInit('avatar', _name, initFlag, initList):
		#VC: init the raw object
		if initFlag&vizconnect.INIT_RAW:
			#VC: set some parameters
			head = True
			rightHand = True
			leftHand = True
			torso = False
			lowerBody = False
			rightArm = False
			leftArm = False
			
			#VC: create the raw object
			# base avatar
			import vizfx
			avatar = vizfx.addChild('mark.cfg')
			avatar._bodyPartDict = {}
			avatar._handModelDict = {}
			avatar.visible(head, r'mark_head.cmf')
			avatar.visible(rightHand, r'mark_hand_r.cmf')
			avatar.visible(leftHand, r'mark_hand_l.cmf')
			avatar.visible(torso, r'mark_torso.cmf')
			avatar.visible(lowerBody, r'mark_legs.cmf')
			avatar.visible(rightArm, r'mark_arm_r.cmf')
			avatar.visible(leftArm, r'mark_arm_l.cmf')
			rawAvatar[_name] = avatar
	
		#VC: init the wrapper (DO NOT EDIT)
		if initFlag&vizconnect.INIT_WRAPPERS:
			vizconnect.addAvatar(rawAvatar[_name], _name, make='WorldViz', model='Mark')
	
		#VC: init the gestures
		if initFlag&vizconnect.INIT_GESTURES:
			#VC: need to get the raw input dict so we have access to signals
			import vizact
			rawInput = vizconnect.getConfiguration().getRawDict('input')
			
			#VC: gestures for the avatar's r_hand
			import hand
			def initHand():
				sensor = hand.InputSensor()
				rawAvatar[_name].handSensor = sensor
				sensor.createHandRenderer = lambda *args,**kw: hand._InputDeviceRenderer(*args,**kw)
				def appliedGetData():
					#VC: set the mappings for the gestures
					if rawInput['right_touch_controller'].isButtonDown(10):# make=Oculus VR, model=Touch Controller, name=right_touch_controller, signal=Button Index
						return (hand.GESTURE_FIST, False, False)# GESTURE_FIST
					#VC: end gesture mappings
					return (hand.GESTURE_FLAT_HAND,False,False)
				sensor.getData = appliedGetData
				return hand.AvatarHandModel(rawAvatar[_name], left=False, type=hand.GLOVE_5DT, sensor=sensor)
			rightHand = initHand()
			rawAvatar[_name]._bodyPartDict[vizconnect.AVATAR_R_HAND] = rightHand
			rawAvatar[_name]._handModelDict[vizconnect.AVATAR_R_HAND] = rightHand
			#VC: gestures for the avatar's l_hand
			import hand
			def initHand():
				sensor = hand.InputSensor()
				rawAvatar[_name].handSensor = sensor
				sensor.createHandRenderer = lambda *args,**kw: hand._InputDeviceRenderer(*args,**kw)
				def appliedGetData():
					#VC: set the mappings for the gestures
					if rawInput['left_touch_controller'].isButtonDown(10):# make=Oculus VR, model=Touch Controller, name=left_touch_controller, signal=Button Index
						return (hand.GESTURE_FIST, False, False)# GESTURE_FIST
					#VC: end gesture mappings
					return (hand.GESTURE_FLAT_HAND,False,False)
				sensor.getData = appliedGetData
				return hand.AvatarHandModel(rawAvatar[_name], left=True, type=hand.GLOVE_5DT, sensor=sensor)
			leftHand = initHand()
			rawAvatar[_name]._bodyPartDict[vizconnect.AVATAR_L_HAND] = leftHand
			rawAvatar[_name]._handModelDict[vizconnect.AVATAR_L_HAND] = leftHand
			
			#VC: gestures may change the raw avatar, so refresh the raw in the wrapper
			vizconnect.getAvatar(_name).setRaw(rawAvatar[_name])
	
		#VC: init the animator
		if initFlag&vizconnect.INIT_ANIMATOR:
			# need to get the raw tracker dict for animating the avatars
			from vizconnect.util.avatar import animator
			from vizconnect.util.avatar import skeleton
			
			# get the skeleton from the avatar
			_skeleton = skeleton.CompleteCharactersHD(rawAvatar[_name])
			
			#VC: set which trackers animate which body part
			# format is: bone: (tracker, parent, degrees of freedom used)
			_trackerAssignmentDict = {
				vizconnect.AVATAR_HEAD:(vizconnect.getTracker('head_tracker').getNode3d(), None, vizconnect.DOF_ORI),
				vizconnect.AVATAR_L_HAND:(vizconnect.getTracker('l_hand_tracker').getNode3d(), None, vizconnect.DOF_6DOF),
				vizconnect.AVATAR_R_HAND:(vizconnect.getTracker('r_hand_tracker').getNode3d(), None, vizconnect.DOF_6DOF),
			}
			
			#VC: create the raw object
			_rawAnimator = animator.Direct(rawAvatar[_name], _skeleton, _trackerAssignmentDict)
			
			#VC: set animator in wrapper (DO NOT EDIT)
			vizconnect.getAvatar(_name).setAnimator(_rawAnimator, make='Virtual', model='Direct')
	
		#VC: set the parent of the node
		if initFlag&vizconnect.INIT_PARENTS:
			vizconnect.getAvatar(_name).setParent(vizconnect.getGroup('transportGroup'))

	#VC: return values can be modified here
	return None


#################################
# Application Settings
#################################

def initSettings():
	#VC: apply general application settings
	viz.mouse.setTrap(False)
	viz.mouse.setVisible(viz.MOUSE_AUTO_HIDE)
	vizconnect.setMouseTrapToggleKey('')
	
	#VC: return values can be modified here
	return None


#################################
# Post-initialization Code
#################################

def postInit():
	"""Add any code here which should be called after all of the initialization of this configuration is complete.
	Returned values can be obtained by calling getPostInitResult for this file's vizconnect.Configuration instance."""
	
	return None


#################################
# Stand alone configuration
#################################

def initInterface():
	#VC: start the interface
	vizconnect.interface.go(__file__,
							live=True,
							openBrowserWindow=True,
							startingInterface=vizconnect.interface.INTERFACE_STARTUP)

	#VC: return values can be modified here
	return None


###############################################

if __name__ == "__main__":
	initInterface()
	viz.add('piazza.osgb')
	viz.add('piazza_animations.osgb')


