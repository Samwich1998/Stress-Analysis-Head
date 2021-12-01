import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
from BaselineRemoval import BaselineRemoval


def normalizePulseBaseline(pulseData, polynomialDegree):
    """
    ----------------------------------------------------------------------
    Input Parameters:
        pulseData:  y-Axis Data for a Single Pulse (Start-End)
        polynomialDegree: Polynomials Used in Baseline Subtraction
    Output Parameters:
        pulseData: y-Axis Data for a Baseline-Normalized Pulse (Start, End = 0)
    Use Case: Shift the Pulse to the x-Axis (Removing non-Horizontal Base)
    Assumption in Function: pulseData is Positive
    ----------------------------------------------------------------------
    Further API Information Can be Found in the Following Link:
    https://pypi.org/project/BaselineRemoval/
    ----------------------------------------------------------------------
    """
    # Perform Baseline Removal Twice to Ensure Baseline is Gone
    for _ in range(2):
        # Baseline Removal Procedure
        baseObj = BaselineRemoval(pulseData)  # Create Baseline Object
        pulseData = baseObj.ModPoly(polynomialDegree) # Perform Modified multi-polynomial Fit Removal

    # Return the Data With Removed Baseline
    return pulseData



def findNearbyMinimum(data, xPointer, binarySearchWindow = 50, maxPointsSearch = 1000):
    """
    Search Right: binarySearchWindow > 0
    Search Left: binarySearchWindow < 0
    """
    # Base Case
    if abs(binarySearchWindow) <= 1:
        return xPointer
    
    maxHeight = data[xPointer]; searchDirection = int(binarySearchWindow/abs(binarySearchWindow))
    # Binary Search Data to the Left to Find Minimum (Skip Over Small Bumps)
    for dataPointer in range(max(xPointer, 0), max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
        # If the Point is Greater Than
        if data[dataPointer] > maxHeight:
            return findNearbyMinimum(data, dataPointer - binarySearchWindow, math.floor(binarySearchWindow/8), maxPointsSearch - (xPointer - dataPointer - binarySearchWindow))
        else:
            xPointer = dataPointer
            maxHeight = data[dataPointer]

    # If Your Binary Search is Too Small, Reduce it
    return findNearbyMinimum(data, xPointer, math.floor(binarySearchWindow/2), maxPointsSearch)

def findBaselineIndex(xData, yData, xPointer, searchDirection = 1):
    
    if searchDirection == 1:
        endSearch = len(yData)
    elif searchDirection == -1:
        endSearch = max(-1, xPointer - 1000)
    else:
        print("Wrong Search Direction")
        sys.exit()
    
    addOn = 5; firstDer = [0]*addOn; skipPoints = 10;
    foundDrop = False; maxSlope = 0
    # Caluclate the Running Slope of the Data
    for peakInd in range(xPointer + searchDirection*(addOn+skipPoints), endSearch, searchDirection):
        # Calculate the First Derivative
        deltaY = np.mean(yData[max(0,peakInd - addOn):peakInd+1]) - np.mean(yData[max(0,peakInd - 2*addOn - 1):peakInd-addOn+1])
        deltaX = max(xData[peakInd] - xData[max(0,peakInd-addOn - 1)], 10E-10)
        firstDeriv = deltaY/deltaX
        firstDer.append(deltaY/deltaX)
        
        # Verify Major Slope Drop
        if abs(firstDeriv) > 0.5:
            foundDrop = True
            maxSlope = max(maxSlope, abs(firstDeriv))
        
        if foundDrop and abs(firstDeriv) < maxSlope/10:
            return findNearbyMinimum(yData, peakInd, binarySearchWindow = searchDirection*20, maxPointsSearch = 50) #peakInd
    return peakInd

def findPeakLines(xData, yData, startSearch, endSearch, pointsPerLine, minChi2, searchDirection = 1):
    lineParams = [0, 0]; startLineInd = 0; endLineInd = 0
    # Loop Through the Left Line Starting Points to Find Best Line
    for startLineI in range(startSearch, endSearch, searchDirection):  
        # Get the Line
        lineX = xData[startLineI:startLineI+pointsPerLine]
        lineY = yData[startLineI:startLineI+pointsPerLine]
        
        # Fit the Line
        lineParamsI = np.polyfit(lineX, lineY, 1)
        lineCHI2 = np.sum(((np.polyval(lineParamsI, lineX) - lineY) ** 2)/lineY)/pointsPerLine
        
        lineParamsI, residuals, _, _, _ = np.polyfit(lineX, lineY, 1, full=True)
        if len(residuals) != 0:
            lineCHI2 = residuals[0] / (len(lineX) - 1)
            
            # Save the Best Line with the Highest Slope
            if lineParams[0]*searchDirection < lineParamsI[0]*searchDirection and lineCHI2 < minChi2:
                lineParams = lineParamsI
                startLineInd = startLineI
                endLineInd = startLineI+pointsPerLine
            
    # Return the Best Line
    return lineParams, startLineInd, endLineInd

def findPeakLines_TWO(xData, yData, baselineIndex, peakInd, minChi2, searchDirection = 1):
    # Find the Starting/Ending Points Representing the Inner 70% of the Peak Amplitude
    peakAmp = yData[peakInd] - yData[baselineIndex]
    startLineY = yData[baselineIndex] + peakAmp*0.3
    endLineY = yData[baselineIndex] + peakAmp*0.8
    
    # Find the Closest Starting Point in the Curve
    if searchDirection == 1:
        # Find Left Line
        startLineInd = baselineIndex + np.argmin(abs(yData[baselineIndex:peakInd+1] - startLineY))
        endLineInd = baselineIndex + np.argmin(abs(yData[baselineIndex:peakInd+1] - endLineY))
    else:
        # Find Right Line
        startLineInd = peakInd + np.argmin(abs(yData[peakInd:baselineIndex+1] - endLineY))
        endLineInd = peakInd + np.argmin(abs(yData[peakInd:baselineIndex+1] - startLineY))
    # If No Line Found, Return 0
    if endLineInd - startLineInd < 5:
        return [0,0],0,0
    
    # Calculate the Line Parameters: [Slope, Y-Cross]
    lineParams, residuals, _, _, _ = np.polyfit(xData[startLineInd:endLineInd+1], yData[startLineInd:endLineInd+1], 1, full=True)
    if len(residuals) != 0:
        # Calculate the Chi2
        lineCHI2 = residuals[0] / (endLineInd - startLineInd - 1)
        # 
        if lineCHI2 < minChi2:
            return lineParams, startLineInd, endLineInd
    return [0,0],0,0
    
            

yDiff1 = []
xDiff1 = []
blinkDurations = []
leftIndices = []
rightIndices = []
finalInds = []
fitPercent = 0.7
minBaselinePoints = 10
pointsPerLine = 20
minChi2 = 5*10E-5
minPeakHeight = 0.1  # No Less Than 0.11
multPeakSepMax = 0.5  # No Less Than 0.25

fromPeakInd = 15

finalSlopesL = []
finalSlopesR = []
curvatures = []
peakShape = []
conservedProp = []
conservedProp2 = []


singlePeaks = []
multPeaks = []

# Rudimentary Peak Detection to Find All Potential Blinks (With Tons of Extra Stuff)
peakIndices = scipy.signal.find_peaks(yData, prominence=.01, width=20)[0];
# Extract the Blinks from the Peaks
for peakInd in peakIndices:
    
    # ------------------------- Find Blink Baselines ------------------------ #
    # Calculate the Left and Right Baseline of the Peak
    leftBaselineIndex = findBaselineIndex(xData, yData, peakInd, searchDirection = -1)
    rightBaselineIndex = findBaselineIndex(xData, yData, peakInd, searchDirection = 1)
    
    # If No Baseline is Found, Ignore the Blink (Too Noisy, Probably Not a Blink)
    if leftBaselineIndex >= peakInd - minBaselinePoints or rightBaselineIndex <= peakInd + minBaselinePoints:
        #print("No Baseline", xData[peakInd])
        continue
    # Minimum Height of a Peak
    elif yData[peakInd] - yData[leftBaselineIndex] < minPeakHeight or yData[peakInd] - yData[rightBaselineIndex] < minPeakHeight:
   #    print("Too Close", xData[peakInd])
        continue
    
    #blinkDuration = xData[rightBaselineIndex] - xData[leftBaselineIndex]
    # ----------------------------------------------------------------------- #
    
    # ---------------- Find leftStroke and rightStroke Lines ---------------- #
    # Define leftStroke and rightStroke Lines
   # leftLineParams, startLeftLineInd, endLeftLineInd = findPeakLines(xData, yData, leftBaselineIndex, peakInd-pointsPerLine, pointsPerLine, minChi2, searchDirection = 1)
   # rightLineParams, startRightLineInd, endRightLineInd = findPeakLines(xData, yData, rightBaselineIndex, peakInd + pointsPerLine, pointsPerLine, minChi2, searchDirection = -1)
    leftLineParams, startLeftLineInd, endLeftLineInd = findPeakLines_TWO(xData, yData, leftBaselineIndex, peakInd, minChi2, searchDirection = 1)
    rightLineParams, startRightLineInd, endRightLineInd = findPeakLines_TWO(xData, yData, rightBaselineIndex, peakInd, minChi2, searchDirection = -1)

    # Remove Peaks Without a Good Line
    if not startLeftLineInd or not startRightLineInd:
        print("No Line Found", xData[peakInd])
        continue
    # ----------------------------------------------------------------------- #
    
    # ------------------------ Extract Blink Features ----------------------- #
    # Find Tent Peak
    peakTentX = (rightLineParams[1] - leftLineParams[1])/(leftLineParams[0] - rightLineParams[0])
    peakTentY = leftLineParams[0]*peakTentX + leftLineParams[1]
    
    # Account for Skewed Baseline - Probably From Eye Movement Alongside Blink
    baseLinesSkewed = abs(yData[rightBaselineIndex] - yData[leftBaselineIndex]) > 0.25*(yData[peakInd] - max(yData[rightBaselineIndex], yData[leftBaselineIndex]))
    # Calculate Baseline of the Peak
    if baseLinesSkewed:
        averageBaselineY = min(yData[rightBaselineIndex], yData[leftBaselineIndex])
    else:
     #   delY = yData[rightBaselineIndex] - yData[leftBaselineIndex]
        delX = xData[rightBaselineIndex] - xData[leftBaselineIndex]
        averageBaselineY = yData[leftBaselineIndex] + (delY/delX)*(xData[peakInd] - xData[leftBaselineIndex])
        
    # Calculate New Baseline Points on the Left Side
    leftBlinkBaselineX = (leftLineParams[1] - averageBaselineY)/(0 -leftLineParams[0])
    leftBlinkBaselineY = averageBaselineY
    # Calculate New Baseline Points on the Right Side
    rightBlinkBaselineX = (averageBaselineY - rightLineParams[1])/(rightLineParams[0] - 0)
    rightBlinkBaselineY = averageBaselineY
    
    # Calculate Blink Amplitudes
    blinkAmpTent = peakTentY - averageBaselineY
    blinkAmpPeak = yData[peakInd] - averageBaselineY
    # Calculate the Blink Duration Parameters
    blinkDuration = rightBlinkBaselineX - leftBlinkBaselineX  # The Total Time of the Blink
    closingTime = peakTentX - leftBlinkBaselineX           # Eye's Closing Time
    openingTime = rightBlinkBaselineX - peakTentX          # Eye's Opening Time
    # Calculate the 1/2 Amp Blink Duration
    blinkAmp50Y = averageBaselineY + blinkAmpPeak*0.5
    blinkAmp50RightInd = np.argmin(yData[peakInd:rightBaselineIndex] - blinkAmp50Y)
    blinkAmp50LeftInd = np.argmin(yData[leftBaselineIndex:peakInd] - blinkAmp50Y)
    halfClosedTime = xData[blinkAmp50RightInd] - xData[blinkAmp50LeftInd]
    # Calculate Time the Eyes are Closed
    blinkAmp90Y = averageBaselineY + blinkAmpPeak*0.9
    blinkAmp90RightInd = np.argmin(yData[peakInd:rightBaselineIndex] - blinkAmp90Y)
    blinkAmp90LeftInd = np.argmin(yData[leftBaselineIndex:peakInd] - blinkAmp90Y)
    eyesClosedTime = xData[blinkAmp90RightInd] - xData[blinkAmp90LeftInd]
    
    # Calculate Shape Parameters
    peakSkew = skew(yData[leftBaselineIndex:rightBaselineIndex], bias=False)
    peakKurtosis = kurtosis(yData[leftBaselineIndex:rightBaselineIndex], fisher=False, bias = False)
    
    # Calculate Curvature
    dx_dt = np.gradient(xData[peakInd - fromPeakInd:peakInd + fromPeakInd + 1])
    dy_dt = np.gradient(yData[peakInd - fromPeakInd:peakInd + fromPeakInd + 1])
    dy_dt = np.gradient(normalizePulseBaseline(yData[peakInd - fromPeakInd:peakInd + fromPeakInd + 1], 1))
    
    velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(len(dx_dt))])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    tangent = np.array([1/ds_dt] * 2).transpose() * velocity


    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5

    

    # ----------------------------------------------------------------------- #
    
    # ------------------------ Cull Potential Blinks ------------------------ #
    # If the Blink is Shorter Than 50ms or Longer Than 2s, Ignore the Blink (Probably Eye Movement)
    if 0.5 < blinkDuration < 0.05:
    #   print("Bad Blink Duration:", blinkDuration)
        continue
    elif closingTime > 0.15:
      #  print("Bad Closing Time:", closingTime)
        continue
    elif abs((peakTentY - yData[peakInd]) + (peakTentX - xData[peakInd])) > 0.1:
       # print("Culled", xData[peakInd])
        continue

    currentShape = yData[peakInd - fromPeakInd:peakInd + fromPeakInd + 1]
    peakDome = np.diff(currentShape)/np.diff(currentShape)[0]
    if max(peakDome) > 1 or min(peakDome) < -1:
        continue
    # ----------------------------------------------------------------------- #
    

    if peakTentY - yData[peakInd] < 3:
        yDiff1.append(peakTentY - yData[peakInd])
        xDiff1.append(peakTentX - xData[peakInd])
        blinkDurations.append(blinkDuration)
        finalInds.append(peakInd)
        leftIndices.append(leftBaselineIndex)
        rightIndices.append(rightBaselineIndex)
        
        finalSlopesL.append(leftLineParams[0])
        finalSlopesR.append(rightLineParams[0])
        curvatures.append((1/curvature[fromPeakInd])*curvature * (-xData[peakInd-fromPeakInd] + xData[peakInd+fromPeakInd]))
       # curvatures.append(curvature * halfClosedTime)
        peakShape.append(yData[peakInd - fromPeakInd:peakInd + fromPeakInd + 1])
        
        currentShape = peakShape[-1] - peakShape[-1][0]
        currentShape = (1/max(currentShape))*currentShape
        
      #  conservedProp.append([leftLineParams[0]/closingTime, rightLineParams[0]/openingTime])

        doublePeak = False
        if not singlePeaks:
            singlePeaks.append(peakInd)
        elif xData[peakInd] - xData[singlePeaks[-1]] < multPeakSepMax:
            doublePeak = True
            lastPeakInd = singlePeaks.pop()
            if not multPeaks:
                multPeaks.append([lastPeakInd, peakInd])
            elif xData[peakInd] - xData[multPeaks[-1][-1]] < multPeakSepMax:
                multPeaks[-1].append(peakInd)
            else:
                multPeaks.append([lastPeakInd, peakInd])
        else:
            singlePeaks.append(peakInd)
        
#        if doublePeak:
#            conservedProp.append([leftLineParams[0]/closingTime, rightLineParams[0]/openingTime])
#        else:
#            conservedProp2.append([leftLineParams[0]/closingTime, rightLineParams[0]/openingTime])
        
        blinkAmpPeak = yData[peakInd] - averageBaselineY
        blinkAmp90Y = averageBaselineY + blinkAmpPeak*0.9
        averageBaselineY = blinkAmp90Y
        
        # Calculate New Baseline Points on the Left Side
        leftBlinkBaselineX = (leftLineParams[1] - averageBaselineY)/(0 -leftLineParams[0])
        leftBlinkBaselineY = averageBaselineY
        # Calculate New Baseline Points on the Right Side
        rightBlinkBaselineX = (averageBaselineY - rightLineParams[1])/(rightLineParams[0] - 0)
        rightBlinkBaselineY = averageBaselineY
        
        blinkDuration = rightBlinkBaselineX - leftBlinkBaselineX  # The Total Time of the Blink
      #  closingTime = peakTentX - leftBlinkBaselineX           # Eye's Closing Time
      #  openingTime = rightBlinkBaselineX - peakTentX          # Eye's Opening Time
        
        blinkAmp = yData[peakInd] - averageBaselineY
        
        conservedProp.append((peakTentY - yData[peakInd])/(peakTentX - xData[peakInd]))
        conservedProp2.append(openingTime/closingTime)
        
        

    if False:
        plt.plot(xData[peakInd], yData[peakInd], 'ko');
        plt.plot(xData, yData); plt.plot(xData[leftBaselineIndex], yData[leftBaselineIndex], 'go');
        plt.plot(xData[rightBaselineIndex], yData[rightBaselineIndex], 'ro');
        plt.plot(peakTentX, peakTentY, 'kx')
        plt.plot(xData, yData); plt.plot([leftBlinkBaselineX, rightBlinkBaselineX], [leftBlinkBaselineY, rightBlinkBaselineY], 'bo');
        plt.plot(xData[startLeftLineInd:endLeftLineInd], leftLineParams[0]*xData[startLeftLineInd:endLeftLineInd] + leftLineParams[1])
        plt.plot(xData[startRightLineInd:endRightLineInd], rightLineParams[0]*xData[startRightLineInd:endRightLineInd] + rightLineParams[1])
        plt.xlim([xData[leftBaselineIndex], xData[rightBaselineIndex]])
        plt.show()
plt.plot(xData, yData); plt.plot(xData[finalInds], yData[finalInds], 'o');
plt.plot(xData[leftIndices], yData[leftIndices], 'go');
plt.plot(xData[rightIndices], yData[rightIndices], 'ro');
#plt.xlim([5, 20])
plt.show()
xDiff1 = np.array(xDiff1); yDiff1 = np.array(yDiff1)
#ax = plt.axes(projection='3d')
plt.plot(xDiff1, yDiff1, 'o'); #plt.xlim([-0.05, 0]); plt.ylim([-.15, 0.05])
plt.show()


"""
for i in range(len(curvatures)):
    plt.plot(curvatures[i], 'o')
plt.show()
for i in range(len(peakShape)):
    plt.plot(normalizePulseBaseline(peakShape[i], 1), 'o')
curvatures = np.array(curvatures)
plt.show()
plt.plot(curvatures[:,10])
plt.show()

for i in range(len(peakShape)):
   # currentShape = normalizePulseBaseline(peakShape[i], 1)
    currentShape = peakShape[i] - peakShape[i][0]
    #plt.plot(currentShape,'o')
    plt.plot((1/max(currentShape))*currentShape, 'o')
plt.show()
"""

conservedProp = np.array(conservedProp)
conservedProp2 = np.array(conservedProp2)
plt.plot(conservedProp, 'o')
plt.plot(conservedProp2, 'o')
plt.show()

a = []
for i in range(len(curvatures)):
    a.append(np.average(curvatures[i]))
    plt.plot(np.average(curvatures[i]), 'o')
plt.ylim([0,0.3])
plt.show();
plt.hist(a, 100)
plt.show()


for i in range(len(peakShape)):
   # currentShape = normalizePulseBaseline(peakShape[i], 1)
    currentShape = np.diff(peakShape[i])
    #plt.plot(currentShape,'o')
    plt.plot(currentShape, 'o')
plt.show()

b = []
for i in range(len(peakShape)):
   # currentShape = normalizePulseBaseline(peakShape[i], 1)
    currentShape = np.diff(peakShape[i])/np.diff(peakShape[i])[0]
    b.append(currentShape[0] + currentShape[-1])
    #plt.plot(currentShape,'o')
    plt.plot(currentShape, 'o')
plt.show()
plt.hist(b)
    
    