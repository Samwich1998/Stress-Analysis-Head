import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis





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
            return peakInd
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
            

yDiff1 = []
xDiff1 = []
blinkDurations = []
leftIndices = []
rightIndices = []
finalInds = []
toBaselines = 15;
fitPercent = 0.7
minBaselinePoints = 10
pointsPerLine = 5
minChi2 = 10E-6

# Rudimentary Peak Detection to Find All Potential Blinks (With Tons of Extra Stuff)
peakIndices = scipy.signal.find_peaks(yData, prominence=.3, width=50)[0];
# Extract the Blinks from the Peaks
for peakInd in peakIndices:
    
    # ------------------------- Find Blink Baselines ------------------------ #
    # Calculate the Left and Right Baseline of the Peak
    leftBaselineIndex = findBaselineIndex(xData, yData, peakInd, searchDirection = -1)
    rightBaselineIndex = findBaselineIndex(xData, yData, peakInd, searchDirection = 1)
    
    # If No Baseline is Found, Ignore the Blink (Too Noisy, Probably Not a Blink)
    if leftBaselineIndex >= peakInd - minBaselinePoints or rightBaselineIndex <= peakInd + minBaselinePoints:
        continue
    
    #blinkDuration = xData[rightBaselineIndex] - xData[leftBaselineIndex]
    # ----------------------------------------------------------------------- #
    
    # ---------------- Find leftStroke and rightStroke Lines ---------------- #
    # Define leftStroke and rightStroke Lines
    leftLineParams, startLeftLineInd, endLeftLineInd = findPeakLines(xData, yData, leftBaselineIndex, peakInd-pointsPerLine, pointsPerLine, minChi2, searchDirection = 1)
    rightLineParams, startRightLineInd, endRightLineInd = findPeakLines(xData, yData, rightBaselineIndex, peakInd + pointsPerLine, pointsPerLine, minChi2, searchDirection = -1)

    # Remove Peaks Without a Good Line
    if not startLeftLineInd or not startRightLineInd:
        continue
    # ----------------------------------------------------------------------- #
    
    # ------------------------ Extract Blink Features ----------------------- #
    # Find Tent Peak
    peakTentX = (rightLineParams[1] - leftLineParams[1])/(leftLineParams[0] - rightLineParams[0])
    peakTentY = leftLineParams[0]*peakTentX + leftLineParams[1]
    
    # Calculate New Baseline Points on the Left Side
    leftBlinkBaselineX = (leftLineParams[1] - yData[leftBaselineIndex])/(0 -leftLineParams[0])
    leftBlinkBaselineY = yData[leftBaselineIndex]
    # Calculate New Baseline Points on the Right Side
    rightBlinkBaselineX = (yData[rightBaselineIndex] - rightLineParams[1])/(rightLineParams[0] - 0)
    rightBlinkBaselineY = yData[rightBaselineIndex]
    # Calculate the Average Baseline
    averageBaselineY = (peakTentIndY + peakTentIndX)/2
    
    # Calculate Blink Amplitudes
    blinkAmpTent = peakTentY - averageBaselineY
    blinkAmpPeak = yData[peakInd] - averageBaselineY
    # Calculate the Blink Times
    blinkDuration = rightBlinkBaselineX - leftBlinkBaselineX  # The Total Time of the Blink
    closingTime = peakTentX - leftBlinkBaselineX           # Eye's Closing Time
    openingTime = rightBlinkBaselineX - peakTentX          # Eye's Opening Time
    # Calculate Time the Eyes are Closed
    blinkAmp90Y = averageBaselineY + blinkAmpPeak*0.9
    blinkAmp90RightInd = np.argmin(yData[peakInd:rightBaselineIndex] - blinkAmp90Y)
    blinkAmp90LeftInd = np.argmin(yData[leftBaselineIndex:peakInd] - blinkAmp90Y)
    eyesClosedTime = xData[blinkAmp90RightInd] - xData[blinkAmp90LeftInd]
    
    # Calculate Shape Parameters
    peakSkew = skew(yData[leftBaselineIndex:rightBaselineIndex], bias=False)
    peakKurtosis = kurtosis(yData[leftBaselineIndex:rightBaselineIndex], fisher=False, bias = False)
    # ----------------------------------------------------------------------- #
    
    # ------------------------ Cull Potential Blinks ------------------------ #

    # If the Blink is Shorter Than 50ms or Longer Than 2s, Ignore the Blink (Probably Eye Movement)
    if 0.5 < blinkDuration < 0.05:
        print(blinkDuration)
        continue
    elif closingTime > 0.15:
        print(closingTime)
        continue
    # ----------------------------------------------------------------------- #
    

    if peakTentY - yData[peakInd] < 0.3:
        yDiff1.append(peakTentY - yData[peakInd])
        xDiff1.append(peakTentX - xData[peakInd])
        blinkDurations.append(blinkDuration)
        finalInds.append(peakInd)
        leftIndices.append(leftBaselineIndex)
        rightIndices.append(rightBaselineIndex)

    if True:
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
plt.xlim([5, 20])
plt.show()
xDiff1 = np.array(xDiff1); yDiff1 = np.array(yDiff1)
#ax = plt.axes(projection='3d')
plt.plot(xDiff1, yDiff1, 'o'); #plt.xlim([-0.05, 0]); plt.ylim([-.15, 0.05])



