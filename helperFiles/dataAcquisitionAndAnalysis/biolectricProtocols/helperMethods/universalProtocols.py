import re
import scipy
import numpy as np


class universalMethods:

    # --------------------- Feature Extraction Methods --------------------- #

    @staticmethod
    def hjorthParameters(timePoints, data, firstDeriv=None, secondDeriv=None, standardized_data=None):
        # Assert the correct data format.
        data = np.asarray(data)
        if standardized_data is not None:
            standardized_data = np.asarray(standardized_data)
        else:
            standardized_data = universalMethods.standardizeData(data)

        # If no derivatives given, calculate the derivatives
        if firstDeriv is None:
            if np.min(np.diff(timePoints)) < 1e-10: print(f"ALERT: Values too small for gradient: {np.min(np.diff(timePoints))}")
            firstDeriv = np.gradient(standardized_data, timePoints)
        if secondDeriv is None:
            secondDeriv = np.gradient(firstDeriv, timePoints)

        # Calculate the hjorthActivity
        activity = np.var(data)  # Cant use standardized data because its variance is already normalized.

        # If activity is zero, complexity and mobility are zero.
        if activity == 0 and len(data.shape) == 1:
            return activity, 0, 0, 0, 0

        # Calculate the hjorthMobility
        firstDerivVariance = np.var(firstDeriv, ddof=1)
        mobility = np.sqrt(firstDerivVariance / activity)

        # Calculate the hjorthComplexity
        secondDerivVariance = np.var(secondDeriv, ddof=1)
        complexity = np.sqrt(secondDerivVariance / firstDerivVariance) / mobility

        return activity, mobility, complexity, firstDerivVariance, secondDerivVariance

    @staticmethod
    def calculatePSD(data, samplingFreq):
        # Calculate the power spectrum density parameters.
        nperseg = int(samplingFreq * 2.5)
        noverlap = nperseg // 2
        nfft = nperseg * 2

        # Calculate the power spectrum density.
        powerSpectrumDensityFreqs, powerSpectrumDensity = scipy.signal.welch(data, fs=samplingFreq, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                                                             detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

        # Normalize the power spectrum density.
        powerSpectrumDensityNormalized = universalMethods.normalizePSD(powerSpectrumDensity)
        # powerSpectrumDensityNormalized is amplitude-invariant to the original data UNLIKE powerSpectrumDensity.

        return powerSpectrumDensityFreqs, powerSpectrumDensity, powerSpectrumDensityNormalized

    @staticmethod
    def standardizeData(data):
        standardDeviation = np.std(data, ddof=1)

        if standardDeviation == 0:
            return np.zeros_like(data)

        return (data - np.mean(data)) / standardDeviation

    @staticmethod
    def bandPower(powerSpectrumDensity, powerSpectrumDensityFreqs, bands, relative=True):
        # Set up the initial parameters.
        dFreq = powerSpectrumDensityFreqs[1] - powerSpectrumDensityFreqs[0]
        bandPowers = []

        for freqBand in bands:
            # Find the indices corresponding to the band of interest
            idx_band = np.logical_and(powerSpectrumDensityFreqs >= freqBand[0], powerSpectrumDensityFreqs <= freqBand[1])
            assert idx_band.sum() != 0, f"You do not have enough sampling frequency to view this band: {freqBand}"

            # Calculate the power in the band of interest
            power = scipy.integrate.simpson(y=powerSpectrumDensity[idx_band], dx=dFreq, even='simpson', axis=-1)

            if relative:
                power /= scipy.integrate.simpson(y=powerSpectrumDensity, dx=dFreq, even='simpson', axis=-1)

            bandPowers.append(power)

        return bandPowers

    @staticmethod
    def spectral_entropy(powerSpectrumDensityNormalized, normalizePSD=True, tolerance=1e-15):
        # Normalize the power spectrum density
        if normalizePSD: powerSpectrumDensityNormalized = universalMethods.normalizePSD(powerSpectrumDensityNormalized)

        # If the power spectrum density is zero, the spectral entropy is zero.
        if np.all(powerSpectrumDensityNormalized == 0):
            return 0

        # Calculate the spectral entropy
        spectralEntropy = -np.sum(np.multiply(powerSpectrumDensityNormalized, np.log2(powerSpectrumDensityNormalized + tolerance)), axis=-1)

        return spectralEntropy

    @staticmethod
    def normalizePSD(powerSpectrumDensity):
        # Normalize the power spectrum density.
        powerSpectrumDensity = np.asarray(powerSpectrumDensity)
        normalizationFactor = powerSpectrumDensity[1:].sum()

        if normalizationFactor == 0:
            return np.zeros_like(powerSpectrumDensity)

        return powerSpectrumDensity / np.sum(powerSpectrumDensity[1:])

    # ------------------------- Formatting Arrays  ------------------------- #

    @staticmethod
    def getEvenlySampledArray(sampling_frequency, num_points, start_time=0):
        end_time = (num_points - 1) / sampling_frequency
        time_array = np.arange(start_time, end_time + 1 / sampling_frequency, 1 / sampling_frequency)
        return time_array

    # ------------------------ Numeric Information  ------------------------ #

    @staticmethod
    def isNumber(string):
        return bool(re.match(r'^[-+]?\d*\.?\d+$', string))

    def convertToOddInt_Positive(self, x):
        return self.convertToOddInt(x, minInt=1)

    @staticmethod
    def convertToOddInt(x, minInt=None, maxInt=None):
        # Round to the nearest odd integer.
        oddInt = 2 * round((x + 1) / 2) - 1

        # Bound the odd integer.
        if minInt is not None:
            assert minInt % 2 == 1
            oddInt = max(oddInt, minInt)
        if minInt is not None:
            assert maxInt % 2 == 1
            oddInt = min(oddInt, maxInt)

        return oddInt

    # -------------------------- Formatting Lines -------------------------- #

    @staticmethod
    def findPointCrossing(array, threshold=0):
        # Shift the threshold to zero
        array = np.array(array) - threshold

        # Return the value right BEFORE the zero crossing.
        return np.where(np.diff(np.sign(array) >= 0))[0]

    @staticmethod
    def findLineIntersectionPoint(leftLineParams, rightLineParams):
        """
        Parameters
        ----------
        leftLineParams: A list of length 2 representing the [slope, y-intercept] ([m, b] in y = mx + b)
        rightLineParams: A list of length 2 representing the [slope, y-intercept] ([m, b] in y = mx + b)
        """
        xPoint = (rightLineParams[1] - leftLineParams[1]) / (leftLineParams[0] - rightLineParams[0])
        yPoint = leftLineParams[0] * xPoint + leftLineParams[1]
        return xPoint, yPoint

    # ---------------------------------------------------------------------- #
    # ------------------------ Find Peak Information ----------------------- #

    def findNearbyMinimum(self, data, xPointer, binarySearchWindow=5, maxPointsSearch=10000):
        """
        Search Right: binarySearchWindow > zero
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            searchSegment = data[max(0, xPointer - 1):min(xPointer + 2, len(data))]
            xPointer -= np.where(searchSegment == data[xPointer])[0][0]
            return xPointer + np.argmin(searchSegment)

        maxHeightPointer = xPointer
        maxHeight = data[xPointer]
        searchDirection = binarySearchWindow // abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(max(xPointer, 0), max(0, min(xPointer + searchDirection * maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] >= maxHeight and xPointer != dataPointer:
                return self.findNearbyMinimum(data, dataPointer - binarySearchWindow, round(binarySearchWindow / 4), maxPointsSearch - searchDirection * (abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                maxHeightPointer = dataPointer
                maxHeight = data[dataPointer]

        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMinimum(data, maxHeightPointer, round(binarySearchWindow / 2), maxPointsSearch - 1)

    def findNearbyMaximum(self, data, xPointer, binarySearchWindow=5, maxPointsSearch=10000):
        """
        Search Right: binarySearchWindow > zero
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        xPointer = min(max(xPointer, 0), len(data) - 1)
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            searchSegment = data[max(0, xPointer - 1):min(xPointer + 2, len(data))]
            xPointer -= np.where(searchSegment == data[xPointer])[0][0]
            return xPointer + np.argmax(searchSegment)

        minHeightPointer = xPointer
        minHeight = data[xPointer]
        searchDirection = binarySearchWindow // abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(xPointer, max(0, min(xPointer + searchDirection * maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] < minHeight and xPointer != dataPointer:
                return self.findNearbyMaximum(data, dataPointer - binarySearchWindow, round(binarySearchWindow / 2), maxPointsSearch - searchDirection * (abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                minHeightPointer = dataPointer
                minHeight = data[dataPointer]

        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMaximum(data, minHeightPointer, round(binarySearchWindow / 2), maxPointsSearch - 1)

    def findRightMinMax(self, data, xPointer, binarySearchWindow=5, maxPointsSearch=10000):
        rightMinimum = self.findNearbyMinimum(data, xPointer, binarySearchWindow, maxPointsSearch)
        rightMaximum = self.findNearbyMaximum(data, xPointer, binarySearchWindow, maxPointsSearch)

        if rightMinimum < rightMaximum:
            return rightMaximum, 1
        else:
            return rightMinimum, -1

    def localOptimization(self, data, points, localType, binarySearchWindow, maxPointsSearch):
        optimizedPoints = []
        if localType == 'min':
            for point in points:
                left = self.findNearbyMinimum(data, point, binarySearchWindow * -1, maxPointsSearch)
                right = self.findNearbyMinimum(data, point, binarySearchWindow, maxPointsSearch)
                if data[left] < data[right]:
                    optimizedPoints.append(left)
                else:
                    optimizedPoints.append(right)
        elif localType == 'max':
            for point in points:
                left = self.findNearbyMaximum(data, point, binarySearchWindow * -1, maxPointsSearch)
                right = self.findNearbyMaximum(data, point, binarySearchWindow, maxPointsSearch)
                if data[left] > data[right]:
                    optimizedPoints.append(left)
                else:
                    optimizedPoints.append(right)
        else:
            return points

        return optimizedPoints

    def findPrevBaselinePointer(self, data, xPointer, binarySearchWindow, samplingFreq, peakHeight_Threshold):
        currBaseLinePointer = xPointer
        newBaseLinePointer = self.findNearbyMinimum(data, currBaseLinePointer, binarySearchWindow, maxPointsSearch=int(samplingFreq * 30))
        print(binarySearchWindow)
        # print(currBaseLinePointer, newBaseLinePointer)
        # print(data[currBaseLinePointer] - data[newBaseLinePointer])

        while currBaseLinePointer > newBaseLinePointer > 0 and data[currBaseLinePointer] - data[newBaseLinePointer] > peakHeight_Threshold:
            currBaseLinePointer = newBaseLinePointer
            newBaseLinePointer = self.findNearbyMinimum(data, currBaseLinePointer, binarySearchWindow, maxPointsSearch=int(samplingFreq * 30))
            # print('walking')
            # print(data[currBaseLinePointer] - data[newBaseLinePointer])
        return currBaseLinePointer

    def findLocalMax(self, data, xPointer, binarySearchWindow=5, maxPointsSearch=10000):
        # Find the local maximum to the left and the right.
        localMax = self.findNearbyMaximum(data, xPointer, binarySearchWindow, maxPointsSearch)
        localMax = self.findNearbyMaximum(data, localMax, -binarySearchWindow, maxPointsSearch)

        return localMax
