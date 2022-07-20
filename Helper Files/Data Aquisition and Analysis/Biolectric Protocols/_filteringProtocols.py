
"""
The following SVD filtering is a modified version of the code:
    https://github.com/nerdull/denoise/blob/0b5d92f446059a70d106f0abf13689dcd40ef89b/denoise.py
Further details of the method can be found here:
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.063320
With a citation:
    X.C. Chen et al., Phys. Rev. E 99, 063320 (2019).
    
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import math
import numpy as np
from scipy.linalg import svd
# Filtering Modules
import scipy
from scipy.signal import butter
from scipy.signal import lfilter
from scipy.signal import savgol_filter
# Fourier Transform Modules
from scipy.fft import rfft,rfftfreq
from scipy.fft import irfft

# -------------------------------------------------------------------------- #
# ------------------------- Filtering Methods Head ------------------------- #

class filteringMethods:

    def __init__(self):
        # Initiate Different Filtering Methods.
        self.bandPassFilter = bandPassFilter()
        self.fourierFilter = fourierFilter()
        self.filterSVD = Denoiser()
        self.savgolFilter = savgolFilter()

# -------------------------------------------------------------------------- #
# ------------------ High/Low/Band Pass Filtering Methods ------------------ #

class bandPassFilter:
    
    def butterParams(self, cutoffFreq = [0.1, 7], samplingFreq = 800, order=3, filterType = 'band'):
        nyq = 0.5 * samplingFreq
        if filterType == "band":
            normal_cutoff = [freq/nyq for freq in cutoffFreq]
        else:
            normal_cutoff = cutoffFreq / nyq
        sos = butter(order, normal_cutoff, btype = filterType, analog = False, output='sos')
        return sos
    
    def butterFilter(self, data, cutoffFreq, samplingFreq, order = 3, filterType = 'band'):
        sos = self.butterParams(cutoffFreq, samplingFreq, order, filterType)
        return scipy.signal.sosfiltfilt(sos, data)
    
    def highPassFilter(self, inputData, Wp, Ws, Rp, Rs):
        """
        data: Data to Filter
        f1: cutOffFreqPassThrough
        f3: cutOffFreqBand
        Rp: attDB (0.1)
        Rs: cutOffDB (30)
        samplingFreq: Frequecy You Take Data
        """            
        [n, wn] = scipy.signal.cheb1ord(Wp/math.pi, Ws/math.pi, Rp, Rs)
        [bz1, az1] = scipy.signal.cheby1(n, Rp, Wp/math.pi, 'High')
        filteredData = lfilter(bz1, az1, inputData)
        return filteredData
    
# -------------------------------------------------------------------------- #
# ------------------- Fourier Transform Filtering Methods ------------------ #

class fourierFilter:
    
    def removeFrequencies(self, f_noise, samplingFreq, cutoffFreq = [0.5, 10]):
        # Prepend the Data with Zeros to be length 2**N for N = 1,2,3,4...
        closestPowerOfTwo = 2**(math.ceil(math.log(len(f_noise))/math.log(2)))
        numZerosToPad = closestPowerOfTwo - len(f_noise)
        f_noisePadded = [0]*numZerosToPad
        f_noisePadded.extend(f_noise)
        # Extra Padding: Mirror the Data on Both Sides
        f_noisePadded.extend(f_noisePadded[::-1])
        # Tranform the Data into the Frequency Domain
        n    = len(f_noisePadded)
        yf   = rfft(f_noisePadded)
        xf   = rfftfreq(n, 1/samplingFreq)
        # Remove the Frequencies Outside the Range
        indices     = np.logical_and(cutoffFreq[0] < xf, xf < cutoffFreq[1])
        yf_clean    = indices * yf # noise frequency will be set to 0
        # Reconstruct the Signal and Return the Data
        return irfft(yf_clean)[numZerosToPad:numZerosToPad+len(f_noise)]

# -------------------------------------------------------------------------- #
# ------------------------ Savgol Filtering Methods ------------------------ #

class savgolFilter:
    
    def savgolFilter(self, noisyData, window_length, polyorder, deriv = 0, mode='nearest'):
        return savgol_filter(noisyData, window_length, polyorder, deriv = deriv)
    
# -------------------------------------------------------------------------- #
# -------------------------- SVD Filtering Methods ------------------------- #

class Denoiser:
    '''
    A class for smoothing a noisy, real-valued data sequence by means of SVD of a partial circulant matrix.
    -----
    Attributes:
        mode: str
            Code running mode: "layman" or "expert".
            In the "layman" mode, the code autonomously tries to find the optimal denoised sequence.
            In the "expert" mode, a user has full control over it.
        s: 1D array of floats
            Singular values ordered decreasingly.
        U: 2D array of floats
            A set of left singular vectors as the columns.
        r: int
            Rank of the approximating matrix of the constructed partial circulant matrix from the sequence.
    '''

    def __init__(self, mode="program"):
        '''
        Class initialization.
        -----
        Arguments:
            mode: str
                Denoising mode. To be selected from ["layman", "expert", "program"]. Default is "program".
                While "layman" grants the code autonomy, "expert" allows a user to experiment.
        -----
        Raises:
            ValueError
                If mode is neither "layman" nor "expert".
        '''
        self._method = {"program": self._denoise_for_consistency, "layman": self._denoise_for_layman, "expert": self._denoise_for_expert}
        if mode not in self._method:
            raise ValueError("unknown mode '{:s}'!".format(mode))
        self.mode = mode

    def _embed(self, x, m):
        '''
        Embed a 1D array into a 2D partial circulant matrix by cyclic left-shift.
        -----
        Arguments:
            x: 1D array of floats
                Input array.
            m: int
                Number of rows of the constructed matrix.
        -----
        Returns:
            X: 2D array of floats
                Constructed partial circulant matrix.
        '''
        x_ext = np.hstack((x, x[:m-1]))
        shape = (m, x.size)
        strides = (x_ext.strides[0], x_ext.strides[0])
        X = np.lib.stride_tricks.as_strided(x_ext, shape, strides)
        return X

    def _reduce(self, A):
        '''
        Reduce a 2D matrix to a 1D array by cyclic anti-diagonal average.
        -----
        Arguments:
            A: 2D array of floats
                Input matrix.
        -----
        Returns:
            a: 1D array of floats
                Output array.
        '''
        m = A.shape[0]
        A_ext = np.hstack((A[:,-m+1:], A))
        strides = (A_ext.strides[0]-A_ext.strides[1], A_ext.strides[1])
        a = np.mean(np.lib.stride_tricks.as_strided(A_ext[:,m-1:], A.shape, strides), axis=0)
        return a

    def _denoise_for_expert(self, sequence, layer, gap, rank):
        '''
        Smooth a noisy sequence by means of low-rank approximation of its corresponding partial circulant matrix.
        -----
        Arguments:
            sequence: 1D array of floats
                Data sequence to be denoised.
            layer: int
                Number of leading rows selected from the matrix.
            gap: float
                Gap between the data levels on the left and right ends of the sequence.
                A positive value means the right level is higher.
            rank: int
                Rank of the approximating matrix.
        -----
        Returns:
            denoised: 1D array of floats
                Smoothed sequence after denoise.
        -----
        Raises:
            AssertionError
                If condition 1 <= rank <= layer <= sequence.size cannot be fulfilled.
        '''
        assert 1 <= rank <= layer <= sequence.size
        self.r = rank
        # linear trend to be deducted
        trend = np.linspace(0, gap, sequence.size)
        X = self._embed(sequence-trend, layer)
        # singular value decomposition
        self.U, self.s, Vh = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)
        # low-rank approximation
        A = self.U[:,:self.r] @ np.diag(self.s[:self.r]) @ Vh[:self.r]
        denoised = self._reduce(A) + trend
        return denoised

    def _cross_validate(self, x, m):
        '''
        Check if the gap of boundary levels of the detrended sequence is within the estimated noise strength.
        -----
        Arguments:
            x: 1D array of floats
                Input array.
            m: int
                Number of rows of the constructed matrix.
        -----
        Returns:
            valid: bool
                Result of cross validation. True means the detrending procedure is valid.
        '''
        X = self._embed(x, m)
        self.U, self.s, self._Vh = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)
        # Search for noise components using the normalized mean total variation of the left singular vectors as an indicator.
        # The procedure runs in batch of every 10 singular vectors.
        self.r = 0
        while True:
            U_sub = self.U[:,self.r:self.r+10]
            NMTV = np.mean(np.abs(np.diff(U_sub,axis=0)), axis=0) / (np.amax(U_sub,axis=0) - np.amin(U_sub,axis=0))
            try:
                # the threshold of 10% can in most cases discriminate noise components
                self.r += np.argwhere(NMTV > .1)[0,0]
                break
            except IndexError:
                self.r += 10
        # estimate the noise strength, while r marks the first noise component
        noise_stdev = np.sqrt(np.sum(self.s[self.r:]**2) / X.size)
        # estimate the gap of boundary levels after detrend
        gap = np.abs(x[-self._k:].mean()-x[:self._k].mean())
        valid = gap < noise_stdev
        return valid

    def _denoise_for_layman(self, sequence, layer):
        '''
        Similar to the "expert" method, except that denoising parameters are optimized autonomously.
        -----
        Arguments:
            sequence: 1D array of floats
                Data sequence to be denoised.
            layer: int
                Number of leading rows selected from the corresponding circulant matrix.
        -----
        Returns:
            denoised: 1D array of floats
                Smoothed sequence after denoise.
        -----
        Raises:
            AssertionError
                If condition 1 <= layer <= sequence.size cannot be fulfilled.
        '''
        assert 1 <= layer <= sequence.size
        # The code takes the mean of a few neighboring data to estimate the boundary levels of the sequence.
        # By default, this number is 11.
        self._k = 11
        # Initially, the code assumes no linear inclination.
        trend = np.zeros_like(sequence)
        # Iterate over the averaging length.
        # In the worst case, iteration must terminate when it is 1.
        while not self._cross_validate(sequence-trend, layer):
            self._k -= 2
            trend = np.linspace(0, sequence[-self._k:].mean()-sequence[:self._k].mean(), sequence.size)
        # low-rank approximation by using only signal components
        A = self.U[:,:self.r] @ np.diag(self.s[:self.r]) @ self._Vh[:self.r]
        denoised = self._reduce(A) + trend
        return denoised
    
    
    def _denoise_for_consistency(self, sequence, layer, k = 20):
        '''
        Similar to the "expert" method, except that denoising parameters are optimized autonomously.
        -----
        Arguments:
            sequence: 1D array of floats
                Data sequence to be denoised.
            layer: int
                Number of leading rows selected from the corresponding circulant matrix.
        -----
        Returns:
            denoised: 1D array of floats
                Smoothed sequence after denoise.
        -----
        Raises:
            AssertionError
                If condition 1 <= layer <= sequence.size cannot be fulfilled.
        '''
        assert 1 <= layer <= sequence.size
        # The code takes the mean of a few neighboring data to estimate the boundary levels of the sequence.
        self._k = k
        # Initially, the code assumes no linear inclination.
        trend = np.linspace(0, sequence[-self._k:].mean()-sequence[:self._k].mean(), sequence.size)
        
        self._cross_validate(sequence-trend, layer)

        # low-rank approximation by using only signal components
        A = self.U[:,:self.r] @ np.diag(self.s[:self.r]) @ self._Vh[:self.r]
        denoised = self._reduce(A) + trend
        return denoised
    
    def _denoise_for_consisten1cy(self, sequence, layer, k = 11, r = 20):
        '''
        Similar to the "expert" method, except that denoising parameters are optimized autonomously.
        -----
        Arguments:
            sequence: 1D array of floats
                Data sequence to be denoised.
            layer: int
                Number of leading rows selected from the corresponding circulant matrix.
        -----
        Returns:
            denoised: 1D array of floats
                Smoothed sequence after denoise.
        -----
        Raises:
            AssertionError
                If condition 1 <= layer <= sequence.size cannot be fulfilled.
        '''
        assert 1 <= layer <= sequence.size
        # The code takes the mean of a few neighboring data to estimate the boundary levels of the sequence.
        self._k = k
        self.r = r
        # Initially, the code assumes no linear inclination.
        trend = np.linspace(0, sequence[-self._k:].mean()-sequence[:self._k].mean(), sequence.size)
        
        # Cross Validate
        X = self._embed(sequence - trend, layer)
        self.U, self.s, self._Vh = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)

        # low-rank approximation by using only signal components
        A = self.U[:,:self.r] @ np.diag(self.s[:self.r]) @ self._Vh[:self.r]
        denoised = self._reduce(A) + trend
        return denoised

    def denoise(self, *args, **kwargs):
        '''
        User interface method.
        It will reference to different denoising methods ad hoc under the fixed name.
        '''
        return self._method[self.mode](*args, **kwargs)


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    signal = np.sinc(x)
    noise = np.random.normal(scale=.1, size=1000)
    sequence = signal + noise
    denoiser = Denoiser()
    denoised = denoiser.denoise(sequence, 200)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x, sequence)
    ax.plot(x, signal)
    ax.plot(x, denoised)
    plt.show()