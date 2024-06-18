# Basic Modules
import math
import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq, irfft
from scipy.linalg import svd


# ------------------------- Filtering Methods Head ------------------------- #

class filteringMethods:

    def __init__(self):
        # Initiate Different Filtering Methods.
        self.bandPassFilter = bandPassFilter()
        self.fourierFilter = fourierFilter()
        self.filterSVD = Denoiser()
        self.savgolFilter = savgolFilter()


# ------------------ High/Low/Band Pass Filtering Methods ------------------ #

class bandPassFilter:

    @staticmethod
    def butterFilter(data, cutoffFreq=(0.1, 7), samplingFreq=800, order=3, filterType='bandpass', fastFilt=True):
        """
        Apply a Butterworth filter to a signal.

        Parameters
        ----------
        data : ndarray
            Input signal to be filtered.
        cutoffFreq : list of float
            Cutoff frequencies of the filter. If filterType is "band", this should be a list of two frequencies.
            Otherwise, this should be a single frequency. Default is [0.1, 7].
        samplingFreq : float
            Sampling frequency of the signal. Default is 800.
        order : int
            Order of the filter. Default is 3.
        filterType : str
            Type of filter. "low", "high", "bandpass", or "notch". Default is "bandpass".

        Returns
        -------
        filteredData : ndarray
            Output filtered signal.
        """
        # If no data to filter, return data
        if cutoffFreq is None:
            return data

        nyq = 0.5 * samplingFreq
        if filterType == "bandpass" and len(cutoffFreq) != 2:
            raise ValueError("cutoffFreq must be a list of two frequencies for bandpass or bandstop filters.")
        normal_cutoff = np.asarray(cutoffFreq) / nyq

        if fastFilt:
            sos = scipy.signal.butter(order, normal_cutoff, btype=filterType, analog=False, output='sos')
            filteredData = scipy.signal.sosfiltfilt(sos, data)
        else:
            b, a = scipy.signal.butter(order, normal_cutoff, btype=filterType, analog=False, output='ba')
            filteredData = scipy.signal.filtfilt(b, a, data)

        return filteredData

    @staticmethod
    def high_pass_filter(data_to_filter, sampling_freq, passband_edge, stopband_edge, passband_ripple, stopband_attenuation, fastFilt=True):
        """
        Applies a Chebyshev type I high-pass filter to the input data.
    
        Parameters:
        -----------
        data_to_filter : array-like
            Input data to filter.
        passband_edge : float
            Passband-edge frequency in Hz.
        stopband_edge : float
            Stopband-edge frequency in Hz.
        passband_ripple : float
            Maximum allowed passband ripple in decibels.
        stopband_attenuation : float
            Minimum required stopband attenuation in decibels.
    
        Returns:
        --------
        filtered_data : array-like
            Filtered data.
        """
        # If no data to filter, return data
        if passband_edge is None:
            return data_to_filter

        # Calculate filter order and cutoff frequency
        nyq_freq = 0.5 * sampling_freq
        Wp = passband_edge / nyq_freq
        Ws = stopband_edge / nyq_freq
        n, wn = scipy.signal.cheb1ord(Wp, Ws, passband_ripple, stopband_attenuation)

        # Design filter and apply to data
        bz, az = scipy.signal.cheby1(n, passband_ripple, Wp, 'highpass')
        if fastFilt:
            filtered_data = scipy.signal.lfilter(bz, az, data_to_filter)
        else:
            filtered_data = scipy.signal.filtfilt(bz, az, data_to_filter)

        return filtered_data


# ------------------- Fourier Transform Filtering Methods ------------------ #

class fourierFilter:

    @staticmethod
    def removeFrequencies(f_noise, samplingFreq, cutoffFreq=(0.5, 10)):
        # Prepend the Data with Zeros to be length 2**N for N = 1,2,3,4...
        closestPowerOfTwo = 2 ** (math.ceil(math.log(len(f_noise)) / math.log(2)))
        numZerosToPad = closestPowerOfTwo - len(f_noise)
        f_noisePadded = [0] * numZerosToPad
        f_noisePadded.extend(f_noise)
        # Extra Padding: Mirror the Data on Both Sides
        f_noisePadded.extend(f_noisePadded[::-1])
        # Tranform the Data into the Frequency Domain
        n = len(f_noisePadded)
        yf = rfft(f_noisePadded)
        xf = rfftfreq(n, 1 / samplingFreq)
        # Remove the Frequencies Outside the Range
        indices = np.logical_and(cutoffFreq[0] < xf, xf < cutoffFreq[1])
        yf_clean = indices * yf  # noise frequency will be set to 0
        # Reconstruct the Signal and Return the Data
        return irfft(yf_clean)[numZerosToPad:numZerosToPad + len(f_noise)]


# ------------------------ Savgol Filtering Methods ------------------------ #

class savgolFilter:

    @staticmethod
    def savgolFilter(noisyData, window_length, polyorder, deriv=0, mode='interp'):
        return scipy.signal.savgol_filter(noisyData, window_length, polyorder, mode=mode, deriv=deriv)


# -------------------------- SVD Filtering Methods ------------------------- #

class Denoiser:
    """
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
    """

    def __init__(self, mode="program"):
        """
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
        """
        self._method = {"program": self._denoise_for_consistency, "layman": self._denoise_for_layman, "expert": self._denoise_for_expert}
        if mode not in self._method:
            raise ValueError("unknown mode '{:s}'!".format(mode))
        self.mode = mode

    @staticmethod
    def _embed(x, m):
        """
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
        """
        x_ext = np.hstack((x, x[:m - 1]))
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
        A_ext = np.hstack((A[:, -m + 1:], A))
        strides = (A_ext.strides[0] - A_ext.strides[1], A_ext.strides[1])
        a = np.mean(np.lib.stride_tricks.as_strided(A_ext[:, m - 1:], A.shape, strides), axis=0)
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
        X = self._embed(sequence - trend, layer)
        # singular value decomposition
        self.U, self.s, Vh = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)
        # low-rank approximation
        A = self.U[:, :self.r] @ np.diag(self.s[:self.r]) @ Vh[:self.r]
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
            U_sub = self.U[:, self.r:self.r + 10]
            NMTV = np.mean(np.abs(np.diff(U_sub, axis=0)), axis=0) / (np.amax(U_sub, axis=0) - np.amin(U_sub, axis=0))
            try:
                # the threshold of 10% can in most cases discriminate noise components
                self.r += np.argwhere(NMTV > .1)[0, 0]
                break
            except IndexError:
                self.r += 10
        # estimate the noise strength, while r marks the first noise component
        noise_stdev = np.sqrt(np.sum(self.s[self.r:] ** 2) / X.size)
        # estimate the gap of boundary levels after detrend
        gap = np.abs(x[-self._k:].mean() - x[:self._k].mean())
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
        while not self._cross_validate(sequence - trend, layer):
            self._k -= 2
            trend = np.linspace(0, sequence[-self._k:].mean() - sequence[:self._k].mean(), sequence.size)
        # low-rank approximation by using only signal components
        A = self.U[:, :self.r] @ np.diag(self.s[:self.r]) @ self._Vh[:self.r]
        denoised = self._reduce(A) + trend
        return denoised

    def _denoise_for_consistency(self, sequence, layer, k=20):
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
        trend = np.linspace(0, sequence[-self._k:].mean() - sequence[:self._k].mean(), sequence.size)

        self._cross_validate(sequence - trend, layer)

        # low-rank approximation by using only signal components
        A = self.U[:, :self.r] @ np.diag(self.s[:self.r]) @ self._Vh[:self.r]
        denoised = self._reduce(A) + trend
        return denoised

    def _denoise_for_consisten1cy(self, sequence, layer, k=11, r=20):
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
        trend = np.linspace(0, sequence[-self._k:].mean() - sequence[:self._k].mean(), sequence.size)

        # Cross Validate
        X = self._embed(sequence - trend, layer)
        self.U, self.s, self._Vh = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)

        # low-rank approximation by using only signal components
        A = self.U[:, :self.r] @ np.diag(self.s[:self.r]) @ self._Vh[:self.r]
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
