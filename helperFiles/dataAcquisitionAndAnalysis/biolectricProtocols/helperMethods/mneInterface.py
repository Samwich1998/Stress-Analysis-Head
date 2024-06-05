import mne_features
import numpy as np


class mneInterface:

    def __init__(self):
        # General parameters.
        self.samplingFreq = None  # The Average Number of Points Steamed Into the Arduino Per Second; Depends on the User's Hardware; If NONE Given, Algorithm will Calculate Based on Initial Data

        # MNE-features parameter information.
        self.allFeaturesMNE = ['app_entropy', 'decorr_time', 'energy_freq_bands', 'higuchi_fd', 'hjorth_complexity', 'hjorth_complexity_spect', 'hjorth_mobility', 'hjorth_mobility_spect',
                               'hurst_exp', 'katz_fd', 'kurtosis', 'line_length', 'mean', 'pow_freq_bands', 'ptp_amp', 'quantile', 'rms', 'samp_entropy', 'skewness', 'spect_edge_freq',
                               'spect_entropy', 'spect_slope', 'std', 'svd_entropy', 'svd_fisher_info', 'teager_kaiser_energy', 'variance', 'wavelet_coef_energy', 'zero_crossings']
        self.kindaSlowFeaturesMNE = ['hjorth_complexity_spect', 'hjorth_mobility_spect', 'kurtosis', 'pow_freq_bands', 'skewness', 'spect_edge_freq', 'spect_entropy', 'spect_slope']
        self.wayTooSlowFeaturesMNE = ['app_entropy', 'energy_freq_bands', 'hurst_exp', 'samp_entropy', 'svd_entropy', 'svd_fisher_info']

        # MNE-features parameters.
        self.frequencyBands = np.array([(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100), (8, 13), (13, 16), (16, 20), (20, 28), (13, 15)])
        self.featuresMNE = ['decorr_time', 'energy_freq_bands', 'higuchi_fd', 'hjorth_complexity', 'hjorth_complexity_spect', 'hjorth_mobility', 'hjorth_mobility_spect',
                            'katz_fd', 'line_length', 'pow_freq_bands', 'ptp_amp', 'rms', 'spect_edge_freq', 'spect_entropy', 'spect_slope',
                            'teager_kaiser_energy', 'variance', 'wavelet_coef_energy', 'zero_crossings']

        self.paramsMNE = {
            # app_entropy parameters.
            'app_entropy__metric': 'chebyshev',
            'app_entropy__emb': 2,
            # samp_entropy parameters.
            'samp_entropy__metric': 'chebyshev',
            'samp_entropy__emb': 2,
            # pow_freq_bands parameters.
            'pow_freq_bands__psd_func': {'welch_n_per_seg': 0, 'welch_n_fft': 0, 'welch_n_overlap': 0},
            'pow_freq_bands__freq_bands': self.frequencyBands,
            'pow_freq_bands__psd_method': 'welch',
            'pow_freq_bands__ratios_triu': False,
            'pow_freq_bands__normalize': True,
            'pow_freq_bands__ratios': None,
            'pow_freq_bands__log': False,
            # hjorth_mobility_spect parameters.
            'hjorth_mobility_spect__psd_params': {'welch_n_per_seg': 0, 'welch_n_fft': 0, 'welch_n_overlap': 0},
            'hjorth_mobility_spect__psd_method': 'welch',
            'hjorth_mobility_spect__normalize': True,
            # hjorth_complexity_spect parameters.
            'hjorth_complexity_spect__psd_params': {'welch_n_per_seg': 0, 'welch_n_fft': 0, 'welch_n_overlap': 0},
            'hjorth_complexity_spect__psd_method': 'welch',
            'hjorth_complexity_spect__normalize': True,
            # higuchi_fd parameters.
            'higuchi_fd__kmax': 10,
            # zero_crossings parameters.
            'zero_crossings__threshold': 2.220446049250313e-16,
            # spect_slope parameters.
            'spect_slope__psd_params': {'welch_n_per_seg': 0, 'welch_n_fft': 0, 'welch_n_overlap': 0},
            'spect_slope__with_intercept': True,
            'spect_slope__psd_method': 'welch',
            'spect_slope__fmax': 100,
            'spect_slope__fmin': 0.5,
            # spect_entropy parameters.
            'spect_entropy__psd_params': {'welch_n_per_seg': 0, 'welch_n_fft': 0, 'welch_n_overlap': 0},
            'spect_entropy__psd_method': 'welch',
            # svd_entropy parameters.
            'svd_entropy__emb': 10,
            'svd_entropy__tau': 1,
            # svd_fisher_info parameters.
            'svd_fisher_info__emb': 10,
            'svd_fisher_info__tau': 1,
            # energy_freq_bands parameters.
            'energy_freq_bands__freq_bands': self.frequencyBands,
            'energy_freq_bands__deriv_filt': True,
            # spect_edge_freq parameters.
            'spect_edge_freq__psd_params': {'welch_n_per_seg': 0, 'welch_n_fft': 0, 'welch_n_overlap': 0},
            'spect_edge_freq__psd_method': 'welch',
            'spect_edge_freq__ref_freq': None,
            'spect_edge_freq__edge': None,
            # wavelet_coef_energy parameters.
            'wavelet_coef_energy__wavelet_name': 'db3',
            # teager_kaiser_energy parameters.
            'teager_kaiser_energy__wavelet_name': 'db3',
        }

        # Remove the features that are not in the selected features
        self.paramsMNE = {key: value for key, value in self.paramsMNE.items() if key.split("__")[0] in self.featuresMNE}

        # MNE-features individual parameters.
        self.psd_params = {}

    def setSamplingFrequencyParams(self, samplingFreq):
        # Set the sampling frequency
        self.samplingFreq = samplingFreq

        # Set the MNE-Features parameters
        n_per_seg = int(samplingFreq * 4)
        n_fft = int(samplingFreq * 8)
        n_overlap = n_per_seg // 2

        # Reset the feature extraction parameters: n_per_seg
        self.paramsMNE['hjorth_complexity_spect__psd_params']['welch_n_per_seg'] = n_per_seg
        self.paramsMNE['hjorth_mobility_spect__psd_params']['welch_n_per_seg'] = n_per_seg
        self.paramsMNE['spect_edge_freq__psd_params']['welch_n_per_seg'] = n_per_seg
        self.paramsMNE['pow_freq_bands__psd_params']['welch_n_per_seg'] = n_per_seg
        self.paramsMNE['spect_entropy__psd_params']['welch_n_per_seg'] = n_per_seg
        self.paramsMNE['spect_slope__psd_params']['welch_n_per_seg'] = n_per_seg

        # Reset the feature extraction parameters: n_fft
        self.paramsMNE['hjorth_complexity_spect__psd_params']['welch_n_fft'] = n_fft
        self.paramsMNE['hjorth_mobility_spect__psd_params']['welch_n_fft'] = n_fft
        self.paramsMNE['spect_edge_freq__psd_params']['welch_n_fft'] = n_fft
        self.paramsMNE['pow_freq_bands__psd_params']['welch_n_fft'] = n_fft
        self.paramsMNE['spect_entropy__psd_params']['welch_n_fft'] = n_fft
        self.paramsMNE['spect_slope__psd_params']['welch_n_fft'] = n_fft

        # Reset the feature extraction parameters: overlap
        self.paramsMNE['hjorth_complexity_spect__psd_params']['welch_n_overlap'] = n_overlap
        self.paramsMNE['hjorth_mobility_spect__psd_params']['welch_n_overlap'] = n_overlap
        self.paramsMNE['spect_edge_freq__psd_params']['welch_n_overlap'] = n_overlap
        self.paramsMNE['pow_freq_bands__psd_params']['welch_n_overlap'] = n_overlap
        self.paramsMNE['spect_entropy__psd_params']['welch_n_overlap'] = n_overlap
        self.paramsMNE['spect_slope__psd_params']['welch_n_overlap'] = n_overlap

        # Reset the feature extraction parameters: individual
        self.psd_params = {'welch_n_per_seg': int(self.samplingFreq * 4), 'welch_n_fft': int(self.samplingFreq * 8), 'welch_n_overlap': 0}

    # --------------------- Feature Extraction Methods --------------------- #

    def extractFeatures(self, data, standardizedData, normalizedData):
        # Specify the general parameters.
        expanded_data = np.expand_dims(data, axis=0)

        # Fast singular feature extraction that is amplitude-invariant.
        zero_crossings = mne_features.univariate.compute_zero_crossings(standardizedData, threshold=2.220446049250313e-16)[0]  # Must be mean centered to be amplitude-invariant. Averages 33.2 μs.
        decorr_time = mne_features.univariate.compute_decorr_time(self.samplingFreq, expanded_data)[0]  # Amplitude-invariant. Averages 89.3 μs.
        higuchi_fd = mne_features.univariate.compute_higuchi_fd(expanded_data, kmax=10)[0]  # Amplitude-invariant. Averages 12 μs.
        katz_fd = mne_features.univariate.compute_katz_fd(expanded_data)[0]  # Amplitude-invariant. Averages 37 μs.
        # Fast singular feature extraction that is amplitude-linear-dependant.
        line_length = mne_features.univariate.compute_line_length(standardizedData)[0]  # Amplitude-linear-dependant when standardized. Averages 16.9 μs.
        ptp_amp = mne_features.univariate.compute_ptp_amp(standardizedData)[0]  # Amplitude-linear-dependant when standardized. Averages 5.25 μs.
        # Fast singular feature extraction that is amplitude-dependant.
        rms = mne_features.univariate.compute_rms(normalizedData)[0]  # Amplitude-dependant. Averages 30.4 μs.

        svd_entropy = mne_features.univariate.compute_svd_entropy(expanded_data, tau=2, emb=10)[0]

        # Slow singular feature extraction: averages 3.4 ms.
        spect_edge_freq = mne_features.univariate.compute_spect_edge_freq(self.samplingFreq, expanded_data, ref_freq=None, edge=None, psd_method='welch', psd_params=self.psd_params)[0]  # Averages 763 μs.
        spect_entropy = mne_features.univariate.compute_spect_entropy(self.samplingFreq, expanded_data, psd_method='welch', psd_params=self.psd_params)[0]  # Averages 739 μs.
        # Slow compact feature extraction.
        pow_freq_bands = mne_features.univariate.compute_pow_freq_bands(self.samplingFreq, expanded_data, freq_bands=self.frequencyBands, normalize=True, ratios=None,
                                                                        ratios_triu=False, psd_method='welch', log=False, psd_params=self.psd_params)  # Averages 865 μs.
        # pow_freq_bands dimension: len(frequencyBands)

        # Extremely slow compact feature extraction.
        energy_freq_bands = mne_features.univariate.compute_energy_freq_bands(self.samplingFreq, expanded_data, freq_bands=self.frequencyBands, deriv_filt=True)  # Averages 20.9 ms.

        # Unpack compressed features.

        return decorr_time, energy_freq_bands, higuchi_fd, katz_fd, line_length, pow_freq_bands, ptp_amp, rms, spect_edge_freq, spect_entropy, zero_crossings
