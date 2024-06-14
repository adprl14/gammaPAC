'''
Code for gamma GLM PAC method

author: Andrew Perley
contact: aperley@stanford.edu

Note: time_segment in fit() might need to be changed to be more intuitive/robust
- i.e., it makes self.phase/amplitude not necessarily the same length as the input signals
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt, windows
from scipy import stats
import cvxpy as cp
#import functions
from .functions import *  


class GammaPAC(object):
    """class for defining a gammaPAC object"""

    def __init__(self, sigPhase, sigAmp, fs):
        sigPhase, sigAmp = self._check_input_signals(sigPhase, sigAmp)
        self.sigPhase = np.copy(sigPhase)
        self.sigAmp = np.copy(sigAmp)
        self.phase = None
        self.amplitude = None
        self.PAC = None
        self.fs = np.copy(float(fs))
        self.kOpt = None
        self.alpha_hat = None
        self.coeffs = None
        self._phaseband = None
        self._ampband = None

    def _check_input_signals(self, sigPhase, sigAmp):
        if not isinstance(sigPhase, np.ndarray) or not isinstance(sigAmp, np.ndarray):
            raise ValueError("Input signals must be NumPy arrays.")
        sigPhase_squeezed = np.squeeze(sigPhase)
        sigAmp_squeezed = np.squeeze(sigAmp)
        if sigPhase_squeezed.ndim == 1 and sigAmp_squeezed.ndim == 1:
            if sigPhase.shape != sigAmp.shape:
                raise ValueError("Input signals must have the same shape.")
        else:
            if sigPhase_squeezed.ndim != 1 or sigAmp_squeezed.ndim != 1:
                raise ValueError("Input signals must be 1-dimensional arrays.")
            if sigPhase_squeezed.shape != sigAmp_squeezed.shape:
                raise ValueError("Input signals must have the same shape.")
            sigPhase = sigPhase_squeezed
            sigAmp = sigAmp_squeezed
            print("Warning: Input signals have different shapes. Squeezing them to 1-dimension.")
        return sigPhase, sigAmp


    def compute_phase(self, signal):
        analytic_signal = hilbert(signal)
        phase = np.angle(analytic_signal)
        return phase

    def compute_amplitude(self, signal):
        analytic_signal = hilbert(signal)
        amplitude = np.abs(analytic_signal)
        return amplitude

    def bandpass_filter(self, signal, freq_band):
        #nyquist_freq = 0.5 * self.fs
        low = freq_band[0] #/ nyquist_freq
        high = freq_band[1] #/ nyquist_freq
        sos = butter(4, [low, high], btype='bandpass', output='sos', fs=self.fs)
        filtered_signal = sosfiltfilt(sos, signal)
        return filtered_signal

    def fit(self, K, freqBandPhase, freqBandAmp, time_segment=None,modelSelection = False,sMethod = 'MDL',penalty=0,solver = 'sp'):
        if self._phaseband == freqBandPhase:
            pFlag = True
        else:
            pFlag = False
            self._phaseband = freqBandPhase
        
        if self._ampband == freqBandAmp:
            aFlag = True
        else:
            aFlag = False
            self._ampband = freqBandAmp

        if time_segment is None:
            sigPhase_segment = self.sigPhase
            sigAmp_segment = self.sigAmp
        else:
            start_time_sec, end_time_sec = time_segment

            # Convert start and end time to sample values
            start_time = int(start_time_sec * self.fs)
            end_time = int(end_time_sec * self.fs)

            # Extract the segment of signals
            sigPhase_segment = self.sigPhase[start_time:end_time]
            sigAmp_segment = self.sigAmp[start_time:end_time]

        # Apply bandpass filters
        if not pFlag:
            filtered_signal1 = self.bandpass_filter(sigPhase_segment, freqBandPhase)
            self.phase = self.compute_phase(filtered_signal1)
        else:
            if self.phase is None:
                filtered_signal1 = self.bandpass_filter(sigPhase_segment, freqBandPhase)
                self.phase = self.compute_phase(filtered_signal1)
        if not aFlag:
            filtered_signal2 = self.bandpass_filter(sigAmp_segment, freqBandAmp)
            self.amplitude = self.compute_amplitude(filtered_signal2)
        else:
            if self.amplitude is None:
                filtered_signal2 = self.bandpass_filter(sigAmp_segment, freqBandAmp)
                self.amplitude = self.compute_amplitude(filtered_signal2)

        #self.phase = self.compute_phase(filtered_signal1)
        #self.amplitude = self.compute_amplitude(filtered_signal2)
        
        #Calculate the PAC values
        self.PAC,self.coeffs,self.alpha_hat,self.kOpt = calc_gammaPAC(self.phase,self.amplitude,K=K,modelSelection = modelSelection,sMethod = sMethod,penalty=penalty,solver = solver)
        #print(f"PAC: {self.PAC:.5f}\nOptimal Alpha: {self.alpha_hat}\nOptimal K: {self.kOpt}")
        return self.PAC
    
    def fit_surrogate(self, K, freqBandPhase, freqBandAmp, time_segment=None,modelSelection = False,sMethod = 'MDL',penalty=0,solver = 'cvx',decimate = None):
        if self._phaseband == freqBandPhase:
            pFlag = True
        else:
            pFlag = False
            self._phaseband = freqBandPhase
        
        if self._ampband == freqBandAmp:
            aFlag = True
        else:
            aFlag = False
            self._ampband = freqBandAmp

        if time_segment is None:
            sigPhase_segment = self.sigPhase
            sigAmp_segment = self.sigAmp
        else:
            start_time_sec, end_time_sec = time_segment

            # Convert start and end time to sample values
            start_time = int(start_time_sec * self.fs)
            end_time = int(end_time_sec * self.fs)

            # Extract the segment of signals
            sigPhase_segment = self.sigPhase[start_time:end_time]
            sigAmp_segment = self.sigAmp[start_time:end_time]

        # Apply bandpass filters
        if not pFlag:
            filtered_signal1 = self.bandpass_filter(sigPhase_segment, freqBandPhase)
            self.phase = self.compute_phase(filtered_signal1)
        else:
            if self.phase is None:
                filtered_signal1 = self.bandpass_filter(sigPhase_segment, freqBandPhase)
                self.phase = self.compute_phase(filtered_signal1)
        if not aFlag:
            filtered_signal2 = self.bandpass_filter(sigAmp_segment, freqBandAmp)
            self.amplitude = self.compute_amplitude(filtered_signal2)
        else:
            if self.amplitude is None:
                filtered_signal2 = self.bandpass_filter(sigAmp_segment, freqBandAmp)
                self.amplitude = self.compute_amplitude(filtered_signal2)

        #self.phase = self.compute_phase(filtered_signal1)
        #self.amplitude = self.compute_amplitude(filtered_signal2)

        #set how much time to shift phase by
        t60 = 60*self.fs
        shift = np.random.randint(t60,self.phase.size-t60)
        
        #Calculate the PAC values
        if decimate is None:
            self.PAC,self.coeffs,self.alpha_hat,self.kOpt = calc_gammaPAC(np.roll(self.phase,shift),self.amplitude,K=K,modelSelection = modelSelection,sMethod = sMethod,penalty=penalty,solver = solver)
        else:
            dec_phase = np.roll(self.phase,shift)[::decimate]
            dec_amp = sp.signal.decimate(self.amplitude,decimate)
            dec_amp[dec_amp < 0] = 0
            self.PAC,self.coeffs,self.alpha_hat,self.kOpt = calc_gammaPAC(dec_phase,dec_amp,K=K,modelSelection = modelSelection,sMethod = sMethod,penalty=penalty,solver = solver)
        
        #print(f"PAC: {self.PAC:.5f}\nOptimal Alpha: {self.alpha_hat}\nOptimal K: {self.kOpt}")
        return self.PAC

    def plotGOF(self,fPlot = True):
        R = buildRFourier(self.phase,self.kOpt)
        Fu_hat,Fu = condQQ(self.amplitude,R,self.coeffs,self.alpha_hat,fPlot)
        return np.array([Fu_hat,Fu])
        
    def generate_comodulogram(self, K, phase_freqs,amp_freqs, phase_bw=4, phase_shift=None, amp_bw=10, amp_shift=None,
                             time_segment=None,fPlot = False):
        
        phase_lower_bound = phase_freqs[0]
        phase_upper_bound = phase_freqs[1]
        amp_lower_bound = amp_freqs[0]
        amp_upper_bound = amp_freqs[1]

        if phase_shift==None:
            phase_shift = phase_bw/2
        if amp_shift == None:
            amp_shift = amp_bw/2


        freq_band_params_phase = []
        freq_band_phase = [phase_lower_bound, phase_lower_bound + phase_bw]
        while freq_band_phase[1] <= phase_upper_bound:
            freq_band_params_phase.append(freq_band_phase.copy())
            freq_band_phase[0] += phase_shift
            freq_band_phase[1] += phase_shift

        freq_band_params_amp = []
        freq_band_amp = [amp_lower_bound, amp_lower_bound + amp_bw]
        while freq_band_amp[1] <= amp_upper_bound:
            freq_band_params_amp.append(freq_band_amp.copy())
            freq_band_amp[0] += amp_shift
            freq_band_amp[1] += amp_shift

        num_freq_bands_phase = len(freq_band_params_phase)
        num_freq_bands_amp = len(freq_band_params_amp)
        pac_values = np.zeros((num_freq_bands_phase, num_freq_bands_amp))

        for i, freqBandPhase in enumerate(freq_band_params_phase):
            for j, freqBandAmp in enumerate(freq_band_params_amp):
                pac_value = self.fit(K, freqBandPhase, freqBandAmp, time_segment)
                pac_values[i, j] = pac_value
    
        # # Plotting the comodulogram
        # fig, ax = plt.subplots()
        # im = ax.imshow(pac_values,origin = "lower", cmap='jet', interpolation='nearest')

        # # Add colorbar
        # cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel('PAC Value', rotation=-90, va="bottom")

        # # Set x and y axis labels
        # freq_band_labels_phase = [f'{freqBand[0]}-{freqBand[1]}' for freqBand in freq_band_params_phase]
        # freq_band_labels_amp = [f'{freqBand[0]}-{freqBand[1]}' for freqBand in freq_band_params_amp]
        # ax.set_xticks(np.arange(num_freq_bands_amp))
        # ax.set_yticks(np.arange(num_freq_bands_phase))
        # ax.set_xticklabels(freq_band_labels_amp)
        # ax.set_yticklabels(freq_band_labels_phase)
        # plt.xlabel('Frequency Bands for Amplitude')
        # plt.ylabel('Frequency Bands for Phase')
        # plt.title('Phase-Amplitude Coupling Comodulogram')

        # # Rotate and align the x-axis tick labels
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # # Display the comodulogram
        # plt.show()
        
        
        if fPlot:
            fig2, ax2 = plt.subplots(figsize = (8,5)) #feel free to change figsize
            im = ax2.imshow(pac_values.T,origin = 'lower',extent = [phase_lower_bound,phase_upper_bound,amp_lower_bound,amp_upper_bound],aspect = 'auto',interpolation = 'gaussian',cmap = 'jet') #heatmap for comodulogram
            ax2.set_ylabel('High Frequency Amplitude (Hz)')
            ax2.set_xlabel('Low Frequency Phase (Hz)')
            ax2.set_title('Comodulogram')
            fig2.colorbar(im,label = 'PAC Value')
            return pac_values.T, fig2, ax2, im
        return pac_values.T
    
    def idPAC(self,filtOutput = True):
        marginalY = calc_fY(self.coeffs,y_vec = self.amplitude,K= self.kOpt,alpha=self.alpha_hat)
        idPAC = np.zeros_like(self.phase)
        for tt in range(self.phase.size):
            likelihood = fYgivenTheta(self.amplitude[tt],self.phase[tt],coeffs=self.coeffs,K= self.kOpt,alpha=self.alpha_hat)
            infoDensity = np.log(likelihood/marginalY[tt])
            idPAC[tt] = infoDensity
        if filtOutput:
            sos = sp.signal.butter(4,self._phaseband[0],'low',output  = 'sos',fs = self.fs)
            idPAC = sp.signal.sosfiltfilt(sos,idPAC)
            idPAC[np.where(idPAC < 0)] = 0
        return idPAC
