
import numpy as np
import scipy as sp
from scipy import signal
from scipy import stats
from scipy.signal import hilbert
from scipy.special import erfinv


'''
Standard PAC methods
- standard implementations of PLV, MVL, Modulation Index

'''

def calc_PLV_phase(low_phase,s_high):
  if low_phase.shape != s_high.shape:
    raise ValueError('signals should be of the same size')
  dim = low_phase.ndim
  length = low_phase.shape[dim-1]
  high_phase = np.angle(sp.signal.hilbert(s_high)) #Hilbert transform to get phase(s_high)
  phase_diff = np.exp(1j*(low_phase-high_phase))   #phase difference
  #constant phase difference means phasors all point in same direction (i.e., phases are 'locked')
  phase_lock_value = np.abs(np.sum(phase_diff)/length) #compute the length of the mean phasor, 

  return phase_lock_value


#Calculates Mean Vector Length between low frequency and high frequency signal
#This is a PAC method
def calc_MVL_phase(low_phase,s_high):
  if low_phase.shape != s_high.shape:
    raise ValueError('signals should be of the same size')
  dim = low_phase.ndim
  length = low_phase.shape[dim-1]
  high_amp = np.abs(sp.signal.hilbert(s_high)) #Get instantaneous amplitude of high frequency signal
  phasor = high_amp*np.exp(1j*(low_phase)) #construct phasors at each point in time
  mean_vector_length = np.abs(np.sum(phasor)/length) #compute the length of the mean phasor

  return mean_vector_length


#Calculates Modulation Index between low frequency and high frequency signal
#This is a PAC method
def calc_MI_phase(low_phase,s_high):
  n_bins = 18 #number of phase bins -> 18 is standard for PAC
  bin_size = 2*np.pi/n_bins #bin size = 2*pi/# of bins
  bin_edges = np.arange(-np.pi,np.pi+bin_size,bin_size) #create bin edges in [0,2*pi], array of n_bins+1 elements (i dont think this is necessary)
  bin_centers = np.arange(-np.pi+bin_size/2,np.pi+bin_size/2,bin_size) #create bin centers, array of n_bin elements
  mean_amp_bin = np.zeros((n_bins,)) #initialize a vector of zeros for average amplitude in each bin

  amplitude = np.abs(sp.signal.hilbert(s_high)) #extract instantaneous amplitude
  #find indices where phase falls in bin i
  for i in range(n_bins):
    '''
    if i != n_bins-1:
      idx_to_bin = np.argwhere(((low_phase>=i*bin_size-np.pi) & (low_phase <(i+1)*bin_size-np.pi))) 
    else:
      idx_to_bin = np.argwhere(((low_phase>=i*bin_size-np.pi) & (low_phase <=(i+1)*bin_size-np.pi))) #edge case: last bin needs to be fully inclusive of edges
    '''

    idx_to_bin = np.argwhere(((low_phase>=i*bin_size-np.pi) & (low_phase <(i+1)*bin_size-np.pi))) #search for indicies of phases within a phase bin
    
    #if nothing has phase in that phase bin -> set amplitude to 0
    if idx_to_bin.size == 0:
      mean_amp_bin[i] = 0
      #print('bin i has nothing')
    else:
      mean_amp_bin[i] = np.nanmean(amplitude[idx_to_bin]) #calculate the mean amplitude in phase bin i

  P_j = mean_amp_bin/np.nansum(mean_amp_bin) #calculate the average amplitude histogram normalized by total to make it resemble a pmf (bins sum to 1)
  H_j = sp.stats.entropy(P_j) #entropy

  KL = np.log(n_bins) - H_j #KL divergence between uniform and amplitude dist: KL(U,P)
  MI = KL/np.log(n_bins) #definition of MI

  # return MI, bin_centers, P_j
  return MI

# normalized direct PAC Ozkurt et al.

def calc_ndPAC_phase(low_phase,s_high, p=.05):
    """
    Parameters
    ----------
    p : float | .05
        P-value to use for thresholding. Sub-threshold PAC values
        will be set to 0. To disable this behavior (no masking), use ``p=1`` or
        ``p=None``.

    Returns
    -------
    pac : array_like
        Array of phase amplitude coupling of shape (n_amp, n_pha, ...)

    References
    ----------
    Ozkurt et al. :cite:`ozkurt2012statistically`
    """
    n_times = s_high.size
    # normalize amplitude
    # use the sample standard deviation, as in original matlab code from author
    amp = np.abs(sp.signal.hilbert(s_high))
    amp = amp.reshape((1,1,amp.size))
    low_phase = low_phase.reshape((1,1,low_phase.size))

    #print(amp.shape,low_phase.shape)
    # normalize amplitude
    # use the sample standard deviation, as in original matlab code from author
    amp = np.subtract(amp, np.mean(amp, axis=-1, keepdims=True))
    amp = np.divide(amp, np.std(amp, ddof=1, axis=-1, keepdims=True))
    # compute pac
    pac = np.abs(np.einsum('i...j, k...j->ik...', amp, np.exp(1j * low_phase)))

    # no thresholding
    if p == 1. or p is None:
        return pac / n_times

    s = pac ** 2
    pac /= n_times
    # set to zero non-significant values
    xlim = n_times * erfinv(1 - p) ** 2
    pac[s <= 2 * xlim] = 0.
    return np.squeeze(pac)
