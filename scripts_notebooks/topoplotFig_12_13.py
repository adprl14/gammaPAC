import numpy as np
import scipy as sp
import sys,os
sys.path.append('../../')
from gammaPAC.gammaPAC import *
from CFC import *
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed, parallel_config
import mne
np.random.seed(0)

# protect against overloading cores on Sherlock
try:
    num_cores = int(os.environ['SLURM_CPUS_ON_NODE'])
    print(f'Using {num_cores} cores')
except:
    num_cores = multiprocessing.cpu_count()

num_cores = 12

####################################################
######## Functions for computing PAC ###############
####################################################

## This function calculates Modulation index for a given input
#  phase and amplitude
def calc_MI_PA(phase,amplitude):
  n_bins = 18 #number of phase bins -> 18 is standard for PAC
  bin_size = 2*np.pi/n_bins #bin size = 2*pi/# of bins
  bin_edges = np.arange(-np.pi,np.pi+bin_size,bin_size) #create bin edges in [0,2*pi], array of n_bins+1 elements (i dont think this is necessary)
  bin_centers = np.arange(-np.pi+bin_size/2,np.pi+bin_size/2,bin_size) #create bin centers, array of n_bin elements
  mean_amp_bin = np.zeros((n_bins,)) #initialize a vector of zeros for average amplitude in each bin
  
  for i in range(n_bins):
    idx_to_bin = np.argwhere(((phase>=i*bin_size-np.pi) & (phase <(i+1)*bin_size-np.pi))) #search for indicies of phases within a phase bin
    #if nothing has phase in that phase bin -> set amplitude to 0
    if idx_to_bin.size == 0:
      mean_amp_bin[i] = 0
      #print('bin i has nothing')
    else:
      mean_amp_bin[i] = np.nanmean(amplitude[idx_to_bin]) #calculate the mean amplitude in phase bin i

  P_j = mean_amp_bin/np.nansum(mean_amp_bin) #calculate the average amplitude histogram normalized by total to make it resemble a pmf (bins sum to 1)
  H_j = -np.nansum(P_j*np.log(P_j)) #entropy

  KL = np.log(n_bins) - H_j #KL divergence between uniform and amplitude dist: KL(U,P)
  MI = KL/np.log(n_bins) #definition of MI

  return MI, bin_centers, P_j
def generate_samples_from_null_timeshift_for_PA(phase,amplitude,fs,N=100):
  samples = np.zeros((N,))
  t60 = fs*60

  for i in range(N):
    shift = np.random.uniform(t60,phase.size-fs*60)
    shift_phase = np.roll(phase,int(shift))
    MI,_,_ = calc_MI_PA(shift_phase,amplitude)
    samples[i] = MI

  return samples

## This function calculates z-scores between the null distribution and the actual MI values
def calc_z_scores(samples,values):
  z_scores = np.zeros((len(samples),))
  for i in range(len(samples)):
    mean = np.mean(samples[i])
    std = np.std(samples[i])
    z = (values[i]-mean)/std
    z_scores[i] = z

  return z_scores

#write a function to calculate PAC for a given mne data and EGG data
def calc_PAC(egg_data,eeg_data,fs,low_band,high_band,n_surrogates,decimate = None):
  pfit = GammaPAC(egg_data,eeg_data,fs)
  surrogate_pacs = np.zeros((n_surrogates,))
  with parallel_config(backend="loky", inner_max_num_threads=1):
    surrogate_pacs =  Parallel(n_jobs=num_cores)(delayed(pfit.fit_surrogate)(1,low_band,high_band,None,False,'MDL',0,'sp',decimate) for i in range(n_surrogates))
  pac_value= pfit.fit(1,low_band,high_band,solver='sp')
  pac_z_value = (pac_value - np.nanmean(surrogate_pacs))/np.nanstd(surrogate_pacs)
  print(f'channel {ch} done')
  return pac_z_value, pac_value, surrogate_pacs


#write a function to calculate MI z-scores for a given mne data and EGG data
def calc_MI_z_scored(egg_data,eeg_data,egg_fs,n_surrogates,downsample = False):
  if downsample:
    egg_data = sp.signal.decimate(egg_data,5)
    egg_fs = egg_fs//5

  sosEGG = sp.signal.butter(4,[.03,.07],btype='bandpass',output = 'sos',fs = egg_fs)
  filtEGG = sp.signal.sosfiltfilt(sosEGG,egg_data)
  phaseEGG = np.angle(sp.signal.hilbert(filtEGG))

  sosEEG = sp.signal.butter(4,[8,12],btype='bandpass',output = 'sos',fs = 125)
  filtEEG = sp.signal.sosfiltfilt(sosEEG,eeg_data)
  amplitudeEEG = np.abs(sp.signal.hilbert(filtEEG))

  if downsample:
    amplitudeEEG = sp.signal.decimate(amplitudeEEG,5)
    amplitudeEEG[amplitudeEEG<0] = 0
  
  n_channels = eeg_data.shape[0]

  MIsall = np.zeros((n_channels,))
  for ch in range(n_channels):
    MI,_,_ = calc_MI_PA(phaseEGG,amplitudeEEG[ch])
    MIsall[ch] = MI

  samples_all = np.zeros((n_channels,n_surrogates))
  with parallel_config(backend="loky", inner_max_num_threads=1):
    samples_all = Parallel(n_jobs=num_cores)(delayed(generate_samples_from_null_timeshift_for_PA)(phaseEGG,amplitudeEEG[ch],egg_fs,n_surrogates) for ch in range(n_channels))

  MI_z_scores = calc_z_scores(samples_all,MIsall)

  return MI_z_scores, MIsall, samples_all

#write a function to plot PAC topoplot
def plot_PAC_topoplot(values,info,fig_name,ax_title = 'Z-scored MI',fsave = True,vlim = (-5,5)):
  fig,ax1 = plt.subplots(ncols=1)
  im,cn = mne.viz.plot_topomap(values,info,ch_type = 'eeg',
                              image_interp = 'cubic',extrapolate='local',size = 5,cmap = 'jet',axes=ax1,show=False,
                              vlim = vlim)

  ax_x_start = 0.95
  ax_x_width = 0.04
  ax_y_start = 0.1
  ax_y_height = 0.9
  cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
  clb = fig.colorbar(im, cax=cbar_ax)
  clb.ax.set_title(ax_title) # title on top of colorbar
  #set colorbar limits
  fig.subplots_adjust(wspace=.05)
  if fsave:
    lgd = ax1.legend()
    plt.savefig(fig_name,bbox_extra_artists=(lgd,),bbox_inches='tight')

#######################
###### Load Data ######
#######################

## Get path to data
data_dir = '../data/'
subj_nums = [1,2]

for subj_num in subj_nums:
    #find the string in subj_list that contains the 'patient'+subj_num
    subj = [s for s in subj_list if 'patient'+str(subj_num) in s][0]
    dataDir = baseDir + exp + '/' + subj + '/'
    saveDir = dataDir + 'dataDeriv/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    print(subj)


    ## Load Data
    fs = 125


    #load EEG data
    dataEEG = np.load(dataDir + f'subj{subj_num}_EEG.npy')
    dataEEG = dataEEG*1e6 #convert to uV


    #load EGG data
    matEGG = sp.io.loadmat(dataDir + f'{subj}_EGG_preprocessed.mat')
    dataEGG = np.squeeze(matEGG['bestEGG'])
    dataEGG = sp.signal.decimate(dataEGG, 2)
    dataEGG = dataEGG[start*60*fs:stop*60*fs]

    ####################################################
    ########## Analysis of PAC and MI #################
    ####################################################

    n_surrogates = 5000

    MI_z_scores, MIsall,samples_all =  calc_MI_z_scored(dataEGG,dataEEG,fs,n_surrogates,downsample=True)
    np.save(saveDir + f'{subj}_surrogateMIs_{methodAR}_{start}-{stop}min.npy',samples_all)
    np.save(saveDir + f'{subj}_MI_z_scores_{methodAR}_{start}-{stop}.npy',MI_z_scores)
    np.save(saveDir + f'{subj}_MI_{methodAR}_{start}-{stop}.npy',MIsall)
    print(f'{subj} MI done!')

    pac_z_scores = np.zeros((dataEEG.shape[0]))
    pac_values = np.zeros((dataEEG.shape[0]))
    surrogate_pacs = np.zeros((dataEEG.shape[0],n_surrogates))

    for ch in range(dataEEG.shape[0]):
        pac_z_scores[ch],pac_values[ch],surrogate_pacs[ch] = calc_PAC(dataEGG,dataEEG[ch],fs,[.03,.07],[8,12],n_surrogates,5)

    np.save(saveDir + f'{subj}_surrogatePACs_{methodAR}_{start}-{stop}min.npy',surrogate_pacs)
    np.save(saveDir + f'{subj}_PAC_z_scores_{methodAR}_{start}-{stop}.npy',pac_z_scores)
    np.save(saveDir + f'{subj}_PAC_values_{methodAR}_{start}-{stop}.npy',pac_values)
    print(f'{subj} PAC done!')
