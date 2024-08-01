import numpy as np
import scipy as sp
import sys,os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert,butter,sosfiltfilt
sns.set(style='ticks', font_scale=1.2)

sys.path.append('../../')
from gammaPAC.gammaPAC import *
from gammaPAC.functions import *
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed, parallel_config
fig_dir = ... #path to save figures

###########################
######## 

# Load EEG data
sf = 1000.
dataFp2 = sp.io.loadmat('../data/mmn_Fp2.mat')['x']
dataFp2 = np.squeeze(dataFp2)
print(dataFp2.shape)
time = np.linspace(0,1,int(sf))-.2

#################################################

fs = sf

sosp = butter(6,[.5,4],'bandpass',output = "sos",fs = fs)
sosa = butter(6,[8,15],'bandpass',output = "sos",fs=fs)


slowERPAC = sosfiltfilt(sosp,dataFp2,axis = -1)
fastERPAC = sosfiltfilt(sosa,dataFp2,axis = -1)
phaseERPAC = np.angle(hilbert(slowERPAC,axis = 0))
ampERPAC = np.abs(hilbert(fastERPAC,axis = 0))


def idPAC_tmp(phase,amp,coeffs,K,alpha,lp = 1,fs = 1,filtOutput = True):
    marginalY = calc_fY(coeffs, amp,K ,alpha)
    idPAC = np.zeros_like(phase)
    for tt in range(phase.size):
        likelihood = fYgivenTheta(amp[tt],phase[tt],coeffs=coeffs,K= K,alpha= alpha)
        infoDensity = np.log(likelihood/marginalY[tt])
        idPAC[tt] = infoDensity
    if filtOutput:
        sos = sp.signal.butter(4,lp,'low',output  = 'sos',fs = fs)
        idPAC = sp.signal.sosfiltfilt(sos,idPAC)
        idPAC[np.where(idPAC < 0)] = 0
    return idPAC

def calc_idpact(phase,amp):
    pac,coeffs,alpha,K = calc_gammaPAC(phase.flatten(),amp.flatten(),K=1,modelSelection = False,sMethod = "MDL",penalty=0)
    idpac = idPAC_tmp(phase,amp,coeffs,K,alpha,lp = 4,fs = fs,filtOutput = False)
    return idpac

def calc_idpact_unpack(args):
  return calc_idpact(*args)

import warnings
warnings.filterwarnings("ignore")

mmn_idpac = np.zeros_like(dataFp2)
N = dataFp2.shape[0]

n_cpu = 12
with parallel_config(backend="loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=n_cpu)(delayed(calc_idpact)(phaseERPAC[n,:],ampERPAC[n,:]) for n in range(N))

for i in range(N):
	mmn_idpac[i,:] = results[i]

np.save('mmn_idpac.npy',mmn_idpac)


########################################
###### PLOT
'''
fig,ax = plt.subplots(2,1,figsize = (10,6),height_ratios = [1,3])
time = np.linspace(0,T/sf,T)
time = time-np.max(time)/2


ax[0].plot(time,erPAC)
ax[0].set_ylabel('ERPAC')

ax2 = ax.twinx()
ax2.plot(time,erPAC,c = "tab:orange")
ax2.set_ylabel('ERPAC',c = "tab:orange")

#fig.supxlabel('Time (s)')

ax[1].imshow(erID,extent = [time[0],time[-1],1,spindles.shape[0]],cmap='jet', interpolation='nearest',aspect = "auto")
ax[1].set_ylabel('Spindle Number')
ax[1].set_xlabel('Time(s)')

plt.savefig(fig_dir+'mmn_ERPAC.png',dpi =400)
'''