import numpy as np
import scipy as sp
import sys,os
import multiprocessing
import matplotlib.pyplot as plt
sys.path.append('../gammaPAC')
sys.path.append('../gammaPAC/ssPAC')
from gammaPAC import *

fig_dir = ... #path to save figures

data = sp.io.loadmat('../data/LFP_HG_HFO.mat')
fs = 1000
LFP_HG = np.squeeze(data["lfpHG"])
LFP_HFO = np.squeeze(data["lfpHFO"])

sos_gamma = sp.signal.butter(2,[60,80],btype = 'bandpass',output = 'sos',fs = fs)
sos_theta = sp.signal.butter(2,[5,10],btype = 'bandpass',output = 'sos',fs = fs)
sos_hfp = sp.signal.butter(2,[120,160],btype = 'bandpass',output = 'sos',fs = fs)

HG_gamma = sp.signal.sosfiltfilt(sos_gamma,LFP_HG)
HG_theta = sp.signal.sosfiltfilt(sos_theta,LFP_HG)

HFO_hfo = sp.signal.sosfiltfilt(sos_gamma,LFP_HFO)
HFO_theta = sp.signal.sosfiltfilt(sos_theta,LFP_HFO)

seg_size = fs
N = LFP_HFO.size//seg_size
hg_ks = np.zeros((N,))
hfo_ks = np.zeros((N,))

hg_ecdf= np.zeros((N,101))
hfo_ecdf = np.zeros((N,101))

for idx in range(N):
    hg_seg = LFP_HG[idx*seg_size:(idx+1)*seg_size]
    hg_tort = GammaPAC(hg_seg,hg_seg,fs)
    pac = hg_tort.fit(1,[5,10],[60,80])
    output = hg_tort.plotGOF(fPlot=False)
    Fu_hat = output[0]
    Fu = output[1]
    ks = np.max(np.abs(Fu_hat - Fu))
    hg_ks[idx] = ks
    hg_ecdf[idx] = Fu_hat

    hfo_seg = LFP_HFO[idx*seg_size:(idx+1)*seg_size]
    hfo_tort = GammaPAC(hfo_seg,hfo_seg,fs)
    pac = hfo_tort.fit(1,[5,10],[120,160])
    output = hfo_tort.plotGOF(fPlot=False)
    Fu_hat = output[0]
    Fu = output[1]
    ks = np.max(np.abs(Fu_hat - Fu))
    hfo_ks[idx] = ks
    hfo_ecdf[idx] = Fu_hat
    if idx % 25 == 0:
        print(f'Iteration {idx} of {N}')

#print summary (mean +- std of ks values)
print(f'High Gamma KS: {np.mean(hg_ks)} +- {np.std(hg_ks)}')
print(f'HFO KS: {np.mean(hfo_ks)} +- {np.std(hfo_ks)}')

#plot histograms of ks values on neighboring subplots
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(hg_ks,bins = 20)
plt.title('High Gamma KS')
plt.ylabel('Count')
plt.xlabel('KS Value')

thresh = 1.36/np.sqrt(fs)
plt.axvline(thresh,ls = '--',color='r',label='Threshold')


plt.subplot(1,2,2)
plt.hist(hfo_ks,bins=20)
plt.title('HFO KS')
plt.ylabel('Count')
plt.xlabel('KS Value')
plt.axvline(thresh,ls = '--',color='r',label='Threshold')


#save figure
plt.savefig(fig_dir+'ks_hist.jpeg',dpi = 300)

#plot average ks plots with shaded error bars
fig,ax = plt.subplots(1,2,figsize=(10,5))

ax[0].plot(Fu,np.mean(hg_ecdf,axis=0))
pm = 1.96*np.std(hg_ecdf,axis=0)
ax[0].fill_between(Fu,np.mean(hg_ecdf,axis=0)-pm,np.mean(hg_ecdf,axis=0)+pm,alpha=0.3)
ax[0].set_title('High Gamma')
ax[0].set_xlabel('F(u)')
ax[0].set_ylabel(r'$\hat{F}(u)$')

#plot dashed lines at diagonal + thresh and diagonal - thresh but only in the range [0,1]
thresh = 1.36/np.sqrt(fs)
ax[0].plot([0,1],[0,1],'k')
ax[0].plot([0,1-thresh],[0+thresh,1],'k--')
ax[0].plot([0+thresh,1],[0,1-thresh],'k--')


ax[1].plot(Fu,np.mean(hfo_ecdf,axis=0))
pm = 1.96*np.std(hfo_ecdf,axis=0)
ax[1].fill_between(Fu,np.mean(hfo_ecdf,axis=0)-pm,np.mean(hfo_ecdf,axis=0)+pm,alpha=0.3)
ax[1].set_title('HFO')
ax[1].set_xlabel('F(u)')
ax[1].set_ylabel(r'$\hat{F}(u)$')
ax[1].plot([0,1],[0,1],'k')
ax[1].plot([0,1-thresh],[0+thresh,1],'k--')
ax[1].plot([0+thresh,1],[0,1-thresh],'k--')

plt.tight_layout()
plt.savefig(fig_dir+'ks_ecdf.jpeg',dpi = 300)





