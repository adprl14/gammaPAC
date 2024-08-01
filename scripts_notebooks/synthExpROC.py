'''
Script to generate synthetic data and build ROC curves for 
gammaPAC methods
'''

import sys,os
sys.path.append('../../')

import numpy as np
import scipy as sp
import h5py as h5
import pickle
from gammaPAC.generalPAC import calc_PLV_phase,calc_MVL_phase,calc_MI_phase,calc_ndPAC_phase
from gammaPAC.gammaPAC import *
from gammaPAC.utils.synthetic_data import generate_coupled_EEG_sig
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
import seaborn as sns
import time
sns.set(rc={'image.cmap': 'jet'})
plt.rc('font', family='serif')


##################################################
########  Set-Up  ################################
# set random seed for reproducibility
np.random.seed(0)


# set up directories and make sure they exist

myDir = ... #whereever you want to work out of
data_dir = myDir+'Data_Deriv/Synthetic'
fig_dir = myDir+'Figures/'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# set up parallel processing
num_cores = cpu_count()  #set this to 1 if you want to run serially

##################################################
########  ROC Function  ##########################


def buildROC_CI(nsamp = 100,coeff = .5,flow = 0.05,fhigh = 10,fs = 100, T = 20,method = calc_MI_phase,noise_var = 0,n_trials = 1,fplot = 0):
    #methods_str= [calc_PLV_phase,calc_MVL_phase,calc_MI_phase,calc_ndPAC_phase,'gammaPAC']
    #if str(method) not in methods_str:
    #    raise ValueError('Please enter a valid method')

    t = np.arange(0,T,1/fs)


    fpr_interp = np.linspace(0,1,101)
    tprs = np.zeros((n_trials,fpr_interp.size))
    aucs = np.zeros((n_trials,))
    for trial_num in range(n_trials):
        labels = np.zeros((2*nsamp,))
        PACs = np.zeros((2*nsamp,))

    for i in range(nsamp):
        for j in range(2):
            if j == 0:
                cc = 0 #coupling coefficient for generation
            else:
                cc = coeff
        
            signal = generate_coupled_EEG_sig(t,flow,fhigh,cc,noise_var)

            #power = np.sum(signal**2)/signal.size
            #print(f'SNR = {10*np.log10((power-noise_var)/noise_var)} dB')

            

            if method != 'gammaPAC':
                sos = sp.signal.butter(2,[flow-.02,flow+.02],btype='bandpass',output = 'sos',fs = fs)
                slow_sig = sp.signal.sosfilt(sos,signal)
                slow_sig = sp.signal.sosfilt(sos,np.flip(slow_sig))
                slow_sig = np.flip(slow_sig)
                slow_phase = np.angle(sp.signal.hilbert(slow_sig))

                sos_fast = sp.signal.butter(2,[fhigh-flow-2,fhigh+flow+2],btype='bandpass',output = 'sos',fs = fs)
                fast_sig = sp.signal.sosfilt(sos_fast,signal)
                fast_sig = sp.signal.sosfilt(sos_fast,np.flip(fast_sig))
                fast_sig = np.flip(fast_sig)
                #fast_amp = np.abs(sp.signal.hilbert(fast_sig))
                #PACstat = globals()[method](slow_phase,fast_sig)
                PACstat = method(slow_phase,fast_sig)
            else:
                pfit = GammaPAC(signal,signal,fs)
                PACstat = pfit.fit(1,[flow - .02, flow+.02],[fhigh-flow-2,fhigh+flow+2],solver='sp')
            
            PACs[2*i+j] = PACstat
            labels[2*i+j] = j

    fpr, tpr, thresholds = metrics.roc_curve(labels,PACs)
    tpr_interp = np.interp(fpr_interp,fpr,tpr)
    tprs[trial_num,:] = tpr_interp
    auc = metrics.roc_auc_score(labels,PACs)
    aucs[trial_num] = auc


    return fpr_interp,tprs, aucs


##################################################
########  Experiment  ############################


# set parameters
nsamp = 50
flow = 0.05
fhigh = 10
fs = 100
T = 20
noise_var = 0.5
n_trials = 200

coupling_coeffs = [.05,.1,.15,.2,.25,.3]
methods_str=[calc_PLV_phase,calc_MVL_phase,calc_MI_phase,calc_ndPAC_phase,'gammaPAC']
method_labels = ['PLV','MVL','MI','ndPAC','gammaPAC']
colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:red']

n_methods = len(method_labels)

#set up arrays to store ROC curve results
store_fprs = np.zeros((len(coupling_coeffs),n_methods,101))
store_tprs = np.zeros((len(coupling_coeffs),n_methods,n_trials,101))
store_aucs = np.zeros((len(coupling_coeffs),n_methods,n_trials))


fig,ax = plt.subplots(2,3,figsize = (20,12))
# loop over coupling coefficients
print('building ROC curves...')
for i,cc in enumerate(coupling_coeffs):
    # loop over methods
    plt.subplot(2,3,i+1)
    print(f'starting cc = {cc}')
    tt = time.time()
    for j,method in enumerate(methods_str):
        print(f'method = {method_labels[j]}')
        #parallelize over trials
        output = Parallel(n_jobs=num_cores)(delayed(buildROC_CI)(nsamp,cc,flow,fhigh,fs,T,method,noise_var,1) for k in range(n_trials))
        for k in range(n_trials):
            store_fprs[i,j,:] = output[k][0]
            store_tprs[i,j,k,:] = output[k][1]
            store_aucs[i,j,k] = output[k][2]
        #plot ROC curve with mean auc in legend and error bars on curve
        mean_auc = np.mean(store_aucs[i,j,:])
        std_auc = np.std(store_aucs[i,j,:])
        mean_tprs = np.mean(store_tprs[i,j,:,:],axis = 0)
        std_tprs = np.std(store_tprs[i,j,:,:],axis = 0)

        ul = mean_tprs+1.96*std_tprs
        ul[np.argwhere(ul>1)] = 1
        ll = mean_tprs-1.96*std_tprs
        ll[np.argwhere(ll<0)] = 0
        
        plt.plot(store_fprs[i,j,:],mean_tprs,label = f'{method_labels[j]}, AUC =  {mean_auc:.3f} $\pm$ {1.96*std_auc:.3f}',color = colors[j])
        plt.plot(store_fprs[i,j,:],ul,color = colors[j],ls = '--')
        plt.plot(store_fprs[i,j,:],ll,color = colors[j],ls = '--')
        plt.fill_between(store_fprs[i,j,:],ll,ul,color = colors[j],alpha = 0.2)
    
    print(f'finished cc = {cc} in {time.time()-tt:.2f} seconds')
    plt.plot([0,1],[0,1],color = 'k',ls = '--')
    plt.legend()
    plt.title(f'Coupling Coefficient = {cc}')

fig.supxlabel('False Positive Rate')
fig.supylabel('True Positive Rate')

# save figure
fig.savefig(fig_dir+'synth_ROC_curves.png',dpi = 300)

# save tprs, fprs, and aucs
np.save(data_dir+'synth_ROC_curves_tprs.npy',store_tprs)
np.save(data_dir+'synth_ROC_curves_fprs.npy',store_fprs)
np.save(data_dir+'synth_ROC_curves_aucs.npy',store_aucs)