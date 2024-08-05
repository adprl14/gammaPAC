import numpy as np
import scipy as sp
import sys,os
from joblib import Parallel, delayed, parallel_config
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
sys.path.append('../../')

from gammaPAC.gammaPAC import *
from CFC import comodulogram
from time import time

#data = sp.io.loadmat('/Users/andrewperley/Desktop/Lab Stuff/Stanford/Gut-Brain/Code/vmPAC/vmPAC_code/Journal_Figures/LFP_HG_HFO.mat')
data = sp.io.loadmat('../data/LFP_HG_HFO.mat')
fs = 1000
LFP_HG = np.squeeze(data["lfpHG"])
LFP_HFO = np.squeeze(data["lfpHFO"])

T = LFP_HG.size
min_shift = 3*int(1/4*fs) #3 cycles of lowest frequency

n_surrs = 100
shifts = np.random.randint(min_shift,T-min_shift,n_surrs)

n_cpus = cpu_count()

n_jobs = np.min([n_cpus,n_surrs])
print(f'Using {n_jobs} CPUs')

#compute comodulograms
tt = time()
comod, fig, ax, im = comodulogram(LFP_HG,LFP_HG,fs,[.5,20.5],[.5,100.5],4,10,2,method = 'MI',plot = True)
fig.savefig('./testing/Data_Deriv/Tort/LFP_HG_comodulogram_MI.png',bbox_inches = 'tight')
print(f'Elapsed time: {time() - tt:.2f} seconds')

tt = time()
p_hg = GammaPAC(LFP_HG,LFP_HG,fs)
comod2, fig2, ax2, im = p_hg.generate_comodulogram(1,[.5,20.5],[.5,100.5],fPlot = True)
fig2.savefig('./testing/Data_Deriv/Tort/LFP_HG_comodulogram_gammaPAC.png',bbox_inches = 'tight')
print(f'Elapsed time: {time() - tt:.2f} seconds')
plt.show()

comod3, fig3, ax3, im3 = comodulogram(LFP_HFO,LFP_HFO,fs,[.5,20.5],[.5,150.5],4,10,2,method = 'MI',plot = True)
fig3.savefig('./testing/Data_Deriv/Tort/LFP_HFO_comodulogram_MI.png',bbox_inches = 'tight')

p_hfo = GammaPAC(LFP_HFO,LFP_HFO,fs)
comod4, fig4, ax4, im4 = p_hfo.generate_comodulogram(1,[.5,20.5],[.5,150.5],fPlot = True)
fig4.savefig('./testing/Data_Deriv/Tort/LFP_HFO_comodulogram_gammaPAC.png',bbox_inches = 'tight')

#save comodulogram data
np.save('./testing/Data_Deriv/Tort/LFP_HG_comodulogram_MI.npz',comod)
np.save('./testing/Data_Deriv/Tort/LFP_HG_comodulogram_gammaPAC.npz',comod2)
np.save('./testing/Data_Deriv/Tort/LFP_HFO_comodulogram_MI.npz',comod3)
np.save('./testing/Data_Deriv/Tort/LFP_HFO_comodulogram_gammaPAC.npz',comod4)


def computeGammaPAC_comod(x1,x2,fs,f_range1,f_range2):
    p = GammaPAC(x1,x2,fs)
    comod = p.generate_comodulogram(1,f_range1,f_range2)
    return comod

np.random.seed(0)

#compute comodulograms of surrogate data
tt = time()
with parallel_config(backend="loky", inner_max_num_threads=1):
    comodMI_hg_surrs = Parallel(n_jobs = n_jobs)(delayed(comodulogram)\
        (np.roll(LFP_HG,shift),LFP_HG,fs,[.5,20.5],[.5,100.5],4,10,2,'MI',False) for shift in shifts)

print('elapsed time: ',time()-tt)

with parallel_config(backend="loky", inner_max_num_threads=1):
    comodPAC_hg_surrs = Parallel(n_jobs = n_jobs)(delayed(computeGammaPAC_comod)\
        (np.roll(LFP_HG,shift),LFP_HG,fs,[.5,20.5],[.5,100.5]) for shift in shifts)

print('elapsed time: ',time()-tt)

with parallel_config(backend="loky", inner_max_num_threads=1):
    comodMI_hfo_surrs = Parallel(n_jobs = n_jobs)(delayed(comodulogram)\
        (np.roll(LFP_HFO,shift),LFP_HFO,fs,[.5,20.5],[.5,150.5],4,10,2,'MI',False) for shift in shifts)

print('elapsed time: ',time()-tt)

with parallel_config(backend="loky", inner_max_num_threads=1):
    comodPAC_hfo_surrs = Parallel(n_jobs = n_jobs)(delayed(computeGammaPAC_comod)\
        (np.roll(LFP_HFO,shift),LFP_HFO,fs,[.5,20.5],[.5,150.5]) for shift in shifts)

#save surrogate data all in one mat file
sp.io.savemat('/scratch/users/aperley/data/Tort/DataDeriv/surrogate_comodulograms.mat',
    {'MI_HG_surrs':comodMI_hg_surrs,
    'PAC_HG_surrs':comodPAC_hg_surrs,
    'MI_HFO_surrs':comodMI_hfo_surrs,
    'PAC_HFO_surrs':comodPAC_hfo_surrs})

print('total time: ',time()-tt)
print('no bugs :)')     