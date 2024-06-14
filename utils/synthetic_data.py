import numpy as np
import scipy as sp
from gammaPAC.functions import buildRFourier


#####################################################################################
####################   Signals from Generative Model   ##############################
#####################################################################################

def generate_latent(K,noise_var = 1,n_timesteps = 10,jump_times = None, jump_steps = None):
    jump_flag = 0
    if (jump_steps is not None) and (jump_times is not None):
        if jump_steps.size != jump_times.size:
            raise ValueError('You need as many jump steps as you have times')
        jump_flag = 1
    elif (jump_steps is not None) or (jump_times is not None):
        raise ValueError('You need both the times and steps to not be Nonetype')

    x = np.zeros((K,n_timesteps))
    cov = noise_var*np.eye(K)

    for t in range(n_timesteps):
        if t == 0:
            x[:,t] = np.random.multivariate_normal(np.zeros((K,)),cov,1)
        else:
            x[:,t] = x[:,t-1] +np.random.multivariate_normal(np.zeros((K,)),cov,1)
    if jump_flag:
        for idx in range(jump_times.size-1):
            ti = jump_times[idx]
            tf = jump_times[idx+1]
            x[:,ti:tf] += jump_steps[idx]
    return x

def generate_observations(K,noise_var = 1,n_timesteps = 10,alpha = 1,jump_times = None,jump_steps= None):
    phases = np.random.uniform(-np.pi,np.pi,n_timesteps)
    R = buildRFourier(phases,K = (K-1)//2)
    xs = generate_latent(K,noise_var,n_timesteps,jump_times,jump_steps)
    p = 2
    
    betas = np.zeros((n_timesteps,))
    ys = np.zeros((n_timesteps,))

    #R, ys = buildRegressorMatrixResponseVec(ys,p,ys.size-p)

    for t in range(n_timesteps):
        betas[t] = alpha*np.exp(-xs[:,t]@R[t,:])

        ys[t] = np.random.gamma(alpha,1/betas[t])
    
    
    return xs,ys,R
    #return xs[:,:-p], ys, R

#####################################################################################
####################   Static Coupled Signals   #####################################
#####################################################################################

def generate_coupled_EEG_sig(time, flow,fhigh, coupling_coefficient,noise_var = 0):
    #time = np.linspace(0,100,10001)
    
    slow_waveform = np.cos(2*np.pi*flow*time)
    
    # this will allow the amplitude to vary from a minumum of 1-coupling
    # coefficient to a maximum of 1.0 at the same frequency as the flow signal
    amplitude_slow = (coupling_coefficient*slow_waveform + 2.0-coupling_coefficient)/2
    
    noise = np.random.normal(0,np.sqrt(noise_var),time.size)
    fast_waveform = amplitude_slow * np.cos(2*np.pi*fhigh*time)
    
    signal = fast_waveform + slow_waveform + noise
    
    return signal

#####################################################################################
####################   Time-Varying Coupled Signals   ###############################
#####################################################################################


def generate_time_varying_coupled_sig(time, flow,fhigh, cc_time_series ,noise_var = 0):
  #time = np.linspace(0,100,10001)

  slow_waveform = np.cos(2*np.pi*flow*time)

  # this will allow the amplitude to vary from a minumum of 1-coupling
  # coefficient to a maximum of 1.0 at the same frequency as the flow signal
  amplitude_slow = (cc_time_series*slow_waveform + 2.0-cc_time_series)/2

  noise = np.random.normal(0,np.sqrt(noise_var),time.size)
  fast_waveform = amplitude_slow * np.cos(2*np.pi*fhigh*time)

  signal = fast_waveform + slow_waveform + noise

  return signal

def generate_ramp(time,period):
  sig = (sp.signal.sawtooth(time*2*np.pi/period)+1)/2
  N = time[np.where(time<period)].size
  zero_array = np.zeros((N,))
  rampfunc = np.array([])
  for i in range(int((time[-1]+time[1]-time[0])/period)):
    rampfunc = np.hstack((rampfunc,zero_array))
    rampfunc = np.hstack((rampfunc,sig[i*N:(i+1)*N]))
  return rampfunc
