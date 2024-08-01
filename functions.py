'''
Contains functions used in the gammaPAC methods
'''

import numpy as np
import scipy as sp
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import cvxpy as cp
import statsmodels.api as sm
from statsmodels.api import GLM
from statsmodels.genmod.families.links import Log

############################################################
############# Model Definition and Fitting #################
############################################################

def buildRFourier(ph,K=1):
    '''
    K is the number of fourier bases
    '''
    m = ph.shape[0]
    RFull = np.zeros((m,2*K+1))
    for k in range(K):
        RFull[:,2*k] = np.sin((k+1)*ph)
        RFull[:,2*k+1] = np.cos((k+1)*ph)
    RFull[:,2*K] = 1.0
    return RFull

def jac(x,y,R):
    T = y.size
    grad = np.zeros((x.size,))
    for i in range(T):
        grad += -y[i]*R[i]*np.exp(-R[i]@x) + R[i]
    return grad
def hess(x,y,R):
    T = y.size
    H = np.zeros((x.size,x.size))
    for i in range(T):
        Rt = R[i]
        H += y[i]*np.outer(Rt,Rt)*np.exp(-Rt.dot(x))
    return H

def objective(y,Rmtx,X, penalty):
    '''
    objective functions used for solving the GLM 
    y: ndarray shape (N,) 
    independent variable (amplitude)
    Rmtx: ndarray shape (N,2*K+1)
    regressor matrix of Fourier basis functions of phase
    penalty: float
    L2 penalty associated with the regression problem. usually set to 0
    '''
    return (y@cp.exp(-Rmtx@X)) + cp.sum(Rmtx@X) + penalty*cp.norm(X, 2)

def modelFitCVX(Rmtx,y,penalty=0,fPrint=False):
    '''model fitting procedure for the GLM using cvxpy'''
    coeffs = np.zeros((Rmtx.shape[1],))
    X = cp.Variable((Rmtx.shape[1],))
    problem = cp.Problem(cp.Minimize(objective(y,Rmtx,X,penalty)))

    try:
        problem.solve()
    except cp.error.SolverError:
        problem.solve(solver='SCS')

    #problem.solve()
    coeffs = X.value
    if fPrint:
        print(problem.status)
    return coeffs

def objective_sp(X,y,Rmtx):
    return (y@np.exp(-Rmtx@X)) + np.sum(Rmtx@X) 

def modelFitSp(Rmtx,y):
    #res = sp.optimize.minimize(objective_sp, np.zeros((Rmtx.shape[1],)),\
    #     jac = jac,hess = hess,args=(y,Rmtx), method = 'Newton-CG', tol=1e-10)
    res = sp.optimize.minimize(objective_sp, np.zeros((Rmtx.shape[1],)),args=(y,Rmtx), tol=1e-10)
    return res.x

def modelFitSM(Rmtx,y):
    '''
    model fitting procedure for the GLM using statsmodels
    '''
    link = Log()
    model = GLM(y,Rmtx,family=sm.families.Gamma(link=link))
    res = model.fit()
    #print(res.summary())
    return res.params

def modelFit(Rmtx,y,solver = 'cvx',penalty=0,fPrint=False):
    if solver == 'cvx':
        coeffs = modelFitCVX(Rmtx,y,penalty,fPrint)
    elif solver == 'sp':
        coeffs = modelFitSp(Rmtx,y)
    else:
        raise ValueError('method must be cvx or sp')
    return coeffs

#negative log-likelihood as a function of alpha
def NLLalpha(alpha,y,Lvec):
    '''
    Lvec is R.dot(coeffs)
    runs faster if we pre calculate Lvec since it never changes
    '''
    #Lvec = R.dot(coeffs)
    T = y.size
    return T*sp.special.loggamma(alpha) - T*alpha*np.log(alpha) + \
        np.sum( -(alpha-1.0)*np.log(y) + alpha*np.multiply(y,np.exp(-Lvec) ) + alpha*Lvec )
    #return T*np.log(sp.special.gamma(alpha)) - T*alpha*np.log(alpha) + \
    #    np.sum( -(alpha-1.0)*np.log(y) + alpha*np.multiply(y,np.exp(-Lvec) ) + alpha*Lvec )

def get_alphastar(y,R,coeffs,method = 'line'):
    '''
    return approximate optimal alpha by line search
    '''
    if method == 'line':
        Lvec = R.dot(coeffs)
        alphas = np.linspace(1e-6,400,1000)
        NLLsAlpha = np.array([NLLalpha(alpha,y,Lvec) for alpha in alphas])
        alphastar = alphas[np.nanargmin(NLLsAlpha)]
    elif method == 'sp':
        res = sp.optimize.minimize(NLLalpha, 1, args=(y,R@coeffs), tol=1e-10)
        alphastar = res.x[0]

    return alphastar


############################################################
############# Model Selection ##############################
############################################################
def calcNNLL(Rmtx,y,coeffs,activeset=None):
  return (np.sum(Rmtx.dot(coeffs)) + (np.exp(-Rmtx.dot(coeffs))).dot(y))/Rmtx.shape[0]

def findNNLLsFourier(phases,y,Ks,solver = 'cvx'):
    Coeffs = {}
    dims = []
    NNLLs=np.zeros((len(Ks),))
    for itr, K in enumerate(Ks):  
        #print('solving for Number of Bases:',K)
        Rall = buildRFourier(phases,K)
        coeffsTmp = modelFit(Rall,y,penalty=0,solver = solver)
        Coeffs[itr] = coeffsTmp
        #NNLL = calcNNLL(Rall,y,coeffsTmp)
        alphaTmp = get_alphastar(y,Rall,coeffsTmp,method='sp')
        print(alphaTmp)
        NNLL = NLLalpha(alphaTmp,y,Rall@coeffsTmp)/y.size
        NNLLs[itr]=NNLL
        dims.append(1+2*K)
    dims = np.array(dims)  
    return Coeffs, NNLLs, dims

def PNNLLsFourier(dims,NNLLs,m,title='',fPlot = False):
    AICpenalties = dims/m
    MDLpenalties = dims*np.log(m)/(2*m)
    PNNLLsAIC=NNLLs+AICpenalties
    PNNLLsMDL=NNLLs+MDLpenalties

    if fPlot:
        plt.figure()
        plt.plot(dims,NNLLs,'o', label='NNLLL')
        plt.plot(dims,PNNLLsAIC,'r*', label='PNNLL AIC')
        plt.plot(dims,PNNLLsMDL,'g*', label='PNNLL MDL')
        plt.xlabel('dimension')
        plt.xticks(range(int(min(dims)),int(max(dims))))
        plt.title(title)
        _=plt.legend()
    return PNNLLsAIC,PNNLLsMDL

############################################################
############# Distributions/PAC ############################
############################################################

'''
Note: the prior f_Theta is uniform over -pi to pi
'''

#likelihood function
def fYgivenTheta(y,theta,coeffs,K= 1,alpha=1):
    regressorvec = np.squeeze(buildRFourier(np.array([theta]),K))
    L = coeffs.dot(regressorvec)
    return np.exp(-NLLalpha(alpha,np.array([y]),L))

#posterior

def fThetagivenY(y,thetas,coeffs,K,alpha=1):
    '''
    must give the function a thetas vector over the whole circle
     to get the normalization constant correct
    '''
    posterior = np.array([fYgivenTheta(y,theta,coeffs,K,alpha)/(2*np.pi) for theta in thetas])
    #print(posterior)
    c = np.sum(posterior)*(thetas[1]-thetas[0])#normalization constant
    posterior = posterior/c
    return posterior


def calc_fY(coeffs,y_vec=None,K = 1,alpha = 1,ntheta = 18):
    if y_vec is None:
        y_vec = np.linspace(0,10,101)
    f_Y = np.zeros_like(y_vec)
    theta_vec = np.arange(-np.pi,np.pi,2*np.pi/ntheta)
    for idx_y,y in enumerate(y_vec):
        f_Y[idx_y] = np.sum(np.array([fYgivenTheta(y,theta,coeffs,K,alpha) for theta in theta_vec]))/(theta_vec.size)

    return f_Y



def calcDKL(y,coeffs,K,alpha = 1,ntheta = 18):
    #calculate KL Divergence between p(theta|y) and p(theta) for a given y based on exponential GLM
    thetas = np.arange(-np.pi,np.pi,2*np.pi/ntheta)
    p_theta_y = fThetagivenY(y, thetas, coeffs,K,alpha)

    #normalize for pmf
    p_theta = np.ones_like(thetas)/thetas.size
    p_theta_y = p_theta_y/np.sum(p_theta_y) #normalize to pmf

    DKL = sp.stats.entropy(p_theta_y,p_theta)
    return DKL

def PACmeasure(high_amp,coeffs,K,alpha = 1):
    #pdata is P(data) = P(Y) = integral(P(y|theta)dtheta)

    if np.min(high_amp) < 1e-4:
        minval = np.min(high_amp)
    else:
        minval = 1e-4

    maxval = np.percentile(high_amp,99.99)

    y_vec = np.linspace(minval,maxval,100)
    DKLs = np.zeros_like(y_vec)

    fY = calc_fY(coeffs,y_vec,K,alpha)
    pY = fY/np.sum(fY)
    for i in range(y_vec.size):
        DKLs[i] = calcDKL(y_vec[i],coeffs,K,alpha)
    PAC = DKLs.dot(pY) #expected value of DKL w respect to Y 
    return PAC, DKLs


########################################################
####### Full Model Fit to PAC output procedure  ########
########################################################

#PAC based on our von mises basis
def calc_gammaPAC(low_phase,high_amp,K=1,modelSelection = False,sMethod = 'MDL',penalty=0,solver = 'sp'):
    if modelSelection:
        Ks = np.arange(0,K+1)
        Coeffs, NNLLs, dims = findNNLLsFourier(low_phase,high_amp,Ks,solver = solver)
        m = low_phase.size
        if sMethod == "MDL":
            MDLpenalties = dims*np.log(m)/(2*m)
            print(MDLpenalties)
            PNNLLsMDL=NNLLs+MDLpenalties
            print(PNNLLsMDL)
            idx_min = np.argmin(PNNLLsMDL)
        elif sMethod == "AIC":
            AICpenalties = dims/m
            PNNLLsAIC = NNLLs+AICpenalties
            idx_min = np.argmin(PNNLLsAIC)
        '''
        BCarray = np.zeros((dims.size,))
        for i in range(BCarray.shape[0]):
        if i == 0:
            if dims[i]==0:
            BCarray[i] = 0
            else:
            BCarray[i] = 1/dims[i]

        else:
            BCarray[i] = 1/dims[i] + BCarray[i-1]

        BCpenalties = np.max(dims)*BCarray/m
        PNNLLsBC = NNLLs+BCpenalties
        idx_min = np.argmin(PNNLLsBC)
        '''
        optimal_K = Ks[idx_min]
        R = buildRFourier(low_phase,optimal_K)
        cvx_coeffs = Coeffs[idx_min]
    else:
        R = buildRFourier(low_phase,K)
        cvx_coeffs = modelFit(R,high_amp,penalty=penalty,solver = solver)
        #alphastar, cvx_coeffs = modelFitJoint(R,high_amp)  #doesnt work
        #cvx_coeffs = modelFitSM(R,high_amp)
        optimal_K = K

    #print('optimal K: ', o0ptimal_K, '\noptimal coeffs: ',cvx_coeffs)
    #print(f'{R}\n{cvx_coeffs}')
    alphastar = get_alphastar(high_amp,R,cvx_coeffs,method='sp')
    if np.isnan(alphastar):
        alphastar = get_alphastar(high_amp,R,cvx_coeffs,method='line')
    gammaPAC,_ = PACmeasure(high_amp,cvx_coeffs,optimal_K,alphastar)
    return gammaPAC,cvx_coeffs,alphastar,optimal_K



############################################################
############# Goodness of Fit  #############################
############################################################

def condQQ(data_amp,Rmtx,coeffs,alpha_hat,fPlot = False):
    T = data_amp.size
    Uplot = np.linspace(0,1,101)
    F_U_hat = np.zeros((Uplot.size,))
    u_t = np.zeros((data_amp.size,))
    for i in range(T):
        if coeffs.ndim == 2:
            Xt = coeffs[:,i]
        else:
            Xt = coeffs
        u_t[i] = sp.stats.gamma.cdf(data_amp[i],alpha_hat,0,np.exp(Rmtx[i,:]@Xt)/alpha_hat)
    for i in np.arange(1,Uplot.size):
        F_U_hat[i] = np.sum(u_t <= Uplot[i])/T
    if fPlot:
        lb = np.where((Uplot-1.36/np.sqrt(T)) >=0 )
        ub = np.where((Uplot+1.36/np.sqrt(T)) <=1 )

        plt.figure(figsize=(6,5))
        plt.plot(Uplot,Uplot,ls= '-',color = 'k' ,label = 'theoretical')
        plt.plot(Uplot,F_U_hat,label = 'data')
        plt.plot(Uplot[ub],Uplot[ub]+1.36/np.sqrt(T),ls= '--',color = 'k' ,label = '')
        plt.plot(Uplot[lb],Uplot[lb]-1.36/np.sqrt(T),ls= '--',color = 'k' ,label = '')
        plt.ylabel('$\hat{F}(U)$')
        plt.xlabel('F(U)')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend()

    return F_U_hat, Uplot


def timeRescaling(y,Rt,Xt,alpha_hat,dt = None):
    beta = alpha_hat*np.exp(-Rt@Xt)
    if dt == None:
        dt = y/100
    t_int = np.arange(dt/2,y-dt/2,dt) #midpoint riemann sum
    pdf_vals = sp.stats.gamma.pdf(t_int,alpha_hat,0,1/beta)
    cdf_vals = sp.stats.gamma.cdf(t_int,alpha_hat,0,1/beta)
    cond_intensity = pdf_vals/(1-cdf_vals)
    e = sp.integrate.trapezoid(cond_intensity,dx = dt)
    return e


def condQQ_timeRescaling(data_amp,Rmtx,coeffs,alpha_hat,fPlot = False):
    T = data_amp.size
    Uplot = np.linspace(0,1,101)
    F_U_hat = np.zeros((Uplot.size,))
    u_t = np.zeros((data_amp.size,))
    for i in range(T):
        if coeffs.ndim == 2:
            Xt = coeffs[:,i]
        else:
            Xt = coeffs
        e = timeRescaling(data_amp[i],Rmtx[i,:],Xt,alpha_hat)
        u_t[i] = sp.stats.expon.cdf(e)
    for i in np.arange(1,Uplot.size):
        F_U_hat[i] = np.sum(u_t <= Uplot[i])/T
    if fPlot:
        lb = np.where((Uplot-1.36/np.sqrt(T)) >=0 )
        ub = np.where((Uplot+1.36/np.sqrt(T)) <=1 )

        plt.figure(figsize=(6,5))
        plt.plot(Uplot,Uplot,ls= '-',color = 'k' ,label = 'theoretical')
        plt.plot(Uplot,F_U_hat,label = 'data')
        plt.plot(Uplot[ub],Uplot[ub]+1.36/np.sqrt(T),ls= '--',color = 'k' ,label = '')
        plt.plot(Uplot[lb],Uplot[lb]-1.36/np.sqrt(T),ls= '--',color = 'k' ,label = '')
        plt.ylabel('$\hat{F}(U)$')
        plt.xlabel('F(U)')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend()

    return F_U_hat, Uplot


