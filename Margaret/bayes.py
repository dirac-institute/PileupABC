import numpy as np
from astropy.table import Table, Column, vstack
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from scipy import integrate
from scipy.special import gammaln
import time

def integrate_lambda(E,A,alpha):
    """ Takes in array of energy bin edges, value of A and alpha and returns finite integral"""
    integral = A/(1-alpha) * np.power(E,1-alpha)
    return integral

def log_likelihood(theta,data,E_max,E_min):
    """
    input theta = paramters where theta[0] = logA, theta[1] = alpha
    Return log(Poisson likelihood) function for dataset
    Poisson probability of the form P(k) = exp(-lambda)*lambda**k/k!
    lambda = integral from E_min to E_max A*E**-a
    input data = piled histogram from previous step
    """
    #E_min = np.arange(0.3,11.0,0.01)
    #E_max = np.arange(0.31,11.01,0.01)
    logA = theta[0]
    a = theta[1]
    lam = integrate_lambda(E_max[1:],np.exp(logA),a) - integrate_lambda(E_min[1:],np.exp(logA),a)
    summation = gammaln(data+1)
    log_like = np.multiply(-1.0,lam) + data*np.log(lam) + summation
    total_log_like = np.sum(log_like)
    if not np.isfinite(total_log_like):
        return -np.inf
    else:
        return total_log_like
    
def neg_log_likelihood(theta,data,E_max,E_min):
    """
    Return log(Poisson likelihood) function for dataset
    Poisson probability of the form P(k) = exp(-lambda)*lambda**k/k!
    lambda = integral from E_min to E_max A*E**-a
    input theta = paramters where theta[0] = logA, theta[1] = alpha
    """
    #E_min = np.arange(0.3,11.0,0.01)
    #E_max = np.arange(0.31,11.01,0.01)
    logA = theta[0]
    a = theta[1]
    lam = integrate_lambda(E_max[1:],np.exp(logA),a) - integrate_lambda(E_min[1:],np.exp(logA),a)
    summation = gammaln(data+1)
    log_like = np.multiply(-1.0,lam) + data*np.log(lam) + summation
    total_log_like = np.sum(log_like)
    if not np.isfinite(total_log_like):
        return -np.inf
    else:
        return -1*total_log_like
    
def log_prior_alpha(a):
    if (a > 0) and (a<5):
        return np.log(0.2)
    else:
        return -np.inf
    
def log_prior_logA(logA):
    if (logA<np.log(1000000))and(logA>np.log(100)):
        return np.log(1.0/(np.log(1000000)-np.log(100)))
    else:
        return -np.inf
    
def log_posterior(log_like,log_priors):
    return log_like + np.sum(log_priors)
