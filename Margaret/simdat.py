import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.table import Table, Column
import time

def test_func():
    print 'test worked!'

def energies_rs(a,b,gamma,N):
    import numpy as np
    import time
    
    ''' Returns an array of photon energies with a power law distribution using rejection sampling.
        a = E_min
        b = E_max
        gamma = photon index
        N = total number of photons to be returned
    '''
    start_time = time.time()
    A = (1.-gamma)/((b**(1.-gamma))-(a**(1.-gamma))) #normalization factor so curve integrates to 1
    pl =  lambda x,A,gamma : A*x**(-1.0*gamma)
    binsize = 0.13 #keV, set by Chandra detector. This will be swapped out later for a more realistic binning
    bins = np.arange(a,b,binsize)
    photons = np.zeros(N)
    count = 0
    while count<N:
        r = np.random.uniform(a,b,1)
        for i in range(len(bins)-1):
            if (r>bins[i])and(r<bins[i+1]):
                edge_lo = bins[i]
                edge_hi = bins[i+1]
                p_r = integrate.quad(pl,edge_lo,edge_hi,args=(A,gamma))[0]
                k = np.random.uniform(0,1,1)
                if (k<=p_r):
                    photons[count] = r
                    count += 1
    #print("Time to generate photon energies = %s seconds" % (time.time() - start_time))
    return photons

def energies_cdf(a,b,gamma,N):
    '''
    Generate photon energies using the cumulative distribution function (CDF)
    CDF taken from https://arxiv.org/pdf/0706.1062.pdf
    a = E_min
    b = E_max
    gamma = photon index
    N = total number of photons to be returned'''
    start_time = time.time()
    r = np.random.uniform(0,1,N)
    x = a*(1-r)**(-1/(gamma-1))
    #print("Time to generate photon energies = %s seconds" % (time.time() - start_time))
    return x

def simulate_data(stop,gamma,handle,arf_bins):
    ''' Generates an un-piled and piled list of photon energies. 
        stop = end time of observation, generally equals total length of obs in seconds.
        gamma = photon index
        handle = 'rs' to generate photons via rejection 
    '''
    #Assign physical parameters
    start = 0. #s
    #stop = 100 #s
    cr = 1. #count rate between 0.1 and 10 photons/second
    read_time = 3.2 #s
    energy_lo = 0.1 #keV
    energy_hi = 10.0 #keV
    K = (stop-start)*cr #Number of photons to generate
    N = np.random.poisson(K) #choose a number of photons from a poisson distribution centered on expected, K
    #print 'number of photons observed =', N
    
    #Assign photon arrival times, random within obervation time
    arrival_times = np.random.uniform(start,stop,N)
    
    #Assign photon energies, random with a power law distribution.
    if handle == 'rs':
        #print 'generating photon energies via rejection sampling'
        energy = energies_rs(energy_lo,energy_hi,gamma,N) #generate photon energies using rejection sampling
    if handle == 'cdf':
        #print 'generating photon energies via CDF method'
        energy = energies_cdf(energy_lo,energy_hi,gamma,N)
    else:
        print 'invalid photon generation method entered'
        
    #Simulate pile up: bin arrival times, if there is more than one photon in a bin, combine their energies
    #and record as one photon, change other photon energies to 0
    piled_energy = np.copy(energy)
    times = Column(arrival_times,name='time')
    energies = Column(energy,name='energy')
    raw = Table([times,energies])
    raw.sort('time')
    time_bins = np.arange(start,stop,read_time)
    time_hist, time_edges = np.histogram(raw['time'],bins=time_bins)
    num_piled = 0
    for i in range(len(time_hist)):
        if time_hist[i] > 1:
            #print time_hist[i]
            num_piled += time_hist[i]-1
            first_photon = np.sum(time_hist[0:i])
            last_photon = first_photon + time_hist[i] - 1
            sum_energy = np.sum(energies[first_photon:last_photon])
            piled_energy[first_photon] = sum_energy
            piled_energy[first_photon+1:last_photon] = 0.0
    #print 'Fraction of photons that are piled (# photons lost) =', np.float(num_piled) / N
    energy = np.array(filter((lambda x: np.isinf(x)==False),energy))
    piled_energy = np.array(filter((lambda x: np.isinf(x)==False),piled_energy))
    #print 'energy',energy
    #print 'piled energy',piled_energy
    #print 'piled energy > 0',piled_energy[piled_energy>0]
    e_hist, e_bins = np.histogram(energy,bins=arf_bins)
    pe_hist, pe_bins = np.histogram(piled_energy[piled_energy>0],bins=arf_bins)
    #print pe_hist
    return e_bins[1:], e_hist, pe_hist

def plot_data(e_bins,e_hist,pe_hist):
    '''
    Takes output data from simulate_data() and plots the raw vs. piled spectra
    Inputs:
    e_bins = energy bins from the ARF
    e_hist = number of photons in each energy bin for the unpiled spectrum
    pe_hist = number of photons in each energy bin for the piled spectrum
    '''
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(e_bins,e_hist,label='un-piled')
    ax.plot(e_bins,pe_hist,label='piled')
    ax.set_xlabel('Energy (keV)')
    ax.set_xlim(-0.5,11)
    ax.set_ylabel('Flux, unscaled')
    ax.set_yscale('log')
    ax.legend()
    #return e_hist, pe_hist