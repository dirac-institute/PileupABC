import numpy as np
from astropy.table import Table, Column, vstack
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from scipy import integrate
from scipy.special import gammaln
import time
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf

import bayes
import simdat
import auto_corr
arf = Table.read('/Users/mlazz/Dropbox/UW/PileupABC/13858/repro/SDSSJ091449.05+085321.corr.arf',format='fits')

data = Table.read('real_data.fits')

count=1
while count < 10:
    print count
    bins, nrgy, new_spectrum = simdat.simulate_data(500000,2.7,'cdf',arf)
    spec_col = Column(new_spectrum,name='simspec'+str(count))
    data.add_column(spec_col)
    count += 1
    
data.write('part1_fake_data.txt',format='ascii')