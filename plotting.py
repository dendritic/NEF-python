'''
Created on 21 Nov 2013

@author: Chris
'''

#import matplotlib.pyplot as plt
import numpy as np

# 
def spike_raster(plt, t, spikes, size=0.01):
    nsets = spikes.shape[0]
    for i, s in enumerate(spikes):
        spike_at = s > 0
        plt.scatter(t[spike_at], i*np.ones(np.sum(spike_at)), s=size, c='b') 
        
