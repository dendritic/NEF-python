'''
Created on 24 Sep 2013

@author: Chris
Butchered pythonic NEF implementation from the nengo webpage
'''
import numpy as np
from numpy import dot, zeros, ones, eye, sum, meshgrid, linspace, exp, log, ndim\
    , product, arange, cos, sin, floor, amax, where
from numpy.random import random_sample as rand
from numpy.random import standard_normal as randn

dt =       0.0005 # simulation time step
#tau_rc =   0.020 # as in Eliasmith, 2005
tau_rc =   0.020 # fast
tau_ref =  0.002 # as in Nengo defaults
#tau_pstc = 0.100 # NMDA territory
def_tau_pstc = 0.005 # 
# scaling factor for the post-synaptic filter
#pstc_scale = 1.0 - exp(-dt/tau_pstc)

''' working entrainable theta oscillator params!!!!:
dt =       0.0005 # simulation time step
#tau_rc =   0.020 # as in Eliasmith, 2005
tau_rc =   0.020 # fast
tau_ref =  0.002 # as in Nengo defaults
#tau_pstc = 0.100 # NMDA territory
tau_pstc = 0.005 # 
# scaling factor for the post-synaptic filter
pstc_scale = 1.0 - exp(-dt/tau_pstc)
'''

prec = np.float32 # default data type

''' Utility functions '''

# Random samples drawn from uniform between min and max of default precision
def rand_uni(low, high, size):
    r = (high - low)*rand(size) + low
    return prec(r)

# Create count unit vectors of random direction in nd dimensions
def rand_unit_vec(count, nd):
    # count random vectors with nd dimensions each
    v = rand_uni(-1, 1, (nd, count))
    v = normalise(v)
    return prec(v)

def normalise(a):
    mag = sum(a**2, 0)**.5 # magnitude of each column
    return a/mag[np.newaxis,:] # divide column element by its magnitude

#matrix pseudo-inverse using SVD with specifiable minimum singular value
def pseudoinv(A, minSV=0):
    U, s, V = np.linalg.svd(A, full_matrices=False)
    #set singular values less than specified to zero
    s[s < minSV] = 0
    #print where(s < 0.01*minSV)
    #pseudo-inverse of singular value matrix
    Si = zeros((s.size,)*2, dtype=s.dtype)
    svidx = where(s > 0)
    Si[svidx,svidx] = 1.0/s[svidx] #pseudo-inverse of S
    # A+ = VS+U+
    return dot(V.conj().T, dot(Si, U.conj().T))

''' Useful transformation functions '''
# You get out what you put in
def iden(x):
    return x

# A step up at t_on and then down at t_off 
def impulse_fun(t_on, t_off, nd, amp=1):
    f = lambda t: amp*((1 if t < t_off else 0) if t >= t_on else 0)*ones(nd)
    return f

def ramp_fun(multiplier):
    return lambda t: multiplier*t

''' Core NEF stuff '''
def lif_params(count, intercept_min, intercept_max, rate_min, rate_max):
    #sample intercept, value at which cell starts firing
    intercept = (intercept_max - intercept_min)*rand(count) + intercept_min
    #sample peak firing rate of neuron (at X.PHI=1)
    rate = (rate_max - rate_min)*rand(count) + rate_min
    #compute cell drive, at peak rate, for LIF neuron
    jmax = 1.0/(1 - np.exp((tau_ref - (1.0/rate))/tau_rc))
    #X.PHI = 1 for jmax, and X.PHI = intercept when J = 1, i.e. at thresh
    gain = (1.0 - jmax)/(intercept - 1.0)
    bias = 1 - gain*intercept
    return prec(gain), prec(bias)

def lif_rate(current):
    A = zeros(current.shape, dtype=prec)
    supra = current > 1
    # set all supra-threshold currents to appropriate rate
    A[supra] = 1.0/(tau_ref - tau_rc*log(1 - 1.0/current[supra]))
    return A

# Advance LIF neuron state one time-step given input
def lif_run(inp, v, ref):
    dV = dt*(inp - v)/tau_rc # LIF dynamics
    v += dV
    #clip V to >= 0
    v[v < 0] = 0
    #V to zero during refractory
    v[ref > 0] = 0
    #dec refractory remaining
    ref -= dt
    #spikes, V above threshold
    spikes = v > 1
    v[spikes] = 0 #reset spiking neurons
    ref[spikes] = tau_ref #spiking neurons enter refractory period
    return spikes

def drive(x, enc, gain, bias, syn=0):
    ncells = gain.shape[0]
    
    # ensure x is a 2D array, i.e. columns are x vectors to compute response for
    x = np.array(x, dtype=prec)
    datashape = x.shape[1:] # save the original shape of x to return response as
    if ndim(x) == 0:
        x.resize((1,1))
    elif ndim(x) == 1:
        x.resize((x.shape[0],1))
    elif ndim(x) > 2:
        x.resize((x.shape[0],product(datashape)))
    
    # compute drive as gain*(enc.x + syn) + bias
    J = gain[:,np.newaxis]*(dot(enc, x) + syn) + bias[:,np.newaxis]
    return J.reshape((ncells,) + datashape)

# returns normalised encoding vectors, with nd dimensions, for count cells
def encoders(count, nd=1):
    return rand_unit_vec(count, nd).T

# responses to encoding of x from simulated run
def responses(x, encoder, gain, bias, duration=0.5):
    #compute drive, J, to cells
    J = drive(x, encoder, gain, bias)
    # cell state variables of parametised precision/datatype
    v = rand_uni(0, 1, J.shape) # random initial voltages
    ref = zeros(J.shape, dtype=prec)
    spike_counts = zeros(J.shape, dtype=prec)
    #iterate timesteps
    t = 0
    while t < duration:
        spike_counts += lif_run(J, v, ref)
        t += dt
    return spike_counts/duration

def tuning_curves(enc, gain, bias, simulate=False, nsamples=1000):
    nd = enc.shape[1] # no. of represented dimensions
    # sample values
    xx = rand_unit_vec(nsamples, nd) # random unit directions
    # randomise magnitude of each column
    mag = rand_uni(0, 1, nsamples)
    xx *= mag[np.newaxis,:]
    # sort the x values for 1d case to make plotting easier
    if nd == 1:
        xx.sort()
    A = responses(xx, enc, gain, bias) if simulate else lif_rate(drive(xx, enc, gain, bias))
    return xx, A

# Compute optimised decoding vectors given firing rates for target values 
def decoder(X, A, function=iden, noise=0.1):
    nd = X.shape[0] #no. represented dimensions
    ncells = A.shape[0] # no. cells in population
    nsamples = product(A.shape[1:]) # no. samples over each dimension
    X = X.reshape((nd, nsamples)).T
    A = A.reshape((ncells, nsamples))
    #make some noise in cell outputs
    if noise > 0:
        # make a copy of A so we don't change the passed in version
        A = A.copy()
        noise_sd = noise*amax(np.abs(A))
        A += prec(noise_sd*randn(A.shape))
    else:
        noise_sd = 0

    # get the desired decoded value at each sample point
    value = np.apply_along_axis(function, 1, X)
    
    # find the optimum linear decoder
    Gamma = dot(A, A.T)/nsamples
    Upsilon = dot(A, value)/nsamples
    print Upsilon.shape
    Ginv = pseudoinv(Gamma, minSV=noise_sd**2)
    decoder = dot(Ginv, Upsilon)/dt
    return decoder

def decoder_from_params(encoder, gain, bias, function=iden, simulate=True,
                        neval_points=1000):
    xx, A = tuning_curves(encoder, gain, bias, simulate, neval_points)
    return decoder(xx, A, function)

def sim(ensembles, duration=1):    
    # simulation info
    times = arange(0, duration, dt)
    nsteps = len(times)
    
    spikes = {}
    value_inp = {}
    decoded = {}
    for k, e in ensembles.items():
        e['V'] = rand_uni(0, 1, e['V'].shape) #randomise initial voltage
        e['ref'] *= zeros(e['ref'].shape, dtype=prec) # reset refractory remain
        ncells = e['V'].shape[0]
        nd = e['enc'].shape[1] # no. represented dimensions
        # cell spikes over time array for each ensemble
        spikes[k] = zeros((ncells, nsteps))
        # external value input over time array for each ensemble
        value_inp[k] = zeros((nd, nsteps), dtype=prec)
        # prepare each ensembles decoded origins
        for k, o in e['origins'].items():
            nd_dec = o['W'].shape[0] # no. dims of decoded origin value
            o['current'] *= 0 # reset currents
            o['pstc_scale'] = 1.0 - exp(-dt/o['tau_pstc'])
            # place to record decoded value at each timestep
            decoded[k] = zeros((nd_dec, nsteps), dtype=prec)

    for ti, t in enumerate(times):
        for ek, e in ensembles.items():
            # compute external value input
            #print e['value_inp'](t).shape, value_inp[k][:,ti].shape
            value_inp[ek][:,ti] = e['value_inp'](t)
            # compute total synaptic cell input
            cell_inp = sum([s['current'] for s in e['syn_inp']], 0)
            cell_drive = drive(value_inp[ek][:,ti:ti + 1], e['enc'], e['gain'], 
                               e['bias'], syn=cell_inp)
            spikes[ek][:,ti:ti + 1] = lif_run(cell_drive, e['V'], e['ref'])
            

            for ok, o in e['origins'].items(): # do each origin's dynamics
                o['current'] *= 1 - o['pstc_scale'] # decay PSTSCs
                # begin new PSTCs from spikes 
                o['current'] += o['pstc_scale']*dot(o['W'], spikes[ek][:,ti:ti+1]) 
                decoded[ok][:,ti] = o['current'][:,0]

    return times, value_inp, spikes, decoded

''' Functional network wiring '''
# Create a decoded origin
def origin(XX, A, function=iden, tau_pstc=def_tau_pstc, transform=None):
    dec = decoder(XX, A, function=function).T
    if transform == None:
        # default transform is identity matrix for rep->rep
        transform = eye(XX.shape[0])
    nd = transform.shape[0] # no. of dimensions of decoded value 
    ori = {'tau_pstc': tau_pstc, # output time constant
           'current': zeros((nd, 1), dtype=prec), # output state var
           'W': dot(transform, dec)}
    return ori

def ensemble(enc, gain, bias, tuning=None):
    ncells = enc.shape[0]
    nd = enc.shape[1] # no. dimension represented
    if tuning == None:
        tuning = tuning_curves(enc, gain, bias)
    statesz = (ncells, 1)
    # default value input is zeros for each dim
    z = zeros(nd)
    zero_inp = lambda t: z
    def_origins = {'X': origin(*tuning)} # default identity decoder
    m = {'enc': enc, 'gain': gain, 'bias': bias, # cell parameters
         'tuning': tuning,
         'V': zeros(statesz, dtype=prec), # cell membrane potential 
         'ref': zeros(statesz, dtype=prec), # cell refractory remaining
         'syn_inp': [], 'value_inp': zero_inp, # inputs
         'origins': def_origins #outputs
         }
    
    return m

# Make feedback connections
def connect_feedback(ensemble, sample_points=None, fb_fun=iden, tau_pstc=def_tau_pstc):
    if sample_points == None:
        # use ensemble default tuning samples points and rates
        XX, R = ensemble['tuning']
    else:
        # use sample points passed in to generate rates for decoders
        XX = sample_points
        R = lif_rate(drive(XX, ensemble['enc'], ensemble['gain'], ensemble['bias']))
        
    # feedback connections decoder feeds back fb_fun(X) to ensemble
    feedback_orig = origin(XX, R, function=fb_fun,
                           transform=ensemble['enc'], tau_pstc=tau_pstc)
    ensemble['origins']['feedback'] = feedback_orig
    ensemble['syn_inp'].append(feedback_orig)

