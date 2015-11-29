'''
Created on 24 Sep 2013

@author: Chris
'''

import numpy as np
from numpy.fft import fft, fftshift, ifftshift
from numpy import diag, tile, insert, roll, r_, exp, zeros, ones, linspace, cos, sin\
    , eye, ndim, dot, amax, sum, prod
from math import pi
import nef

''' Gaussian function exp(-x^2/2s^2) '''
def gaussian(x, sigma=1):
    return exp(-x**2/(2*sigma**2))

''' Returns Fourier coefficents upto fmax for 1D Gaussians tiled over unit
    length spaced evenly with nsamples points '''
def tiled_gaussian_coeff(sigma=0.05, nsamples=400, fmax=2):
    t = linspace(-0.5, 0.5, nsamples, False)
    g = gaussian(t, sigma)
    #Gaussians circularly shifted to be centred at each point
    g_tiled = np.array([roll(g, -i) for i in range(nsamples)]).T
    f = r_[0:fmax + 1];
    fg = fftshift(fft(ifftshift(g_tiled, axes=0), axis=0), axes=0)

    # extract complex coefficients for used frequencies
    freq_idx = f + nsamples/2
    coeff = 2.0/nsamples*fg[freq_idx,:]
    coeff[f == 0,:] *= 0.5
    #coeff[f < 0] *= -1
    return coeff, f, t

''' Reconstruct time domain signal given components c, of frequencies, f at 
    time points tt. '''
def freq_sig(c, f, tt):
    cos_comp = c[np.newaxis,:].real*cos(2*pi*f[np.newaxis,:]*tt[:,np.newaxis])
    sin_comp = c[np.newaxis,:].imag*sin(2*pi*f[np.newaxis,:]*tt[:,np.newaxis])
    return cos_comp + sin_comp

''' Generate an array of alternating real & imaginary components from 
    each complex coefficient of zc, with frequencies f. The zero frequency
    coefficient will only generate a real element '''
def interlace(zc, f):
    #interlace,row-wise real and imaginary components into one array
    comp_is_real = tile([[True], [False]], (len(f), 1)).flatten()
    comp_freq = tile(f, (2,1)).T.flatten()
    zero_imag = (comp_is_real == False) & (comp_freq == 0)
    #get rid of the zero imaginary row
    comp_is_real = comp_is_real[zero_imag == False]
    comp_freq = comp_freq[zero_imag == False]
    coeff = zeros((len(comp_freq), zc.shape[1]))
    coeff[comp_is_real,:] = zc.real
    coeff[comp_is_real == False,:] = zc[f != 0,:].imag
    
    return coeff, comp_freq, comp_is_real

''' Returns the transformation matrix for rotating each alternating real/imag
    pair together given the frequency f. '''
def rot_pairs_mat(f, delta):
    ##NB: assumes zero-freq term is first, remaining are real/imag pairs
    # take one of each nonzero pair
    f = f[1::2]
    n = 2*len(f) + 1
    phase = 2*pi*f*delta
    dv0 = zeros(n)
    dv0[0] = 1 # constant term stays the same
    dv0[1::2] = cos(phase)
    dv0[2::2] = cos(phase)
    dv1 = zeros(n - 1)
    dv1[1::2] = -sin(phase)
    dvn1 = zeros(n - 1)
    dvn1[1::2] = sin(phase)
    T = diag(dv0, 0) + diag(dv1, 1) + diag(dvn1, -1)
    #print T[1:3,1:3]
    return T

''' Takes alternating real and imaginary component pairs and returns
    them as single complex coefficients '''
def deinterlace(c):
    # insert a zero f0 imag component
    #zero_sin_idx = (c.shape[0] + 1)/2
    zero_sin_idx = 0
    c = insert(c, zero_sin_idx, 0,  axis=0)
    print c
    return c[::2,:] + c[1::2,:]*1j

''' Testing ground for ring oscillators '''
# feedback implementing bump oscillator dynamics
def bump_rot(x, rot_mat, tau_pstc, speed=1):
    I = eye(len(x))
    If = I.copy()
    #print If
    If[[5,6],[5,6]] = 0
    
    T = (rot_mat - I)*speed*tau_pstc/nef.dt + 1.15*If
    return dot(T, x)

# oscillator input with time varying frequency
def var_freq_osc(t, nd):
    off_time = 4
    if ndim(t) == 0:
        scalar = True
        t = np.array([t])
    else:
        scalar = False
    target_freq = zeros(t.shape)
    phase_off = zeros(t.shape)
    
    time_groups = [(t > i) & (t < i + 1) for i in range(off_time)]
    for i, s in enumerate(time_groups):
        target_freq[s] = 5 + 2*i
        phase_off[s] = 13*i # some randomish offset
    phase = 2*pi*t*target_freq*3 + phase_off
    nelems = len(t)
    y = zeros((nd, nelems))
    mag = 1.1
    y[-2,:] = mag*cos(phase)
    y[-1,:] = mag*sin(phase)
    y[-2,t >= off_time] = 0
    if scalar:
        return y[:,0]
    else:
        return y

# multiple oscillating frequencies entraining
def multi_osc(t, nd, f):
    if ndim(t) == 0:
        scalar = True
        t = np.array([t])
    else:
        scalar = False

    phase = 2*pi*t[np.newaxis,:]*f[:,np.newaxis]
    #print phase.shape
    y = zeros((nd, phase.shape[1]))
    mag = 1.3/len(f)
    y[-2,:] = mag*sum(cos(phase), 0)
    y[-1,:] = mag*sum(sin(phase), 0)
    if scalar:
        return y[:,0]
    else:
        return y

def attractor_ensemble(ncells=300, speed=8, fmax=3, sigma=0.045):
    nsamples = 2*ncells
    #fmax = 4
    #sigma = 0.08
    #sigma = 0.045
    #tau_pstc = 0.005
    tau_pstc = 0.005
    
    gain, bias = nef.lif_params(ncells, -1, 1, 100, 200)
    
    # compute coefficients "directions" for each cell   
    zc_cell, zf, tt = tiled_gaussian_coeff(nsamples=ncells, fmax=fmax, sigma=sigma)
    enc, cell_freq, _ = interlace(zc_cell, zf)
    enc = nef.normalise(enc).T
    # compute sampling coefficients, same as directions
    zc_samples, _, _ = tiled_gaussian_coeff(nsamples=nsamples, fmax=fmax, sigma=sigma)
    xx, _, _ = interlace(zc_samples, zf)
    xx = xx/amax(xx)
    
    nd = enc.shape[1]
    
    #print zc_cell[:,0].real
    
    # compute cell rates for each sample point
    R = nef.lif_rate(nef.drive(xx, enc, gain, bias))
    
    #N = coeff_by_cell.shape[1]
    #coeff_by_cell = coeff_by_cell/amax(coeff_by_cell)
    #coeff_by_cell = vstack((coeff_by_cell, rand_unit_vec(N, 1).T))
    #enc = array(normalise(coeff_by_cell).T)#array(normalise(coeff_by_cell).T)
    
    #gain,bias = lif_params(N, -1, 1, 200, 400)
    #xx = coeff_by_cell
    #xx[-1,:] *= rand_uni(0, 1, N)
    #A = responses(xx, enc, gain, bias)
    #A = lif_rate(drive(xx, enc, gain, bias))
    osc_rot = rot_pairs_mat(cell_freq, nef.dt*speed)
    bump_fun = lambda x: bump_rot(x, rot_mat=osc_rot, tau_pstc=tau_pstc, speed=1)
    #figure()
    #plot()
    #update_plot(tt, zc_cell[:,0], zf)
    ens = nef.ensemble(enc, gain, bias, (xx, R))
    nef.connect_feedback(ens, xx, bump_fun, tau_pstc)
    
    init_value = enc[0,:]
    ens['value_inp'] = lambda t: init_value if t < 0.010 else zeros(nd)

    return ens

def test_bump(duration=1, ncells=300):
    # run three bumps, each entrained to an increasing freq
    off = 0
    base_freq = [5,6,7]
    freq_grad = 3.5
    vcos = []
    ref_osc = []
    dec_f1 = []
    
    for f in base_freq:
        ba = attractor_ensemble()
        nd = ba['enc'].shape[1]
        phase_fun = lambda t: 2*pi*t*((f + off) + freq_grad*t)
        cos_mask = zeros(nd)
        cos_mask[1] = 1.0
        sin_mask = zeros(nd)
        sin_mask[2] = 1.0
        ba['value_inp'] = lambda t: cos(phase_fun(t))*cos_mask + sin(phase_fun(t))*sin_mask
        vcos.append(ba)
        tt,xin,s,do = nef.sim({'ring': ba}, duration=duration)
        
        f1 = do['X'][[1,2],:]

        ent_osc = cos(phase_fun(tt)) + 1j*sin(phase_fun(tt))
        ref_osc.append(ent_osc)
        dec_f1.append(f1[0,:] + 1j*f1[1,:])
        
#         plot(t, osc.T)
#         plot(t, ent_osc.real)
    
    fsum_dec_osc = prod(np.array(dec_f1), axis=0)
    #fsum_dec_osc = fsum_dec_osc/np.absolute(fsum_dec_osc)
    fsum_dec_osc = fsum_dec_osc/8.25
    fsum_ref_osc = prod(np.array(ref_osc), axis=0)
    
    # now use decoded oscillation with summed freq to entrain third harmonic
    ba = attractor_ensemble(ncells=ncells, fmax=6, sigma=0.045, speed=9.25)
    #ba['value_inp'] = lambda t: 0.72*np.array([0, 0, 0, 0, 0, fsum_dec_osc[tt == t][0].real, fsum_dec_osc[tt == t][0].imag])
    ba['value_inp'] = lambda t: (0*t + 1.0)*np.array([0, 0, 0, 0, 0, fsum_dec_osc[tt == t][0].real, fsum_dec_osc[tt == t][0].imag, 0, 0, 0, 0, 0, 0])
    #(-.35*t + 0.9)*
    tt,xin,s,do = nef.sim({'ring': ba}, duration=duration)
    
    from pylab import figure, plot, gca, xlabel, ylabel, xlim, ylim
    figure()
    
    
#     plot(tt, fsum_dec_osc.real)
#     plot(tt, fsum_dec_osc.imag)
#     plot(tt, np.absolute(fsum_dec_osc))
    plot(tt, 0.5*dec_f1[1].real, label='VCO phase mean')
    plot(tt, xin['ring'][5,:], label='Input: VCO phase sum')
    plot(tt, do['X'][[1],:].T, label='F1 cosine')
    #plot(tt, do['X'][[3],:].T, label='F2 cosine')
    plot(tt, do['X'][[5],:].T, label='F3 cosine')
    ylabel('Amplitude')
    xlabel('Time (s)')
    xlim(0, duration)
    ylim(-1, 1)
    
    
    figure()
    import plotting
    plotting.spike_raster(gca(), tt, s['ring'])
    ylabel('Cell')
    xlabel('Time (s)')
    xlim(0, duration)
    ylim(0, ncells)

    data = {'t': tt,
             'vcophasesum': xin['ring'][5,:] + 1j*xin['ring'][6,:],
             'vcophasemean': 0.5*dec_f1[1],
             'cosF1': do['X'][1,:],
             'spikes': s['ring']}
    return data
#     
#         plot(t, ent_osc.real)
#     return fsum_dec_osc, fsum_ref_osc, tt
    #enc, gain, bias, X, A, fb_fun = bump_attractor(speed=8)
#     ens = attractor_ensemble()
    #inp, ori = nef.basic_attractor(enc, X, A, fb_fun)
    #ext_inp = lambda t,n: multi_osc(t,n,np.array(ent_freq))
    #ext_inp = nef.impulse_fun(0, 0.25, 1)
    #t, spikes, dec_ori = nef.sim(enc, gain, bias, inp, ori, duration, ext_inp)
    #return t, spikes, dec_ori
    
    