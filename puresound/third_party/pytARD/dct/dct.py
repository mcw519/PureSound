# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:00:03 2022

@author: smailnik@students.zhaw.ch
"""
import numpy as np
from numpy import pi, exp
from numpy.fft import fft as cpu_fft
import cupy as cp
from cupy.fft import fft as gpu_fft
from scipy.fft import dct as scipy_dct


def dct_cpu(x):
    v = np.zeros_like(x)
    N = len(x) # N is length of input signal x
    k = np.arange(N)
    
    v[:(N-1)//2+1] = x[::2]
    
    if N % 2: # odd length
        v[(N-1)//2+1:] = x[-2::-2]
    else: # even length
        v[(N-1)//2+1:] = x[::-2]
    
    V = cpu_fft(v)
    
    V *= 2 * exp(-1j*pi*k/(2*N))
    return V.real

def dct_gpu(x):
    v = cp.zeros_like(x)
    N = len(x) # N is length of input signal x
    k = cp.arange(N)
    
    v[:(N-1)//2+1] = x[::2]
    
    if N % 2: # odd length
        v[(N-1)//2+1:] = x[-2::-2]
    else: # even length
        v[(N-1)//2+1:] = x[::-2]
    
    V = gpu_fft(v)
    
    V *= 2 * exp(-1j*pi*k/(2*N))
    return V.real


# https://docs.cupy.dev/en/stable/user_guide/fft.html
import unittest
import time
class TestDCT(unittest.TestCase):
    # @staticmethod
    def test_dct_results(self):
        samples = 1000000
        t = np.arange(samples)
        #x_cpu = np.sin(t)*np.cos(t) # signal
        x_cpu = np.random.rand(samples) # signal
        x_gpu = cp.asarray(x_cpu)
        s_dct = []
        R=10
        print(f'average time from {R} runs with {samples} samples:')
        
        gpu_res = None
        cpu_res = None
        scipy_dct_res = None
        
        times = []
        # for t in range(R):
        tick = time.time()
        gpu_res = cp.asnumpy(dct_gpu(x_gpu))
        tack = time.time()
        times.append(tack-tick)
        print('gpu:',np.average(times))
        
        times = []
        for t in range(R):
            tick = time.time()
            cpu_res = dct_cpu(x_cpu)
            tack = time.time()
            times.append(tack-tick)
        print('cpu:',np.average(times))


        times = []
        for t in range(R):
            tick = time.time()
            scipy_dct_res = scipy_dct(x_cpu)
            tack = time.time()
            times.append(tack-tick)
        print('scipy:',np.average(times))

        
        np.testing.assert_almost_equal(scipy_dct_res, gpu_res, decimal=10)
        np.testing.assert_almost_equal(scipy_dct_res, cpu_res, decimal=10)
        np.testing.assert_almost_equal(cpu_res, gpu_res, decimal=10)

        print(gpu_res[:10])
        print(cpu_res[:10])
        print(scipy_dct_res[:10])

# use conda
#  python -m unittest dct.py

# average time from 10 runs with 500 samples:
# gpu: 0.02030034065246582
# cpu: 0.00010449886322021485

# average time from 10 runs with 10000 samples:
# gpu: 0.02060539722442627
# cpu: 0.0008929252624511718

# average time from 10 runs with 10000000 samples:
# gpu: 0.2829333543777466
# cpu: 1.7449397802352906

# average time from 10 runs with 15000000 samples:
# gpu: 0.4354936361312866
# cpu: 2.506287670135498

# average time from 10 runs with 17000000 samples:
# gpu: 0.5117461919784546
# cpu: 3.3482783794403077

# average time from 10 runs with 18000000 samles:
# gpu: 0.5012950897216797
# cpu: 3.723789358139038

# t = np.arange(256)
# x_cpu = np.sin(t) # signal
# x_gpu = cp.asarray(x_cpu)
# # print(dct_cpu(x_cpu))   
# print(dct_gpu(x_gpu))   
