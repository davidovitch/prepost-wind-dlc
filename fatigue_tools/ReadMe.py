'''
Created on 10/07/2014

@author: MMPE

Functions to calculate fatigue loads for python 3

The low level methods are compiled using cython for 64 bit
'''

import os

from fatigue.Fatigue import eq_load, rainflow_windap, cycle_matrix, \
    rainflow_astm
import numpy as np


os.chdir("../")  # When compile is needed current working directory must be parent of "fatigue"
print (os.getcwd())
signal = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])
print (eq_load(signal, no_bins=50, neq=17, rainflow_func=rainflow_windap))
print (eq_load(signal, no_bins=50, neq=17, rainflow_func=rainflow_astm))
print (cycle_matrix(signal, 4, 4, rainflow_func=rainflow_windap))
print (cycle_matrix(signal, 4, 4, rainflow_func=rainflow_astm))
print (cycle_matrix([(.5, signal), (.5, signal + 2)], 4, 8, rainflow_func=rainflow_astm))