'''
Created on 16/07/2013

@author: mmpe
'''

from fatigue_tools.fatigue import eq_load, rainflow_astm, rainflow_windap, \
    cycle_matrix
import unittest
import numpy as np
from hawc2 import Hawc2io

class Test(unittest.TestCase):


    def test_astm1(self):

        signal = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])

        ampl, mean = rainflow_astm(signal)
        np.testing.assert_array_equal(np.histogram2d(ampl, mean, [6, 4])[0], np.array([[ 0., 1., 0., 0.],
                                                                                                           [ 1., 0., 0., 2.],
                                                                                                           [ 0., 0., 0., 0.],
                                                                                                           [ 0., 0., 0., 1.],
                                                                                                           [ 0., 0., 0., 0.],
                                                                                                           [ 0., 0., 1., 2.]]))

    def test_windap1(self):
        signal = np.array([-2.0, 0.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -1.0, 0.0, 3.0, 0.0, -4.0, 0.0, 4.0, 0.0, -2.0])
        ampl, mean = rainflow_windap(signal, 18, 2)
        np.testing.assert_array_equal(np.histogram2d(ampl, mean, [6, 4])[0], np.array([[ 0., 0., 1., 0.],
                                                                                                   [ 1., 0., 0., 2.],
                                                                                                   [ 0., 0., 0., 0.],
                                                                                                   [ 0., 0., 0., 1.],
                                                                                                   [ 0., 0., 0., 0.],
                                                                                                   [ 0., 0., 2., 1.]]))

    def test_windap2(self):
        data = Hawc2io.ReadHawc2("test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(eq_load(data, neq=61), np.array([1.356, 1.758, 2.370, 2.784, 3.077, 3.296]), 0.001)


    def test_astm2(self):
        data = Hawc2io.ReadHawc2("test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(eq_load(data, neq=61, rainflow_func=rainflow_astm), np.array([1.361, 1.765, 2.378, 2.791, 3.083, 3.302]), 0.001)


    def test_windap3(self):
        data = Hawc2io.ReadHawc2("test").ReadBinary([2]).flatten()
        np.testing.assert_array_equal(cycle_matrix(data, 4, 4, rainflow_func=rainflow_windap)[0], np.array([[  14., 65., 39., 24.],
                                                                   [  0., 1., 4., 0.],
                                                                   [  0., 0., 0., 0.],
                                                                   [  0., 1., 2., 0.]]))


    def test_astm3(self):
        data = Hawc2io.ReadHawc2("test").ReadBinary([2]).flatten()
        np.testing.assert_allclose(cycle_matrix(data, 4, 4, rainflow_func=rainflow_astm)[0], np.array([[ 24., 83., 53., 26.],
                                                                                                           [  0., 1., 4., 0.],
                                                                                                           [  0., 0., 0., 0.],
                                                                                                           [  0., 1., 2., 0.]]), 0.001)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
