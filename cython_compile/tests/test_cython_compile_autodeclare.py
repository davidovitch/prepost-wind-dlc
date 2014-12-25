'''
Created on 11/07/2013

@author: mmpe
'''
from mmpe.cython_compile import cython_import, is_compiled, cython_compile, \
    cython_compile_autodeclare
import os
import unittest
from mmpe.cython_compile.tests.common.test import peak_trough
import numpy as np
from mmpe.functions.timing import get_time


class Test_cython_compile_autodeclare(unittest.TestCase):

#    def test_compiled(self):
#
#        clean("test_cython_compile_autodeclare_CyTest")
#        self.assertTrue(CyTest(3)[0])
#        self.assertEqual(CyTest(3)[1], 6)
#
#
#    def test_for_loop(self):
#        clean("test_cython_compile_autodeclare_CyTest2")
#        CyTest2(1)
#        res1, t_cy = get_time(CyTest2)(10000000)
#        res2, t_py = get_time(PyTest2)(10000000)
#        self.assertTrue(res1, res2)
#        self.assertTrue(t_cy * 20 < t_py, "t_cy, t_py: %s, %s" % (t_cy, t_py))
#
#    def test_undef_var(self):
#        clean("test_cython_compile_autodeclare_CyTest3")
#        CyTest3(1)
#
#    def test_longlong(self):
#        clean("test_cython_compile_autodeclare_PrimeCheck")
#        self.assertTrue(PrimeCheck(32416190071))
#
#    def test_for_loop2(self):
#        clean("test_cython_compile_autodeclare_CyTest4")
#        CyTest4(np.array([1, 2]))
#
#
#    def test_numpy(self):
#        clean("test_cython_compile_autodeclare_Cy_numpy")
#        self.assertEqual(list(Cy_numpy()), [1, 2])
#
#    def test_np(self):
#        clean("test_cython_compile_autodeclare_Cy_np")
#        self.assertEqual(list(Cy_np()), [1, 2])


    def test_compiled_peak_trough(self):

        clean("common/test/peak_trough_peak_trough")
        peak_trough.peak_trough(np.array([-2, 0, 1, 0, -3, 0, 5, 0, -1, 0, 3, 0, -4, 0, 4, 0, -2]), 5)



def clean(filename):
    for ext in ['.py', '.pyd']:
        if os.path.isfile(filename + ext):
            try:
                os.remove(filename + ext)
            except WindowsError:
                pass

@cython_compile_autodeclare
def CyTest(n):
    return __file__.endswith(".pyd"), n * 2


@cython_compile_autodeclare
def CyTest2(n):
    count = 0
    for i in range(n):
        count = count + i % 100
    return count


def PyTest2(n):
    count = 0
    for i in range(n):
        count = count + i % 100
    return count

@cython_compile_autodeclare
def CyTest3(n):
    k = 0
    for j in range(n):
        count = (j + k) % 100
    return count

@cython_compile_autodeclare
def CyTest4(A):
    count = 0
    for i in A:
        count = count + i % 100
    return count

@cython_compile_autodeclare
def PrimeCheck(p):
    import math
    for y in range(2, int(math.sqrt(p)) + 1):
        if p % y == 0:
            return False
    return True

@cython_compile_autodeclare
def Cy_numpy():
    import numpy
    x = numpy.array([1, 2])
    return x

@cython_compile_autodeclare
def Cy_np():
    x = np.array([1, 2])
    return x

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
