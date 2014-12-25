'''
Created on 11/07/2013

@author: mmpe
'''
from io import StringIO
import inspect
import math
import os
import subprocess
import sys
import time
import unittest

from mmpe.cython_compile import cython_import, cython_compile
from mmpe.cython_compile.tests import cleanup
from mmpe.functions.timing import get_time



class Test_cython_compile(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):

        subprocess.Popen(r'python cleanup.py', cwd=os.path.realpath("."))
        super(Test_cython_compile, cls).tearDownClass()

#
#    def test_compiled(self):
#
#        clean("test_cython_compile_CyTest")
#        self.assertTrue(CyTest(3)[0])
#        self.assertEqual(CyTest(3)[1], 6)
#
#    def test_compiled_package(self):
#        clean("pack/cy_test_CyTest")
#        from pack import cy_test
#        self.assertTrue(cy_test.CyTest(3)[0])
#        self.assertEqual(cy_test.CyTest(3)[1], 6)
#
#
#    def test_for_loop(self):
#        clean("test_cython_compile_CyTest1")
#        CyTest1(1)
#        _, t_cy = get_time(CyTest1)(10000000)
#        _, t_py = get_time(PyTest1)(10000000)
#        self.assertTrue(t_cy * 1.2 < t_py, (t_cy, t_py))
#
#    def test_for_loop2(self):
#        clean("test_cython_compile_CyTest2")
#        CyTest2(1)
#        res1, t_cy = get_time(CyTest2)(10000000)
#        res2, t_py = get_time(PyTest2)(10000000)
#        self.assertTrue(res1, res2)
#        self.assertTrue(t_cy * 20 < t_py, "20*%s=%s not < %s" % (t_cy, 20 * t_cy, t_py))
#
#    def test_not_compiled(self):
#
#        sys.stderr = StringIO()
#        self.assertEqual(CyFail(9), 3)
#        sys.stderr.seek(0)
#
#        self.assertIn("UserWarning: Compilation or import of", sys.stderr.read())
#        sys.stderr = sys.__stderr__
#
#    def test_CyCheck(self):
#        print(CyCheck(17))
#
#    def test_pycheck_cdef(self):
#        print(pycheck_cdef(17))
#
#
#    def test_numpy(self):
#        clean("test_cython_compile_Cy_numpy")
#        self.assertEqual(list(Cy_numpy()), [1, 2])
#
#    def test_np(self):
#        clean("test_cython_compile_Cy_np")
#        self.assertEqual(list(Cy_np()), [1, 2])


    def test_char(self):
        self.assertEqual(Cy_char('a'.encode()), b'A')


    def compare(self, module_name, func, *args):
            clean(module_name)
            with open(module_name + '.py', 'w') as fid:
                fid.write(inspect.getsource(func))


            res1, t_py = get_time(func)(*args)

            cython_import(module_name)
            cmodule = __import__(module_name)
            cfunc = getattr(cmodule, func.__name__)
            res2, t_cy = get_time(cfunc)(*args)
            self.assertEqual(res1, res2, "%s - %s" % (module_name, func))
            clean(module_name)
            return t_py, t_cy


def clean(filename):
    for ext in ['.pyd']:
        if os.path.isfile(filename + ext):
            try:
                os.remove(filename + ext)
            except WindowsError:
                pass

@cython_compile
def CyTest(n):
    return __file__.endswith(".pyd"), n * 2



@cython_compile
def CyTest1(n):
    for i in range(n):
        pass

def PyTest1(n):
    for i in range(n):
        pass

@cython_compile
def CyTest2(n):  #cpdef CyTest2(int n):
    #cdef int i
    #cdef int count
    count = 0
    for i in range(n):
        count = count + i % 100
    return count

@cython_compile
def CyFail(p):
    return math.sqrt(p)


@cython_compile
def CyCheck(p):
    import math
    for i in range(10):
        for y in range(2, int(math.sqrt(p)) + 1):
            if p % y == 0:
                return False
    return True


def PyTest2(n):
    count = 0
    for i in range(n):
        count = count + i % 100
    return count

@cython_compile
def pycheck_cdef(p):  #cpdef pycheck_cdef(unsigned long long p):
    import math
    #cdef int y
    for y in range(2, int(math.sqrt(p)) + 1):
        if p % y == 0:
            return False
    return True

@cython_compile
def Cy_numpy():
    import numpy
    x = numpy.array([1, 2])
    return x

@cython_compile
def Cy_np():
    import numpy as np
    x = np.array([1, 2])
    return x

@cython_compile
def Cy_char(c):  #cpdef bytes Cy_char(char* c):
    return c.upper()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
