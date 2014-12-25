'''
Created on 11/07/2013

@author: mmpe
'''
from mmpe.cython_compile import cython_import, is_compiled

import inspect
import os
import unittest


import sys
from mmpe.functions.timing import get_time
class Test_cython_import(unittest.TestCase):


    def setUp(self):
        sys.path.append(".")

    def test_compiled(self):
        name = 'cy_test'
        self.compare(name, CyTest, 1)
        import cy_test
        self.assertEqual(cy_test.CyTest(2), 4)
        self.assertTrue(is_compiled(cy_test))

    def test_for_loop(self):
        t_py, t_cy = self.compare("for_loop1", CyTest1, 1000000)
        print (t_py, t_cy)
        self.assertTrue(t_cy * 1.2 < t_py)



    def test_for_loop2(self):
        t_py, t_cy = self.compare("for_loop2", CyTest2, 1000000)
        self.assertTrue(t_cy * 10 < t_py)


    def test_for_loop2_in_pack(self):
        t_py, t_cy = self.compare("pack.for_loop2", CyTest2, 1000000)
        self.assertTrue(t_cy * 10 < t_py, (t_cy, t_py))


    def compare(self, module_name, func, *args):
            clean(module_name)
            with open(module_name.replace(".", os.path.sep) + '.py', 'w') as fid:
                fid.write(inspect.getsource(func))

            res1, t_py = get_time(func)(*args)

            cython_import(module_name)
            cmodule = __import__(module_name, fromlist=module_name.split(".")[-1])
            self.assertTrue(is_compiled(cmodule), module_name)
            cfunc = getattr(cmodule, func.__name__)
            res2, t_cy = get_time(cfunc)(*args)
            self.assertEqual(res1, res2, "%s - %s" % (module_name, func))
            #clean(module_name)
            return t_py, t_cy


def clean(filename):
    for ext in ['.py', '.pyd']:
        if os.path.isfile(filename + ext):
            try:
                os.remove(filename + ext)
            except WindowsError:
                pass



def CyTest(n):
    return n * 2

def CyTest1(n):
    for i in range(n):
        pass

def CyTest2(n):  #cpdef CyTest2(int n):
    #cdef int i
    #cdef int count
    count = 0
    for i in range(n):
        count = count + i % 100
    return count

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
