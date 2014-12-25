'''
Created on 11/07/2013

@author: mmpe
'''

from mmpe.cython_compile.cython_compile import py2pyx, py2pyx_autodeclare
import inspect
import numpy as np
import unittest


class Test_Py2Pyx(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.t1py = inspect.getsourcelines(t1)[0]
        cls.t2py = inspect.getsourcelines(t2)[0]
        cls.pyx_header = ['import cython\n', 'import numpy as np\n', 'cimport numpy as np\n']

    def tearDown(self):
        pass

    def test_no_defs(self):
        self.assertEqual(self.pyx_header + self.t1py, py2pyx(self.t1py))

    def test_with_defs(self):
        self.assertEqual(self.pyx_header + ['cpdef t2(int a, float b):\n',
                                            '    cdef int c\n',
                                            '    c = 1.\n',
                                            '    pass\n'], py2pyx(self.t2py))

    def test_with_defaults(self):
        def temp(a=1., b=2.):
            pass

        self.assertEqual(self.pyx_header + ['        cpdef temp(long long a=1., long long b=2.):\n',
                                            '            pass\n'],
                         py2pyx_autodeclare(temp, (1,), {'b':2}))




    def test_autodeclare_arguments(self):
        self.assertEqual(self.pyx_header + ['cpdef t1(long long a, double b):\n',
                                            "    pass\n"],
                         py2pyx_autodeclare(t1, (1, 2.), {}))


    def test_autodeclare_cdef(self):
        def temp():
            #cdef long long int a
            #cdef float _b # comment
            #cdef double _cc#comment
            #cdef np.ndarray[sdfl] d
            a = 1
            _b = 2.
            _cc = 3
            d = 4

        indent = " " * 8
        self.assertEqual(self.pyx_header + [indent + 'cpdef temp():\n',
                                            indent + '    cdef long long int a\n',
                                            indent + '    cdef float _b # comment\n',
                                            indent + '    cdef double _cc#comment\n',
                                            indent + '    cdef np.ndarray[sdfl] d\n',
                                            indent + '    a = 1\n',
                                            indent + '    _b = 2.\n',
                                            indent + '    _cc = 3\n',
                                            indent + '    d = 4\n'],
                         py2pyx_autodeclare(temp, [], {}))

    def test_autodeclare_assign(self):
        def temp():
            #cdef float c
            a = 1
            b = 2.
            c = 3.  # declared by cdef
            b = 4  # declared by b= 2.
        indent = " " * 8
        self.assertEqual(self.pyx_header + [indent + 'cpdef temp():\n',
                                            indent + '    cdef long long a\n',
                                            indent + '    cdef double b\n',
                                            indent + '    cdef float c\n',
                                            indent + '    a = 1\n',
                                            indent + '    b = 2.\n',
                                            indent + '    c = 3.  # declared by cdef\n',
                                            indent + '    b = 4  # declared by b= 2.\n'],
                         py2pyx_autodeclare(temp, [], {}))


    def test_autodeclare_in(self):
        def temp():
            #cdef float c
            for a in range(5): pass
            for b in range(5):
                pass
            for c in [.1, .2]: pass
            for d, e in list(zip(['a', 'b'], [.1, .2])):
                pass

        indent = " "*8
        self.assertEqual(self.pyx_header + [indent + 'cpdef temp():\n',
                                            indent + '    cdef long long a\n',
                                            indent + '    cdef long long b\n',
                                            indent + '    cdef double e\n',
                                            indent + '    cdef float c\n',
                                            indent + '    for a in range(5): pass\n',
                                            indent + '    for b in range(5):\n',
                                            indent + '        pass\n',
                                            indent + '    for c in [.1, .2]: pass\n',
                                            indent + "    for d, e in list(zip(['a', 'b'], [.1, .2])):\n",
                                            indent + '        pass\n'],
                         py2pyx_autodeclare(temp, [], {}))




    def test_autodeclare_all_types(self):
        def temp(a, b, c, d, e):
            pass
        self.assertEqual(self.pyx_header + ['        cpdef temp(long long a, double b, long long c, np.ndarray[np.int32_t,ndim=1] d, np.ndarray[np.float64_t,ndim=2] e):\n',
                                            '            pass\n'],
                         py2pyx_autodeclare(temp, (1, 2., 3, np.array([4]), np.array([[5], [6]], dtype=np.float64)), {}))


    def test_autodeclare_complex(self):
        pyx = self.pyx_header + [
               'cpdef t3(long long a, double b):\n',
               '    cdef long long c\n',
               '    cdef long long d\n',
               '    cdef long long e\n',
               '    cdef double f\n',
               '    cdef np.ndarray[np.float64_t,ndim=2] g\n',
               '    c = 1\n',
               '    for d in range(5):\n',
               '        pass\n',
               '    for e, f in list(zip(range(5), [0.1])):\n',
               '        pass\n',
               '    g = np.array([[3, 4], [5., 6]])\n']


        self.assertEqual(pyx, py2pyx_autodeclare(t3, [1, 2.], {}))

    def test_autodeclare_and_defs(self):
        pyx = self.pyx_header + [
               'cpdef t2(int a, float b):\n',
               '    cdef int c\n',
               '    c = 1.\n',
               '    pass\n']
        self.assertEqual(pyx, py2pyx_autodeclare(t2, [1., 1], {}))

def t1(a, b):
    pass

def t2(a, b):  #cpdef t2(int a, float b):
    #cdef int c
    c = 1.
    pass


def t3(a, b):
    c = 1
    for d in range(5):
        pass
    for e, f in list(zip(range(5), [0.1])):
        pass
    g = np.array([[3, 4], [5., 6]])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testpy2pyx']
    unittest.main()
