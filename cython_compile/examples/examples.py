'''
Created on 11/07/2013

@author: Mads M. Pedersen (mmpe@dtu.dk)
'''
from tests.mytimeit import print_time, get_time
import math
from mmpe.cython_compile import cython_import, cython_compile, \
    cython_compile_autodeclare

def pycheck(p):
    for i in range(10):
        for y in range(2, int(math.sqrt(p)) + 1):
            if p % y == 0:
                return False
    return True


@cython_compile
def cycheck_compile(p):
    import math
    for i in range(10):
        for y in range(2, int(math.sqrt(p)) + 1):
            if p % y == 0:
                return False
    return True


@cython_compile_autodeclare
def cycheck_compile_autodeclare(p):
    import math
    for i in range(10):
        for y in range(2, int(math.sqrt(p)) + 1):
            if p % y == 0:
                return False
    return True


p = 17

print pycheck(p)

cython_import('cycheck')
import cycheck
print cycheck.cycheck(p)
print cycheck.cycheck_pure(p)
print cycheck.cycheck_cdef(p)

print cycheck_compile(p)

print cycheck_compile_autodeclare(p)


