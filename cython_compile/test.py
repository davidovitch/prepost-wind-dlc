from mmpe.cython_compile import is_compiled, cython_import
cy_test = cython_import("cy_test")
name = 'cy_test'

print (cy_test.CyTest(2))  #4
print (is_compiled(cy_test))
