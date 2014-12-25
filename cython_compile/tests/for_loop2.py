def CyTest2(n):  #cpdef CyTest2(int n):
    #cdef int i
    #cdef int count
    count = 0
    for i in range(n):
        count = count + i % 100
    return count
