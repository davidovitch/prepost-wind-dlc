'''
Created on 21/11/2013

@author: mmpe
'''
import os
import sys
import time


def cleanup(folder="."):
    for f in os.listdir():
        _, ext = os.path.splitext(f)
        if ext in ['.c', '.pyx', '.pyd']:
            os.remove(f)
            print (f)
#    tmp_lst = lambda : [f for f in os.listdir(".") if f.endswith('.pyd')]
#    for i in range(3):
#        try:
#            for f in tmp_lst():
#                print ("try remove %s" % f,)
#                os.remove(f)
#                print ("Done")
#            sys.exit(0)
#        except IOError:
#            print ("Failed")
#            time.sleep(1)
if __name__ == "__main__":
    cleanup()
