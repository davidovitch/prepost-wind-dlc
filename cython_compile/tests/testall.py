'''
Created on 12/07/2013

@author: mmpe
'''

import glob
import os
import unittest

def module_strings():
    test_file_strings = glob.glob('test_*.py')
    test_file_strings.extend(glob.glob('*/test_*.py'))
    return [s[0:len(s) - 3].replace("\\", ".") for s in test_file_strings]


def suite():
    suites = [unittest.defaultTestLoader.loadTestsFromName(s) for s in module_strings()]
    return unittest.TestSuite(suites)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = suite()
    runner.run (test_suite)
else:
    # for run as pydev unit-test
    for mstr in module_strings():
        exec("from %s import *" % mstr.replace("\\" , "."))



for root, folder, files in os.walk("."):
    for f in files:
        if f.endswith(".pyd"):
            try:
                os.remove(os.path.join(root, f))
            except:
                pass
