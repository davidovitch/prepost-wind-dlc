#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:43:25 2014

This code will run in the clusters native Python 2.4.3 environment. A PBS
script will be created and saved in pbs_in. Afterwards, this PBS script will
be submitted to the cluster's que with qsub, and will execute the defined
Python script in a Miniconda Python environment. The default environment is
called anaconda.

@author: dave
"""

__author__ = "David Verelst <dave@dtu.dk>"
__license__ = "GPL-2+"

#from __future__ import division
#from __future__ import print_function

import sys
import os
import subprocess as sproc

# only python 2.7 and up
#from argparse import ArgumentParser
# python 2.4
from optparse import OptionParser

PBS_TEMP = """
### Standard Output
#PBS -N [sim_id]_[fname]_qsub-py
#PBS -o ./pbs_out/qsub-py_[fname].out
### Standard Error
#PBS -e ./pbs_out/qsub-py_[fname].err
#PBS -W umask=003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=03:00:00
#PBS -lnodes=1:ppn=1
### Queue name
#PBS -q workq
### Browse to current working dir
cd $PBS_O_WORKDIR
pwd
### ===========================================================================
### run the job

export PATH=/home/MET/STABCON/miniconda/bin:$PATH

# activate the custom python environment:
source activate [py_env]

# add custom libraries to the python path
export PYTHONPATH=/home/MET/STABCON/repositories/prepost:$PYTHONPATH
export PYTHONPATH=/home/MET/STABCON/repositories/fatigue_tools:$PYTHONPATH

python [fpath_full]

# change permissions to read/write for the group of the files and folders: 
chmod 775 -R *
# deactivate the python environment
source deactivate

### ===========================================================================
### wait for jobs to finish
wait
exit
"""

def submit_pbs(fpath, fname, py_env):
    """
    Set the configurable PBS elements, and submit the pbs file
    """
    fpath_full = os.path.join(fpath, fname)
    pbs_script = PBS_TEMP.replace('[fpath_full]', fpath_full)
    pbs_script = pbs_script.replace('[fname]', fname)
    pbs_script = pbs_script.replace('[py_env]', py_env)
    # if we run several cases in parallel, also have the pbs_out file indicate
    # on which directory/sim_id it is working
    sim_id = os.getcwd().split(os.path.sep)[-1]
    pbs_script = pbs_script.replace('[sim_id]', sim_id)

    # write the pbs_script
    FILE = open('pbs_in/%s.p' % fname, 'w')
    FILE.write(pbs_script)
    FILE.close()
    # and submit
    cmd = 'qsub pbs_in/%s.p' % (fname)
    print 'submitting to cluster as:'
    print cmd
    p = sproc.Popen(cmd, stdout=sproc.PIPE, stderr=sproc.STDOUT, shell=True)

    # p.wait() will lock the current shell until p is done
    # p.stdout.readlines() checks if there is any output, but also locks
    # the thread if nothing comes back
    stdout = p.stdout.readlines()
    for line in stdout:
        print line
    # wait until qsub is finished doing its magic
    p.wait()

    return pbs_script, stdout


if __name__ == '__main__':

    # default_path = '/home/MET/STABCON/repositories/prepost'
    default_path = '/home/leob/bin/prepost'

    # parse the arguments, only relevant when using as a command line utility
    parser = OptionParser(usage = "%prog -f pythonfile")
    parser.add_argument = parser.add_option
    parser.add_argument('-f', '--file', type='string', dest='fname',
                        action='store', default=None,
                        help='python file name that should be run on cluster')
    parser.add_argument('-p', '--path', type='string', dest='fpath',
                        action='store', default=default_path,
                        help='path of the python file')
    parser.add_argument('--py_env', type='string', dest='py_env',
                        help='name of the python environment',
                        default='anaconda')

    # TODO: configure flags for default actions such post-process, plot, launch
    # in those cases the default folder layout is assumed

    # make sure a filename is given
    (options, args) = parser.parse_args()
    if options.fname is None:
        parser.print_usage()
        sys.stderr.write("error: specify a file name with -f" + os.linesep)
        sys.exit(1)
    if options.fpath == default_path:
        rpl = (options.fpath, os.linesep)
        sys.stderr.write("assuming default path: %s %s" % rpl)
    else:
        rpl = (options.fpath, os.linesep)
        sys.stderr.write("path to python file: %s %s" % rpl)

    # create the PBS file based on the template
    # write and launch the pbs script with qsub
    pbs_script, stdout = submit_pbs(options.fpath, options.fname, options.py_env)
