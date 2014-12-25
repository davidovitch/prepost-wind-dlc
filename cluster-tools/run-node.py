#!python
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 25 22:38:43 2014

@author: dave
"""

import os
#from argparse import ArgumentParser
import argparse
import paramiko

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="python script to execute")
    parser.add_argument("-n", "--node",  default='g-080',
                    help="gorm node hoste name, between g-001 and g-080")
    args = parser.parse_args()

    # connect to a node for the post processing
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname='g-080', username=os.environ['USER'])

    stdin, stdout, stderr = client.exec_command('python %s' % args.script)
    for line in stdout:
        print '... ' + line.strip('\n')
    client.close()