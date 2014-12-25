# -*- coding: utf-8 -*-
#!python
"""
Created on Thu Sep 25 22:14:11 2014

@author: dave
"""

import os

import dlc_analysis


if __name__ == '__main__':

    # find the current path
    path = os.getcwd()
    # and assume we are in a simulation case of a certain turbine/project
    dlc_analysis.P_RUN = os.sep.join(path.split(os.sep)[:-2])
    # MODEL SOURCES, exchanche file sources
    dlc_analysis.P_SOURCE = dlc_analysis.P_RUN

    dlc_analysis.PROJECT = path.split(os.sep)[-2]
    sim_id = path.split(os.sep)[-1]

    dlc_analysis.MASTERFILE = 'nrel_5mw_master_%s.htc' % sim_id

    print '   P_RUN: %s' % dlc_analysis.P_RUN
    print 'P_SOURCE: %s' % dlc_analysis.P_SOURCE
    print ' PROJECT: %s' % dlc_analysis.PROJECT
    print '  sim_id: %s' % sim_id

    dlc_analysis.launch_dlcs(sim_id)