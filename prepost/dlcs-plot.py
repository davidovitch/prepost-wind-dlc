# -*- coding: utf-8 -*-
#!python
"""
Created on Fri Sep 26 12:02:17 2014

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

    resdir = '%s/%s/%s/' % (dlc_analysis.P_RUN, dlc_analysis.PROJECT, sim_id)

    # and post process
    dlc_analysis.plot_stats(sim_id, fig_dir_base=resdir)
