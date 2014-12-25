# -*- coding: utf-8 -*-
#!python
"""
Created on Thu Sep 25 22:21:06 2014

@author: dave
"""

import os
import dlc_analysis

if __name__ == '__main__':

	# DEBUG mode from Spyder: Change working directory
    # os.chdir('\\\\mimer\\hawc2sim\\DTU10MW\\C0002\\')
    # find the current path
    path = os.getcwd()
    # and assume we are in a simulation case of a certain turbine/project
    dlc_analysis.P_RUN = os.sep.join(path.split(os.sep)[:-2])
    # MODEL SOURCES, exchanche file sources
    dlc_analysis.P_SOURCE = dlc_analysis.P_RUN
    dlc_analysis.PROJECT = path.split(os.sep)[-2]
    sim_id = path.split(os.sep)[-1]

    # and post process
    resdir = '%s/%s/%s/' % (dlc_analysis.P_RUN, dlc_analysis.PROJECT, sim_id)
      # Do the whole thing:
    df_stats = dlc_analysis.post_launch(sim_id, statistics=True, check_logs=True, force_dir=resdir)
	# Skip Log analysis:
#    df_stats = dlc_analysis.post_launch(sim_id, statistics=True, check_logs=False, force_dir=resdir)
	# Skip statistic analysis:
#    df_stats = dlc_analysis.post_launch(sim_id, statistics=False, check_logs=True, force_dir=resdir)

    # # and plotting
    fig_dir = resdir
    dlc_analysis.plot_stats(sim_id, fig_dir_base=fig_dir)
