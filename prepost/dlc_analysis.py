# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:21:11 2014

@author: dave
"""

from __future__ import division
from __future__ import print_function
#print(*objects, sep=' ', end='\n', file=sys.stdout)

import sys
#import logging
import os
import shutil
import socket

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigCanvas
#from scipy import interpolate as interp
#from scipy.optimize import fmin_slsqp
#from scipy.optimize import minimize
#from scipy.interpolate import interp1d
#import scipy.integrate as integrate
#http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
import pandas as pd

#import openpyxl as px
#import numpy as np

#sys.path.append("/home/dave/PhD/Projects/Hawc2Dev/")
#sys.path.append('/home/dave/Repositories/DTU/prepost')
#import windIO
#import plotting
import Simulations as sim

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)
if not socket.gethostname()[:2] == 'g-':
    plt.rc('text', usetex=True)
plt.rc('legend', fontsize=11)
plt.rc('legend', numpoints=1)
plt.rc('legend', borderaxespad=0)

def make_fig(nrows=1, ncols=1, figsize=(12,8), dpi=120):
    fig = mpl.figure.Figure(figsize=figsize, dpi=dpi)
    canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig)
    fig.set_canvas(canvas)
    axes = np.ndarray((nrows, ncols), dtype=np.object)
    plt_nr = 1
    for row in range(nrows):
        for col in range(ncols):
            axes[row,col] = fig.add_subplot(row+1, col+1, plt_nr)
            plt_nr += 1
    return fig, canvas, axes

def convert_to_utf8(filename):
    # gather the encodings you think that the file may be
    # encoded inside a tuple
    encodings = ('windows-1253', 'iso-8859-7', 'macgreek')

    # try to open the file and exit if some IOError occurs
    try:
        f = open(filename, 'r').read()
    except Exception:
        sys.exit(1)

    # now start iterating in our encodings tuple and try to
    # decode the file
    for enc in encodings:
        try:
            # try to decode the file with the first encoding
            # from the tuple.
            # if it succeeds then it will reach break, so we
            # will be out of the loop (something we want on
            # success).
            # the data variable will hold our decoded text
            data = f.decode(enc)
            break
        except Exception:
            # if the first encoding fail, then with the continue
            # keyword will start again with the second encoding
            # from the tuple an so on.... until it succeeds.
            # if for some reason it reaches the last encoding of
            # our tuple without success, then exit the program.
            if enc == encodings[-1]:
                sys.exit(1)
            continue

    # now get the absolute path of our filename and append .bak
    # to the end of it (for our backup file)
    fpath = os.path.abspath(filename)
    newfilename = fpath + '.bak'
    # and make our backup file with shutil
    shutil.copy(filename, newfilename)

    # and at last convert it to utf-8
    f = open(filename, 'w')
    try:
        f.write(data.encode('utf-8'))
    except Exception, e:
        print(e)
    finally:
        f.close()

def to_lower_case(p_root, sim_id):
    """
    """
    p_htc = 'htc'
    p_target = os.path.join(p_root, sim_id, p_htc)
    # find all dlc defintions in the subfolders
    for root, dirs, files in os.walk(p_target):
        for fname in files:
#            print root
#            print fname
            os.rename(root+'/'+fname, root+'/'+fname.lower())
        base = root.split('/')[-1]
        if base[:3] == 'DLC':
            new = root.replace(base, base.lower())
            os.rename(root, new)

def read_excel_files(p_root, sim_id):

    df_list = {}

    p_htc = 'htc'
    p_target = os.path.join(p_root, sim_id, p_htc)
    # find all dlc defintions in the subfolders
    for root, dirs, files in os.walk(p_target):
        if not root[-14:].lower() == '_iec61400-1ed3':
#            print('ignoring', root)
            continue
        for file_name in files:
            if not file_name[-5:] == '.xlsx':
                continue
            f_target = os.path.join(root, file_name)
            print(f_target, end='')
            try:
                xl = pd.ExcelFile(f_target)
                df = xl.parse("Sheet1")
    #            print df.head()
                df_list[f_target.replace('.xlsx', '')] = df
                print('')
            except:
                print('     XXXXX ERROR')

    return df_list

# =============================================================================
### MODEL
# =============================================================================

def master_tags(sim_id, runmethod='local', silent=False, verbose=False):
    """
    Create HtcMaster() object
    =========================

    the HtcMaster contains all the settings to start creating htc files.
    It holds the master file, server paths and more.

    The master.tags dictionary holds those tags who do not vary for different
    cases. Variable tags, i.e. tags who are a function of other variables
    or other tags, are defined in the function variable_tag_func().

    It is considered as good practice to define the default values for all
    the variable tags in the master_tags

    Members
    -------

    Returns
    -------

    """

    project = PROJECT

    # MODEL SOURCES, exchanche file sources
    p_local = P_SOURCE
    # target run dir
    p_root = P_RUN

    # TODO: write a lot of logical tests for the tags!!

    # FIXME: some tags are still variable! Only static tags here that do
    # not depent on any other variable that can change

    master = sim.HtcMaster(verbose=verbose, silent=silent)

    # =========================================================================
    # SOURCE FILES
    # =========================================================================
    # TODO: move to variable_tag
    rpl = (p_root, project, sim_id)
    rpl_loc = (p_local, project, sim_id)
    if runmethod in ['local', 'local-script', 'none', 'local-ram']:
        master.tags['[run_dir]'] = '%s/%s/%s/' % rpl
    elif runmethod == 'windows-script':
        master.tags['[run_dir]'] = '%s/%s/%s/' % rpl
    elif runmethod == 'gorm':
        master.tags['[run_dir]'] = '%s/%s/%s/' % rpl
    elif runmethod == 'jess':
        master.tags['[run_dir]'] = '%s/%s/%s/' % rpl
    else:
        msg='unsupported runmethod, options: none, local, gorm or opt'
        raise ValueError, msg

    master.tags['[master_htc_file]'] = MASTERFILE

    master.tags['[master_htc_dir]'] = '%s/%s/%s/htc/_master/' % rpl_loc

    # directory to data, htc, SOURCE DIR
    master.tags['[model_dir_local]']  = '%s/%s/%s/' % rpl_loc

    master.tags['[post_dir]'] = '%s/%s/python-prepost-data/' % (p_root, project)

    master.tags['[st_file]'] = 'NREL_5MW_st.txt'
    master.tags['[ae_file]'] = 'AeDist_Flap_01.dat'
    master.tags['[pc_file]'] = 'NREL_5MW_pc.txt'

    # -------------------------------------------------------------------------
    # semi variable tags that only change per simulation series
    # TODO: create a stand alone function for this? As in variable_tag_func?
    master.tags['[sim_id]'] = sim_id
    # folder names for the saved results, htc, data, zip files
    # Following dirs are relative to the model_dir_server and they specify
    # the location of where the results, logfiles, animation files that where
    # run on the server should be copied to after the simulation has finished.
    # on the node, it will try to copy the turbulence files from these dirs
    master.tags['[animation_dir]'] = 'animation/'
    master.tags['[control_dir]']   = 'control/'
    master.tags['[data_dir]']      = 'data/'
    master.tags['[eigenfreq_dir]'] = False
    master.tags['[htc_dir]']       = 'htc/'
    master.tags['[log_dir]']       = 'logfiles/'
    master.tags['[meander_dir]']   = False
    master.tags['[opt_dir]']       = False
    master.tags['[pbs_out_dir]']   = 'pbs_out/'
    master.tags['[res_dir]']       = 'res/'
    master.tags['[iter_dir]']      = 'iter/'
    master.tags['[turb_dir]']      = 'turb/'
    master.tags['[turb_db_dir]']   = '../turb/'
    master.tags['[wake_dir]']      = False
    master.tags['[hydro_dir]']     = False
    master.tags['[mooring_dir]']   = False

    # zip_root_files only is used when copy to run_dir and zip creation, define
    # in the HtcMaster object
    master.tags['[zip_root_files]'] = []
    # only active on PBS level, so files have to be present in the run_dir
    master.tags['[copyback_files]'] = []
    master.tags['[copyback_frename]'] = []
    master.tags['[copyto_files]'] = ['data/AeDist_Flap_01.dat',
                                    'data/FlapInp_NacaThk17.ds']
    master.tags['[copyto_generic]'] = ['data/AeDist_Flap_01.dat',
                                    'data/FlapInp_NacaThk17.ds']

    # set the model_zip tag to include the sim_id
    master.tags['[model_zip]'] = project
    master.tags['[model_zip]'] += '_' + master.tags['[sim_id]'] + '.zip'
    # -------------------------------------------------------------------------

    master.tags['[dt_sim]'] = 0.02

    # =========================================================================
    # basic required tags by HtcMaster and PBS in order to function properly
    # =========================================================================
    # case_id will be set with variable_tag_func
#    master.tags['[Case id.]'] = None
    #master.tags['[run_dir]'] = '/home/dave/tmp/'

    master.tags['[pbs_queue_command]'] = '#PBS -q workq'
#    master.tags['[pbs_queue_command]'] = '#PBS -q xpresq'
    # the express queue has 2 thyra nodes with max walltime of 1h
    #master.tags['[pbs_queue_command]'] = '#PBS -q xpresq'
    # walltime should have following format: hh:mm:ss
    #master.tags['[walltime]'] = '00:20:00'
    master.tags['[walltime]'] = '01:00:00'
    master.tags['[auto_walltime]'] = False

    return master

def variable_tag_func(master, case_id_short=False):
    """
    Function which updates HtcMaster.tags and returns an HtcMaster object

    Only use lower case characters for case_id since a hawc2 result and
    logfile are always in lower case characters.

    BE CAREFULL: if you change a master tag that is used to dynamically
    calculate an other tag, that change will be propageted over all cases,
    for example:
    master.tags['tag1'] *= master.tags[tag2]*master.tags[tag3']
    it will accumlate over each new case. After 20 cases
    master.tags['tag1'] = (master.tags[tag2]*master.tags[tag3'])^20
    which is not wanted, you should do
    master.tags['tag1'] = tag1_base*master.tags[tag2]*master.tags[tag3']
    """

    mt = master.tags

    # -------------------------------------------------------------------------
#    import code # To have keyboard()
#    code.interact(local=locals())
    mt['[duration]'] = str(float(mt['[time_stop]']) - float(mt['[t0]']))

    return master

# =============================================================================
### STAT PLOTS
# =============================================================================

def plot_stats(sim_ids, fig_dir_base=None):
    """
    For each wind speed, take the max of the max
    """

#    project = 'NREL5MW'
    project = PROJECT
#    proot = '/mnt/gorm/HAWC2'
    proot = P_RUN
#    proot = '/mnt/hawc2sim'
#    proot = '/mnt/vea-group/AED/STABCON/SIM'

    post_dir = '%s/%s/python-prepost-data/' % (proot, project)
#    fig_dir = '%s/%s/%s/figures/' % (proot, project, sim_id)
#    post_dir = '/home/dave/Desktop/'

    # if sim_id is a list, combine the two dataframes into one
    df_stats = pd.DataFrame()
    if type(sim_ids).__name__ == 'list':
        for ii, sim_id in enumerate(sim_ids):
            cc = sim.Cases(post_dir, sim_id, rem_failed=True)
            if ii == 0:
                df_stats = cc.load_stats()
            else:
                # because there is no unique index, we will ignore it
                df_stats = pd.concat([df_stats, cc.load_stats()], ignore_index=True)
    else:
        sim_id = sim_ids
        sim_ids = False
        cc = sim.Cases(post_dir, sim_id, rem_failed=True)
        df_stats = cc.load_stats()

#    if force_dir:
#        cc.change_results_dir(resdir=force_dir)
#        for case in cc.cases:
#            sim_id = cc.cases[case]['[post_dir]']
#            cc.cases[case]['[post_dir]'] = post_dir

#    # add DLC category
#    f = lambda x: x.split('_')[0]
#    df_stats['DLC'] = df_stats['[Case id.]'].map(f)

#    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,8), num=1)

    # define the number of positions you want to have the color for
    N = 22
    # select a color map
    cmap = mpl.cm.get_cmap('jet', N)
    # convert to array
    cmap_arr = cmap(np.arange(N))
    # color=cmap_arr[icol][0:3]

    # make a stastics plot for each channel
    gb_ch = df_stats.groupby(df_stats.channel)

    # channel selection
    plot_chans = {}
    plot_chans['DLL-2-inpvec-2'] = 'P_e'
    # plot_chans['bearing-shaft_rot-angle_speed-rpm'] = 'RPM'

    plot_chans['tower-tower-node-001-momentvec-x'] = 'M_x T_B'
    plot_chans['tower-tower-node-001-momentvec-y'] = 'M_y T_B'
    # plot_chans['tower-tower-node-001-momentvec-z'] = 'M_z T_B'

    # plot_chans['tower-tower-node-008-momentvec-z'] = 'M_x T_T'
    # plot_chans['tower-tower-node-008-momentvec-z'] = 'M_y T_T'
    # plot_chans['tower-tower-node-008-momentvec-z'] = 'M_z T_T'

    # plot_chans['shaft-shaft-node-004-momentvec-x'] = 'M_x Shaft_{MB}'
    # plot_chans['shaft-shaft-node-004-momentvec-y'] = 'M_y Shaft_{MB}'
    # plot_chans['shaft-shaft-node-004-momentvec-z'] = 'M_z Shaft_{MB}'

    plot_chans['blade1-blade1-node-003-momentvec-x'] = 'M_x B1_{root}'
    # plot_chans['blade1-blade1-node-003-momentvec-y'] = 'M_y B1_{root}'
    # plot_chans['blade1-blade1-node-003-momentvec-z'] = 'M_z B1_{root}'
    plot_chans['blade2-blade2-node-003-momentvec-x'] = 'M_x B2_{root}'
    # plot_chans['blade2-blade2-node-003-momentvec-y'] = 'M_y B2_{root}'
    # plot_chans['blade2-blade2-node-003-momentvec-z'] = 'M_z B2_{root}'
    plot_chans['blade3-blade3-node-003-momentvec-x'] = 'M_x B3_{root}'
    # plot_chans['blade3-blade3-node-003-momentvec-y'] = 'M_y B3_{root}'
    # plot_chans['blade3-blade3-node-003-momentvec-z'] = 'M_z B3_{root}'

    # plot_chans['global-blade1-elem-018-zrel-1.00-State pos-y'] = 'B1 U_y'
    # plot_chans['global-blade2-elem-018-zrel-1.00-State pos-y'] = 'B2 U_y'
    # plot_chans['global-blade3-elem-018-zrel-1.00-State pos-y'] = 'B3 U_y'

    # plot_chans['bearing-pitch1-angle-deg'] = 'B1_{pitch}'
    # plot_chans['bearing-pitch2-angle-deg'] = 'B2_{pitch}'
    # plot_chans['bearing-pitch3-angle-deg'] = 'B3_{pitch}'

    # plot_chans['setbeta-bladenr-1-flapnr-1'] = 'B1_{flap}'
    # plot_chans['setbeta-bladenr-2-flapnr-1'] = 'B2_{flap}'
    # plot_chans['setbeta-bladenr-3-flapnr-1'] = 'B3_{flap}'

#    plot_chans['DLL-5-inpvec-1'] = 'd tow-tip'
#    plot_chans['DLL-1-inpvec-27'] = 'tow acc lp'

    mfcs1 = ['k', 'w']
    mfcs2 = ['b', 'w']
    mfcs3 = ['r', 'w']
    stds = ['r', 'b']

    for nr, (ch_name, gr_ch) in enumerate(gb_ch):
        if ch_name not in plot_chans:
            continue
        for dlc_name, gr_ch_dlc in gr_ch.groupby(df_stats['[DLC]']):
            print('start plotting:  %s %s' % (str(dlc_name).ljust(7), ch_name))

            fig, canvas, axes = make_fig(nrows=1, ncols=1, figsize=(7,5))
            ax = axes[0,0]
            # seperate figure for the standard deviations
            fig2, canvas2, axes2 = make_fig(nrows=1, ncols=1, figsize=(7,5))
            ax2 = axes2[0,0]

            if fig_dir_base is None and not sim_ids:
                res_dir = gr_ch_dlc['[res_dir]'][:1].values[0]
                run_dir = gr_ch_dlc['[run_dir]'][:1].values[0]
                fig_dir = fig_dir_base + res_dir
            elif fig_dir_base and not sim_ids:
                res_dir = gr_ch_dlc['[res_dir]'][:1].values[0]
                fig_dir = fig_dir_base + res_dir
            elif sim_ids and fig_dir_base is not None:
                # create the compare directory if not defined
                fig_dir = fig_dir_base

            # if we have a list of different cases, we also need to group those
            # because the sim_id wasn't saved before in the data frame,
            # we need to derive that from the run dir
            # if there is only one run dir nothing changes
            ii = 0
            sid_names = []
            for run_dir, gr_ch_dlc_sid in gr_ch_dlc.groupby(df_stats['[run_dir]']):
                sid_name = run_dir.split('/')[-2]
                sid_names.append(sid_name)
                print(sid_name)
                wind = gr_ch_dlc_sid['[Windspeed]'].values
                dmin = gr_ch_dlc_sid['min'].values
                dmean = gr_ch_dlc_sid['mean'].values
                dmax = gr_ch_dlc_sid['max'].values
                dstd = gr_ch_dlc_sid['std'].values
                if not sim_ids:
                    lab1 = 'mean'
                    lab2 = 'min'
                    lab3 = 'max'
                    lab4 = 'std'
                else:
                    lab1 = 'mean %s' % sid_name
                    lab2 = 'min %s' % sid_name
                    lab3 = 'max %s' % sid_name
                    lab4 = 'std %s' % sid_name
                mfc1 = mfcs1[ii]
                mfc2 = mfcs2[ii]
                mfc3 = mfcs3[ii]
                ax.plot(wind, dmean, mec='k', marker='o', mfc=mfc1, ls='',
                        label=lab1, alpha=0.7)
                ax.plot(wind, dmin, mec='b', marker='^', mfc=mfc2, ls='',
                        label=lab2, alpha=0.7)
                ax.plot(wind, dmax, mec='r', marker='v', mfc=mfc3, ls='',
                        label=lab3, alpha=0.7)

                ax2.plot(wind, dstd, mec=stds[ii], marker='s', mfc=stds[ii], ls='',
                        label=lab4, alpha=0.7)

                ii += 1

#            for wind, gr_wind in  gr_ch_dlc.groupby(df_stats['[Windspeed]']):
#                wind = gr_wind['[Windspeed]'].values
#                dmin = gr_wind['min'].values#.mean()
#                dmean = gr_wind['mean'].values#.mean()
#                dmax = gr_wind['max'].values#.mean()
##                dstd = gr_wind['std'].mean()
#                ax.plot(wind, dmean, 'ko', label='mean', alpha=0.7)
#                ax.plot(wind, dmin, 'b^', label='min', alpha=0.7)
#                ax.plot(wind, dmax, 'rv', label='max', alpha=0.7)
##                ax.errorbar(wind, dmean, c='k', ls='', marker='s', mfc='w',
##                        label='mean and std', yerr=dstd)
            ax.grid()
            ax.set_xlim([3, 27])
            leg = ax.legend(loc='best', ncol=2)
            leg.get_frame().set_alpha(0.7)
            ax.set_title(r'{DLC%s} $%s$' % (dlc_name, plot_chans[ch_name]))
            ax.set_xlabel('Wind speed [m/s]')
            fig.tight_layout()
            fig.subplots_adjust(top=0.92)
            if not sim_ids:
                fig_path = fig_dir + ch_name.replace(' ', '_') + '.png'
            else:
                sids = '_'.join(sid_names)
#                fig_dir = run_dir.split('/')[:-1] + 'figures/'
                fname = '%s_%s.png' % (ch_name.replace(' ', '_'), sids)
                fig_path = fig_dir + 'dlc%s/' % dlc_name
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)
                fig_path = fig_path + fname
            fig.savefig(fig_path)#.encode('latin-1')
#            canvas.close()
            fig.clear()
            print('saved: %s' % fig_path)


            ax2.grid()
            ax2.set_xlim([3, 27])
            leg = ax2.legend(loc='best', ncol=2)
            leg.get_frame().set_alpha(0.7)
            ax2.set_title(r'{DLC%s} $%s$' % (dlc_name, plot_chans[ch_name]))
            ax2.set_xlabel('Wind speed [m/s]')
            fig2.tight_layout()
            fig2.subplots_adjust(top=0.92)
            if not sim_ids:
                fig_path = fig_dir + ch_name.replace(' ', '_') + '_std.png'
            else:
                sids = '_'.join(sid_names)
                fname = '%s_std_%s.png' % (ch_name.replace(' ', '_'), sids)
                fig_path = fig_dir + 'dlc%s/' % dlc_name
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)
                fig_path = fig_path + fname
            fig2.savefig(fig_path)#.encode('latin-1')
#            canvas.close()
            fig2.clear()
            print('saved: %s' % fig_path)

#def plot_stats():


# =============================================================================
### PRE- POST
# =============================================================================

def launch_dlcs(sim_id):
    """
    """
    # MODEL SOURCES, exchanche file sources
#    p_local = '/mnt/vea-group/AED/STABCON/SIM/NREL5MW'
    p_local = '%s/%s' % (P_SOURCE, PROJECT)
    # target run dir (is defined in the master_tags)
#    p_root = '/mnt/gorm/HAWC2/NREL5MW'

    df_list = read_excel_files(p_local, sim_id)

    iter_dict = dict()
    iter_dict['[empty]'] = [False]

    opt_tags = []
#    nan_dict = {}

    for dlc, df in df_list.iteritems():
#        # only consider one case
#        if not((dlc.find('dlc22y') > 0) or (dlc.find('dlc62') > 0)):
#            continue

        dlc_case = dlc.split(os.path.sep)[-1]
#        print dlc
#        print dlc_case.lower()
#        if dlc_case[:5] not in ['dlc22y']:  # Obs, doesn't work for some of the strings
#            continue
        for count, row in df.iterrows():
            # make sure we have all strings in the dictionary
            tags_dict = {key:str(val).lower() for key, val in dict(row).iteritems()}
            tags_dict['[res_dir]'] = 'res/%s/' % dlc_case
            tags_dict['[log_dir]'] = 'logfiles/%s/' % dlc_case
            tags_dict['[htc_dir]'] = 'htc/%s/' % dlc_case
            tags_dict['[case_id]'] = tags_dict['[Case id.]']
            tags_dict['[time_stop]'] = tags_dict['[time stop]']
            tags_dict['[turb_base_name]'] = tags_dict['[Turb base name]']
            tags_dict['[DLC]'] = tags_dict['[Case id.]'].split('_')[0][3:]
            tags_dict['[pbs_out_dir]'] = 'pbs_out/%s/' % dlc_case
            tags_dict['[pbs_in_dir]'] = 'pbs_in/%s/' % dlc_case
            tags_dict['[iter_dir]'] = 'iter/%s/' % dlc_case
            tags_dict['[sim_id]'] = sim_id
            # replace nan with empty
            for ii, jj in tags_dict.iteritems():
                if jj == 'nan':
                    tags_dict[ii] = ''
            opt_tags.append(tags_dict.copy())

#    print '#'*50
#    print nan_dict
#    print '#'*50

    runmethod = 'gorm'
#    runmethod = 'local-script'
#    runmethod = 'windows-script'
#    runmethod = 'jess'
    master = master_tags(sim_id, runmethod=runmethod)
    master.tags['[hawc2_exe]'] = 'hawc2mb.exe' # CHECK it matches wine

    # TODO: copy master and DLC exchange files to p_root too!!

    # all tags set in master_tags will be overwritten by the values set in
    # variable_tag_func(), iter_dict and opt_tags
    # values set in iter_dict have precedence over opt_tags
    # variable_tag_func() has precedense over iter_dict, which has precedence
    # over opt_tags. So opt_tags comes last
    sim.prepare_launch(iter_dict, opt_tags, master, variable_tag_func,
                    write_htc=True, runmethod=runmethod, verbose=False,
                    copyback_turb=True, msg='', update_cases=False,
                    ignore_non_unique=False, run_only_new=False)

def post_launch(sim_id, statistics=True, rem_failed=True, check_logs=True,
                force_dir=False):

#    project = 'NREL5MW'
#    proot = '/mnt/gorm/HAWC2'
#    proot = '/mnt/hawc2sim'
#    proot = '/mnt/vea-group/AED/STABCON/SIM'
    proot = P_RUN
    project = PROJECT
    post_dir = '%s/%s/python-prepost-data/' % (proot, project)

    # =========================================================================
    # check logfiles, results files, pbs output files
    # logfile analysis is written to a csv file in logfiles directory
    # =========================================================================
    # load the file saved in post_dir
    cc = sim.Cases(post_dir, sim_id, rem_failed=rem_failed)
    cc.force_lower_case_id()

    if force_dir:
        for case in cc.cases:
            sim_id = cc.cases[case]['[post_dir]']
            cc.cases[case]['[post_dir]'] = post_dir
            cc.cases[case]['[run_dir]'] = force_dir

    if check_logs:
        cc.post_launch()
    elif rem_failed:
        cc.remove_failed()

    if statistics:
        # for the default load case analysis, add mechanical power
#        add = {'ch1_name':'floater-floater-node-001-momentvec-z',
#               'ch2_name':'bearing-shaft_rot-angle_speed-rad/s',
#               'ch_name_add':'mechanical-power-floater-floater-001',
#               'factor':1.0, 'operator':'*'}
        i0, i1 = 0, -1

        tags = cc.cases[cc.cases.keys()[0]].keys()
        add = None

        df_stats = cc.statistics(calc_mech_power=False, i0=i0, i1=i1,
                                 tags=tags, add_sensor=add, ch_fatigue=None)

        return df_stats

def testing_dataframe():

    proot = P_RUN
    project = PROJECT
    post_dir = '%s/%s/python-prepost-data/' % (proot, project)
    cc = sim.Cases(post_dir, sim_id, rem_failed=True)
    cc.force_lower_case_id()
    df = cc.load_stats()
    dict2 = {}
    for key in df.keys():
        print(key.ljust(30), type(key).__name__)
        qqq = {type(k).__name__:None for k in df[key]}
        print(qqq, end='\n\n')
        tmp = df[key].values
        if qqq.keys()[0][:7] == 'unicode':
            tmp = np.array(tmp, dtype=np.str)
        elif qqq.keys()[0] == 'NoneType':
            tmp = np.array(tmp, dtype=np.str)
        dfs = pd.DataFrame({str(key):tmp})
        fpath = '/home/dave/SimResults/NREL5MW/python-prepost-data/test'
        dfs.to_hdf('%s.h5' % fpath, 'table', mode='w')
        dict2[str(key)] = tmp.copy()
    dfs2 = pd.DataFrame(dict2)
    dfs2.to_hdf('%s.h5' % fpath, 'table', mode='w')

if __name__ == '__main__':
#    p_root = '/mnt/hawc2sim/NREL5MW'
#    p_root = '/mnt/gorm/HAWC2/NREL5MW'
#    p_root = '/mnt/vea-group/AED/STABCON/SIM/NREL5MW'

    # -- Define Case Main Dir: P_RUN/PROJECT/sim_id -- #
        # target run dir
    P_RUN = '//mimer/hawc2sim'
    # when executing on gorm, this has to be
#    P_RUN = '/mnt/mimer/hawc2sim'
        # . Simulation ID:
    sim_id = 'C0002'
        # . Project directory:
#    PROJECT = 'NREL5MW'
    PROJECT = 'DTU10MW'
        # . Master File
#    MASTERFILE = 'nrel_5mw_master_%s.htc' % sim_id
    MASTERFILE = 'dtu10mw_master_%s.htc' % sim_id

    # -- Assign sources (exchange files sources) and output directories -- #
    P_SOURCE = P_RUN
    resdir = '%s/%s/%s/' % (P_RUN, PROJECT, sim_id)

    # LAUNCH SIMULATIONS
    launch_dlcs(sim_id)

    # POST PROCESS SIMULATIONS
#    df_stats = post_launch(sim_id, check_logs=True, force_dir=resdir)

    # PLOT STATS, only one case
#    figdir = '%s/%s/%s/' % (P_RUN, PROJECT, sim_id)
#    plot_stats(sim_id, fig_dir=figdir)

#    # PLOT STATS, when comparing cases
#    sim_ids = ['C0010', 'C0011']
#    figdir = '%s/%s/%s/' % (P_RUN, PROJECT, 'figures/C0010-C0011')
#    plot_stats(sim_ids, fig_dir_base=figdir)


