# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:16:34 2011

@author: dave
"""

from __future__ import division
from __future__ import print_function
#print(*objects, sep=' ', end='\n', file=sys.stdout)

# standard python library
import os
import subprocess as sproc
import copy
import zipfile
import shutil
import datetime
import math
import pickle
# what is actually the difference between warnings and logging.warn?
# for which context is which better?
#import warnings
import logging
from operator import itemgetter
from time import time

# numpy and scipy only used in HtcMaster._all_in_one_blade_tag
import numpy as np
import scipy
import scipy.interpolate as interpolate
#import matplotlib.pyplot as plt
import pandas as pd

# custom libraries
import misc
import windIO

def load_pickled_file(source):
    FILE = open(source, 'rb')
    result = pickle.load(FILE)
    FILE.close()
    return result

def save_pickle(source, variable):
    FILE = open(source, 'wb')
    pickle.dump(variable, FILE, protocol=2)
    FILE.close()

def write_file(file_path, file_contents, mode):
    """
    INPUT:
        file_path: path/to/file/name.csv
        string   : file contents is a string
        mode     : reading (r), writing (w), append (a),...
    """

    FILE = open(file_path, mode)
    FILE.write(file_contents)
    FILE.close()

def create_multiloop_list(iter_dict, debug=False):
    """
    Create a list based on multiple nested loops
    ============================================

    Considerd the following example

    >>> for v in range(V_start, V_end, V_delta):
    ...     for y in range(y_start, y_end, y_delta):
    ...         for c in range(c_start, c_end, c_delta):
    ...             print v, y, c

    Could be replaced by a list with all these combinations. In order to
    replicate this with create_multiloop_list, iter_dict should have
    the following structure

    >>> iter_dict = dict()
    >>> iter_dict['v'] = range(V_start, V_end, V_delta)
    >>> iter_dict['y'] = range(y_start, y_end, y_delta)
    >>> iter_dict['c'] = range(c_start, c_end, c_delta)
    >>> iter_list = create_multiloop_list(iter_dict)
    >>> for case in iter_list:
    ...     print case['v'], case['y'], case['c']

    Parameters
    ----------

    iter_dict : dictionary
        Key holds a valid tag as used in HtcMaster.tags. The corresponding
        value shouuld be a list of values to be considered.

    Output
    ------

    iter_list : list
        List containing dictionaries. Each entry is a combination of the
        given iter_dict keys.

    Example
    -------

    >>> iter_dict={'[wind]':[5,6,7],'[coning]':[0,-5,-10]}
    >>> create_multiloop_list(iter_dict)
    [{'[wind]': 5, '[coning]': 0},
     {'[wind]': 5, '[coning]': -5},
     {'[wind]': 5, '[coning]': -10},
     {'[wind]': 6, '[coning]': 0},
     {'[wind]': 6, '[coning]': -5},
     {'[wind]': 6, '[coning]': -10},
     {'[wind]': 7, '[coning]': 0},
     {'[wind]': 7, '[coning]': -5},
     {'[wind]': 7, '[coning]': -10}]
    """

    iter_list = []

    # fix the order of the keys
    key_order = iter_dict.keys()
    nr_keys = len(key_order)
    nr_values,indices = [],[]
    # determine how many items on each key
    for key in key_order:
        # each value needs to be an iterable! len() will fail if it isn't
        # count how many values there are for each key
        if type(iter_dict[key]).__name__ != 'list':
            print('%s does not hold a list' % key)
            raise ValueError, 'Each value in iter_dict has to be a list!'
        nr_values.append(len(iter_dict[key]))
        # create an initial indices list
        indices.append(0)

    if debug: print(nr_values, indices)

    go_on = True
    # keep track on which index you are counting, start at the back
    loopkey = nr_keys -1
    cc = 0
    while go_on:
        if debug: print(indices)

        # Each entry on the list is a dictionary with the parameter combination
        iter_list.append(dict())

        # save all the different combination into one list
        for keyi in range(len(key_order)):
            key = key_order[keyi]
            # add the current combination of values as one dictionary
            iter_list[cc][key] = iter_dict[key][indices[keyi]]

        # +1 on the indices of the last entry, the overflow principle
        indices[loopkey] += 1

        # cycle backwards thourgh all dimensions and propagate the +1 if the
        # current dimension is full. Hence overflow.
        for k in range(loopkey,-1,-1):
            # if the current dimension is over its max, set to zero and change
            # the dimension of the next. Remember we are going backwards
            if not indices[k] < nr_values[k] and k > 0:
                # +1 on the index of the previous dimension
                indices[k-1] += 1
                # set current loopkey index back to zero
                indices[k] = 0
                # if the previous dimension is not on max, break out
                if indices[k-1] < nr_values[k-1]:
                    break
            # if we are on the last dimension, break out if that is also on max
            elif k == 0 and not indices[k] < nr_values[k]:
                if debug: print(cc)
                go_on = False

        # fail safe exit mechanism...
        if cc > 20000:
            raise UserWarning, 'multiloop_list has already '+str(cc)+' items..'
            go_on = False

        cc += 1

    return iter_list

def local_shell_script(htc_dict, sim_id):
    """
    """
    shellscript = ''
    breakline = '"' + '*'*80 + '"'
    nr_cases = len(htc_dict)
    nr = 1
    for case in htc_dict:
        shellscript += 'echo ""' + '\n'
        shellscript += 'echo ' + breakline + '\n' + 'echo '
        shellscript += '" ===> Progress:'+str(nr)+'/'+str(nr_cases)+'"\n'
        # get a shorter version for the current cases tag_dict:
        scriptpath = htc_dict[case]['[run_dir]'] + 'runall.sh'
        try:
            hawc2_exe = htc_dict[case]['[hawc_exe]']
        except KeyError:
            hawc2_exe = 'hawc2mb.exe'
        htc_dir = htc_dict[case]['[htc_dir]']
        #shellscript += 'cd /home/dave/Projects/0_Risø_NDA/HAWC2/run'
        # log all warning messages: WINEDEBUG=-all
#        wine = 'WINEARCH=win32 WINEPREFIX=~/.wine32 wine'
        wine = 'wine'
        shellscript += '%s %s %s %s \n' % (wine, hawc2_exe, htc_dir, case)
        shellscript += 'echo ' + breakline + '\n'
        nr+=1

    write_file(scriptpath, shellscript, 'w')
    print('\nrun local shell script written to:')
    print(scriptpath)

def local_windows_script(cases, sim_id, nr_cpus=2):
    """
    """

    tot_cases = len(cases)
    i_script = 1
    i_case_script = 1
    cases_per_script = int(math.ceil(float(tot_cases)/float(nr_cpus)))
    # header of the new script, each process has its own copy
    header = ''
    header += 'rem\nrem\n'
    header += 'mkdir _%i_\n'
    header += 'robocopy .\ .\_%i_ /e /xf *.log /xf *.dat /xf *.sel /xd _*_\n'
    header += 'cd _%i_\n'
    header += 'rem\nrem\n'
    footer = ''
    footer += 'rem\nrem\n'
    footer += 'cd ..\n'
    footer += 'robocopy .\_%i_\ /e .\ /move\n'
    footer += 'rem\nrem\n'
    shellscript = header % (i_script, i_script, i_script)

    stop = False

    for i_case, (cname, case) in enumerate(cases.iteritems()):
#    for i_case, case in enumerate(sorted(cases.keys())):

        shellscript += 'rem\nrem\n'
        shellscript += 'rem ===> Progress: %3i / %3i\n' % (i_case+1, tot_cases)
        # copy turbulence from data base, if applicable
        if case['[turb_db_dir]'] is not None:
            # we are one dir up in cpu exe dir
            turb = case['[turb_base_name]'] + '*.bin'
            dbdir = os.path.join('./../', case['[turb_db_dir]'], turb)
            dbdir = dbdir.replace('/', '\\')
            rpl = (dbdir, case['[turb_dir]'].replace('/', '\\'))
            shellscript += 'copy %s %s\n' % rpl

        # get a shorter version for the current cases tag_dict:
        scriptpath = '%srunall-%i.bat' % (case['[run_dir]'], i_script)
        htcpath = case['[htc_dir]'][:-1].replace('/', '\\') # ditch the /
        try:
            hawc2_exe = case['[hawc2_exe]']
        except KeyError:
            hawc2_exe = 'hawc2mb.exe'
        rpl = (hawc2_exe.replace('/', '\\'), htcpath, cname.replace('/', '\\'))
        shellscript += "%s .\\%s\\%s\n" % rpl
        # copy back to data base directory if they do not exists there
        # remove turbulence file again, if copied from data base
        if case['[turb_db_dir]'] is not None:
            # copy back if it does not exist in the data base
            # IF EXIST "c:\test\file.ext"  (move /y "C:\test\file.ext" "C:\quality\" )
            turbu = case['[turb_base_name]'] + 'u.bin'
            turbv = case['[turb_base_name]'] + 'v.bin'
            turbw = case['[turb_base_name]'] + 'w.bin'
            dbdir = os.path.join('./../', case['[turb_db_dir]'])
            for tu in (turbu, turbv, turbw):
                tu_db = os.path.join(dbdir, tu).replace('/', '\\')
                tu_run = os.path.join(case['[turb_dir]'], tu).replace('/', '\\')
                rpl = (tu_db, tu_run, dbdir.replace('/', '\\'))
                shellscript += 'IF NOT EXIST "%s" move /y "%s" "%s"\n' % rpl
            # remove turbulence from run dir
            allturb = os.path.join(case['[turb_dir]'], '*.*')
            allturb = allturb.replace('/', '\\')
            # do not prompt for delete confirmation: /Q
            shellscript += 'del /Q "%s"\n' % allturb

        if i_case_script >= cases_per_script:
            # footer: copy all files back
            shellscript += footer % i_script
            stop = True
            write_file(scriptpath, shellscript, 'w')
            print('\nrun local shell script written to:')
            print(scriptpath)

            # header of the new script, each process has its own copy
            # but only if there are actually jobs left
            if i_case+1 < tot_cases:
                i_script += 1
                i_case_script = 1
                shellscript = header % (i_script, i_script, i_script)
                stop = False
        else:
            i_case_script += 1

    # we might have missed the footer of a partial script
    if not stop:
        shellscript += footer % i_script
        write_file(scriptpath, shellscript, 'w')
        print('\nrun local shell script written to:')
        print(scriptpath)

def run_local_ram(cases, check_log=True):

    ram_root = '/tmp/HAWC2/'

    if not os.path.exists(ram_root):
        os.makedirs(ram_root)

    print('copying data from run_dir to RAM...', end='')

    # first copy everything to RAM
    for ii, case in enumerate(cases):
        # all tags for the current case
        tags = cases[case]
        run_dir = copy.copy(tags['[run_dir]'])
        run_dir_ram = ram_root + tags['[sim_id]']
        if not os.path.exists(run_dir_ram):
            os.makedirs(run_dir_ram)
        # and also change the run dir so we can launch it easily
        tags['[run_dir]'] = run_dir_ram + '/'
        for root, dirs, files in os.walk(run_dir):
            run_dir_base = os.path.commonprefix([root, run_dir])
            cdir = root.replace(run_dir_base, '')
            dstbase = os.path.join(run_dir_ram, cdir)
            if not os.path.exists(dstbase):
                os.makedirs(dstbase)
            for fname in files:
                src = os.path.join(root, fname)
                dst = os.path.join(dstbase, fname)
                shutil.copy2(src, dst)

    print('done')

    # launch from RAM
    run_local(cases, check_log=check_log)
    # change run_dir back to original
    for ii, case in enumerate(cases):
        tags = cases[case]
        tags['[run_dir]'] = run_dir

    print('copying data from RAM back to run_dir')
    print('run_dir: %s' % run_dir)

    # and copy everything back
    for root, dirs, files in os.walk(run_dir_ram):
        run_dir_base = os.path.commonprefix([root, run_dir_ram])
        cdir = root.replace(run_dir_base, '')
        # in case it is the same
        if len(cdir) == 0:
            pass
        # join doesn't work if cdir has a leading / ?? so drop it
        elif cdir[0] == '/':
            dstbase = os.path.join(run_dir, cdir[1:])
        for fname in files:
            src = os.path.join(root, fname)
            dst = os.path.join(dstbase, fname)
            if not os.path.exists(dstbase):
                os.makedirs(dstbase)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print('src:', src)
                print('dst:', dst)
                print(e)
                print()
                pass

    print('...done')

    return cases


def run_local(cases, silent=False, check_log=True):
    """
    Run all HAWC2 simulations locally from cases
    ===============================================

    Run all case present in a cases dict locally and wait until HAWC2 is ready.

    In verbose mode, each HAWC2 simulation is also timed

    Parameters
    ----------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary holding
        all the tags/value pairs as used for that case

    check_log : boolean, default=False
        Check the log file emmidiately after execution of the HAWC2 case

    silent : boolean, default=False
        When False, usefull information will be printed and the HAWC2
        simulation time will be calculated from the Python perspective. The
        silent variable is also passed on to logcheck_case

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Update cases with the STDOUT of the respective HAWC2 simulation

    """

    # remember the current working directory
    cwd = os.getcwd()
    nr = len(cases)
    if not silent:
        print('')
        print('='*79)
        print('Be advised, launching %i HAWC2 simulation(s) sequentially' % nr)
        print('run dir: %s' % cases[cases.keys()[0]]['[run_dir]'])
        print('')

    if check_log:
        errorlogs = ErrorLogs(silent=silent)

    for ii, case in enumerate(cases):
        # all tags for the current case
        tags = cases[case]
        # for backward compatibility assume default HAWC2 executable
        try:
            hawc2_exe = tags['[hawc2_exe]']
        except KeyError:
            hawc2_exe = 'hawc2-115'
        # TODO: if a turbulence data base is set, copy the files from there

        # the launch command
#        cmd  = 'WINEDEBUG=-all WINEARCH=win32 WINEPREFIX=~/.wine32 wine'
        cmd  = 'wine'
        cmd += " %s %s%s" % (hawc2_exe, tags['[htc_dir]'], case)
        # remove any escaping in tags and case for security reasons
        cmd = cmd.replace('\\','')
        # browse to the correct launch path for the HAWC2 simulation
        os.chdir(tags['[run_dir]'])

        if not silent:
            start = time()
            progress = '%4i/%i  : %s%s' % (ii+1, nr, tags['[htc_dir]'], case)
            print('*'*75)
            print(progress)

        # and launch the HAWC2 simulation
        p = sproc.Popen(cmd,stdout=sproc.PIPE,stderr=sproc.STDOUT,shell=True)

        # p.wait() will lock the current shell until p is done
        # p.stdout.readlines() checks if there is any output, but also locks
        # the thread if nothing comes back
        # save the output that HAWC2 sends to the shell to the cases
        # note that this is a list, each item holding a line
        cases[case]['sim_STDOUT'] = p.stdout.readlines()
        # wait until HAWC2 finished doing its magic
        p.wait()

        if not silent:
            # print(the simulation command line output
            print(' ' + '-'*75)
            print(''.join(cases[case]['sim_STDOUT']))
            print(' ' + '-'*75)
            # caclulation time
            stp = time() - start
            stpmin = stp/60.
            print('HAWC2 execution time: %8.2f sec (%8.2f min)' % (stp,stpmin))

        # where there any errors in the output? If yes, abort
        for k in cases[case]['sim_STDOUT']:
            kstart = k[:14]
            if kstart in [' *** ERROR ***', 'forrtl: severe']:
                cases[case]['[hawc2_sim_ok]'] = False
                #raise UserWarning, 'Found error in HAWC2 STDOUT'
            else:
                cases[case]['[hawc2_sim_ok]'] = True

        # check the log file strait away if required
        if check_log:
            start = time()
            errorlogs = logcheck_case(errorlogs, cases, case, silent=silent)
            stop = time() - start
            if case.endswith('.htc'):
                kk = case[:-4] + '.log'
            else:
                kk = case + '.log'
            errors = errorlogs.MsgListLog2[kk][0]
            exitok = errorlogs.MsgListLog2[kk][1]
            if not silent:
                print('log checks took %5.2f sec' % stop)
                print('    found error: ', errors)
                print(' exit correctly: ', exitok)
                print('*'*75)
                print()
            # also save in cases
            if not errors and exitok:
                cases[case]['[hawc2_sim_ok]'] = True
            else:
                cases[case]['[hawc2_sim_ok]'] = False

    if check_log:
        # take the last case to determine sim_id, run_dir and log_dir
        sim_id = cases[case]['[sim_id]']
        run_dir = cases[case]['[run_dir]']
        log_dir = cases[case]['[log_dir]']
        # save the extended (.csv format) errorlog list?
        # but put in one level up, so in the logfiles folder directly
        errorlogs.ResultFile = sim_id + '_ErrorLog.csv'
        # use the model path of the last encoutered case in cases
        errorlogs.PathToLogs = run_dir + log_dir
        errorlogs.save()

    # just in case, browse back the working path relevant for the python magic
    os.chdir(cwd)
    if not silent:
        print('\nHAWC2 has done all of its sequential magic!')
        print('='*79)
        print('')

    return cases


def prepare_launch(iter_dict, opt_tags, master, variable_tag_func,
                write_htc=True, runmethod='local', verbose=False,
                copyback_turb=True, msg='', silent=False, check_log=True,
                update_cases=False, ignore_non_unique=False,
                run_only_new=False, windows_nr_cpus=2):
    """
    Create the htc files, pbs scripts and replace the tags in master file
    =====================================================================

    Do not use any uppercase letters in the filenames, since HAWC2 will
    convert all of them to lower case results file names (.sel, .dat, .log)

    create sub folders according to sim_id, in order to not create one
    folder for the htc, results, logfiles which grows very large in due
    time!!

    opt_tags is a list of dictionaries of tags:
        [ {tag1=12,tag2=23,..},{tag1=11, tag2=33, tag9=5,...},...]
    for each wind, yaw and coning combi, each tag dictionary in the list
    will be set.

    Make sure to always define all dictionary keys in each list, otherwise
    the value of the first appareance will remain set for the remaining
    simulations in the list.
    For instance, in the example above, if tag9=5 is not set for subsequent
    lists, tag9 will remain having value 5 for these subsequent sets

    The tags for each case are consequently set in following order (or
    presedence):
        * master
        * opt_tags
        * iter_dict
        * variable_tag_func

    Parameters
    ----------

    iter_dict : dict

    opt_tags : dict

    master : HtcMaster object

    variable_tag_func : function object

    write_htc : boolean, default=True

    verbose : boolean, default=False

    runmethod : {'local' (default),'thyra','gorm','local-script','none'}
        Specify how/what to run where. For local, each case in cases is
        run locally via python directly. If set to 'local-script' a shell
        script is written to run all cases locally sequential. If set to
        'thyra' or 'gorm', PBS scripts are written to the respective server.

    msg : str, default=''
        A descriptive message of the simulation series is saved at
        "post_dir + master.tags['[sim_id]'] + '_tags.txt'". Additionally, this
         tagfile also holds the opt_tags and iter_dict values.

    update_cases : boolean, default=False
        If True, a current cases dictionary can be updated with new simulations

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary holding
        all the tags/value pairs as used for that case

    """

    post_dir = master.tags['[post_dir]']
    # either take a currently existing cases dictionary, or create a new one
    if update_cases:
        try:
            FILE = open(post_dir + master.tags['[sim_id]'] + '.pkl', 'rb')
            cases = pickle.load(FILE)
            FILE.close()
            print('updating cases for %s' % master.tags['[sim_id]'])
        except IOError:
            print(79*'=')
            print("failed to load cases dict for updating simd_id at:")
            print(post_dir + master.tags['[sim_id]'] + '.pkl')
            print(79*'=')
            cases = {}
        # but only run the new cases
        cases_to_run = {}
    else:
        cases = {}

    # if empty, just create a dummy item so we get into the loops
    if len(iter_dict) == 0:
        iter_dict = {'__dummy__': [0]}
    combi_list = create_multiloop_list(iter_dict)

    # load the master htc file as a string under the master.tags
    master.loadmaster()

    # ignore if the opt_tags is empty, will result in zero
    if len(opt_tags) > 0:
        sim_total = len(combi_list)*len(opt_tags)
    else:
        sim_total = len(combi_list)
        # if no opt_tags specified, create an empty dummy tag
        opt_tags = [dict({'__DUMMY_TAG__' : 0})]
    sim_nr = 0

    # make sure all the required directories are in place at run_dir
    master.create_run_dir()

    # cycle thourgh all the combinations
    for it in combi_list:
        for ot in opt_tags:
            sim_nr += 1
            # update the tags from the opt_tags list
            if not '__DUMMY_TAG__' in ot:
                master.tags.update(ot)
            # update the tags set in the combi_list
            master.tags.update(it)
            # -----------------------------------------------------------
            # start variable tags update
            master = variable_tag_func(master)
            # end variable tags
            # -----------------------------------------------------------
            if not silent:
                print('htc progress: ' + format(sim_nr, '3.0f') + '/' + \
                       format(sim_total, '3.0f'))

            if verbose:
                print('===master.tags===\n', master.tags)

            # returns a dictionary with all the tags used for this
            # specific case
            htc = master.createcase(write_htc=write_htc)
            #htc=master.createcase_check(cases_repo,write_htc=write_htc)

            # make sure the current cases is unique!
            if not ignore_non_unique:
                if htc.keys()[0] in cases:
                    msg = 'non unique case in cases: %s' % htc.keys()[0]
                    raise KeyError, msg

            # save in the big cases. Note that values() gives a copy!
            cases[htc.keys()[0]] = htc.values()[0]
            # if we have an update scenario, keep track of the cases we want
            # to run again. This prevents us from running all cases on every
            # update
            if run_only_new:
                cases_to_run[htc.keys()[0]] = htc.values()[0]

            if verbose:
                print('created cases for: %s.htc\n' % master.tags['[case_id]'])

    # only copy data and create zip after all htc files have been created.
    # Note that createcase could also creat other input files
    # create the execution folder structure and copy all data to it
    master.copy_model_data()

    # create the zip file
    master.create_model_zip()

    # create directory if post_dir does not exists
    try:
        os.mkdir(post_dir)
    except OSError:
        pass
    FILE = open(post_dir + master.tags['[sim_id]'] + '.pkl', 'wb')
    pickle.dump(cases, FILE, protocol=2)
    FILE.close()

    if not silent:
        print('\ncases saved at:')
        print(post_dir + master.tags['[sim_id]'] + '.pkl')

    # also save the iter_dict and opt_tags in a text file for easy reference
    # or quick checks on what each sim_id actually contains
    # sort the taglist for convienent reading/comparing
    tagfile = msg + '\n\n'
    tagfile += '='*79 + '\n'
    tagfile += 'iter_dict\n'.rjust(30)
    tagfile += '='*79 + '\n'
    iter_dict_list = sorted(iter_dict.iteritems(), key=itemgetter(0))
    for k in iter_dict_list:
        tagfile += str(k[0]).rjust(30) + ' : ' + str(k[1]).ljust(20) + '\n'

    tagfile += '\n'
    tagfile += '='*79 + '\n'
    tagfile += 'opt_tags\n'.rjust(30)
    tagfile += '='*79 + '\n'
    for k in opt_tags:
        tagfile += '\n'
        tagfile += '-'*79 + '\n'
        tagfile += 'opt_tags set\n'.rjust(30)
        tagfile += '-'*79 + '\n'
        opt_dict = sorted(k.iteritems(), key=itemgetter(0), reverse=False)
        for kk in opt_dict:
            tagfile += str(kk[0]).rjust(30)+' : '+str(kk[1]).ljust(20) + '\n'
    if update_cases:
        mode = 'a'
    else:
        mode = 'w'
    write_file(post_dir + master.tags['[sim_id]'] + '_tags.txt', tagfile, mode)

    if run_only_new:
        cases = cases_to_run

    launch(cases, runmethod=runmethod, verbose=verbose,
           copyback_turb=copyback_turb, check_log=check_log,
           windows_nr_cpus=windows_nr_cpus)

    return cases

def prepare_relaunch(cases, runmethod='gorm', verbose=False, write_htc=True,
                     copyback_turb=True, silent=False, check_log=True):
    """
    Instead of redoing everything, we know recreate the HTC file for those
    in the given cases dict. Nothing else changes. The data and zip files
    are not updated, the convience tagfile is not recreated. However, the
    saved (pickled) cases dict corresponding to the sim_id is updated!

    This method is usefull to correct mistakes made for some cases.

    It is adviced to not change the case_id, sim_id, from the cases.
    """

    # initiate the HtcMaster object, load the master file
    master = HtcMaster()
    # for invariant tags, load random case. Necessary before we can load
    # the master file, otherwise we don't know which master to load
    master.tags = cases[cases.keys()[0]]
    master.loadmaster()

    # load the original cases dict
    post_dir = master.tags['[post_dir]']
    FILE = open(post_dir + master.tags['[sim_id]'] + '.pkl', 'rb')
    cases_orig = pickle.load(FILE)
    FILE.close()

    sim_nr = 0
    sim_total = len(cases)
    for case, casedict in cases.iteritems():
        sim_nr += 1

        # set all the tags in the HtcMaster file
        master.tags = casedict
        # returns a dictionary with all the tags used for this
        # specific case
        htc = master.createcase(write_htc=write_htc)
        #htc=master.createcase_check(cases_repo,write_htc=write_htc)

        if not silent:
            print('htc progress: ' + format(sim_nr, '3.0f') + '/' + \
                   format(sim_total, '3.0f'))

        if verbose:
            print('===master.tags===\n', master.tags)

        # make sure the current cases already exists, otherwise we are not
        # relaunching!
        if case not in cases_orig:
            msg = 'relaunch only works for existing cases: %s' % case
            raise KeyError, msg

        # save in the big cases. Note that values() gives a copy!
        # remark, what about the copying done at the end of master.createcase?
        # is that redundant then?
        cases[htc.keys()[0]] = htc.values()[0]

        if verbose:
            print('created cases for: %s.htc\n' % master.tags['[case_id]'])

    launch(cases, runmethod=runmethod, verbose=verbose, check_log=check_log,
           copyback_turb=copyback_turb, silent=silent)

    # update the original file: overwrite the newly set cases
    FILE = open(post_dir + master.tags['[sim_id]'] + '.pkl', 'wb')
    cases_orig.update(cases)
    pickle.dump(cases_orig, FILE, protocol=2)
    FILE.close()

def prepare_launch_cases(cases, runmethod='gorm', verbose=False,write_htc=True,
                         copyback_turb=True, silent=False, check_log=True,
                         variable_tag_func=None, sim_id_new=None):
    """
    Same as prepare_launch, but now the input is just a cases object (cao).
    If relaunching some earlier defined simulations, make sure to at least
    rename the sim_id, otherwise it could become messy: things end up in the
    same folder, sim_id post file get overwritten, ...

    In case you do not use a variable_tag_fuc, make sure all your tags are
    defined in cases. First and foremost, this means that the case_id does not
    get updated to have a new sim_id, the path's are not updated, etc

    When given a variable_tag_func, make sure it is properly
    defined: do not base a variable tag's value on itself to avoid value chains

    The master htc file will be loaded and alls tags defined in the cases dict
    will be applied to it as is.
    """

    # initiate the HtcMaster object, load the master file
    master = HtcMaster()
    # for invariant tags, load random case. Necessary before we can load
    # the master file, otherwise we don't know which master to load
    master.tags = cases[cases.keys()[0]]
    # load the master htc file as a string under the master.tags
    master.loadmaster()
    # create the execution folder structure and copy all data to it
    # but reset to the correct launch dirs first
    sim_id = master.tags['[sim_id]']
    if runmethod in ['local', 'local-script', 'none']:
        path = '/home/dave/PhD_data/HAWC2_results/ojf_post/%s/' % sim_id
        master.tags['[run_dir]'] = path
    elif runmethod == 'jess':
        master.tags['[run_dir]'] = '/mnt/jess/HAWC2/ojf_post/%s/' % sim_id
    elif runmethod == 'gorm':
        master.tags['[run_dir]'] = '/mnt/gorm/HAWC2/ojf_post/%s/' % sim_id
    else:
        msg='unsupported runmethod, options: none, local, thyra, gorm, opt'
        raise ValueError, msg
    master.create_run_dir()
    master.copy_model_data()
    # create the zip file
    master.create_model_zip()

    sim_nr = 0
    sim_total = len(cases)

    # for safety, create a new cases dict. At the end of the ride both cases
    # and cases_new should be identical!
    cases_new = {}

    # cycle thourgh all the combinations
    for case, casedict in cases.iteritems():
        sim_nr += 1

        sim_id = casedict['[sim_id]']
        # reset the launch dirs
        if runmethod in ['local', 'local-script', 'none']:
            path = '/home/dave/PhD_data/HAWC2_results/ojf_post/%s/' % sim_id
            casedict['[run_dir]'] = path
        elif runmethod == 'thyra':
            casedict['[run_dir]'] = '/mnt/thyra/HAWC2/ojf_post/%s/' % sim_id
        elif runmethod == 'gorm':
            casedict['[run_dir]'] = '/mnt/gorm/HAWC2/ojf_post/%s/' % sim_id
        else:
            msg='unsupported runmethod, options: none, local, thyra, gorm, opt'
            raise ValueError, msg

        # -----------------------------------------------------------
        # set all the tags in the HtcMaster file
        master.tags = casedict
        # apply the variable tags if applicable
        if variable_tag_func:
            master = variable_tag_func(master)
        elif sim_id_new:
            # TODO: finish this
            # replace all the sim_id occurences with the updated one
            # this means also the case_id tag changes!
            pass
        # -----------------------------------------------------------

        # returns a dictionary with all the tags used for this specific case
        htc = master.createcase(write_htc=write_htc)

        if not silent:
            print('htc progress: ' + format(sim_nr, '3.0f') + '/' + \
                   format(sim_total, '3.0f'))

        if verbose:
            print('===master.tags===\n', master.tags)

        # make sure the current cases is unique!
        if htc.keys()[0] in cases_new:
            msg = 'non unique case in cases: %s' % htc.keys()[0]
            raise KeyError, msg
        # save in the big cases. Note that values() gives a copy!
        # remark, what about the copying done at the end of master.createcase?
        # is that redundant then?
        cases_new[htc.keys()[0]] = htc.values()[0]

        if verbose:
            print('created cases for: %s.htc\n' % master.tags['[case_id]'])

    post_dir = master.tags['[post_dir]']

    # create directory if post_dir does not exists
    try:
        os.mkdir(post_dir)
    except OSError:
        pass
    FILE = open(post_dir + master.tags['[sim_id]'] + '.pkl', 'wb')
    pickle.dump(cases_new, FILE, protocol=2)
    FILE.close()

    if not silent:
        print('\ncases saved at:')
        print(post_dir + master.tags['[sim_id]'] + '.pkl')

    launch(cases_new, runmethod=runmethod, verbose=verbose,
           copyback_turb=copyback_turb, check_log=check_log)

    return cases_new



def launch(cases, runmethod='local', verbose=False, copyback_turb=True,
           silent=False, check_log=True, windows_nr_cpus=2):
    """
    The actual launching of all cases in the Cases dictionary. Note that here
    only the PBS files are written and not the actuall htc files.

    Parameters
    ----------

    cases : dict
        Dictionary with the case name as key and another dictionary as value.
        The latter holds all the tag/value pairs used in the respective
        simulation.

    verbose : boolean, default=False

    runmethod : {'local' (default),'thyra','gorm','linux-script','none',
                 'windows-script'}
        Specify how/what to run where. For local, each case in cases is
        run locally via python directly. If set to 'linux-script' a shell
        script is written to run all cases locally sequential. If set to
        'thyra' or 'gorm', PBS scripts are written to the respective server.
    """

    random_case = cases.keys()[0]
    sim_id = cases[random_case]['[sim_id]']
    pbs_out_dir = cases[random_case]['[pbs_out_dir]']

    if runmethod == 'local-script' or runmethod == 'linux-script':
        local_shell_script(cases, sim_id)
    elif runmethod == 'windows-script':
        local_windows_script(cases, sim_id, nr_cpus=windows_nr_cpus)
    elif runmethod in ['jess','gorm']:
        # create the pbs object
        pbs = PBS(cases, server=runmethod)
        pbs.copyback_turb = copyback_turb
        pbs.verbose = verbose
        pbs.pbs_out_dir = pbs_out_dir
        pbs.create()
    elif runmethod == 'local':
        cases = run_local(cases, silent=silent, check_log=check_log)
    elif runmethod =='local-ram':
        cases = run_local_ram(cases, check_log=check_log)
    elif runmethod == 'none':
        pass
    else:
        msg = 'unsupported runmethod, valid options: local, thyra, gorm or opt'
        raise ValueError, msg

def post_launch(cases):
    """
    Do some basics checks: do all launched cases have a result and LOG file
    and are there any errors in the LOG files?

    Parameters
    ----------

    cases : either a string (path to file) or the cases itself
    """

    # TODO: finish support for default location of the cases and file name
    # two scenario's: either pass on an cases and get from their the
    # post processing path or pass on the simid and load from the cases
    # from the default location
    # in case run_local, do not check PBS!

    # in case it is a path, load the cases
    if type(cases).__name__ == 'str':
        cases = load_pickled_file(cases)

    # saving output to textfile and print(at the same time
    LOG = Log()
    LOG.print_logging = True

    # load one case dictionary from the cases to get data that is the same
    # over all simulations in the cases
    try:
        master = cases.keys()[0]
    except IndexError:
        print('there are no cases, aborting...')
        return None
    post_dir = cases[master]['[post_dir]']
    sim_id = cases[master]['[sim_id]']
    run_dir = cases[master]['[run_dir]']
    log_dir = cases[master]['[log_dir]']

    # for how many of the created cases are there actually result, log files
    pbs = PBS(cases)
    pbs.cases = cases
    cases_fail = pbs.check_results(cases)

    # add the failed cases to the LOG:
    LOG.add(['number of failed cases: ' + str(len(cases_fail))])
    LOG.add(list(cases_fail))
    # for k in cases_fail:
    #    print(k

    # initiate the object to check the log files
    errorlogs = ErrorLogs(cases=cases)
    LOG.add(['checking ' + str(len(cases)) + ' LOG files...'])
    nr = 1
    nr_tot = len(cases)

    tmp = cases.keys()[0]
    print('checking logs, path (from a random item in cases):')
    print(run_dir + log_dir)

    for k in sorted(cases.keys()):
        # a case could not have a result, but a log file might still exist
        if k.endswith('.htc'):
            kk = k[:-4] + '.log'
        else:
            kk = k + '.log'
        # note that if errorlogs.PathToLogs is a file, it will only check that
        # file. If it is a directory, it will check all that is in the dir
        run_dir = cases[k]['[run_dir]']
        log_dir = cases[k]['[log_dir]']
        errorlogs.PathToLogs = run_dir + log_dir + kk
        try:
            errorlogs.check()
            print('checking logfile progress: ' + str(nr) + '/' + str(nr_tot))
        except IOError:
            print('           no logfile for:  %s' % (run_dir + log_dir + kk))
        except Exception as e:
            print('  log analysis failed for: %s' % kk)
            print(e)
        nr += 1

        # if simulation did not ended correctly, put it on the fail list
        try:
            if not errorlogs.MsgListLog2[kk][1]:
                cases_fail[k] = cases[k]
        except KeyError:
            pass

    # now see how many cases resulted in an error and add to the general LOG
    # determine how long the first case name is
    try:
        spacing = len(errorlogs.MsgListLog2.keys()[0]) + 9
    except Exception as e:
        print('nr of OK cases: %i' % (len(cases) - len(cases_fail)))
        raise e
    LOG.add(['display log check'.ljust(spacing) + 'found_error?'.ljust(15) + \
            'exit_correctly?'])
    for k in errorlogs.MsgListLog2:
        LOG.add([k.ljust(spacing)+str(errorlogs.MsgListLog2[k][0]).ljust(15)+\
            str(errorlogs.MsgListLog2[k][1]) ])
    # save the extended (.csv format) errorlog list?
    # but put in one level up, so in the logfiles folder directly
    errorlogs.ResultFile = sim_id + '_ErrorLog.csv'
    # save the log file analysis in the run_dir instead of the log_dir
    errorlogs.PathToLogs = run_dir# + log_dir
    errorlogs.save()

    # save the error LOG list, this is redundant, since it already exists in
    # the general LOG file (but only as a print, not the python variable)
    tmp = post_dir + sim_id + '_MsgListLog2'
    save_pickle(tmp, errorlogs.MsgListLog2)

    # save the list of failed cases
    save_pickle(post_dir + sim_id + '_fail.pkl', cases_fail)

    return cases_fail

def logcheck_case(errorlogs, cases, case, silent=False):
    """
    Check logfile of a single case
    ==============================

    Given the cases and a case, check that single case on errors in the
    logfile.

    """

    #post_dir = cases[case]['[post_dir]']
    #sim_id = cases[case]['[sim_id]']
    run_dir = cases[case]['[run_dir]']
    log_dir = cases[case]['[log_dir]']
    if case.endswith('.htc'):
        caselog = case[:-4] + '.log'
    else:
        caselog = case + '.log'
    errorlogs.PathToLogs = run_dir + log_dir + caselog
    errorlogs.check()

    # in case we find an error, abort or not?
    errors = errorlogs.MsgListLog2[caselog][0]
    exitcorrect = errorlogs.MsgListLog2[caselog][1]
    if errors:
        # print all error messages
        #logs.MsgListLog : [ [case, line nr, error1, line nr, error2, ....], ]
        # difficult: MsgListLog is not a dict!!
        #raise UserWarning, 'HAWC2 simulation has errors in logfile, abort!'
        #warnings.warn('HAWC2 simulation has errors in logfile!')
        logging.warn('HAWC2 simulation has errors in logfile!')
    elif not exitcorrect:
        #raise UserWarning, 'HAWC2 simulation did not ended correctly, abort!'
        #warnings.warn('HAWC2 simulation did not ended correctly!')
        logging.warn('HAWC2 simulation did not ended correctly!')

    # no need to do that, aborts on failure anyway and OK log check will be
    # printed in run_local when also printing how long it took to check
    #if not silent:
        #print 'log checks ok'
        #print '   found error: %s' % errorlogs.MsgListLog2[caselog][0]
        #print 'exit correctly: %s' % errorlogs.MsgListLog2[caselog][1]

    return errorlogs

    ## save the extended (.csv format) errorlog list?
    ## but put in one level up, so in the logfiles folder directly
    #errorlogs.ResultFile = sim_id + '_ErrorLog.csv'
    ## use the model path of the last encoutered case in cases
    #errorlogs.PathToLogs = run_dir + log_dir
    #errorlogs.save()

def get_htc_dict(post_dir, simid):
    """
    Load the htc_dict, remove failed cases
    """
    htc_dict = load_pickled_file(post_dir + simid + '.pkl')

    # if the post processing is done on simulations done by thyra/gorm, and is
    # downloaded locally, change path to results
    for case in htc_dict:
        if htc_dict[case]['[run_dir]'][:4] == '/mnt':
            path = '/home/dave/PhD_data/HAWC2_results/ojf_post/' +simid +'/'
            htc_dict[case]['[run_dir]'] = path

    try:
        htc_dict_fail = load_pickled_file(post_dir + simid + '_fail.pkl')
    except IOError:
        return htc_dict

    # ditch all the failed cases out of the htc_dict
    # otherwise we will have fails when reading the results data files
    for k in htc_dict_fail:
        del htc_dict[k]
        print('removed from htc_dict due to error: ' + k)

    return htc_dict

class Log:
    """
    Class for convinient logging. Create an instance and add lines to the
    logfile as a list with the function add.
    The added items will be printed if
        self.print_logging = True. Default value is False

    Create the instance, add with .add('lines') (lines=list), save with
    .save(target), print(current log to screen with .printLog()
    """
    def __init__(self):
        self.log = []
        # option, should the lines added to the log be printed as well?
        self.print_logging = False
        self.file_mode = 'a'

    def add(self, lines):
        # the input is a list, where each entry is considered as a new line
        for k in lines:
            self.log.append(k)
            if self.print_logging:
                print(k)

    def save(self, target):
        # tread every item in the log list as a new line
        FILE = open(target, self.file_mode)
        for k in self.log:
            FILE.write(k + '\n')
        FILE.close()
        # and empty the log again
        self.log = []

    def printscreen(self):
        for k in self.log:
            print(k)

class HtcMaster:
    """
    """

    def __init__(self, verbose=False, silent=False):
        """
        """

        # TODO: make HtcMaster callable, so that when called you actually
        # set a value for a certain tag or add a new one. In doing so,
        # you can actually warn when you are overwriting a tag, or when
        # a different tag has the same name, etc

        # create a dictionary with the tag name as key as the default value
        self.tags = dict()

        # should we print(where the file is written?
        self.verbose = verbose
        self.silent = silent

        # following tags are required
        #---------------------------------------------------------------------
        self.tags['[case_id]'] = None

        self.tags['[master_htc_file]'] = None
        self.tags['[master_htc_dir]'] = None
        # path to model zip file, needs to accessible from the server
        # relative from the directory where the pbs files are launched on the
        # server. Suggestions is to always place the zip file in the model
        # folder, so only the zip file name has to be defined
        self.tags['[model_zip]'] = None

        # path to HAWTOPT blade result file: quasi/res/blade.dat
        self.tags['[blade_hawtopt_dir]'] = None
        self.tags['[blade_hawtopt]'] = None
        self.tags['[zaxis_fact]'] = 1.0
        # TODO: rename to execution dir, that description fits much better!
        self.tags['[run_dir]'] = None
        #self.tags['[run_dir]'] = '/home/dave/tmp/'

        # following dirs are relative to the run_dir!!
        # they indicate the location of the SAVED (!!) results, they can be
        # different from the execution dirs on the node which are set in PBS
        self.tags['[hawc2_exe]'] = 'hawc2mb.exe'
        self.tags['[res_dir]'] = 'results/'
        self.tags['[iter_dir]'] = 'iter/'
        self.tags['[log_dir]'] = 'logfiles/'
        self.tags['[turb_dir]'] = 'turb/'
        self.tags['[turb_db_dir]'] = None
        self.tags['[control_dir]'] = 'control/'
        self.tags['[animation_dir]'] = 'animation/'
        self.tags['[eigenfreq_dir]'] = 'eigenfreq/'
        self.tags['[wake_dir]'] = 'wake/'
        self.tags['[meander_dir]'] = 'meander/'
        self.tags['[htc_dir]'] = 'htc/'
        self.tags['[mooring_dir]'] = 'mooring/'
        self.tags['[hydro_dir]'] = 'htc_hydro/'
        self.tags['[pbs_out_dir]'] = 'pbs_out/'
        self.tags['[turb_base_name]'] = 'turb_'
        self.tags['[wake_base_name]'] = 'turb_'
        self.tags['[meand_base_name]'] = 'turb_'
        self.tags['[zip_root_files]'] = []

        self.tags['[pbs_queue_command]'] = '#PBS -q workq'
        # the express que has 2 thyra nodes with max walltime of 1h
#        self.tags['[pbs_queue_command]'] = '#PBS -q xpresq'
        # walltime should have following format: hh:mm:ss
        self.tags['[walltime]'] = '04:00:00'

    def create_run_dir(self):
        """
        If non existent, create run_dir and all required model sub directories
        """

        # create the remote folder structure
        if not os.path.exists(self.tags['[run_dir]']):
            os.makedirs(self.tags['[run_dir]'])
        # the data folder
        data_run = self.tags['[run_dir]'] + self.tags['[data_dir]']
        if not os.path.exists(data_run):
            os.makedirs(data_run)
        # the htc folder
        path = self.tags['[run_dir]'] + self.tags['[htc_dir]']
        if not os.path.exists(path):
            os.makedirs(path)
        # if the results dir does not exists, create it!
        path = self.tags['[run_dir]'] + self.tags['[res_dir]']
        if not os.path.exists(path):
            os.makedirs(path)
        # if the logfile dir does not exists, create it!
        path = self.tags['[run_dir]'] + self.tags['[log_dir]']
        if not os.path.exists(path):
            os.makedirs(path)
        # if the eigenfreq dir does not exists, create it!

        if self.tags['[eigenfreq_dir]']:
            path = self.tags['[run_dir]'] + self.tags['[eigenfreq_dir]']
            if not os.path.exists(path):
                os.makedirs(path)
        # if the animation dir does not exists, create it!
        path = self.tags['[run_dir]'] + self.tags['[animation_dir]']
        if not os.path.exists(path):
            os.makedirs(path)

        path = self.tags['[run_dir]'] + self.tags['[turb_dir]']
        if not os.path.exists(path):
            os.makedirs(path)

        if self.tags['[wake_dir]']:
            path = self.tags['[run_dir]'] + self.tags['[wake_dir]']
            if not os.path.exists(path):
                os.makedirs(path)

        if self.tags['[meander_dir]']:
            path = self.tags['[run_dir]'] + self.tags['[meander_dir]']
            if not os.path.exists(path):
                os.makedirs(path)

        if self.tags['[opt_dir]']:
            path = self.tags['[run_dir]'] + self.tags['[opt_dir]']
            if not os.path.exists(path):
                os.makedirs(path)

        path = self.tags['[run_dir]'] + self.tags['[control_dir]']
        if not os.path.exists(path):
            os.makedirs(path)

        if self.tags['[mooring_dir]']:
            path = self.tags['[run_dir]'] + self.tags['[mooring_dir]']
            if not os.path.exists(path):
                os.makedirs(path)

        if self.tags['[hydro_dir]']:
            path = self.tags['[run_dir]'] + self.tags['[hydro_dir]']
            if not os.path.exists(path):
                os.makedirs(path)

        path = self.tags['[run_dir]'] + 'externalforce/'
        if not os.path.exists(path):
            os.makedirs(path)

    def copy_model_data(self):
        """

        Copy the model data to the execution folder

        """

        # in case we are running local and the model dir is the server dir
        # we do not need to copy the data files, they are already on location
        data_local = self.tags['[model_dir_local]'] + self.tags['[data_dir]']
        data_run = self.tags['[run_dir]'] + self.tags['[data_dir]']
        if not data_local == data_run:

            # copy root files
            model_root = self.tags['[model_dir_local]']
            run_root = self.tags['[run_dir]']
            for fname in self.tags['[zip_root_files]']:
                shutil.copy2(model_root + fname, run_root + fname)

            # copy special files with changing file names
            if '[ESYSMooring_init_fname]' in self.tags:
                if self.tags['[ESYSMooring_init_fname]'] is not None:
                    fname_source = self.tags['[ESYSMooring_init_fname]']
                    fname_target = 'ESYSMooring_init.dat'
                    shutil.copy2(model_root + fname_source,
                                 run_root + fname_target)

            # copy all content of the following dirs
            dirs = [self.tags['[control_dir]'], self.tags['[hydro_dir]'],
                    self.tags['[mooring_dir]'], 'externalforce/',
                    self.tags['[data_dir]']]
            plocal = self.tags['[model_dir_local]']
            prun = self.tags['[run_dir]']

            # copy all files present in the specified folders
            for path in dirs:
                if not path:
                    continue
                for root, dirs, files in os.walk(plocal+path):
                    for file_name in files:
                        src = os.path.join(root, file_name)
                        dst = prun + path + file_name
                        shutil.copy2(src, dst)

    def create_model_zip(self):
        """

        Create the model zip file based on the master tags file settings.

        Paremeters
        ----------

        master : HtcMaster object


        """

        # FIXME: all directories should be called trough their appropriate tag!

        #model_dir = HOME_DIR + 'PhD/Projects/Hawc2Models/'+MODEL+'/'
        model_dir_server = self.tags['[run_dir]']
        model_dir_local = self.tags['[model_dir_local]']

        # ---------------------------------------------------------------------
        # create the zipfile object locally
        zf = zipfile.ZipFile(model_dir_local + self.tags['[model_zip]'],'w')

        # empty folders, the'll hold the outputs
        # zf.write(source, target in zip, )
        # TODO: use user defined directories here and in PBS
        # note that they need to be same as defined in the PBS script. We
        # manually set these up instead of just copying the original.

        animation_dir = self.tags['[animation_dir]']
        control_dir = self.tags['[control_dir]']
        eigenfreq_dir = self.tags['[eigenfreq_dir]']
        htc_dir = self.tags['[htc_dir]']
        data_dir = self.tags['[data_dir]']
        logfiles_dir = self.tags['[log_dir]']
        results_dir = self.tags['[res_dir]']
        turb_dir = self.tags['[turb_dir]']
        wake_dir = self.tags['[wake_dir]']
        meander_dir = self.tags['[meander_dir]']
        mooring_dir = self.tags['[mooring_dir]']
        hydro_dir = self.tags['[hydro_dir]']

        zf.write('.', animation_dir+'.', zipfile.ZIP_DEFLATED)
        zf.write('.', control_dir+'.', zipfile.ZIP_DEFLATED)
        if eigenfreq_dir:
            zf.write('.', eigenfreq_dir+'.', zipfile.ZIP_DEFLATED)
        zf.write('.', htc_dir+'.', zipfile.ZIP_DEFLATED)
        zf.write('.', logfiles_dir+'.', zipfile.ZIP_DEFLATED)
        zf.write('.', results_dir+'.', zipfile.ZIP_DEFLATED)
        zf.write('.', turb_dir+'.', zipfile.ZIP_DEFLATED)
        if wake_dir:
            zf.write('.', wake_dir+'.', zipfile.ZIP_DEFLATED)
        if meander_dir:
            zf.write('.', meander_dir+'.', zipfile.ZIP_DEFLATED)
        if mooring_dir:
            zf.write('.', mooring_dir+'.', zipfile.ZIP_DEFLATED)
        if hydro_dir:
            zf.write('.', hydro_dir+'.', zipfile.ZIP_DEFLATED)
        # external force dll has a hard coded path for the input
        zf.write('.', 'externalforce/.', zipfile.ZIP_DEFLATED)

        # if any, add files that should be added to the root of the zip file
        for file_name in self.tags['[zip_root_files]']:
            zf.write(model_dir_local+file_name, file_name, zipfile.ZIP_DEFLATED)

        if '[ESYSMooring_init_fname]' in self.tags:
            if self.tags['[ESYSMooring_init_fname]'] is not None:
                fname_source = self.tags['[ESYSMooring_init_fname]']
                fname_target = 'ESYSMooring_init.dat'
                zf.write(model_dir_local + fname_source, fname_target,
                         zipfile.ZIP_DEFLATED)

        # manually add all that resides in control, mooring and hydro
        paths = [control_dir, mooring_dir, hydro_dir, 'externalforce/', data_dir]
        for target_path in paths:
            if not target_path:
                continue
            for root, dirs, files in os.walk(model_dir_local+target_path):
                for file_name in files:
                    #print 'adding', file_name
                    zf.write(os.path.join(root,file_name),
                             target_path + file_name, zipfile.ZIP_DEFLATED)

        # also add the master file to the root of the zip file
        fmaster = self.tags['[master_htc_dir]'] + self.tags['[master_htc_file]']
        zf.write(fmaster, self.tags['[master_htc_file]'], zipfile.ZIP_DEFLATED)

        # and close again
        zf.close()

        # ---------------------------------------------------------------------
        # copy zip file to the server, this will be used on the nodes
        src = model_dir_local  + self.tags['[model_zip]']
        dst = model_dir_server + self.tags['[model_zip]']

        # in case we are running local and the model dir is the server dir
        # we do not need to copy the zip file, it is already on location
        if not src == dst:
            shutil.copy2(src, dst)

        ## copy to zip data file to sim_id htc folder on the server dir
        ## so we now have exactly all data to relaunch any htc file later
        #dst  = model_dir_server + self.tags['[htc_dir]']
        #dst += self.tags['[model_zip]']
        #shutil.copy2(src, dst)

    def _sweep_tags(self):
        """
        The original way with all tags in the htc file for each blade node
        """
        # set the correct sweep cruve, these values are used
        a = self.tags['[sweep_amp]']
        b = self.tags['[sweep_exp]']
        z0 = self.tags['[sweep_curve_z0]']
        ze = self.tags['[sweep_curve_ze]']
        nr = self.tags['[nr_nodes_blade]']
        # format for the x values in the htc file
        ff = ' 1.03f'
        for zz in range(nr):
            it_nosweep = '[x'+str(zz+1)+'-nosweep]'
            item = '[x'+str(zz+1)+']'
            z = self.tags['[z'+str(zz+1)+']']
            if z >= z0:
                curve = eval(self.tags['[sweep_curve_def]'])
                # new swept position = original + sweep curve
                self.tags[item]=format(self.tags[it_nosweep]+curve,ff)
            else:
                self.tags[item]=format(self.tags[it_nosweep], ff)

    def _staircase_windramp(self, nr_steps, wind_step, ramptime, septime):
        """Create a stair case wind ramp


        """

        pass

    def _all_in_one_blade_tag(self, radius_new=None):
        """
        Create htc input based on a HAWTOPT blade result file

        Automatically get the number of nodes correct in master.tags based
        on the number of blade nodes

        WARNING: initial x position of the half chord point is assumed to be
        zero

        zaxis_fact : int, default=1.0 --> is member of default tags
            Factor for the htc z-axis coordinates. The htc z axis is mapped to
            the HAWTOPT radius. If the blade radius develops in negative z
            direction, set to -1

        Parameters
        ----------

        radius_new : ndarray(n), default=False
            z coordinates of the nodes. If False, a linear distribution is
            used and the tag [nr--of-nodes-per-blade] sets the number of nodes


        """
        # TODO: implement support for x position to be other than zero

        # TODO: This is not a good place, should live somewhere else. Or
        # reconsider inputs etc so there is more freedom in changing the
        # location of the nodes, set initial x position of the blade etc

        # and save under tag [blade_htc_node_input] in htc input format

        nr_nodes = self.tags['[nr_nodes_blade]']

        blade = self.tags['[blade_hawtopt]']
        # in the htc file, blade root =0 and not blade hub radius
        blade[:,0] = blade[:,0] - blade[0,0]

        if type(radius_new).__name__ == 'NoneType':
            # interpolate to the specified number of nodes
            radius_new = np.linspace(blade[0,0], blade[-1,0], nr_nodes)

        # Data checks on radius_new
        elif not type(radius_new).__name__ == 'ndarray':
            raise ValueError, 'radius_new has to be either NoneType or ndarray'
        else:
            if not len(radius_new.shape) == 1:
                raise ValueError, 'radius_new has to be 1D'
            elif not len(radius_new) == nr_nodes:
                msg = 'radius_new has to have ' + str(nr_nodes) + ' elements'
                raise ValueError, msg

        # save the nodal positions in the tag cloud
        self.tags['[blade_nodes_z_positions]'] = radius_new

        # make sure that radius_hr is just slightly smaller than radius low res
        radius_new[-1] = blade[-1,0]-0.00000001
        twist_new = interpolate.griddata(blade[:,0], blade[:,2], radius_new)
        # blade_new is the htc node input part:
        # sec 1   x     y     z   twist;
        blade_new = scipy.zeros((len(radius_new),4))
        blade_new[:,2] = radius_new*self.tags['[zaxis_fact]']
        # twist angle remains the same in either case (standard/ojf rotation)
        blade_new[:,3] = twist_new*-1.

        # set the correct sweep cruve, these values are used
        a = self.tags['[sweep_amp]']
        b = self.tags['[sweep_exp]']
        z0 = self.tags['[sweep_curve_z0]']
        ze = self.tags['[sweep_curve_ze]']
        tmp = 'nsec ' + str(nr_nodes) + ';'
        for k in range(nr_nodes):
            tmp += '\n'
            i = k+1
            z = blade_new[k,2]
            y = blade_new[k,1]
            twist = blade_new[k,3]
            # x position, sweeping?
            if z >= z0:
                x = eval(self.tags['[sweep_curve_def]'])
            else:
                x = 0.0

            # the node number
            tmp += '        sec ' + format(i, '2.0f')
            tmp += format(x, ' 11.03f')
            tmp += format(y, ' 11.03f')
            tmp += format(z, ' 11.03f')
            tmp += format(twist, ' 11.03f')
            tmp += ' ;'

        self.tags['[blade_htc_node_input]'] = tmp

        # and create the ae file
        #5	Blade Radius [m] 	Chord[m]  T/C[%]  Set no. of pc file
        #1 25 some comments
        #0.000     0.100    21.000   1
        nr_points = blade.shape[0]
        tmp2 = '1  Blade Radius [m] Chord [m] T/C [%] pc file set nr\n'
        tmp2 += '1  %i auto generated by _all_in_one_blade_tag()' % nr_points

        for k in range(nr_points):
            tmp2 += '\n'
            tmp2 += '%9.3f %9.3f %9.3f' % (blade[k,0], blade[k,1], blade[k,3])
            tmp2 += ' %4i' % (k+1)
        # end with newline
        tmp2 += '\n'

        # TODO: finish writing file, implement proper handling of hawtopt path
        # and save the file
        #if self.tags['aefile']
        #write_file(file_path, tmp2, 'w')

    def loadmaster(self):
        """
        Load the master file, path to master file is defined in
        __init__(): target, master
        """

        # what is faster, load the file in one string and do replace()?
        # or the check error log approach?

        path_to_master  = self.tags['[master_htc_dir]']
        path_to_master += self.tags['[master_htc_file]']

        # load the file:
        if not self.silent:
            print('loading master: ' + path_to_master)
        FILE = open(path_to_master, 'r')
        lines = FILE.readlines()
        FILE.close()

        # convert to string:
        self.master_str = ''
        for line in lines:
            self.master_str += line

    def createcase_check(self, htc_dict_repo, \
                            tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        Check if a certain case name already exists in a specified htc_dict.
        If true, return a message and do not create the case. It can be that
        either the case name is a duplicate and should be named differently,
        or that the simulation is a duplicate and it shouldn't be repeated.
        """

        # is the [case_id] tag unique, given the htc_dict_repo?
        if self.verbose:
            print('checking if following case is in htc_dict_repo: ')
            print(self.tags['[case_id]'] + '.htc')

        if htc_dict_repo.has_key(self.tags['[case_id]'] + '.htc'):
            # if the new case_id already exists in the htc_dict_repo
            # do not add it again!
            # print('case_id key is not unique in the given htc_dict_repo!'
            raise UserWarning, \
                'case_id key is not unique in the given htc_dict_repo!'
        else:
            htc = self.createcase(tmp_dir=tmp_dir, write_htc=write_htc)
            return htc

    def createcase(self, tmp_dir='/tmp/HawcPyTmp/', write_htc=True):
        """
        replace all the tags from the master file and save the new htc file
        """

        htc = self.master_str

        # and now replace all the tags in the htc master file
        # when iterating over a dict, it will give the key, given in the
        # corresponding format (string keys as strings, int keys as ints...)
        for k in self.tags:
            value = self.tags[k]
            # TODO: give error if a default is not defined, like null
            # if it is a boolean, replace with ; or blank
            if type(self.tags[k]).__name__ == 'bool' and self.tags[k]:
                # we have a boolean that is True, switch it on
                value = ''
            elif type(self.tags[k]).__name__ == 'bool' and not self.tags[k]:
                value = ';'
            # if string is not found, it will do nothing
            htc = htc.replace(str(k), str(value))

        # and save the the case htc file:
        case = self.tags['[case_id]'] + '.htc'

        htc_target = self.tags['[run_dir]'] + self.tags['[htc_dir]']
        if not self.silent:
            print('htc will be written to: ')
            print('  ' + htc_target)
            print('  ' + case)

        # and write the htc file to the temp dir first
        if write_htc:
            # create subfolder if necesarrt
            if not os.path.exists(htc_target):
                os.makedirs(htc_target)
            write_file(htc_target + case, htc, 'w')
            # write_file(tmp_dir + case, htc, 'w')

        # return the used tags, some parameters can be used later, such as the
        # turbulence name in the pbs script
        # return as a dictionary, to be used in htc_dict
        tmp = dict()
        # return a copy of the tags, otherwise you will not catch changes
        # made to the different tags in your sim series
        tmp[case] = copy.copy(self.tags)
        return tmp

class PBS:
    """
    The part where the actual pbs script is writtin in this class (functions
    create(), starting() and ending() ) is based on the MS Excel macro
    written by Torben J. Larsen

    input a list with htc file names, and a dict with the other paths,
    such as the turbulence file and folder, htc folder and others
    """

    def __init__(self, htc_dict, server='gorm'):
        """
        Define the settings here. This should be done outside, but how?
        In a text file, paramters list or first create the object and than set
        the non standard values??

        where htc_dict is a dictionary with
            [key=case name, value=used_tags_dict]

        where tags as outputted by MasterFile (dict with the chosen options)

        For gorm, maxcpu is set to 1, do not change otherwise you might need to
        change the scratch dir handling.
        """
        self.server = server
        self.verbose = True

#        if server == 'thyra':
#            self.maxcpu = 4
#            self.secperiter = 0.020
        if server == 'gorm':
            self.maxcpu = 1
            self.secperiter = 0.012
        elif server == 'jess':
            self.maxcpu = 1
            self.secperiter = 0.012
        else:
            raise UserWarning, 'server support only for jess or gorm'

        # the output channels comes with a price tag. Each time step
        # will have a penelty depending on the number of output channels

        self.iterperstep = 8.0 # average nr of iterations per time step
        # lead time: account for time losses when starting a simulation,
        # copying the turbulence data, generating the turbulence
        self.tlead = 5.0*60.0

        # pbs script prefix, this name will show up in the qstat listings
        self.pref = 'HAWC2_'
        # the actual script starts empty
        self.pbs = ''

#        self.wine = 'WINEARCH=win32 WINEPREFIX=~/.wine32 wine'
        self.wine = 'wine'  # the full line seems to have issues
        self.wine_dir = '/home/leob/.wine32/drive_c/bin'
        # /dev/shm should be the RAM of the cluster
#        self.node_run_root = '/dev/shm'
        self.node_run_root = '/scratch'

        self.htc_dict = htc_dict

        # location of the output messages .err and .out created by the node
        self.pbs_out_dir = 'pbs_out/'
        self.pbs_in_dir = 'pbs_in/'

        # for the start number, take hour/minute combo
        d = datetime.datetime.today()
        tmp = int( str(d.hour)+format(d.minute, '02.0f') )*100
        self.pbs_start_number = tmp
        self.copyback_turb = True
        self.copyback_fnames = []
        self.copyback_fnames_rename = []
        self.copyto_generic = []
        self.copyto_fname = []

    def create(self):
        """
        Main loop for creating the pbs scripts, based on the htc_dict, which
        contains the case name as key and tag dictionairy as value
        """

        # dynamically set walltime based on the number of time steps
        # for thyra, make a list so we base the walltime on the slowest case
        self.nr_time_steps = []
        self.duration = []
        self.t0 = []
        # '[time_stop]' '[dt_sim]'

        # REMARK: this i not realy consistent with how the result and log file
        # dirs are allowed to change for each individual case...
        # first check if the pbs_out_dir exists, this dir is considered to be
        # the same for all cases present in the htc_dict
        # self.tags['[run_dir]']
        case0 = self.htc_dict.keys()[0]
        path = self.htc_dict[case0]['[run_dir]'] + self.pbs_out_dir
        if not os.path.exists(path):
            os.makedirs(path)

        # create pbs_in base dir
        path = self.htc_dict[case0]['[run_dir]'] + self.pbs_in_dir
        if not os.path.exists(path):
            os.makedirs(path)

        # number the pbs jobs:
        count2 = self.pbs_start_number
        # initial cpu count is zero
        count1 = 1
        # scan through all the cases
        i, i_tot = 1, len(self.htc_dict)
        ended = True

        for case in self.htc_dict:

            # get a shorter version for the current cases tag_dict:
            tag_dict = self.htc_dict[case]

            # group all values loaded from the tag_dict here, to keep overview
            # the directories to SAVE the results/logs/turb files
            # load all relevant dir settings: the result/logfile/turbulence/zip
            # they are now also available for starting() and ending() parts
            hawc2_exe = tag_dict['[hawc2_exe]']
            self.results_dir = tag_dict['[res_dir]']
            self.eigenfreq_dir = tag_dict['[eigenfreq_dir]']
            self.logs_dir = tag_dict['[log_dir]']
            self.animation_dir = tag_dict['[animation_dir]']
            self.TurbDirName = tag_dict['[turb_dir]']
            self.TurbDb = tag_dict['[turb_db_dir]']
            self.WakeDirName = tag_dict['[wake_dir]']
            self.MeanderDirName = tag_dict['[meander_dir]']
            self.ModelZipFile = tag_dict['[model_zip]']
            self.htc_dir = tag_dict['[htc_dir]']
            self.hydro_dir = tag_dict['[hydro_dir]']
            self.mooring_dir = tag_dict['[mooring_dir]']
            self.model_path = tag_dict['[run_dir]']
            self.turb_base_name = tag_dict['[turb_base_name]']
            self.wake_base_name = tag_dict['[wake_base_name]']
            self.meand_base_name = tag_dict['[meand_base_name]']
            self.pbs_queue_command = tag_dict['[pbs_queue_command]']
            self.walltime = tag_dict['[walltime]']
            self.dyn_walltime = tag_dict['[auto_walltime]']

            # create the pbs_out_dir if necesary
            try:
                path = tag_dict['[run_dir]'] + tag_dict['[pbs_out_dir]']
                if not os.path.exists(path):
                    os.makedirs(path)
                self.pbs_out_dir = tag_dict['[pbs_out_dir]']
            except:
                pass

            # create pbs_in subdirectories if necessary
            try:
                path = tag_dict['[run_dir]'] + tag_dict['[pbs_in_dir]']
                if not os.path.exists(path):
                    os.makedirs(path)
                self.pbs_in_dir = tag_dict['[pbs_in_dir]']
            except:
                pass

            try:
                self.copyback_files = tag_dict['[copyback_files]']
                self.copyback_frename = tag_dict['[copyback_frename]']
            except KeyError:
                pass

            try:
                self.copyto_generic = tag_dict['[copyto_generic]']
                self.copyto_files = tag_dict['[copyto_files]']
            except KeyError:
                pass

            # related to the dynamically setting the walltime
            duration = float(tag_dict['[time_stop]'])
            dt = float(tag_dict['[dt_sim]'])
            self.nr_time_steps.append(duration/dt)
            self.duration.append(float(tag_dict['[duration]']))
            self.t0.append(float(tag_dict['[t0]']))

            if self.verbose:
                print('htc_dir in pbs.create:')
                print(self.htc_dir)
                print(self.model_path)

            # we only start a new case, if we have something that ended before
            # the very first case has to start with starting
            if ended:
                count1 = 1
                # define the path for the new pbs script
                jobid = self.pref + str(count2)
                pbs_in_fname = '%s_%s.p' % (tag_dict['[case_id]'], jobid)
                pbs_path = self.model_path + self.pbs_in_dir + pbs_in_fname
                # Start a new pbs script, we only need the tag_dict here
                self.starting(tag_dict, jobid)
                ended = False

            # -----------------------------------------------------------------
            # WRITING THE ACTUAL JOB PARAMETERS

            # output the current scratch directory
            self.pbs += "pwd\n"
            # zip file has been copied to the node before (in start_pbs())
            # unzip now in the node
            self.pbs += "/usr/bin/unzip " + self.ModelZipFile + '\n'
            # create all directories, especially relevant if there are case
            # dependent sub directories that are not present in the ZIP file
            self.pbs += "mkdir " + self.htc_dir + '\n'
            self.pbs += "mkdir " + self.results_dir + '\n'
            self.pbs += "mkdir " + self.logs_dir + '\n'
            self.pbs += "mkdir " + self.TurbDirName + '\n'
            if self.WakeDirName:
                self.pbs += "mkdir " + self.WakeDirName + '\n'
            if self.MeanderDirName:
                self.pbs += "mkdir " + self.MeanderDirName + '\n'
            if self.hydro_dir:
                self.pbs += "mkdir " + self.hydro_dir + '\n'
            # create the eigen analysis dir just in case that is necessary
            if self.eigenfreq_dir:
                self.pbs += 'mkdir %s \n' % self.eigenfreq_dir

            # and copy the htc file to the node
            self.pbs += "cp -R $PBS_O_WORKDIR/" + self.htc_dir \
                + case +" ./" + self.htc_dir + '\n'

            # if there is a turbulence file data base dir, copy from there
            if self.TurbDb is not None:
                tmp = (self.TurbDb, self.turb_base_name, self.TurbDirName)
                self.pbs += "cp -R $PBS_O_WORKDIR/%s%s*.bin %s \n" % tmp
            else:
                # turbulence files basenames are defined for the case
                self.pbs += "cp -R $PBS_O_WORKDIR/" + self.TurbDirName + \
                    self.turb_base_name + "*.bin ./"+self.TurbDirName + '\n'

            if self.WakeDirName:
                self.pbs += "cp -R $PBS_O_WORKDIR/" + self.WakeDirName + \
                    self.wake_base_name + "*.bin ./"+self.WakeDirName + '\n'

            if self.MeanderDirName:
                self.pbs += "cp -R $PBS_O_WORKDIR/" + self.MeanderDirName + \
                    self.meand_base_name + "*.bin ./"+self.MeanderDirName + '\n'

            # copy and rename input files with given versioned name to the
            # required non unique generic version
            for fname, fgen in zip(self.copyto_files, self.copyto_generic):
                self.pbs += "cp -R $PBS_O_WORKDIR/%s ./%s \n" % (fname, fgen)

            # the hawc2 execution commands via wine
            param = (self.wine, hawc2_exe, self.htc_dir+case)
            self.pbs += "%s %s ./%s &\n" % param

            #self.pbs += "wine get_mac_adresses" + '\n'
            # self.pbs += "cp -R ./*.mac  $PBS_O_WORKDIR/." + '\n'
            # -----------------------------------------------------------------

            # and we end when the cpu's per node are full
            if int(count1/self.maxcpu) == 1:
                # write the end part of the pbs script
                self.ending(pbs_path, jobid)
                ended = True
                # print progress:
                replace = ((i/self.maxcpu), (i_tot/self.maxcpu), self.walltime)
                print('pbs script %3i/%i walltime=%s' % replace)

            count2 += 1
            i += 1
            # the next cpu
            count1 += 1

        # it could be that the last node was not fully loaded. In that case
        # we do not have had a succesfull ending, and we still need to finish
        if not ended:
            # write the end part of the pbs script
            self.ending(pbs_path, jobid)
            # progress printing
            replace = ( (i/self.maxcpu), (i_tot/self.maxcpu), self.walltime )
            print('pbs script %3i/%i walltime=%s, partially loaded' % replace)
#            print 'pbs progress, script '+format(i/self.maxcpu,'2.0f')\
#                + '/' + format(i_tot/self.maxcpu, '2.0f') \
#                + ' partially loaded...'

    def starting(self, tag_dict, jobid):
        """
        First part of the pbs script
        """

        # a new clean pbs script!
        self.pbs = ''
        self.pbs += "### Standard Output" + ' \n'

        case_id = tag_dict['[case_id]']

        # PBS job name
        self.pbs += "#PBS -N %s \n" % (jobid)
        self.pbs += "#PBS -o ./" + self.pbs_out_dir + case_id + ".out" + '\n'
        # self.pbs += "#PBS -o ./pbs_out/" + jobid + ".out" + '\n'
        self.pbs += "### Standard Error" + ' \n'
        self.pbs += "#PBS -e ./" + self.pbs_out_dir + case_id + ".err" + '\n'
        # self.pbs += "#PBS -e ./pbs_out/" + jobid + ".err" + '\n'
        self.pbs += '#PBS -W umask=644\n'
        self.pbs += "### Maximum wallclock time format HOURS:MINUTES:SECONDS\n"
#        self.pbs += "#PBS -l walltime=" + self.walltime + '\n'
        self.pbs += "#PBS -l walltime=[walltime]\n"
        self.pbs += "#PBS -a [start_time]" + '\n'
        # in case of gorm, we need to make it work correctly. Now each job
        # has a different scratch dir. If we set maxcpu to 12 they all have
        # the same scratch dir. In that case there should be done something
        # differently

        # specify the number of nodes and cpu's per node required
        if self.maxcpu > 1:
            # Number of nodes and cpus per node (ppn)
            lnodes = int(math.ceil(len(self.htc_dict)/float(self.maxcpu)))
            lnodes = 1
            self.pbs += "#PBS -lnodes=%i:ppn=%i\n" % (lnodes, self.maxcpu)
        else:
            self.pbs += "#PBS -lnodes=1:ppn=1\n"
            # Number of nodes and cpus per node (ppn)

        self.pbs += "### Queue name" + '\n'
        # queue names for Thyra are as follows:
        # short walltime queue (shorter than an hour): '#PBS -q xpresq'
        # or otherwise for longer jobs: '#PBS -q workq'
        self.pbs += self.pbs_queue_command + '\n'

        self.pbs += "### Create scratch directory and copy data to it \n"
        # output the current directory
        self.pbs += "cd $PBS_O_WORKDIR" + '\n'
        self.pbs += "pwd \n"
        # The batch system on Gorm allows more than one job per node.
        # Because of this the scratch directory name includes both the
        # user name and the job ID, that is /scratch/$USER/$PBS_JOBID
        # if not scratch, make the dir
        if self.node_run_root != '/scratch':
            self.pbs += 'mkdir %s/$USER\n' % self.node_run_root
            self.pbs += 'mkdir %s/$USER/$PBS_JOBID\n' % self.node_run_root

        # copy the zip files to the scratch dir on the node
        self.pbs += "cp -R ./" + self.ModelZipFile + \
            ' %s/$USER/$PBS_JOBID\n' % (self.node_run_root)

        self.pbs += "### Execute commands on scratch nodes \n"
        self.pbs += 'cd %s/$USER/$PBS_JOBID\n' % self.node_run_root
#        # also copy all the HAWC2 exe's to the scratch dir
#        self.pbs += "cp -R %s/* ./\n" % self.wine_dir
#        # custom name hawc2 exe
#        self.h2_new = tag_dict['[hawc2_exe]'] + '-' + jobid + '.exe'
#        self.pbs += "mv %s.exe %s\n" % (tag_dict['[hawc2_exe]'], self.h2_new)

    def ending(self, pbs_path, jobid):
        """
        Last part of the pbs script, including command to write script to disc
        COPY BACK: from node to
        """

        self.pbs += "### wait for jobs to finish \n"
        self.pbs += "wait\n"
        self.pbs += "### Copy back from scratch directory \n"
        for i in range(1,self.maxcpu+1,1):

            # navigate to the cpu dir on the node
            # The batch system on Gorm allows more than one job per node.
            # Because of this the scratch directory name includes both the
            # user name and the job ID, that is /scratch/$USER/$PBS_JOBID
            # NB! This is different from Thyra!
            self.pbs += "cd %s/$USER/$PBS_JOBID\n" % self.node_run_root

            # create the log, res etc dirs in case they do not exist
            self.pbs += "mkdir $PBS_O_WORKDIR/" + self.results_dir + "\n"
            self.pbs += "mkdir $PBS_O_WORKDIR/" + self.logs_dir + "\n"
            if self.animation_dir:
                self.pbs += "mkdir $PBS_O_WORKDIR/" + self.animation_dir + "\n"
            if self.WakeDirName:
                self.pbs += "mkdir $PBS_O_WORKDIR/" + self.WakeDirName + "\n"
            if self.MeanderDirName:
                self.pbs += "mkdir $PBS_O_WORKDIR/" + self.MeanderDirName + "\n"

            # and copy the results and log files frome the node to the
            # thyra home dir
            self.pbs += "cp -R " + self.results_dir + \
                ". $PBS_O_WORKDIR/" + self.results_dir + ".\n"
            self.pbs += "cp -R " + self.logs_dir + \
                ". $PBS_O_WORKDIR/" + self.logs_dir + ".\n"
            if self.animation_dir:
                self.pbs += "cp -R " + self.animation_dir + \
                    ". $PBS_O_WORKDIR/" + self.animation_dir + ".\n"

            if self.eigenfreq_dir:
                # just in case the eig dir has subdirs for the results, only
                # select the base path and cp -r will take care of the rest
                p1 = self.eigenfreq_dir.split('/')[0]
                self.pbs += "cp -R %s/. $PBS_O_WORKDIR/%s/. \n" % (p1, p1)
                # for eigen analysis with floater, modes are in root
                eig_dir_sys = '%ssystem/' % self.eigenfreq_dir
                self.pbs += 'mkdir $PBS_O_WORKDIR/%s \n' % eig_dir_sys
                self.pbs += "cp -R mode* $PBS_O_WORKDIR/%s. \n" % eig_dir_sys

            # copy back turbulence file?
            if self.copyback_turb and self.TurbDb is not None:
                self.pbs += "cp -R " + self.TurbDirName + \
                    ". $PBS_O_WORKDIR/" + self.TurbDb + ".\n"
                if self.WakeDirName:
                    self.pbs += "cp -R " + self.WakeDirName + \
                        ". $PBS_O_WORKDIR/" + self.WakeDirName + ".\n"
                if self.MeanderDirName:
                    self.pbs += "cp -R " + self.MeanderDirName + \
                        ". $PBS_O_WORKDIR/" + self.MeanderDirName + ".\n"
            elif self.copyback_turb:
                self.pbs += "cp -R " + self.TurbDirName + \
                    ". $PBS_O_WORKDIR/" + self.TurbDirName + ".\n"
                if self.WakeDirName:
                   self.pbs += "cp -R " + self.WakeDirName + \
                        ". $PBS_O_WORKDIR/" + self.WakeDirName + ".\n"
                if self.MeanderDirName:
                    self.pbs += "cp -R " + self.MeanderDirName + \
                        ". $PBS_O_WORKDIR/" + self.MeanderDirName + ".\n"

            # copy back any other kind of file specified
            if len(self.copyback_frename) == 0:
                self.copyback_frename = self.copyback_files
            for fname, fnew in zip(self.copyback_files, self.copyback_frename):
                self.pbs += "cp -R %s $PBS_O_WORKDIR/%s \n" % (fname, fnew)

            # check what is left
            self.pbs += 'ls -lah\n'

            # and delete it all
            self.pbs += 'cd ..\n'
            self.pbs += 'ls -lah\n'
            self.pbs += 'echo $PBS_JOBID\n'
            self.pbs += 'rm -r $PBS_JOBID \n'

            # Delete the batch file at the end. However, is this possible since
            # the batch file is still open at this point????
            # self.pbs += "rm "

        # base walltime on the longest simulation in the batch
        nr_time_steps = max(self.nr_time_steps)
        # TODO: take into acccount the difference between time steps with
        # and without output. This penelaty also depends on the number of
        # channels outputted. So from 0 until t0 we have no penalty,
        # from t0 until t0+duration we have the output penalty.

        # always a predifined lead time to account for startup losses
        tmax = int(nr_time_steps*self.secperiter*self.iterperstep + self.tlead)
        if self.dyn_walltime:
            dt_seconds = datetime.datetime.fromtimestamp(tmax)
            self.walltime = dt_seconds.strftime('%H:%M:%S')
            self.pbs = self.pbs.replace('[walltime]', self.walltime)
        else:
            self.pbs = self.pbs.replace('[walltime]', self.walltime)
        # and reset the nr_time_steps list for the next pbs job file
        self.nr_time_steps = []
        self.t0 = []
        self.duration = []

        # TODO: add logfile checking support directly here. In that way each
        # node will do the logfile checking and statistics calculations right
        # after the simulation. Figure out a way how to merge the data from
        # all the different cases afterwards

        self.pbs += "exit\n"

        if self.verbose:
            print('writing pbs script to path: ' + pbs_path)

        # and write the script to a file:
        write_file(pbs_path,self.pbs, 'w')
        # make the string empty again, for memory
        self.pbs = ''

    def check_results(self, htc_dict):
        """
        Cross-check if all simulations on the list have returned a simulation.
        Combine with ErrorLogs to identify which errors occur where.
        """

        htc_dict_fail = {}

        print('checking if all log and result files are present...', end='')

        # check for each case if we have results and a log file
        for cname, case in htc_dict.iteritems():
            run_dir = case['[run_dir]']
            res_dir = case['[res_dir]']
            log_dir = case['[log_dir]']
            cname_ = cname.replace('.htc', '')
            if not os.path.exists(run_dir + log_dir + cname_ + '.log'):
                htc_dict_fail[cname] = copy.copy(htc_dict[cname])
                continue
            try:
                size_sel = os.stat(run_dir + res_dir + cname_ + '.sel').st_size
                size_dat = os.stat(run_dir + res_dir + cname_ + '.dat').st_size
            except OSError:
                size_sel = 0
                size_dat = 0
            if size_sel < 5 or size_dat < 5:
                htc_dict_fail[cname] = copy.copy(htc_dict[cname])

        print('done!')

        # length will be zero if there are no failures
        return htc_dict_fail

# TODO: rewrite the error log analysis to something better. Take different
# approach: start from the case and see if the results are present. Than we
# also have the tags_dict available when log-checking a certain case
class ErrorLogs:
    """
    Analyse all HAWC2 log files in any given directory
    ==================================================

    Usage:
    logs = ErrorLogs()
    logs.MsgList    : list with the to be checked messages. Add more if required
    logs.ResultFile : name of the result file (default is ErrorLog.csv)
    logs.PathToLogs : specify the directory where the logsfile reside,
                        the ResultFile will be saved in the same directory.
                        It is also possible to give the path of a specific
                        file, the logfile will not be saved in this case. Save
                        when all required messages are analysed with save()
    logs.check() to analyse all the logfiles and create the ResultFile
    logs.save() to save after single file analysis

    logs.MsgListLog : [ [case, line nr, error1, line nr, error2, ....], [], ...]
    holding the error messages, empty if no err msg found
    will survive as long as the logs object exists. Keep in
    mind that when processing many messages with many error types (as defined)
    in MsgList might lead to an increase in memory usage.

    logs.MsgListLog2 : dict(key=case, value=[found_error, exit_correct]
        where found_error and exit_correct are booleans. Found error will just
        indicate whether or not any error message has been found

    All files in the speficied folder (PathToLogs) will be evaluated.
    When Any item present in MsgList occurs, the line number of the first
    occurance will be displayed in the ResultFile.
    If more messages are required, add them to the MsgList
    """

    # TODO: move to the HAWC2 plugin for cases

    def __init__(self, silent=False, cases=None):

        self.silent = silent
        # specify folder which contains the log files
        self.PathToLogs = ''
        self.ResultFile = 'ErrorLog.csv'

        self.cases = cases

        # the total message list log:
        self.MsgListLog = []
        # a smaller version, just indication if there are errors:
        self.MsgListLog2 = dict()

        # specify which message to look for. The number track's the order.
        # this makes it easier to view afterwards in spreadsheet:
        # every error will have its own column

        # error messages that appear during initialisation
        self.err_init = {}
        self.err_init[' *** ERROR ***  in command '] = len(self.err_init.keys())
        #  *** WARNING *** A comma "," is written within the command line
        self.err_init[' *** WARNING *** A comma ",'] = len(self.err_init.keys())
        #  *** ERROR *** Not correct number of parameters
        self.err_init[' *** ERROR *** Not correct '] = len(self.err_init.keys())
        #  *** INFO *** End of file reached
        self.err_init[' *** INFO *** End of file r'] = len(self.err_init.keys())
        #  *** ERROR *** No line termination in command line
        self.err_init[' *** ERROR *** No line term'] = len(self.err_init.keys())
        #  *** ERROR *** MATRIX IS NOT DEFINITE
        self.err_init[' *** ERROR *** MATRIX IS NO'] = len(self.err_init.keys())
        #  *** ERROR *** There are unused relative
        self.err_init[' *** ERROR *** There are un'] = len(self.err_init.keys())
        #  *** ERROR *** Error finding body based
        self.err_init[' *** ERROR *** Error findin'] = len(self.err_init.keys())
        #  *** ERROR *** In body actions
        self.err_init[' *** ERROR *** In body acti'] = len(self.err_init.keys())
        #  *** ERROR *** Command unknown
        self.err_init[' *** ERROR *** Command unkn'] = len(self.err_init.keys())
        #  *** ERROR *** ERROR - More bodies than elements on main_body: tower
        self.err_init[' *** ERROR *** ERROR - More'] = len(self.err_init.keys())
        #  *** ERROR *** The program will stop
        self.err_init[' *** ERROR *** The program '] = len(self.err_init.keys())
        #  *** ERROR *** Unknown begin command in topologi.
        self.err_init[' *** ERROR *** Unknown begi'] = len(self.err_init.keys())
        #  *** ERROR *** Not all needed topologi main body commands present
        self.err_init[' *** ERROR *** Not all need'] = len(self.err_init.keys())
        #  *** ERROR ***  in command line
        self.err_init[' *** ERROR ***  in command '] = len(self.err_init.keys())
        #  *** ERROR ***  opening timoschenko data file
        self.err_init[' *** ERROR ***  opening tim'] = len(self.err_init.keys())
        #  *** ERROR *** Error opening AE data file
        self.err_init[' *** ERROR *** Error openin'] = len(self.err_init.keys())
        #  *** ERROR *** Requested blade _ae set number not found in _ae file
        self.err_init[' *** ERROR *** Requested bl'] = len(self.err_init.keys())
        #  Error opening PC data file
        self.err_init[' Error opening PC data file'] = len(self.err_init.keys())
        #  *** ERROR *** error reading mann turbulence
        self.err_init[' *** ERROR *** error readin'] = len(self.err_init.keys())
        #  *** INFO *** The DLL subroutine
        self.err_init[' *** INFO *** The DLL subro'] = len(self.err_init.keys())
        #  ** WARNING: FROM ESYS ELASTICBAR: No keyword
        self.err_init[' ** WARNING: FROM ESYS ELAS'] = len(self.err_init.keys())
        #  *** ERROR *** DLL ./control/killtrans.dll could not be loaded - error!
        self.err_init[' *** ERROR *** DLL'] = len(self.err_init.keys())
        # *** ERROR *** The DLL subroutine
        self.err_init[' *** ERROR *** The DLL subr'] = len(self.err_init.keys())

        # error messages that appear during simulation
        self.err_sim = {}
        #  *** ERROR *** Wind speed requested inside
        self.err_sim[' *** ERROR *** Wind speed r'] = len(self.err_sim.keys())
        #  Maximum iterations exceeded at time step:
        self.err_sim[' Maximum iterations exceede'] = len(self.err_sim.keys())
        #  Solver seems not to converge:
        self.err_sim[' Solver seems not to conver'] = len(self.err_sim.keys())
        #  *** ERROR *** Out of x bounds:
        self.err_sim[' *** ERROR *** Out of x bou'] = len(self.err_sim.keys())

        # TODO: error message from a non existing channel output/input
        # add more messages if required...

        self.init_cols = len(self.err_init.keys())
        self.sim_cols = len(self.err_sim.keys())

    def check(self):

        # MsgListLog = []

        # load all the files in the given path
        FileList = []
        for files in os.walk(self.PathToLogs):
            FileList.append(files)

        # if the instead of a directory, a file path is given
        # the generated FileList will be empty!
        try:
            NrFiles = len(FileList[0][2])
        # input was a single file:
        except:
            NrFiles = 1
            # simulate one entry on FileList[0][2], give it the file name
            # and save the directory on in self.PathToLogs
            tmp = self.PathToLogs.split('/')[-1]
            # cut out the file name from the directory
            self.PathToLogs = self.PathToLogs.replace(tmp, '')
            FileList.append([ [],[],[tmp] ])
            single_file = True
        i=1

        # walk trough the files present in the folder path
        for fname in FileList[0][2]:
            fname_lower = fname.lower()
            # progress indicator
            if NrFiles > 1:
                if not self.silent:
                    print('progress: ' + str(i) + '/' + str(NrFiles))

            # open the current log file
            FILE = open(self.PathToLogs+str(fname_lower), 'r')
            lines = FILE.readlines()
            FILE.close()

            # keep track of the messages allready found in this file
            tempLog = []
            tempLog.append(fname)
            exit_correct, found_error = False, False
            # create empty list item for the different messages and line
            # number. Include one column for non identified messages
            for j in range(self.init_cols + self.sim_cols + 1):
                tempLog.append('')
                tempLog.append('')

            # if there is a cases object, see how many time steps we expect
            if self.cases is not None:
                case = self.cases[fname.replace('.log', '.htc')]
                dt = float(case['[dt_sim]'])
                time_steps = float(case['[time_stop]']) / dt
                iterations = np.ndarray( (time_steps,3), dtype=np.float32 )
            else:
                iterations = np.ndarray( (len(lines),3), dtype=np.float32 )
                dt = False
            iterations[:,0:2] = -1
            iterations[:,2] = 0

            # keep track of the time_step number
            time_step, init_block = -1, True
            # check for messages in the current line
            # for speed: delete from message watch list if message is found
            for j, line in enumerate(lines):
                # all id's of errors are 27 characters long
                msg = line[:27]

                # keep track of the number of iterations
                if line[:12] == ' Global time':
                    time_step += 1
                    iterations[time_step,0] = float(line[14:40])
                    iterations[time_step,1] = int(line[-6:-2])
                    # time step is the first time stamp
                    if not dt:
                        dt = float(line[15:40])
                    # no need to look for messages if global time is mentioned
                    continue

                elif line[:20] == ' Starting simulation':
                    init_block = False

                elif init_block:
                    # if string is shorter, we just get a shorter string.
                    # checking presence in dict is faster compared to checking
                    # the length of the string
                    if msg in self.err_init:
                        col_nr = self.err_init[msg]
                        # 2nd item is the column position of the message
                        tempLog[2*(col_nr+1)] = line[:-2]
                        # line number of the message
                        tempLog[2*col_nr+1] += '%i, ' % j
                        found_error = True

                # find errors that can occur during simulation
                elif msg in self.err_sim:
                    col_nr = self.err_sim[msg] + self.init_cols
                    # 2nd item is the column position of the message
                    tempLog[2*(col_nr+1)] = line[:-2]
                    # in case stuff already goes wrong on the first time step
                    if time_step == -1:
                        time_step = 0
                    # line number of the message
                    tempLog[2*col_nr+1] += '%i, ' % time_step
                    found_error = True
                    iterations[time_step,2] = 1

                # method of last resort, we have no idea what message
                elif line[:13] == ' *** ERROR ***' or line[:10]==' ** WARNING':
                    tempLog[-2] = line[:-2]
                    # line number of the message
                    tempLog[-1] = j
                    found_error = True

            # see if the last line holds the sim time
            if line[:15] ==  ' Elapsed time :':
                exit_correct = True
                elapsed_time = float(line[15:-3])
                tempLog.append( elapsed_time )
            else:
                elapsed_time = -1
                tempLog.append('')

            # give the last recorded time step
            tempLog.append('%1.11f' % iterations[time_step,0])

            # simulation and simulation output time
            if self.cases is not None:
                t_stop = float(case['[time_stop]'])
                tempLog.append('%1.01f' % t_stop)
                tempLog.append('%1.04f' % (t_stop/elapsed_time))
                tempLog.append('%1.01f' % float(case['[duration]']))
            else:
                tempLog.append('')
                tempLog.append('')
                tempLog.append('')

            # as last element, add the total number of iterations
            itertotal = np.nansum(iterations[:,1])
            tempLog.append('%i' % itertotal)

            # the delta t used for the simulation
            if dt:
                tempLog.append('%1.7f' % dt)
            else:
                tempLog.append('failed to find dt')

            # number of time steps
            tempLog.append('%i' % len(iterations) )

            # if the simulation didn't end correctly, the elapsed_time doesn't
            # exist. Add the average and maximum nr of iterations per step
            # or, if only the structural and eigen analysis is done, we have 0
            try:
                ratio = float(elapsed_time)/float(itertotal)
                tempLog.append('%1.6f' % ratio)
            except (UnboundLocalError, ZeroDivisionError, ValueError) as e:
                tempLog.append('')
            # when there are no time steps (structural analysis only)
            try:
                tempLog.append('%1.2f' % iterations[:,1].mean() )
                tempLog.append('%1.2f' % iterations[:,1].max() )
            except ValueError:
                tempLog.append('')
                tempLog.append('')

            # save the iterations in the results folder
            fiter = fname.replace('.log', '.iter')
            fmt = ['%12.06f', '%4i', '%4i']
            if self.cases is not None:
                fpath = case['[run_dir]'] + case['[iter_dir]']
                # in case it has subdirectories
                for tt in [3,2,1]:
                    tmp = os.path.sep.join(fpath.split(os.path.sep)[:-tt])
                    if not os.path.exists(tmp):
                        os.makedirs(tmp)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                np.savetxt(fpath + fiter, iterations, fmt=fmt)
            else:
                np.savetxt(self.PathToLogs + fiter, iterations, fmt=fmt)

            # append the messages found in the current file to the overview log
            self.MsgListLog.append(tempLog)
            self.MsgListLog2[fname] = [found_error, exit_correct]
            i += 1

#            # if no messages are found for the current file, than say so:
#            if len(MsgList2) == len(self.MsgList):
#                tempLog[-1] = 'NO MESSAGES FOUND'

        # if we have only one file, don't save the log file to disk. It is
        # expected that if we analyse many different single files, this will
        # cause a slower script
        if single_file:
            # now we make it available over the object to save and let it grow
            # over many analysis
            # self.MsgListLog = copy.copy(MsgListLog)
            pass
        else:
            self.save()

    def save(self):

        # write the results in a file, start with a header
        contents = 'file name;' + 'lnr;msg;'*(self.init_cols)
        contents += 'iter_nr;msg;'*(self.sim_cols)
        contents += 'lnr;msg;'
        # and add headers for elapsed time, nr of iterations, and sec/iteration
        contents += 'Elapsted time;last time step;Simulation time;'
        contents += 'real sim time;Sim output time;'
        contents += 'total iterations;dt;nr time steps;'
        contents += 'seconds/iteration;average iterations/time step;'
        contents += 'maximum iterations/time step;\n'
        for k in self.MsgListLog:
            for n in k:
                contents = contents + str(n) + ';'
            # at the end of each line, new line symbol
            contents = contents + '\n'

        # write csv file to disk, append to facilitate more logfile analysis
        if not self.silent:
            print('Error log analysis saved at:')
            print(self.PathToLogs+str(self.ResultFile))
        FILE = open(self.PathToLogs+str(self.ResultFile), 'a')
        FILE.write(contents)
        FILE.close()


class ModelData:
    """
    Second generation ModelData function. The HawcPy version is crappy, buggy
    and not mutch of use in the optimisation context.
    """
    class st_headers:
        """
        Indices to the respective parameters in the HAWC2 st data file
        """
        r     = 0
        m     = 1
        x_cg  = 2
        y_cg  = 3
        ri_x  = 4
        ri_y  = 5
        x_sh  = 6
        y_sh  = 7
        E     = 8
        G     = 9
        Ixx   = 10
        Iyy   = 11
        I_p   = 12
        k_x   = 13
        k_y   = 14
        A     = 15
        pitch = 16
        x_e   = 17
        y_e   = 18

    def __init__(self, verbose=False, silent=False):
        self.verbose = verbose
        self.silent = silent
        # define the column width for printing
        self.col_width = 13
        # formatting and precision
        self.float_hi = 9999.9999
        self.float_lo =  0.01
        self.prec_float = ' 9.05f'
        self.prec_exp =   ' 8.04e'
        self.prec_loss = 0.01

        #0 1  2    3    4    5    6    7   8 9 10   11
        #r m x_cg y_cg ri_x ri_y x_sh y_sh E G I_x  I_y
        #12    13  14  15  16  17  18
        #I_p/K k_x k_y A pitch x_e y_e
        # 19 cols
        self.st_column_header_list = ['r', 'm', 'x_cg', 'y_cg', 'ri_x', \
            'ri_y', 'x_sh', 'y_sh', 'E', 'G', 'I_x', 'I_y', 'J', 'k_x', \
            'k_y', 'A', 'pitch', 'x_e', 'y_e']

        self.st_column_header_list_latex = ['r','m','x_{cg}','y_{cg}','ri_x',\
            'ri_y', 'x_{sh}','y_{sh}','E', 'G', 'I_x', 'I_y', 'J', 'k_x', \
            'k_y', 'A', 'pitch', 'x_e', 'y_e']

        # make the column header
        self.column_header_line = 19 * self.col_width * '=' + '\n'
        for k in self.st_column_header_list:
            self.column_header_line += k.rjust(self.col_width)
        self.column_header_line += '\n' + (19 * self.col_width * '=') + '\n'

    def fromline(self, line, separator=' '):
        # TODO: move this to the global function space (dav-general-module)
        """
        split a line, but ignore any blank spaces and return a list with only
        the values, not empty places
        """
        # remove all tabs, new lines, etc? (\t, \r, \n)
        line = line.replace('\t',' ').replace('\n','').replace('\r','')
        # trailing and ending spaces
        line = line.strip()
        line = line.split(separator)
        values = []
        for k in range(len(line)):
            if len(line[k]) > 0: #and k == item_nr:
                values.append(line[k])
                # break

        return values

    def load_st(self, file_path, file_name):
        """
        Now a better format: st_dict has following key/value pairs
            'nset'    : total number of sets in the file (int).
                        This should be autocalculated every time when writing
                        a new file.
            '007-000-0' : set number line in one peace
            '007-001-a' : comments for set-subset nr 07-01 (str)
            '007-001-b' : subset nr and number of data points, should be
                        autocalculate every time you generate a file
            '007-001-d' : data for set-subset nr 07-01 (ndarray(n,19))

        NOW WE ONLY CONSIDER SUBSET COMMENTS, SET COMMENTS, HOW ARE THEY
        TREADED NOW??

        st_dict is for easy remaking the same file. We need a different format
        for easy reading the comments as well. For that we have the st_comments
        """

        # TODO: store this in an HDF5 format! This is perfect for that.

        # read all the lines of the file into memory
        self.st_path, self.st_file = file_path, file_name
        FILE = open(file_path + file_name)
        lines = FILE.readlines()
        FILE.close()

        subset = False
        st_dict = dict()
        st_comments = dict()
        for i, line in enumerate(lines):

            # convert line to list space seperated list
            line_list = self.fromline(line)

            # see if the first character is marking something
            if i == 0:
                # it is possible that the NSET line is not defined
                parts = line.split(' ')
                try:
                    for k in xrange(10):
                        parts.remove(' ') # throws error when can't find
                except ValueError:
                    pass
                # for a valid 7 NSET line, we now only have 2 elements left
                try:
                    int(parts[0]) # raises ValueError if not an int
                    if parts[1].lower() != 'nset':
                        raise ValueError
                    # first item is the number of sets enclosed in the file
                    #nset = line_list[0]
                    set_nr = 0
                    subset_nr = 0
                    st_dict['000-000-0'] = line
                except ValueError:
                    pass

            # marks the start of a set
            if line[0] == '#':
                #sett = True
                # first character is the #, the rest is the number
                set_nr = int(line_list[0][1:])
                st_dict['%03i-000-0' % set_nr] = line
                # and reset subset nr to zero now
                subset_nr = 0
                subset_nr_track = 0
                # and comments only format, back to one string
                st_comments['%03i-000-0' % set_nr] = ' '.join(line_list[1:])

            # marks the start of a subset
            elif line[0] == '$':
                subset_nr_track += 1
                subset = True
                subset_nr = int(line_list[0][1:])
                # and comments only format, back to one string
                setid = '%03i-%03i-b' % (set_nr, subset_nr)
                st_comments[setid] = ' '.join(line_list[2:])

                # check if the number read corresponds to tracking
                if subset_nr is not subset_nr_track:
                    msg = 'subset_nr and subset_nr_track do not match'
                    raise UserWarning, msg

                nr_points = int(line_list[1])
                st_dict[setid] = line
                # prepare read data points
                sub_set_arr = scipy.zeros((nr_points,19), dtype=np.float64)
                # keep track of where we are on the data array, initialize
                # to 0 for starters
                point = 0

            # in case we are not in subset mode, we only have comments left
            elif not subset:
                # FIXME: how are we dealing with set comments now?
                # subset comments are coming before the actual subset
                # so we account them to one set later than we are now
                #if subset_nr > 0 :
                key = '%03i-%03i-a' % (set_nr, subset_nr+1)
                # in case it is not the first comment line
                if st_dict.has_key(key): st_dict[key] += line
                else: st_dict[key]  = line
                ## otherwise we have the set comments
                #else:
                    #key = '%03i-%03i-a' % (set_nr, subset_nr)
                    ## in case it is not the first comment line
                    #if st_dict.has_key(key): st_dict[key] += line
                    #else: st_dict[key]  = line

            # in case we have the data points, make sure there are enough
            # data poinst present, raise an error if it doesn't
            elif len(line_list)==19 and subset:
                # we can store it in the array
                sub_set_arr[point,:] = line_list
                # on the last entry:
                if point == nr_points-1:
                    # save to the dict:
                    st_dict['%03i-%03i-d' % (set_nr, subset_nr)]= sub_set_arr
                    # and indicate we're done subsetting, next we can have
                    # either set or subset comments
                    subset = False
                point += 1

            #else:
                #msg='error in st format: don't know where to put current line'
                #raise UserWarning, msg

        self.st_dict = st_dict
        self.st_comments = st_comments

    def _format_nr(self, number):
        """
        Automatic format the number

        prec_loss : float, default=0.01
            acceptible precision loss expressed in %

        """

        # the formatting of the number
        numabs = abs(number)
        # just a float precision defined in self.prec_float
        if (numabs < self.float_hi and numabs > self.float_lo):
            numfor = format(number, self.prec_float)
        # if it is zero, just simply print as 0.0
        elif number == 0.0:
            numfor = format(number, ' 1.1f')
        # exponentional, precision defined in self.prec_exp
        else:
            numfor = format(number, self.prec_exp)

        try:
            loss = 100.0*abs(1 - (float(numfor)/number))
        except ZeroDivisionError:
            if abs(float(numfor)) > 0.00000001:
                msg = 'precision loss, from %1.10f to %s' \
                            % (number, numfor.strip())
                raise ValueError, 'precesion loss for new st file'
            else:
                loss = 0
        if loss > self.prec_loss:
            msg = 'precision loss, from %1.10f to %s (%f pc)' \
                        % (number, numfor.strip(), loss)
            raise ValueError, msg

        return numfor


    def write_st(self, file_path, file_name, print_header=False):
        """
        prec_loss : float, default=0.01
            acceptible precision loss expressed in %
        """
        # TODO: implement all the tests when writing on nset, number of data
        # points, subsetnumber sequence etc

        content = ''

        # sort the key list
        keysort = self.st_dict.keys()
        keysort.sort()

        for key in keysort:

            # in case we are just printing what was recorded before
            if not key.endswith('d'):
                content += self.st_dict[key]
            # else we have an array
            else:
                # cycle through data points and print them orderly: control
                # precision depending on the number, keep spacing constant
                # so it is easy to read the textfile
                for m in range(self.st_dict[key].shape[0]):
                    for n in range(self.st_dict[key].shape[1]):
                        # TODO: check what do we lose here?
                        # we are coming from a np.float64, as set in the array
                        # but than it will not work with the format()
                        number = float(self.st_dict[key][m,n])
                        numfor = self._format_nr(number)
                        content += numfor.rjust(self.col_width)
                    content += '\n'

                if print_header:
                    content += self.column_header_line

        # and write file to disk again
        FILE = open(file_path + file_name, 'w')
        FILE.write(content)
        FILE.close()
        if not self.silent:
            print('st file written:', file_path + file_name)


    def write_latex(self, fpath, selection=[]):
        """
        Write a table in Latex format based on the data in the st file.

        selection : list
            [ [setnr, subsetnr, table caption], [setnr, subsetnr, caption],...]
            if not specified, all subsets will be plotted

        """

        cols_p1 = ['r [m]', 'm [kg/m]', 'm(ri{_x})^2 [kgNm^2]',
                   'm(ri{_y})^2 [kgNm^2]', 'EI_x [Nm^2]', 'EI_y [Nm^2]',
                   'EA [N]', 'GJ [\\frac{Nm^2}{rad}]']

        cols_p2 = ['r [m]', 'x_cg [m]', 'y_cg [m]', 'x_sh [m]', 'y_sh [m]',
                'x_e [m]', 'y_e [m]', 'k_x [-]', 'k_y [-]', 'pitch [deg]']

        if len(selection) < 1:
            for key in self.st_dict.keys():
                # but now only take the ones that hold data
                if key[-1] == 'd':
                    selection.append([int(key[:3]), int(key[4:7])])

        for i,j, caption in selection:
            # get the data
            try:
                # set comment should be the name of the body
                set_comment = self.st_comments['%03i-000-0' % (i)]
#                subset_comment = self.st_comments['%03i-%03i-b' % (i,j)]
                st_arr = self.st_dict['%03i-%03i-d' % (i,j)]
            except AttributeError:
                msg = 'ModelData object md is not loaded properly'
                raise AttributeError, msg

            # build the latex table header
#            textable = u"\\begin{table}[b!]\n"
#            textable += u"\\begin{center}\n"
            textable_p1 = u"\\centering\n"
            textable_p1 += u"\\begin{tabular}"
            # configure the column properties
            tmp = [u'C{2.0 cm}' for k in cols_p1]
            tmp = u"|".join(tmp)
            textable_p1 += u'{|' + tmp + u'|}'
            textable_p1 += u'\hline\n'
            # add formula mode for the headers
            tmp = []
            for k in cols_p1:
                k1, k2 = k.split(' ')
                tmp.append(u'$%s$ $%s$' % (k1,k2) )
#            tmp = [u'$%s$' % k for k in cols_p1]
            textable_p1 += u' & '.join(tmp)
            textable_p1 += u'\\\\ \n'
            textable_p1 += u'\hline\n'

            textable_p2 = u"\\centering\n"
            textable_p2 += u"\\begin{tabular}"
            # configure the column properties
            tmp = [u'C{1.5 cm}' for k in cols_p2]
            tmp = u"|".join(tmp)
            textable_p2 += u'{|' + tmp + u'|}'
            textable_p2 += u'\hline\n'
            # add formula mode for the headers
            tmp = []
            for k in cols_p2:
                k1, k2 = k.split(' ')
                tmp.append(u'$%s$ $%s$' % (k1,k2) )
#            tmp = [u'$%s$ $%s$' % (k1, k2) for k in cols_p2]
            # hack: spread the last element over two lines
#            tmp[-1] = '$pitch$ $[deg]$'
            textable_p2 += u' & '.join(tmp)
            textable_p2 += u'\\\\ \n'
            textable_p2 += u'\hline\n'

            for row in xrange(st_arr.shape[0]):
                r    = st_arr[row, self.st_headers.r]
                m    = st_arr[row,self.st_headers.m]
                x_cg = st_arr[row,self.st_headers.x_cg]
                y_cg = st_arr[row,self.st_headers.y_cg]
                ri_x = st_arr[row,self.st_headers.ri_x]
                ri_y = st_arr[row,self.st_headers.ri_y]
                x_sh = st_arr[row,self.st_headers.x_sh]
                y_sh = st_arr[row,self.st_headers.y_sh]
                E    = st_arr[row,self.st_headers.E]
                G    = st_arr[row,self.st_headers.G]
                Ixx  = st_arr[row,self.st_headers.Ixx]
                Iyy  = st_arr[row,self.st_headers.Iyy]
                I_p  = st_arr[row,self.st_headers.I_p]
                k_x  = st_arr[row,self.st_headers.k_x]
                k_y  = st_arr[row,self.st_headers.k_y]
                A    = st_arr[row,self.st_headers.A]
                pitch = st_arr[row,self.st_headers.pitch]
                x_e   = st_arr[row,self.st_headers.x_e]
                y_e   = st_arr[row,self.st_headers.y_e]
                # WARNING: same order as the labels defined in variable "cols"!
                p1 = [r, m, m*ri_x*ri_x, m*ri_y*ri_y, E*Ixx, E*Iyy, E*A,I_p*G]
                p2 = [r, x_cg, y_cg, x_sh, y_sh, x_e, y_e, k_x, k_y, pitch]

                textable_p1 += u" & ".join([self._format_nr(k) for k in p1])
                textable_p1 += u'\\\\ \n'

                textable_p2 += u" & ".join([self._format_nr(k) for k in p2])
                textable_p2 += u'\\\\ \n'

            # default caption
            if caption == '':
                caption = 'HAWC2 cross sectional parameters for body: %s' % set_comment

            textable_p1 += u"\hline\n"
            textable_p1 += u"\end{tabular}\n"
            textable_p1 += u"\caption{%s}\n" % caption
#            textable += u"\end{center}\n"
#            textable += u"\end{table}\n"

            fname = '%s-%s-%03i-%03i_p1' % (self.st_file, set_comment, i, j)
            fname = fname.replace('.', '') + '.tex'
            with open(fpath + fname, 'w') as f:
                f.write(textable_p1)

            textable_p2 += u"\hline\n"
            textable_p2 += u"\end{tabular}\n"
            textable_p2 += u"\caption{%s}\n" % caption
#            textable += u"\end{center}\n"
#            textable += u"\end{table}\n"

            fname = '%s-%s-%03i-%03i_p2' % (self.st_file, set_comment, i, j)
            fname = fname.replace('.', '') + '.tex'
            with open(fpath + fname, 'w') as f:
                f.write(textable_p2)


class Cases:
    """
    Class for the old htc_dict
    ==========================

    Formerly known as htc_dict: a dictionary with on the key a case identifier
    (case name) and the value is a dictionary holding all the different tags
    and value pairs which define the case

    TODO:

    define a public API so that plugin's can be exposed in a standarized way
    using pre defined variables:

    * pandas DataFrame backend instead of a dictionary

    * generic, so not bound to HAWC2. Goal: manage a lot of simulations
      and their corresponding inputs/outus

    * integration with OpenMDAO?

    * case id (hash)

    * case name (which is typically created with variable_tag_name method)

    * results

    * inputs

    * outputs

    a variable tags that has a dictionary mirror for database alike searching

    launch, post_launch, prepare_(re)launch should be methods of this or
    inheret from Cases

    Create a method to add and remove cases from the pool so you can perform
    some analysis on them. Maybe make a GUI that present a list with current
    cases in the pool and than checkboxes to remove them.

    Remove the HAWC2 specific parts to a HAWC2 plugin. The HAWC2 plugin will
    inheret from Cases. Proposed class name: HAWC2Cases, XFOILCases

    Rename cases to pool? A pool contains several cases, mixing several
    sim_id's?

    create a unique case ID based on the hash value of all the tag+values?
    """

    # TODO: add a method that can reload a certain case_dict, you change
    # some parameters for each case (or some) and than launch again

    #def __init__(self, post_dir, sim_id, resdir=False):
    def __init__(self, *args, **kwargs):
        """
        Either load the cases dictionary if post_dir and sim_id is given,
        otherwise the input is a cases dictionary

        Paramters
        ---------

        cases : dict
            The cases dictionary in case there is only one argument

        post_dir : str
            When using two arguments

        sim_id : str or list
            When using two arguments

        resdir : str, default=False

        loadstats : boolean, default=False

        rem_failed : boolean, default=True

        """

        resdir = kwargs.get('resdir', False)
        self.loadstats = kwargs.get('loadstats', False)
        self.rem_failed = kwargs.get('rem_failed', True)

        # determine the input argument scenario
        if len(args) == 1:
            if type(args[0]).__name__ == 'dict':
                self.cases = args[0]
                sim_id = False
            else:
                raise ValueError, 'One argument input should be a cases dict'
        elif len(args) == 2:
            self.post_dir = args[0]
            sim_id = args[1]
        else:
            raise ValueError, 'Only one or two arguments are allowed.'

        # if sim_id is a list, than merge all sim_id's of that list
        if type(sim_id).__name__ == 'list':
            # stats, dynprop and fail are empty dictionaries if they do not
            # exist
            self.merge_sim_ids(sim_id)
            # and define a new sim_id based on all items from the list
            self.sim_id = '_'.join(sim_id)
        # in case we still need to load the cases dict
        elif type(sim_id).__name__ == 'str':
            self.sim_id = sim_id
            self._get_cases_dict(self.post_dir, sim_id)
            # load the statistics if applicable
            if self.loadstats:
                self.stats_df = self.load_stats()

        # change the results directory if applicable
        if resdir:
            self.change_results_dir(resdir)

#        # try to load failed cases and remove them
#        try:
#            self.load_failed(sim_id)
#            self.remove_failed()
#        except IOError:
#            pass

        #return self.cases

    def select(self, search_keyval=False, search_key=False):
        """
        Select only a sub set of the cases

        Select either search_keyval or search_key. Using both is not supported
        yet. Run select twice to achieve the same effect. If both are False,
        cases will be emptied!

        Parameters
        ----------

        search_keyval : dictionary, default=False
            Keys are the column names. If the values match the ones in the
            database, the respective row gets selected. Each tag is hence
            a unique row identifier

        search_key : dict, default=False
            The key is the string that should either be inclusive (value TRUE)
            or exclusive (value FALSE) in the case key
        """

        db = misc.DictDB(self.cases)
        if search_keyval:
            db.search(search_keyval)
        elif search_key:
            db.search_key(search_keyval)
        else:
            db.dict_sel = {}
        # and remove all keys that are not in the list
        remove = set(self.cases) - set(db.dict_sel)
        for k in remove:
            self.cases.pop(k)


    def launch(self, runmethod='local', verbose=False, copyback_turb=True,
           silent=False, check_log=True):
        """
        Launch all cases
        """

        launch(self.cases, runmethod=runmethod, verbose=verbose, silent=silent,
               check_log=check_log, copyback_turb=copyback_turb)

    def post_launch(self):
        """
        Post Launching Maintenance

        check the logs files and make sure result files are present and
        accounted for.
        """
        # TODO: integrate global post_launch in here
        self.cases_fail = post_launch(self.cases)

        if self.rem_failed:
            self.remove_failed()

    def load_case(self, case):
        try:
            iterations = self.load_iterations(case)
        except IOError:
            iterations = None
        res = self.load_result_file(case)
        return res, iterations

    def load_iterations(self, case):

        respath = case['[run_dir]'] + case['[iter_dir]']
        resfile = case['[case_id]']
        return np.loadtxt(respath + resfile + '.iter')

    # TODO: HAWC2 result file reading should be moved to Simulations
    # and we should also switch to faster HAWC2 reading!
    def load_result_file(self, case, _slice=False):
        """
        Set the correct HAWC2 channels

        Parameters
        ----------

        case : dict
            a case dictionary holding all the tags set for this specific
            HAWC2 simulation

        Returns
        -------

        res : object
            A HawcPy LoadResults instance with attributes such as sig, ch_dict,
            and much much more

        """

        respath = case['[run_dir]'] + case['[res_dir]']
        resfile = case['[case_id]']
        self.res = windIO.LoadResults(respath, resfile)
        if not _slice:
            _slice = np.r_[0:len(self.res.sig)]
        self.time = self.res.sig[_slice,0]
        self.sig = self.res.sig[_slice,:]
        self.case = case

        return self.res

    def load_struct_results(self, case, max_modes=500, nrmodes=1000):
        """
        Load the structural analysis result files
        """
        fpath = '%s%s' % (case['[run_dir]'], case['[eigenfreq_dir]'])

        # BEAM OUTPUT
        fname = '%s_beam_output.txt' % case['[case_id]']
        beam = None

        # BODY OUTPUT
        fname = '%s_body_output.txt' % case['[case_id]']
        body = None

        # EIGEN BODY
        fname = '%s_eigen_body.txt' % case['[case_id]']
        try:
            eigen_body, rs2 = windIO.ReadEigenBody(fpath, fname, debug=False,
                                              nrmodes=nrmodes)
        except Exception as e:
            eigen_body = None
            print('failed to load eigen_body')
            print(e)

        # EIGEN STRUCT
        fname = '%s_eigen_struct.txt' % case['[case_id]']
        try:
            eigen_struct = windIO.ReadEigenStructure(fpath, fname, debug=False,
                                                     max_modes=max_modes)
        except Exception as e:
            eigen_struct = None
            print('failed to load eigen_struct')
            print(e)

        # STRUCT INERTIA
        fname = '%s_struct_inertia.txt' % case['[case_id]']
        struct_inertia = None

        return beam, body, eigen_body, eigen_struct, struct_inertia

    def change_results_dir(self, resdir):
        """
        if the post processing concerns simulations done by thyra/gorm, and
        is downloaded locally, change path to results accordingly

        """
        for case in self.cases:
            sim_id = self.cases[case]['[sim_id]']
            newpath = resdir + sim_id + '/'
            self.cases[case]['[run_dir]'] = newpath

        #return cases

    def force_lower_case_id(self):
        tmp_cases = {}
        for cname, case in self.cases.iteritems():
            tmp_cases[cname.lower()] = case.copy()
        self.cases = tmp_cases

    def _get_cases_dict(self, post_dir, sim_id):
        """
        Load the pickled dictionary containing all the cases and their
        respective tags.

        Returns
        -------

        cases : Cases object
            cases with failures removed. Failed cases are kept in
            self.cases_fail

        """
        self.cases = load_pickled_file(post_dir + sim_id + '.pkl')
        self.cases_fail = {}

        self.force_lower_case_id()

        if self.rem_failed:
            try:
                self.load_failed(sim_id)
                # ditch all the failed cases out of the htc_dict otherwise
                #  we will have fails when reading the results data files
                self.remove_failed()
            except IOError:
                print("couldn't find pickled failed dictionary")

        return

    def merge_sim_ids(self, sim_id_list, silent=False):
        """
        Load and merge for a list of sim_id's cases, fail, dynprop and stats
        ====================================================================

        For all sim_id's in the sim_id_list the cases, stats, fail and dynprop
        dictionaries are loaded. If one of them doesn't exists, an empty
        dictionary is returned.

        Currently, there is no warning given when a certain case will be
        overwritten upon merging.

        """

        cases_merged = {}
        cases_fail_merged = {}

        for ii, sim_id in enumerate(sim_id_list):

            # TODO: give a warning if we have double entries or not?
            self.sim_id = sim_id
            self._get_cases_dict(self.post_dir, sim_id)
            cases_fail_merged.update(self.cases_fail)

            # and copy to htc_dict_merged. Note that non unique keys will be
            # overwritten: each case has to have a unique name!
            cases_merged.update(self.cases)

            # merge the statistics if applicable
            # self.stats_dict[channels] = df
            if self.loadstats:
                if ii == 0:
                    self.stats_df = self.load_stats()
                else:
                    self.stats_df = self.stats_df.append(self.load_stats())

        self.cases = cases_merged
        self.cases_fail = cases_fail_merged

    def printall(self, scenario, figpath=''):
        """
        For all the cases, get the average value of a certain channel
        """
        self.figpath = figpath

        # plot for each case the dashboard
        for k in self.cases:

            if scenario == 'blade_deflection':
                self.blade_deflection(self.cases[k], self.figpath)

    def diff(self, refcase_dict, cases):
        """
        See wich tags change over the given cases of the simulation object
        """

        # there is only one case allowed in refcase dict
        if not len(refcase_dict) == 1:
            return ValueError, 'Only one case allowed in refcase dict'

        # take an arbritrary case as baseline for comparison
        refcase = refcase_dict[refcase_dict.keys()[0]]
        #reftags = sim_dict[refcase]

        diffdict = dict()
        adddict = dict()
        remdict = dict()
        print()
        print('*'*80)
        print('comparing %i cases' % len(cases))
        print('*'*80)
        print()
        # compare each case with the refcase and see if there are any diffs
        for case in sorted(cases.keys()):
            dd = misc.DictDiff(refcase, cases[case])
            diffdict[case] = dd.changed()
            adddict[case] = dd.added()
            remdict[case] = dd.removed()
            print('')
            print('='*80)
            print(case)
            print('='*80)
            for tag in sorted(diffdict[case]):
                print(tag.rjust(20),':', cases[case][tag])

        return diffdict, adddict, remdict

    def blade_deflection(self, case, **kwargs):
        """
        """

        # read the HAWC2 result file
        self.load_result_file(case)

        # select all the y deflection channels
        db = misc.DictDB(self.res.ch_dict)

        db.search({'sensortype' : 'state pos', 'component' : 'z'})
        # sort the keys and save the mean values to an array/list
        chiz, zvals = [], []
        for key in sorted(db.dict_sel.keys()):
            zvals.append(-self.sig[:,db.dict_sel[key]['chi']].mean())
            chiz.append(db.dict_sel[key]['chi'])

        db.search({'sensortype' : 'state pos', 'component' : 'y'})
        # sort the keys and save the mean values to an array/list
        chiy, yvals = [], []
        for key in sorted(db.dict_sel.keys()):
            yvals.append(self.sig[:,db.dict_sel[key]['chi']].mean())
            chiy.append(db.dict_sel[key]['chi'])

        return np.array(zvals), np.array(yvals)

    def remove_failed(self):

        # don't do anything if there is nothing defined
        if self.cases_fail == None:
            return

        # ditch all the failed cases out of the htc_dict
        # otherwise we will have fails when reading the results data files
        for k in self.cases_fail:
            try:
                self.cases_fail[k] = copy.copy(self.cases[k])
                del self.cases[k]
                print('removed from htc_dict due to error: ' + k)
            except KeyError:
                print('WARNING: failed case does not occur in cases')
                print('   ', k)

    def load_failed(self, sim_id):

        FILE = open(self.post_dir + sim_id + '_fail.pkl', 'rb')
        self.cases_fail = pickle.load(FILE)
        FILE.close()

    def load_stats(self, **kwargs):
        """
        Load an existing statistcs file
        """
        post_dir = kwargs.get('post_dir', self.post_dir)
        sim_id = kwargs.get('sim_id', self.sim_id)
        fpath = post_dir + sim_id + '_statistics.h5'

        try:
            stats_df = pd.read_hdf(fpath, 'table')
#            FILE = open(post_dir + sim_id + '_statistics.pkl', 'rb')
#            stats_dict = pickle.load(FILE)
#            FILE.close()
        except IOError:
            stats_df = None
            print('NO STATS FOUND FOR', sim_id)

        return stats_df

    def statistics(self, new_sim_id=False, silent=False, ch_sel=None,
                   tags=['[turb_seed]','[windspeed]'], calc_mech_power=False,
                   save=True, m=[3, 4, 6, 8, 10, 12], neq=None, no_bins=46,
                   ch_fatigue={}, update=False, add_sensor=None,
                   add_sens_divide=None, i0=0, i1=-1):
        """
        Calculate statistics and save them in a pandas dataframe

        Parameters
        ----------

        ch_sel : list, default=None
            If defined, only add defined channels to the output data frame.
            The list should contain valid channel names as defined in ch_dict.

        tags : list, default=['[turb_seed]','[windspeed]']
            Select which tag values from cases should be included in the
            dataframes. This will help in selecting and identifying the
            different cases.

        ch_fatigue : list, default=[]
            Valid ch_dict channel names for which the equivalent fatigue load
            needs to be calculated. When set to None, ch_fatigue = ch_sel.

        Returns
        -------

        dfs : dict
            Dictionary of dataframes, where the key is the channel name of
            the output (that was optionally defined in ch_sel), and the value
            is the dataframe containing the statistical values for all the
            different selected cases.

        """

        # in case the output changes, remember the original ch_sel
        if ch_sel is not None:
            ch_sel_init = ch_sel.copy()
        else:
            ch_sel_init = None

        if ch_fatigue is None:
            ch_fatigue_init = None
        else:
            ch_fatigue_init = ch_fatigue

        # TODO: should the default tags not be all the tags in the cases dict?
        tag_default = ['[case_id]', '[sim_id]']
        tag_chan = 'channel'
        # merge default with other tags
        for tag in tag_default:
            if tag not in tags:
                tags.append(tag)

        # tags can only be unique, when there the same tag appears twice
        # it will break the DataFrame creation
        if len(tags) is not len(set(tags)):
            raise ValueError, 'tags can only contain unique entries'

        # the dictionary that will be used to create a pandas dataframe
        df_dict = { tag:[] for tag in tags }
        df_dict[tag_chan] = []
        # add more columns that will help with IDing the channel
        df_dict['channel_name'] = []
        df_dict['channel_units'] = []
        df_dict['channel_nr'] = []
        df_dict['channel_desc'] = []

        # get some basic parameters required to calculate statistics
        try:
            case = self.cases.keys()[0]
        except IndexError:
            print('no cases to select so no statistics, aborting ...')
            return None

        post_dir = self.cases[case]['[post_dir]']
        if not new_sim_id:
            # select the sim_id from a random case
            sim_id = self.cases[case]['[sim_id]']
        else:
            sim_id = new_sim_id

        if not silent:
            nrcases = len(self.cases)
            print('='*79)
            print('statistics for %s, nr cases: %i' % (sim_id, nrcases))

        for ii, (cname, case) in enumerate(self.cases.iteritems()):

            if not silent:
                print('stats progress: %4i/%i' % (ii, nrcases))

            # make sure the selected tags exist
            if len(tags) != len(set(case) and tags):
                raise KeyError, 'not all selected tags exist in cases'

            self.load_result_file(case)
            ch_dict = self.res.ch_dict.copy()
            # calculate the statistics values
            stats = self.res.calc_stats(self.sig, i0=i0, i1=i1)

            # Because each channel is a new row, it doesn't matter how many
            # data channels each case has, and this approach does not brake
            # when different cases have a different number of output channels
            # By default, just take all channels in the result file.
            if ch_sel_init is None:
                ch_sel = ch_dict.keys()
                print('selecting all channels for statistics')

            if add_sensor is not None:
                chi1 = self.res.ch_dict[add_sensor['ch1_name']]['chi']
                chi2 = self.res.ch_dict[add_sensor['ch2_name']]['chi']
                name = add_sensor['ch_name_add']
                factor = add_sensor['factor']
                operator = add_sensor['operator']

                irange = int(300.0*self.res.N/700.0)
                sig_add = np.ndarray((irange, 1))
                p1 = self.res.sig[-irange:,chi1]
                p2 = self.res.sig[-irange:,chi2]
                if operator == '*':
                    sig_add[:,0] = p1*p2*factor
                elif operator == '/':
                    sig_add[:,0] = factor*p1/p2
                else:
                    raise ValueError, 'Operator needs to be either * or /'
                add_stats = self.res.calc_stats(sig_add)
                add_stats_i = stats['max'].shape[0]
                # add a new channel description for the mechanical power
                ch_dict[name] = {}
                ch_dict[name]['chi'] = add_stats_i
                # and append to all the statistics types
                for key, stats_arr in stats.iteritems():
                    stats[key] = np.append(stats_arr, add_stats[key])

            # calculate mechanical power first before deriving statistics
            # from it
            if calc_mech_power:
                name = 'stats-shaft-power'
                P_mech = self.shaft_power()
                sig_tmp = np.ndarray((self.res.N, 1))
                sig_tmp[:,0] = P_mech
                P_mech_stats = self.res.calc_stats(sig_tmp)
                mech_stats_i = stats['max'].shape[0]
                # add a new channel description for the mechanical power
                ch_dict[name] = {}
                ch_dict[name]['chi'] = mech_stats_i
                # and append to all the statistics types
                for key, stats_arr in stats.iteritems():
                    stats[key] = np.append(stats_arr, P_mech_stats[key])

            # calculate the fatigue properties from selected channels
            fatigue, tags_fatigue = {}, []
            if ch_fatigue_init is None:
                ch_fatigue = ch_sel
                print('selecting all channels for fatigue')
            else:
                ch_fatigue = ch_fatigue_init

            for ch_id in ch_fatigue:
                chi = ch_dict[ch_id]['chi']
                signal = self.res.sig[:,chi]
                if neq is None:
                    neq = float(case['[duration]'])
                eq = self.res.calc_fatigue(signal, no_bins=no_bins, neq=neq, m=m)
                fatigue[ch_id] = {}
                # when calc_fatigue succeeds, we should have as many items
                # as in m
                if len(eq) == len(m):
                    for eq_, m_ in zip(eq, m):
                        fatigue[ch_id]['m=%2.01f' % m_] = eq_
                # when it fails, we get an empty list back
                else:
                    for m_ in m:
                        fatigue[ch_id]['m=%2.01f' % m_] = np.nan

            # build the fatigue tags
            for m_ in m:
                tag = 'm=%2.01f' % m_
                tags_fatigue.append(tag)

            # -----------------------------------------------------------------
            # define the pandas data frame dict on first run
            # -----------------------------------------------------------------
            # Only build the ch_sel collection once. By definition, the
            # statistics, fatigue and htc tags will not change
            if ii == 0:
                # statistical parameters
                for statparam in stats.keys():
                    df_dict[statparam] = []
#                # additional tags
#                for tag in tags:
#                    df_dict[tag] = []
                # fatigue data
                for tag in tags_fatigue:
                    df_dict[tag] = []

            for ch_id in ch_sel:

                chi = ch_dict[ch_id]['chi']

                # sig_stat = [(0=value,1=index),statistic parameter, channel]
                # stat params = 0 max, 1 min, 2 mean, 3 std, 4 range, 5 abs max
                # note that min, mean, std, and range are not relevant for index
                # values. Set to zero there.

                # -------------------------------------------------------------
                # Fill in all the values for the current data entry
                # -------------------------------------------------------------

                # the auxiliry columns
                try:
                    name = self.res.ch_details[chi,0]
                    unit = self.res.ch_details[chi,1]
                    desc = self.res.ch_details[chi,2]
                except (IndexError, AttributeError) as e:
                    name = ''
                    desc = ''
                    unit = ''
                df_dict['channel_name'].append(name)
                df_dict['channel_units'].append(unit)
                df_dict['channel_desc'].append(desc)
                df_dict['channel_nr'].append(chi)

                # each df line is a channel of case that needs to be id-eed
                df_dict[tag_chan].append(ch_id)

                # for all the statistics keys, save the values for the
                # current channel
                for statparam in stats.keys():
                    df_dict[statparam].append(stats[statparam][chi])
                # and save the tags from the input htc file in order to
                # label each different case properly
                for tag in tags:
                    df_dict[tag].append(case[tag])
                # append any fatigue channels if applicable, otherwise nan
                if ch_id in fatigue:
                    for m_fatigue, eq_ in fatigue[ch_id].iteritems():
                        df_dict[m_fatigue].append(eq_)
                else:
                    for tag in tags_fatigue:
                        # TODO: or should this be NaN?
                        df_dict[tag].append(np.nan)

        # there might be a mix of strings and numbers now, see if we can have
        # the same data type throughout a column
        # nasty hack: because of the unicode -> string conversion we might not
        # overwrite the same key in the dict.
        # FIXME: this approach will result in twice the memory useage though...
        # we can not pop/delete items from a dict while iterating over it
        df_dict2 = {}
        for colkey, col in df_dict.iteritems():
            # if we have a list, convert to string
            if type(col[0]).__name__ == 'list':
                for ii, item in enumerate(col):
                    col[ii] = '**'.join(item)
            # if we already have an array (statistics) or a list of numbers
            # do not try to cast into another data type, because downcasting
            # in that case will not raise any exception
            elif type(col[0]).__name__[:3] in ['flo', 'int', 'nda']:
                df_dict2[str(colkey)] = np.array(col)
                continue
            # in case we have unicodes instead of strings, we need to convert
            # to strings otherwise the saved .h5 file will have pickled elements
            try:
                df_dict2[str(colkey)] = np.array(col, dtype=np.int32)
            except OverflowError:
                try:
                    df_dict2[str(colkey)] = np.array(col, dtype=np.int64)
                except OverflowError:
                    df_dict2[str(colkey)] = np.array(col, dtype=np.float64)
            except ValueError:
                try:
                    df_dict2[str(colkey)] = np.array(col, dtype=np.float64)
                except ValueError:
                    df_dict2[str(colkey)] = np.array(col, dtype=np.str)
            except TypeError:
                # in all other cases, make sure we have converted them to
                # strings and NOT unicode
                df_dict2[str(colkey)] = np.array(col, dtype=np.str)
            except Exception as e:
                print('failed to convert column %s to single data type' % colkey)
                raise e

        # in case converting to dataframe fails, fall back
        try:
            dfs = pd.DataFrame(df_dict2)
        except Exception as e:
            fpath = post_dir + sim_id + '_statistics'
            FILE = open(fpath + '.pkl', 'wb')
            pickle.dump(df_dict2, FILE, protocol=2)
            FILE.close()
            print('failed to convert to data frame, saved as dict')
            raise e

        # and save/update the statistics database
        if save:
            fpath = post_dir + sim_id + '_statistics'
            print('saving statistics: %s ...' % (post_dir + sim_id), end='')
#            FILE = open(fpath + '.pkl', 'wb')
#            pickle.dump(dfs, FILE, protocol=2)
#            FILE.close()
            if update:
                dfs.to_hdf('%s.h5' % fpath, 'table', mode='r+')
            else:
                dfs.to_hdf('%s.h5' % fpath, 'table', mode='w')
                dfs.to_csv('%s.csv' % fpath)

            print('DONE!!\n')

        return dfs

    def stats2dataframe(self, ch_sel=None, tags=['[turb_seed]','[windspeed]']):
        """
        Convert the archaic statistics dictionary of a group of cases to
        a more convienent pandas dataframe format.

        DEPRICATED, use statistics instead!!

        Parameters
        ----------

        ch_sel : dict, default=None
            Map short names to the channel id's defined in ch_dict in order to
            have more human readable column names in the pandas dataframe. By
            default, if ch_sel is None, a dataframe for each channel in the
            ch_dict (so in the HAWC2 output) will be created. When ch_sel is
            defined, only those channels are considered.
            ch_sel[short name] = full ch_dict identifier

        tags : list, default=['[turb_seed]','[windspeed]']
            Select which tag values from cases should be included in the
            dataframes. This will help in selecting and identifying the
            different cases.

        Returns
        -------

        dfs : dict
            Dictionary of dataframes, where the key is the channel name of
            the output (that was optionally defined in ch_sel), and the value
            is the dataframe containing the statistical values for all the
            different selected cases.
        """

        df_dict = {}

        for cname, case in self.cases.iteritems():

            # make sure the selected tags exist
            if len(tags) != len(set(case) and tags):
                raise KeyError, 'not all selected tags exist in cases'

            sig_stats = self.stats_dict[cname]['sig_stats']
            ch_dict = self.stats_dict[cname]['ch_dict']

            if ch_sel is None:
                ch_sel = { (i, i) for i in ch_dict.keys() }

            for ch_short, ch_name in ch_sel.iteritems():

                chi = ch_dict[ch_name]['chi']
                # sig_stat = [(0=value,1=index),statistic parameter, channel]
                # stat params = 0 max, 1 min, 2 mean, 3 std, 4 range, 5 abs max
                # note that min, mean, std, and range are not relevant for index
                # values. Set to zero there.
                try:
                    df_dict[ch_short]['case name'].append(cname)
                    df_dict[ch_short]['max'].append(   sig_stats[0,0,chi])
                    df_dict[ch_short]['min'].append(   sig_stats[0,1,chi])
                    df_dict[ch_short]['mean'].append(  sig_stats[0,2,chi])
                    df_dict[ch_short]['std'].append(   sig_stats[0,3,chi])
                    df_dict[ch_short]['range'].append( sig_stats[0,4,chi])
                    df_dict[ch_short]['absmax'].append(sig_stats[0,5,chi])
                    for tag in tags:
                        df_dict[ch_short][tag].append(case[tag])
                except KeyError:
                    df_dict[ch_short] = {'case name' : [cname]}
                    df_dict[ch_short]['max']    = [sig_stats[0,0,chi]]
                    df_dict[ch_short]['min']    = [sig_stats[0,1,chi]]
                    df_dict[ch_short]['mean']   = [sig_stats[0,2,chi]]
                    df_dict[ch_short]['std']    = [sig_stats[0,3,chi]]
                    df_dict[ch_short]['range']  = [sig_stats[0,4,chi]]
                    df_dict[ch_short]['absmax'] = [sig_stats[0,5,chi]]
                    for tag in tags:
                        df_dict[ch_short][tag] = [ case[tag] ]

        # and create for each channel a dataframe
        dfs = {}
        for ch_short, df_values in df_dict.iteritems():
            dfs[ch_short] = pd.DataFrame(df_values)

        return dfs

    def load_azimuth(self, azi, load, sectors=360):
        """
        Establish load dependency on rotor azimuth angle
        """

        # sort on azimuth angle
        isort = np.argsort(azi)
        azi = azi[isort]
        load = load[isort]

        azi_sel = np.linspace(0, 360, num=sectors)
        load_sel = np.interp(azi_sel, azi, load)


    def shaft_power(self):
        """
        Return the mechanical shaft power based on the shaft torsional loading
        """
        try:
            irpm = self.res.ch_dict['bearing-shaft_rot-angle_speed-rpm']['chi']
            rads = self.res.sig[:,irpm]*np.pi/30.0
        except KeyError:
            irads = self.res.ch_dict['Omega']['chi']
            rads = self.res.sig[:,irads]

        itorque = self.res.ch_dict['shaft-shaft-node-004-momentvec-z']['chi']
        torque = self.res.sig[:,itorque]

        return torque*rads

    def calc_torque_const(self, save=False, name='ojf'):
        """
        If we have constant RPM over the simulation, calculate the torque
        constant. The current loaded HAWC2 case is considered. Consequently,
        first load a result file with load_result_file

        Parameters
        ----------

        save : boolean, default=False

        name : str, default='ojf'
            File name of the torque constant result. Default to using the
            ojf case name. If set to hawc2, it will the case_id. In both
            cases the file name will be extended with '.kgen'

        Returns
        -------

        [windspeed, rpm, K] : list

        """
        # make sure the results have been loaded previously
        try:
            # get the relevant index to the wanted channels
            # tag: coord-bodyname-pos-sensortype-component
            tag = 'bearing-shaft_nacelle-angle_speed-rpm'
            irpm = self.res.ch_dict[tag]['chi']
            chi_rads = self.res.ch_dict['Omega']['chi']
            tag = 'shaft-shaft-node-001-momentvec-z'
            chi_q = self.res.ch_dict[tag]['chi']
        except AttributeError:
            msg = 'load results first with Cases.load_result_file()'
            raise ValueError, msg

#        if not self.case['[fix_rpm]']:
#            print
#            return

        windspeed = self.case['[windspeed]']
        rpm = self.res.sig[:,irpm].mean()
        # and get the average rotor torque applied to maintain
        # constant rotor speed
        K = -np.mean(self.res.sig[:,chi_q]*1000./self.res.sig[:,chi_rads])

        result = np.array([windspeed, rpm, K])

        # optionally, save the values and give the case name as file name
        if save:
            fpath = self.case['[post_dir]'] + 'torque_constant/'
            if name == 'hawc2':
                fname = self.case['[case_id]'] + '.kgen'
            elif name == 'ojf':
                fname = self.case['[ojf_case]'] + '.kgen'
            else:
                raise ValueError, 'name should be either ojf or hawc2'
            # create the torque_constant dir if it doesn't exists
            try:
                os.mkdir(fpath)
            except OSError:
                pass

#            print('gen K saving at:', fpath+fname
            np.savetxt(fpath+fname, result, header='windspeed, rpm, K')

        return result

# TODO: implement this
class Results():
    """
    Move all Hawc2io to here? NO: this should be the wrapper, to interface
    the htc_dict with the io functions

    There should be a bare metal module/class for those who only want basic
    python support for HAWC2 result files and/or launching simulations.

    How to properly design this module? Change each class into a module? Or
    leave like this?
    """

    # OK, for now use this to do operations on HAWC2 results files

    def __init___(self):
        """
        """
        pass

    def m_equiv(self, st_arr, load, pos):
        r"""Centrifugal corrected equivalent moment

        Convert beam loading into a single equivalent bending moment. Note that
        this is dependent on the location in the cross section. Due to the
        way we measure the strain on the blade and how we did the calibration
        of those sensors.

        .. math::

            \epsilon = \frac{M_{x_{equiv}}y}{EI_{xx}} = \frac{M_x y}{EI_{xx}}
            + \frac{M_y x}{EI_{yy}} + \frac{F_z}{EA}

            M_{x_{equiv}} = M_x + \frac{I_{xx}}{I_{yy}} M_y \frac{x}{y}
            + \frac{I_{xx}}{Ay} F_z

        Parameters
        ----------

        st_arr : np.ndarray(19)
            Only one line of the st_arr is allowed and it should correspond
            to the correct radial position of the strain gauge.

        load : list(6)
            list containing the load time series of following components
            .. math:: load = F_x, F_y, F_z, M_x, M_y, M_z
            and where each component is an ndarray(m)

        pos : np.ndarray(2)
            x,y position wrt neutral axis in the cross section for which the
            equivalent load should be calculated

        Returns
        -------

        m_eq : ndarray(m)
            Equivalent load, see main title

        """

        F_z = load[2]
        M_x = load[3]
        M_y = load[4]

        x, y = pos[0], pos[1]

        A = st_arr[ModelData.st_headers.A]
        I_xx = st_arr[ModelData.st_headers.Ixx]
        I_yy = st_arr[ModelData.st_headers.Iyy]

        M_x_equiv = M_x + ( (I_xx/I_yy)*M_y*(x/y) ) + ( F_z*I_xx/(A*y) )
        # or ignore edgewise moment
        #M_x_equiv = M_x + ( F_z*I_xx/(A*y) )

        return M_x_equiv


def eigenbody(cases, debug=False):
    """
    Read HAWC2 body eigenalysis result file
    =======================================

    This is basically a cases convience wrapper around Hawc2io.ReadEigenBody

    Parameters
    ----------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case.

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case. For each
        case, it is updated with the results, results2 of the eigenvalue
        analysis performed for each body using the following respective
        tags: [eigen_body_results] and [eigen_body_results2].

    """

    #Body data for body number : 3 with the name :nacelle
    #Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #Mode nr:  1:   1.45388E-21    1.74896E-03    6.28319E+02

    for case in cases:
        # tags for the current case
        tags = cases[case]
        file_path = tags['[run_dir]'] + tags['[eigenfreq_dir]']
        # FIXME: do not assuem anything about the file name here, should be
        # fully defined in the tags/dataframe
        file_name = tags['[case_id]'] + '_body_eigen'
        # and load the eigenfrequency body results
        results, results2 = windIO.ReadEigenBody(file_path, file_name,
                                                  nrmodes=10)
        # add them to the htc_dict
        cases[case]['[eigen_body_results]'] = results
        cases[case]['[eigen_body_results2]'] = results2

    return cases

def eigenstructure(cases, debug=False):
    """
    Read HAWC2 structure eigenalysis result file
    ============================================

    This is basically a cases convience wrapper around
    Hawc2io.ReadEigenStructure

    Parameters
    ----------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case.

    Returns
    -------

    cases : dict{ case : dict{tag : value} }
        Dictionary where each case is a key and its value a dictionary
        holding all the tags/value pairs as used for that case. For each
        case, it is updated with the modes_arr of the eigenvalue
        analysis performed for the structure.
        The modes array (ndarray(3,n)) holds fd, fn and damping.
    """

    for case in cases:
        # tags for the current case
        tags = cases[case]
        file_path = tags['[run_dir]'] + tags['[eigenfreq_dir]']
        # FIXME: do not assuem anything about the file name here, should be
        # fully defined in the tags/dataframe
        file_name = tags['[case_id]'] + '_strc_eigen'
        # and load the eigenfrequency structure results
        modes = windIO.ReadEigenStructure(file_path, file_name, max_modes=500)
        # add them to the htc_dict
        cases[case]['[eigen_structure]'] = modes

    return cases

if __name__ == '__main__':
    pass

