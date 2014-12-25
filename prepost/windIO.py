# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:53:59 2014

@author: dave
"""

from __future__ import division # always devide as floats
from __future__ import print_function
#print(*objects, sep=' ', end='\n', file=sys.stdout)

__author__ = 'David Verelst'
__license__ = 'GPL'
__version__ = '0.5'

#import sys
from time import time
import scipy
import scipy.io as sio
import array
import numpy as np
import os
import logging
#import sys
import copy
import unittest
import struct
import math

import misc
import fatigue

class LoadResults:
    """Read a HAWC2 result data file

    Usage:
    obj = LoadResults(file_path, file_name)

    This class is called like a function:
    HawcResultData() will read the specified file upon object initialization.

    Available output:
    obj.sig[timeStep,channel]   : complete result file in a numpy array
    obj.ch_details[channel,(0=ID; 1=units; 2=description)] : np.array
    obj.error_msg: is 'none' if everything went OK, otherwise it holds the
    error

    The ch_dict key/values pairs are structured differently for different
        type of channels. Currently supported channels are:

        For forcevec, momentvec, state commands:
            key:
                coord-bodyname-pos-sensortype-component
                global-tower-node-002-forcevec-z
                local-blade1-node-005-momentvec-z
                hub1-blade1-elem-011-zrel-1.00-state pos-z
            value:
                ch_dict[tag]['coord']
                ch_dict[tag]['bodyname']
                ch_dict[tag]['pos'] = pos
                ch_dict[tag]['sensortype']
                ch_dict[tag]['component']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the DLL's this is:
            key:
                DLL-dll_name-io-io_nr
                DLL-yaw_control-outvec-3
                DLL-yaw_control-inpvec-1
            value:
                ch_dict[tag]['dll_name']
                ch_dict[tag]['io']
                ch_dict[tag]['io_nr']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the bearings this is:
            key:
                bearing-bearing_name-output_type-units
                bearing-shaft_nacelle-angle_speed-rpm
            value:
                ch_dict[tag]['bearing_name']
                ch_dict[tag]['output_type']
                ch_dict[tag]['chi']
                ch_dict[tag]['units']

    """

    # start with reading the .sel file, containing the info regarding
    # how to read the binary file and the channel information
    def __init__(self, file_path, file_name, debug=False, usecols=None,
                 readdata=True):

        self.debug = debug

        # timer in debug mode
        if self.debug:
            start = time()

        self.file_path = file_path
        # remove .log, .dat, .sel extensions who might be accedental left
        if file_name[-4:] in ['.htc','.sel','.dat','.log']:
            file_name = file_name[:-4]
        # FIXME: since HAWC2 will always have lower case output files, convert
        # any wrongly used upper case letters to lower case here
        self.file_name = file_name.lower()
        self.read_sel()
        # create for any supported channel the
        # continue if the file has been succesfully read
        if self.error_msg == 'none':
            # load the channel id's and scale factors
            self.scale_factors = self.data_sel()
            # with the sel file loaded, we have all the channel names to
            # squeeze into a more consistant naming scheme
            self._unified_channel_names()
            # only read when asked for
            if readdata:
                # if there is sel file but it is empty or whatever else
                # FilType will not exists
                try:
                    # read the binary file
                    if self.FileType == 'BINARY':
                        self.read_bin(self.scale_factors, usecols=usecols)
                    # read the ASCII file
                    elif self.FileType == 'ASCII':
                        self.read_ascii(usecols=usecols)
                    else:
                        print('='*79)
                        print('unknown file type: ' + self.FileType)
                        print('='*79)
                        self.error_msg = 'error: unknown file type'
                        self.sig = []
                except:
                    print('='*79)
                    print('couldn\'t determine FileType')
                    print('='*79)
                    self.error_msg = 'error: no file type'
                    self.sig = []

        if self.debug:
            stop = time() - start
            print('time to load HAWC2 file:', stop, 's')

    def read_sel(self):
        # anticipate error on file reading
        try:
            # open file, read and close
            go_sel = self.file_path + self.file_name + '.sel'
            FILE = open(go_sel, "r")
            self.lines = FILE.readlines()
            FILE.close()
            self.error_msg = 'none'

        # error message if the file does not exists
        except:
            # print(26*' ' + 'ERROR'
            print(50*'=')
            print(self.file_path)
            print(self.file_name + '.sel could not be found')
            print(50*'=')
            self.error_msg = 'error: file not found'

    def data_sel(self):

        # scan through all the lines in the file
        line_nr = 1
        # channel counter for ch_details
        ch = 0
        for line in self.lines:
            # on line 9 we can read following paramaters:
            if line_nr == 9:
                # remove the end of line character
                line = line.replace('\n','')

                settings = line.split(' ')
                # delete all empty string values
                for k in range(settings.count('')):
                    settings.remove('')

                # and assign proper values with correct data type
                self.N = int(settings[0])
                self.Nch = int(settings[1])
                self.Time = float(settings[2])
                # there are HEX values at the end of this line...
                # On Linux they will show up in the last variable, so don't inc
                if os.name == 'posix':
                    nrchars = len(settings[3])-1
                elif os.name == 'nt':
                    nrchars = len(settings[3])
                else:
                    raise UserWarning, \
                    'Untested platform:', os.name
                settings[3] = settings[3][0:nrchars]
                self.FileType = settings[3]
                self.Freq = self.N/self.Time

                # prepare list variables
                self.ch_details = np.ndarray(shape=(self.Nch,3),dtype='<U100')
                # it seems that float64 reeds the data correctly from the file
                scale_factors = scipy.zeros(self.Nch, dtype='Float64')
                #self.scale_factors_dec = scipy.zeros(self.Nch, dtype='f8')
                i = 0

            # starting from line 13, we have the channels info
            if line_nr > 12:
                # read the signal details
                if line_nr < 13 + self.Nch:
                    # remove leading and trailing whitespaces from line parts
                    self.ch_details[ch,0] = str(line[12:43]).strip() # chID
                    self.ch_details[ch,1] = str(line[43:54]).strip() # chUnits
                    self.ch_details[ch,2] = str(line[54:-1]).strip() # chDescr
                    ch += 1
                # read the signal scale parameters for binary format
                elif line_nr > 14 + self.Nch:
                    scale_factors[i] = line
                    # print(scale_factors[i]
                    #self.scale_factors_dec[i] = D.Decimal(line)
                    i = i + 1
                # stop going through the lines if at the end of the file
                if line_nr == 2*self.Nch + 14:
                    self.scale_factors = scale_factors

                    if self.debug:
                        print('N       ', self.N)
                        print('Nch     ', self.Nch)
                        print('Time    ', self.Time)
                        print('FileType', self.FileType)
                        print('Freq    ', self.Freq)
                        print('scale_factors', scale_factors.shape)

                    return scale_factors
                    break

            # counting the line numbers
            line_nr = line_nr + 1

    def read(self, usecols=False):
        """
        This whole LoadResults needs to be refactered because it is crap.
        Keep the old ones for backwards compatibility
        """

        if self.FileType == 'ASCII':
            self.read_ascii(usecols=usecols)
        elif self.FileType == 'BINARY':
            self.read_bin(self.scale_factors, usecols=usecols)

    def read_bin(self, scale_factors, usecols=False):
        if not usecols:
            usecols = range(0, self.Nch)
        fid = open(self.file_path + self.file_name + '.dat', 'rb')
        self.sig = np.zeros( (self.N, len(usecols)) )
        for j, i in enumerate(usecols):
            fid.seek(i*self.N*2,0)
            self.sig[:,j] = np.fromfile(fid, 'int16', self.N)*scale_factors[i]

    def read_bin_old(self, scale_factors):
        # if there is an error reading the binary file (for instance if empty)
        try:
            # read the binary file
            go_binary = self.file_path + self.file_name + '.dat'
            FILE = open(go_binary, mode='rb')

            # create array, put all the binary elements as one long chain in it
            binvalues = array.array('h')
            binvalues.fromfile(FILE, self.N * self.Nch)
            FILE.close()
            # convert now to a structured numpy array
            # sig = np.array(binvalues, np.float)
#            sig = np.array(binvalues)
            # this is faster! the saved bin values are only of type int16
            sig = np.array(binvalues, dtype='int16')

            if self.debug: print(self.N, self.Nch, sig.shape)

#            sig = np.reshape(sig, (self.Nch, self.N))
#            # apperently Nch and N had to be reversed to read it correctly
#            # is this because we are reading a Fortran array with Python C
#            # code? so now transpose again so we have sig(time, channel)
#            sig = np.transpose(sig)

            # reshape the array to 2D and transpose (Fortran to C array)
            sig = sig.reshape((self.Nch, self.N)).T

            # create diagonal vector of size (Nch,Nch)
            dig = np.diag(scale_factors)
            # now all rows of column 1 are multiplied with dig(1,1)
            sig = np.dot(sig,dig)
            self.sig = sig
            # 'file name;' + 'lnr;msg;'*(len(MsgList)) + '\n'
        except:
            self.sig = []
            self.error_msg = 'error: reading binary file failed'
            print('========================================================')
            print(self.error_msg)
            print(self.file_path)
            print(self.file_name)
            print('========================================================')

    def read_ascii(self, usecols=None):

        try:
            go_ascii = self.file_path + self.file_name + '.dat'
#            self.sig = np.genfromtxt(go_ascii)
            self.sig = np.loadtxt(go_ascii, usecols=usecols)
#            self.sig = np.fromfile(go_ascii, dtype=np.float32, sep='  ')
#            self.sig = self.sig.reshape((self.N, self.Nch))
        except:
            self.sig = []
            self.error_msg = 'error: reading ascii file failed'
            print('========================================================')
            print(self.error_msg)
            print(self.file_path)
            print(self.file_name)
            print('========================================================')

#        print '========================================================'
#        print 'ASCII reading not implemented yet'
#        print '========================================================'
#        self.sig = []
#        self.error_msg = 'error: ASCII reading not implemented yet'

    def reformat_sig_details(self):
        """Change HAWC2 output description of the channels short descriptive
        strings, usable in plots

        obj.ch_details[channel,(0=ID; 1=units; 2=description)] : np.array
        """

        # CONFIGURATION: mappings between HAWC2 and short good output:
        change_list = []
        change_list.append( ['original','new improved'] )

#        change_list.append( ['Mx coo: hub1','blade1 root bending: flap'] )
#        change_list.append( ['My coo: hub1','blade1 root bending: edge'] )
#        change_list.append( ['Mz coo: hub1','blade1 root bending: torsion'] )
#
#        change_list.append( ['Mx coo: hub2','blade2 root bending: flap'] )
#        change_list.append( ['My coo: hub2','blade2 root bending: edge'] )
#        change_list.append( ['Mz coo: hub2','blade2 root bending: torsion'] )
#
#        change_list.append( ['Mx coo: hub3','blade3 root bending: flap'] )
#        change_list.append( ['My coo: hub3','blade3 root bending: edge'] )
#        change_list.append( ['Mz coo: hub3','blade3 root bending: torsion'] )

        change_list.append( ['Mx coo: blade1','blade1 flap'] )
        change_list.append( ['My coo: blade1','blade1 edge'] )
        change_list.append( ['Mz coo: blade1','blade1 torsion'] )

        change_list.append( ['Mx coo: blade2','blade2 flap'] )
        change_list.append( ['My coo: blade2','blade2 edge'] )
        change_list.append( ['Mz coo: blade2','blade2 torsion'] )

        change_list.append( ['Mx coo: blade3','blade3 flap'] )
        change_list.append( ['My coo: blade3','blade3 edeg'] )
        change_list.append( ['Mz coo: blade3','blade3 torsion'] )

        change_list.append( ['Mx coo: hub1','blade1 out-of-plane'] )
        change_list.append( ['My coo: hub1','blade1 in-plane'] )
        change_list.append( ['Mz coo: hub1','blade1 torsion'] )

        change_list.append( ['Mx coo: hub2','blade2 out-of-plane'] )
        change_list.append( ['My coo: hub2','blade2 in-plane'] )
        change_list.append( ['Mz coo: hub2','blade2 torsion'] )

        change_list.append( ['Mx coo: hub3','blade3 out-of-plane'] )
        change_list.append( ['My coo: hub3','blade3 in-plane'] )
        change_list.append( ['Mz coo: hub3','blade3 torsion'] )
        # this one will create a false positive for tower node nr1
        change_list.append( ['Mx coo: tower','tower top momemt FA'] )
        change_list.append( ['My coo: tower','tower top momemt SS'] )
        change_list.append( ['Mz coo: tower','yaw-moment'] )

        change_list.append( ['Mx coo: chasis','chasis momemt FA'] )
        change_list.append( ['My coo: chasis','yaw-moment chasis'] )
        change_list.append( ['Mz coo: chasis','chasis moment SS'] )

        change_list.append( ['DLL inp  2:  2','tower clearance'] )

        self.ch_details_new = np.ndarray(shape=(self.Nch,3),dtype='<U100')

        # approach: look for a specific description and change it.
        # This approach is slow, but will not fail if the channel numbers change
        # over different simulations
        for ch in range(self.Nch):
            # the change_list will always be slower, so this loop will be
            # inside the bigger loop of all channels
            self.ch_details_new[ch,:] = self.ch_details[ch,:]
            for k in range(len(change_list)):
                if change_list[k][0] == self.ch_details[ch,0]:
                    self.ch_details_new[ch,0] =  change_list[k][1]
                    # channel description should be unique, so delete current
                    # entry and stop looking in the change list
                    del change_list[k]
                    break

#        self.ch_details_new = ch_details_new

    def _unified_channel_names(self):
        """
        Make certain channels independent from their index.

        The unified channel dictionary ch_dict holds consequently named
        channels as the key, and the all information is stored in the value
        as another dictionary.

        The ch_dict key/values pairs are structured differently for different
        type of channels. Currently supported channels are:

        For forcevec, momentvec, state commands:
            node numbers start with 0 at the root
            element numbers start with 1 at the root
            key:
                coord-bodyname-pos-sensortype-component
                global-tower-node-002-forcevec-z
                local-blade1-node-005-momentvec-z
                hub1-blade1-elem-011-zrel-1.00-state pos-z
            value:
                ch_dict[tag]['coord']
                ch_dict[tag]['bodyname']
                ch_dict[tag]['pos']
                ch_dict[tag]['sensortype']
                ch_dict[tag]['component']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the DLL's this is:
            key:
                DLL-dll_name-io-io_nr
                DLL-yaw_control-outvec-3
                DLL-yaw_control-inpvec-1
            value:
                ch_dict[tag]['dll_name']
                ch_dict[tag]['io']
                ch_dict[tag]['io_nr']
                ch_dict[tag]['chi']
                ch_dict[tag]['sensortag']
                ch_dict[tag]['units']

        For the bearings this is:
            key:
                bearing-bearing_name-output_type-units
                bearing-shaft_nacelle-angle_speed-rpm
            value:
                ch_dict[tag]['bearing_name']
                ch_dict[tag]['output_type']
                ch_dict[tag]['chi']
                ch_dict[tag]['units']

        For many of the aero sensors:
            'Cl', 'Cd', 'Alfa', 'Vrel'
            key:
                sensortype-blade_nr-pos
                Cl-1-0.01
            value:
                ch_dict[tag]['sensortype']
                ch_dict[tag]['blade_nr']
                ch_dict[tag]['pos']
                ch_dict[tag]['chi']
                ch_dict[tag]['units']


        """
        # save them in a dictionary, use the new coherent naming structure
        # as the key, and as value again a dict that hols all the different
        # classifications: (chi, channel nr), (coord, coord), ...
        self.ch_dict = dict()

        # some channel ID's are unique, use them
        ch_unique = set(['Omega', 'Ae rot. torque', 'Ae rot. power',
                     'Ae rot. thrust', 'Time', 'Azi  1'])
        ch_aero = set(['Cl', 'Cd', 'Alfa', 'Vrel'])

        # scan through all channels and see which can be converted
        # to sensible unified name
        for ch in range(self.Nch):
            items = self.ch_details[ch,2].split(' ')
            # remove empty values in the list
            items = misc.remove_items(items, '')

            dll = False

            # be carefull, identify only on the starting characters, because
            # the signal tag can hold random text that in some cases might
            # trigger a false positive

            # -----------------------------------------------------------------
            # check for all the unique channel descriptions
            if self.ch_details[ch,0].strip() in ch_unique:
                tag = self.ch_details[ch,0].strip()
                channelinfo = {}
                channelinfo['units'] = self.ch_details[ch,1]
                channelinfo['sensortag'] = self.ch_details[ch,2]
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # or in the long description:
            #    0          1        2      3  4    5     6 and up
            # MomentMz Mbdy:blade nodenr:   5 coo: blade  TAG TEXT
            elif self.ch_details[ch,2].startswith('MomentM'):
                coord = items[5]
                bodyname = items[1].replace('Mbdy:', '')
                # set nodenr to sortable way, include leading zeros
                # node numbers start with 0 at the root
                nodenr = '%03i' % int(items[3])
                # skip the attached the component
                #sensortype = items[0][:-2]
                # or give the sensor type the same name as in HAWC2
                sensortype = 'momentvec'
                component = items[0][-1:len(items[0])]
                # the tag only exists if defined
                if len(items) > 6:
                    sensortag = ' '.join(items[6:])

                # and tag it
                pos = 'node-%s' % nodenr
                tagitems = (coord,bodyname,pos,sensortype,component)
                tag = '%s-%s-%s-%s-%s' % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['pos'] = pos
                channelinfo['sensortype'] = sensortype
                channelinfo['component'] = component
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch,1]

            # -----------------------------------------------------------------
            #   0    1      2        3       4  5     6     7 and up
            # Force  Fx Mbdy:blade nodenr:   2 coo: blade  TAG TEXT
            elif self.ch_details[ch,2].startswith('Force'):
                coord = items[6]
                bodyname = items[2].replace('Mbdy:', '')
                nodenr = '%03i' % int(items[4])
                # skipe the attached the component
                #sensortype = items[0]
                # or give the sensor type the same name as in HAWC2
                sensortype = 'forcevec'
                component = items[1][1]
                if len(items) > 7:
                    sensortag = ' '.join(items[7:])
                else:
                    sensortag = ''

                # and tag it
                pos = 'node-%s' % nodenr
                tagitems = (coord,bodyname,pos,sensortype,component)
                tag = '%s-%s-%s-%s-%s' % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['pos'] = pos
                channelinfo['sensortype'] = sensortype
                channelinfo['component'] = component
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch,1]

            # -----------------------------------------------------------------
            #   0    1  2      3       4      5   6         7    8
            # State pos x  Mbdy:blade E-nr:   1 Z-rel:0.00 coo: blade
            #   0           1     2    3           4    5   6         7     8
            # State_rot proj_ang tx Mbdy:mc_dummy E-nr: 1 Z-rel:0.00 coo: global
            elif self.ch_details[ch,0].startswith('State') \
                 or self.ch_details[ch,0].startswith('euler') \
                 or self.ch_details[ch,0].startswith('ax') \
                 or self.ch_details[ch,0].startswith('proj'):
                coord = items[8]
                bodyname = items[3].replace('Mbdy:', '')
                # element numbers start with 1 at the root
                elementnr = '%03i' % int(items[5])
                zrel = '%04.2f' % float(items[6].replace('Z-rel:', ''))
                # skip the attached the component
                #sensortype = ''.join(items[0:2])
                # or give the sensor type the same name as in HAWC2
                tmp = self.ch_details[ch,0].split(' ')
                sensortype = tmp[0]
                if sensortype.startswith('State'):
                    sensortype += ' ' + tmp[1]
                component = items[2]
                if len(items) > 7:
                    sensortag = ' '.join(items[7:])

                # and tag it
                pos = 'elem-%s-zrel-%s' % (elementnr, zrel)
                tagitems = (coord,bodyname,pos,sensortype,component)
                tag = '%s-%s-%s-%s-%s' % tagitems
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['bodyname'] = bodyname
                channelinfo['pos'] = pos
                channelinfo['sensortype'] = sensortype
                channelinfo['component'] = component
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch,1]

            # -----------------------------------------------------------------
            # DLL CONTROL I/O
            # there are two scenario's on how the channel description is formed
            # the channel id is always the same though
            # id for both cases:
            #          DLL out  1:  3
            #          DLL inp  2:  3
            # description case 1 ("dll type2_dll b2h2 inpvec 30" in htc output)
            #               0         1    2   3     4+
            #          yaw_control outvec  3  yaw_c input reference angle
            # description case 2 ("dll inpvec 2 1" in htc output):
            #           0  1 2     3  4  5  6+
            #          DLL : 2 inpvec :  4  mgen hss
            elif self.ch_details[ch,0].startswith('DLL'):

                # case 2: no reference to dll name
                if self.ch_details[ch,2].startswith('DLL'):
                    dll_nr = items[2]
                    io = items[3]
                    io_nr = items[5]
                    sensortag = ' '.join(items[6:])
                    # and tag it
                    tag = 'DLL-%s-%s-%s' % (dll_nr,io,io_nr)
                # case 1: type2 dll name is given
                else:
                    dll_name = items[0]
                    io = items[1]
                    io_nr = items[2]
                    sensortag = ' '.join(items[3:])
                    tag = 'DLL-%s-%s-%s' % (dll_name,io,io_nr)

                # save all info in the dict
                channelinfo = {}
                channelinfo['dll'] = dll
                channelinfo['io'] = io
                channelinfo['io_nr'] = io_nr
                channelinfo['chi'] = ch
                channelinfo['sensortag'] = sensortag
                channelinfo['units'] = self.ch_details[ch,1]

            # -----------------------------------------------------------------
            # BEARING OUTPUS
            # bea1 angle_speed       rpm      shaft_nacelle angle speed
            elif self.ch_details[ch,0].startswith('bea'):
                output_type = self.ch_details[ch,0].split(' ')[1]
                bearing_name = items[0]
                units = self.ch_details[ch,1]
                # there is no label option for the bearing output

                # and tag it
                tag = 'bearing-%s-%s-%s' % (bearing_name,output_type,units)
                # save all info in the dict
                channelinfo = {}
                channelinfo['bearing_name'] = bearing_name
                channelinfo['output_type'] = output_type
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # AERO CL, CD, CM, VREL, ALFA, LIFT, DRAG, etc
            # Cl, R=  0.5     deg      Cl of blade  1 at radius   0.49
            # Azi  1          deg      Azimuth of blade  1
            elif self.ch_details[ch,0].split(',')[0] in ch_aero:
                dscr_list = self.ch_details[ch,2].split(' ')
                dscr_list = misc.remove_items(dscr_list, '')

                sensortype = self.ch_details[ch,0].split(',')[0]
                radius = dscr_list[-1]
                # is this always valid?
                blade_nr = self.ch_details[ch,2].split('blade  ')[1][0]
                # sometimes the units for aero sensors are wrong!
                units = self.ch_details[ch,1]
                # there is no label option

                # and tag it
                tag = '%s-%s-%s' % (sensortype,blade_nr,radius)
                # save all info in the dict
                channelinfo = {}
                channelinfo['sensortype'] = sensortype
                channelinfo['radius'] = float(radius)
                channelinfo['blade_nr'] = int(blade_nr)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # TODO: wind speed
            # some spaces have been trimmed here
            # WSP gl. coo.,Vy          m/s
            # // Free wind speed Vy, gl. coo, of gl. pos   0.00,  0.00,  -2.31
            # WSP gl. coo.,Vdir_hor          deg
            # Free wind speed Vdir_hor, gl. coo, of gl. pos  0.00,  0.00, -2.31

            # -----------------------------------------------------------------
            # WATER SURFACE gl. coo, at gl. coo, x,y=   0.00,   0.00
            elif self.ch_details[ch,2].startswith('Water'):
                units = self.ch_details[ch,1]

                # but remove the comma
                x = items[-2][:-1]
                y = items[-1]

                # and tag it
                tag = 'watersurface-global-%s-%s' % (x, y)
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = 'global'
                channelinfo['pos'] = (float(x), float(y))
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # WIND SPEED
            # WSP gl. coo.,Vx
            elif self.ch_details[ch,0].startswith('WSP gl.'):
                units = self.ch_details[ch,1]
                direction = self.ch_details[ch,0].split(',')[1]
                tmp = self.ch_details[ch,2].split('pos')[1]
                x, y, z = tmp.split(',')
                x, y, z = x.strip(), y.strip(), z.strip()

                # and tag it
                tag = 'windspeed-global-%s-%s-%s-%s' % (direction, x, y, z)
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = 'global'
                channelinfo['pos'] = (x, y, z)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # WIND SPEED AT BLADE
            # 0: WSP Vx, glco, R= 61.5
            # 2: Wind speed Vx of blade  1 at radius  61.52, global coo.
            elif self.ch_details[ch,0].startswith('WSP V'):
                units = self.ch_details[ch,1].strip()
                direction = self.ch_details[ch,0].split(' ')[1].strip()
                blade_nr = self.ch_details[ch,2].split('blade')[1].strip()[:2]
                radius = self.ch_details[ch,2].split('radius')[1].split(',')[0]
                coord = self.ch_details[ch,2].split(',')[1].strip()

                radius = radius.strip()
                blade_nr = blade_nr.strip()

                # and tag it
                rpl = (direction, blade_nr, radius, coord)
                tag = 'wsp-blade-%s-%s-%s-%s' % rpl
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['direction'] = direction
                channelinfo['blade_nr'] = int(blade_nr)
                channelinfo['radius'] = float(radius)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # FLAP ANGLE
            # 2: Flap angle for blade  3 flap number  1
            elif self.ch_details[ch,0][:7] == 'setbeta':
                units = self.ch_details[ch,1].strip()
                blade_nr = self.ch_details[ch,2].split('blade')[1].strip()
                blade_nr = blade_nr.split(' ')[0].strip()
                flap_nr = self.ch_details[ch,2].split(' ')[-1].strip()

                radius = radius.strip()
                blade_nr = blade_nr.strip()

                # and tag it
                tag = 'setbeta-bladenr-%s-flapnr-%s' % (blade_nr, flap_nr)
                # save all info in the dict
                channelinfo = {}
                channelinfo['coord'] = coord
                channelinfo['flap_nr'] = int(flap_nr)
                channelinfo['blade_nr'] = int(blade_nr)
                channelinfo['units'] = units
                channelinfo['chi'] = ch

            # -----------------------------------------------------------------
            # ignore all the other cases we don't know how to deal with
            else:
                # if we get here, we don't have support yet for that sensor
                # and hence we can't save it. Continue with next channel
                continue

            # -----------------------------------------------------------------
            # ignore if we have a non unique tag
            if self.ch_dict.has_key(tag):
                msg = 'non unique tag for HAWC2 results, ignoring: %s' % tag
                logging.warn(msg)
            else:
                self.ch_dict[tag] = copy.copy(channelinfo)



    def _data_window(self, nr_rev=None, time=None):
        """
        Based on a time interval, create a proper slice object
        ======================================================

        The window will start at zero and ends with the covered time range
        of the time input.

        Paramters
        ---------

        nr_rev : int, default=None
            NOT IMPLEMENTED YET

        time : list, default=None
            time = [time start, time stop]

        Returns
        -------

        slice_

        window

        zoomtype

        time_range
            time_range = [0, time[1]]

        """

        # -------------------------------------------------
        # determine zome range if necesary
        # -------------------------------------------------
        time_range = None
        if nr_rev:
            raise NotImplementedError
            # input is a number of revolutions, get RPM and sample rate to
            # calculate the required range
            # TODO: automatich detection of RPM channel!
            time_range = nr_rev/(self.rpm_mean/60.)
            # convert to indices instead of seconds
            i_range = int(self.Freq*time_range)
            window = [0, time_range]
            # in case the first datapoint is not at 0 seconds
            i_zero = int(self.sig[0,0]*self.Freq)
            slice_ = np.r_[i_zero:i_range+i_zero]

            zoomtype = '_nrrev_' + format(nr_rev, '1.0f') + 'rev'

        elif time.any():
            time_range = time[1] - time[0]

            i_start = int(time[0]*self.Freq)
            i_end = int(time[1]*self.Freq)
            slice_ = np.r_[i_start:i_end]
            window = [time[0], time[1]]

            zoomtype = '_zoom_%1.1f-%1.1fsec' %  (time[0], time[1])

        return slice_, window, zoomtype, time_range

    def calc_stats(self, sig, i0=0, i1=-1):

        stats = {}
        # calculate the statistics values:
        stats['max'] = sig[i0:i1,:].max(axis=0)
        stats['min'] = sig[i0:i1,:].min(axis=0)
        stats['mean'] = sig[i0:i1,:].mean(axis=0)
        stats['std'] = sig[i0:i1,:].std(axis=0)
        stats['range'] = stats['max'] - stats['min']
        stats['absmax'] = np.absolute(sig[i0:i1,:]).max(axis=0)
        stats['rms'] = np.sqrt(np.mean(sig[i0:i1,:]*sig[i0:i1,:], axis=0))

        return stats

    def calc_fatigue(self, signal, no_bins=46, m=[3, 4, 6, 8, 10, 12], neq=1):
        """
        signal is 1D
        """

        try:
            sig_rf = fatigue.rainflow_astm(signal)
        except:
            return []

        if len(sig_rf) < 1 and not sig_rf:
            return []

        hist_data, x, bin_avg =  fatigue.rfc_hist(sig_rf, no_bins)

        m = np.atleast_1d(m)

        eq = []
        for i in range(len(m)):
            eq.append(np.power(np.sum(0.5 * hist_data *\
                                    np.power(bin_avg, m[i])) / neq, 1. / m[i]))
        return eq

    def blade_deflection(self):
        """
        """

        # select all the y deflection channels
        db = misc.DictDB(self.ch_dict)

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

    def save_csv(self, fname, fmt='%.18e', delimiter=','):
        """
        Save to csv and use the unified channel names as columns
        """
        map_sorting = {}
        # first, sort on channel index
        for ch_key, ch in self.ch_dict.iteritems():
            map_sorting[ch['chi']] = ch_key

        header = []
        # not all channels might be present...iterate again over map_sorting
        for chi in map_sorting:
            try:
                sensortag = self.ch_dict[map_sorting[chi]]['sensortag']
                header.append(map_sorting[chi] + ' // ' + sensortag)
            except:
                header.append(map_sorting[chi])

        # and save
        print('saving...', end='')
        np.savetxt(fname, self.sig[:,map_sorting.keys()], fmt=fmt,
                   delimiter=delimiter, header=delimiter.join(header))
        print(fname)


def ReadEigenBody(file_path, file_name, debug=False, nrmodes=1000):
    """
    Read HAWC2 body eigenalysis result file
    =======================================

    Parameters
    ----------

    file_path : str

    file_name : str



    Returns
    -------

    results : dict{body : ndarray(3,1)}
        Dictionary with body name as key and an ndarray(3,1) holding Fd, Fn
        [Hz] and the logarithmic damping decrement [%]

    results2 : dict{body : dict{Fn : [Fd, damp]}  }
        Dictionary with the body name as keys and another dictionary holding
        the eigenfrequency and damping information. The latter has the
        natural eigenfrequncy as key (hence all duplicates are ignored) with
        the damped eigenfrequency and logarithmic damping decrement as values.

    """

    #Body data for body number : 3 with the name :nacelle
    #Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #Mode nr:  1:   1.45388E-21    1.74896E-03    6.28319E+02
    FILE = open(file_path + file_name)
    lines = FILE.readlines()
    FILE.close()

    results = dict()
    results2 = dict()
    for line in lines:
        # identify for which body we will read the data
        if line.startswith('Body data for body number'):
            body = line.split(':')[2].rstrip().lstrip()
            # remove any annoying characters
            body = body.replace('\n','').replace('\r','')
        # identify mode number and read the eigenfrequencies
        elif line.startswith('Mode nr:'):
            # stop if we have found a certain amount of
            if results.has_key(body) and len(results[body]) > nrmodes:
                continue

            linelist = line.replace('\n','').replace('\r','').split(':')
            #modenr = linelist[1].rstrip().lstrip()
            eigenmodes = linelist[2].rstrip().lstrip().split('   ')
            if debug: print(eigenmodes)
            # in case we have more than 3, remove all the empty ones
            # this can happen when there are NaN values
            if not len(eigenmodes) == 3:
                eigenmodes = linelist[2].rstrip().lstrip().split(' ')
                eigmod = []
                for k in eigenmodes:
                    if len(k) > 1:
                        eigmod.append(k)
                #eigenmodes = eigmod
            else:
                eigmod = eigenmodes
            # remove any trailing spaces for each element
            for k in range(len(eigmod)):
                eigmod[k] = eigmod[k].lstrip().rstrip()
            eigmod_arr = np.array(eigmod,dtype=np.float64).reshape(3,1)
            if debug: print(eigmod_arr)
            if results.has_key(body):
                results[body] = np.append(results[body],eigmod_arr,axis=1)
            else:
                results[body] = eigmod_arr

            # or alternatively, save in a dict first so we ignore all the
            # duplicates
            #if results2.has_key(body):
                #results2[body][eigmod[1]] = [eigmod[0], eigmod[2]]
            #else:
                #results2[body] = {eigmod[1] : [eigmod[0], eigmod[2]]}

    return results, results2

def ReadEigenStructure(file_path, file_name, debug=False, max_modes=500):
    """
    Read HAWC2 structure eigenalysis result file
    ============================================

    The file looks as follows:
    #0 Version ID : HAWC2MB 11.3
    #1 ___________________________________________________________________
    #2 Structure eigenanalysis output
    #3 ___________________________________________________________________
    #4 Time : 13:46:59
    #5 Date : 28:11.2012
    #6 ___________________________________________________________________
    #7 Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #8 Mode nr:  1:   3.58673E+00    3.58688E+00    5.81231E+00
    #...
    #302  Mode nr:294:   0.00000E+00    6.72419E+09    6.28319E+02

    Parameters
    ----------

    file_path : str

    file_name : str

    debug : boolean, default=False

    max_modes : int
        Stop evaluating the result after max_modes number of modes have been
        identified

    Returns
    -------

    modes_arr : ndarray(3,n)
        An ndarray(3,n) holding Fd, Fn [Hz] and the logarithmic damping
        decrement [%] for n different structural eigenmodes

    """

    #0 Version ID : HAWC2MB 11.3
    #1 ___________________________________________________________________
    #2 Structure eigenanalysis output
    #3 ___________________________________________________________________
    #4 Time : 13:46:59
    #5 Date : 28:11.2012
    #6 ___________________________________________________________________
    #7 Results:         fd [Hz]       fn [Hz]       log.decr [%]
    #8 Mode nr:  1:   3.58673E+00    3.58688E+00    5.81231E+00
    #  Mode nr:294:   0.00000E+00    6.72419E+09    6.28319E+02

    FILE = open(file_path + file_name)
    lines = FILE.readlines()
    FILE.close()

    header_lines = 8

    # we now the number of modes by having the number of lines
    nrofmodes = len(lines) - header_lines

    modes_arr = np.ndarray((3,nrofmodes))

    for i, line in enumerate(lines):
        if i > max_modes:
            # cut off the unused rest
            modes_arr = modes_arr[:,:i]
            break

        # ignore the header
        if i < header_lines:
            continue

        # split up mode nr from the rest
        parts = line.split(':')
        #modenr = int(parts[1])
        # get fd, fn and damping, but remove all empty items on the list
        modes_arr[:,i-header_lines]=misc.remove_items(parts[2].split(' '),'')

    return modes_arr

class Veer:
    """
    """

    def __init__(self):
        pass

    def veer_ekman_mod(self, z, z_h, h_ME=1000.0, a_phi=0.5):
        """
        Modified Ekman veer profile, as defined by Mark C. Kelly in email on
        10 October 2014 15:10 (RE: veer profile)

        .. math::
            \\varphi(z) - \\varphi(z_H) \\approx a_{\\varphi}
            e^{-\sqrt{z_H/h_{ME}}}
            \\frac{z-z_H}{\sqrt{z_H*h_{ME}}}
            \\left( 1 - \\frac{z-z_H}{2 \sqrt{z_H h_{ME}}}
            - \\frac{z-z_H}{4z_H} \\right)

        where:
        :math:`h_{ME} \\equiv \\frac{\\kappa u_*}{f}`
        and :math:`f = 2 \Omega \sin \\varphi` is the coriolis parameter,
        and :math:`\\kappa = 0.41` as the von Karman constant,
        and :math:`u_\\star = \\sqrt{\\frac{\\tau_w}{\\rho}}` friction velocity.

        For on shore, :math:`h_{ME} \\approx 1000`, for off-shore,
        :math:`h_{ME} \\approx 500`

        :math:`a_{\\varphi} \\approx 0.5`

        """

        t1 = np.exp(-math.sqrt(z_h / h_ME))
        t2 = (z - z_h) / math.sqrt(z_h * h_ME)
        t3 = ( 1.0 - (z-z_h)/(2.0*math.sqrt(z_h*h_ME)) - (z-z_h)/(4.0*z_h) )

        return a_phi * t1 * t2 * t3

    def create_input_ekman_mod(self, z_h, r_blade_tip, fname, h_ME=500.0,
                               a_phi=0.5, nr_vert=20, nr_hor=3):
        """
        Create a user defined veer wind profile based on Mark's modified
        Ekman model

        Paramters
        ---------

        method : str, default=linear
            'linear' for a linear veer, 'ekman_mod' for modified ekman method
        """

        # take 15% extra space after the blade tip
        z = np.linspace(0, z_h + r_blade_tip*1.15, nr_vert)
        # along the horizontal, coordinates with 0 at the rotor center
        x = np.linspace(-r_blade_tip*1.15, r_blade_tip*1.15, nr_hor)
        # veer angles (phi-phi_z) in radians
        phi = self.veer_ekman_mod(z, z_h, h_ME=h_ME, a_phi=a_phi)
        tan_phi = np.tan(phi)

        # convert veer angles to veer components in v, u. Make sure the
        # normalized wind speed remains 1!
#        u = sympy.Symbol('u')
#        v = sympy.Symbol('v')
#        tan_phi = sympy.Symbol('tan_phi')
#        eq1 = u**2.0 + v**2.0 - 1.0
#        eq2 = (tan_phi*v/u) - 1.0
#        sol = sympy.solvers.solve([eq1, eq2], [u,v], dict=True)
        u = tan_phi*np.sqrt( 1.0/(tan_phi*tan_phi + 1.0) )
        v = np.sqrt(1.0/(tan_phi*tan_phi + 1.0))

        u_full = u[:,np.newaxis] + np.zeros((3,))[np.newaxis,:]
        v_full = v[:,np.newaxis] + np.zeros((3,))[np.newaxis,:]
        w_full = np.zeros((v.shape[0],nr_hor))

        # and create the input file
        with open(fname, 'w') as f:
            f.write('# User defined shear file\n')
            f.write('%i %i # nr_hor (v), nr_vert (w)' % (nr_hor, nr_vert))
            h1 = 'normalized with U_mean, nr_hor (v) rows, nr_vert (w) columns'
            f.write('# v component, %s\n' % h1)
            np.savetxt(f, v_full, fmt='% 08.05f', delimiter='  ')
            f.write('# u component, %s\n' % h1)
            np.savetxt(f, u_full, fmt='% 08.05f', delimiter='  ')
            f.write('# w component, %s\n' % h1)
            np.savetxt(f, w_full, fmt='% 08.05f', delimiter='  ')
            h2 = '# v coordinates (along the horizontal, nr_hor, 0 rotor center)'
            f.write('%s\n' % h2)
            np.savetxt(f, x.reshape((x.size,1)), fmt='% 8.02f')
            h3 = '# w coordinates (zero is at ground level, height, nr_hor)'
            f.write('%s\n' % h3)
            np.savetxt(f, z.reshape((z.size,1)), fmt='% 8.02f')

class Turbulence:
    """
    Untested and incomplete class!
    """

    def __init__(self):

        pass

    def read_hawc2(self, fpath, shape):
        """
        Read the HAWC2 turbulence format
        """

        fid = open(fpath, 'rb')
        tmp = np.fromfile(fid, 'float32', shape[0]*shape[1]*shape[2])
        turb = np.reshape(tmp, shape)

        return turb

    def read_bladed(self, fpath, basename):

        fid = open(fpath + basename + '.wnd', 'rb')
        R1 = struct.unpack('h', fid.read(2))[0]
        R2 = struct.unpack('h', fid.read(2))[0]
        turb = struct.unpack('i', fid.read(4))[0]
        lat = struct.unpack('f', fid.read(4))[0]
        rough = struct.unpack('f', fid.read(4))[0]
        refh = struct.unpack('f', fid.read(4))[0]
        longti = struct.unpack('f', fid.read(4))[0]
        latti = struct.unpack('f', fid.read(4))[0]
        vertti = struct.unpack('f', fid.read(4))[0]
        dv = struct.unpack('f', fid.read(4))[0]
        dw = struct.unpack('f', fid.read(4))[0]
        du = struct.unpack('f', fid.read(4))[0]
        halfalong = struct.unpack('i', fid.read(4))[0]
        mean_ws = struct.unpack('f', fid.read(4))[0]
        VertLongComp = struct.unpack('f', fid.read(4))[0]
        LatLongComp = struct.unpack('f', fid.read(4))[0]
        LongLongComp = struct.unpack('f', fid.read(4))[0]
        Int = struct.unpack('i', fid.read(4))[0]
        seed = struct.unpack('i', fid.read(4))[0]
        VertGpNum = struct.unpack('i', fid.read(4))[0]
        LatGpNum = struct.unpack('i', fid.read(4))[0]
        VertLatComp = struct.unpack('f', fid.read(4))[0]
        LatLatComp = struct.unpack('f', fid.read(4))[0]
        LongLatComp = struct.unpack('f', fid.read(4))[0]
        VertVertComp = struct.unpack('f', fid.read(4))[0]
        LatVertComp = struct.unpack('f', fid.read(4))[0]
        LongVertComp = struct.unpack('f', fid.read(4))[0]

        points = np.fromfile(fid, 'int16', 2*halfalong*VertGpNum*LatGpNum*3)
        fid.close()
        return points

    def convert2bladed(self, fpath, basename, shape=(4096,32,32)):
        """
        Convert turbulence box to BLADED format
        """

        u = self.read_hawc2(fpath + basename + 'u.bin', shape)
        v = self.read_hawc2(fpath + basename + 'v.bin', shape)
        w = self.read_hawc2(fpath + basename + 'w.bin', shape)

        # mean velocity components at the center of the box
        v1, v2 = (shape[1]/2)-1, shape[1]/2
        w1, w2 = (shape[2]/2)-1, shape[2]/2
        ucent = (u[:,v1,w1] + u[:,v1,w2] + u[:,v2,w1] + u[:,v2,w2]) / 4.0
        vcent = (v[:,v1,w1] + v[:,v1,w2] + v[:,v2,w1] + v[:,v2,w2]) / 4.0
        wcent = (w[:,v1,w1] + w[:,v1,w2] + w[:,v2,w1] + w[:,v2,w2]) / 4.0

        # FIXME: where is this range 351:7374 coming from?? The original script
        # considered a box of lenght 8192
        umean = np.mean(ucent[351:7374])
        vmean = np.mean(vcent[351:7374])
        wmean = np.mean(wcent[351:7374])

        ustd = np.std(ucent[351:7374])
        vstd = np.std(vcent[351:7374])
        wstd = np.std(wcent[351:7374])

        # gives a slight different outcome, but that is that significant?
#        umean = np.mean(u[351:7374,15:17,15:17])
#        vmean = np.mean(v[351:7374,15:17,15:17])
#        wmean = np.mean(w[351:7374,15:17,15:17])

        # this is wrong since we want the std on the center point
#        ustd = np.std(u[351:7374,15:17,15:17])
#        vstd = np.std(v[351:7374,15:17,15:17])
#        wstd = np.std(w[351:7374,15:17,15:17])

        iu = np.zeros(shape)
        iv = np.zeros(shape)
        iw = np.zeros(shape)

        iu[:,:,:] = (u - umean)/ustd*1000.0
        iv[:,:,:] = (v - vmean)/vstd*1000.0
        iw[:,:,:] = (w - wmean)/wstd*1000.0

        # because MATLAB and Octave do a round when casting from float to int,
        # and Python does a floor, we have to round first
        np.around(iu, decimals=0, out=iu)
        np.around(iv, decimals=0, out=iv)
        np.around(iw, decimals=0, out=iw)

        return iu.astype(np.int16), iv.astype(np.int16), iw.astype(np.int16)

    def write_bladed(self, fpath, basename, shape):
        """
        Write turbulence BLADED file
        """
        # TODO: get these parameters from a HAWC2 input file
        seed = 6
        mean_ws = 11.4
        turb = 3
        R1 = -99
        R2 = 4

        du = 0.974121094
        dv = 4.6875
        dw = 4.6875

        longti = 14
        latti = 9.8
        vertti = 7

        iu, iv, iw = self.convert2bladed(fpath, basename, shape=shape)

        fid = open(fpath + basename + '.wnd', 'wb')
        fid.write(struct.pack('h', R1)) # R1
        fid.write(struct.pack('h', R2)) # R2
        fid.write(struct.pack('i', turb)) # Turb
        fid.write(struct.pack('f', 999)) # Lat
        fid.write(struct.pack('f', 999)) # rough
        fid.write(struct.pack('f', 999)) # refh
        fid.write(struct.pack('f', longti)) # LongTi
        fid.write(struct.pack('f', latti)) # LatTi
        fid.write(struct.pack('f', vertti)) # VertTi
        fid.write(struct.pack('f', dv)) # VertGpSpace
        fid.write(struct.pack('f', dw)) # LatGpSpace
        fid.write(struct.pack('f', du)) # LongGpSpace
        fid.write(struct.pack('i', shape[0]/2)) # HalfAlong
        fid.write(struct.pack('f', mean_ws)) # meanWS
        fid.write(struct.pack('f', 999.)) # VertLongComp
        fid.write(struct.pack('f', 999.)) # LatLongComp
        fid.write(struct.pack('f', 999.)) # LongLongComp
        fid.write(struct.pack('i', 999)) # Int
        fid.write(struct.pack('i', seed)) # Seed
        fid.write(struct.pack('i', shape[1])) # VertGpNum
        fid.write(struct.pack('i', shape[2])) # LatGpNum
        fid.write(struct.pack('f', 999)) # VertLatComp
        fid.write(struct.pack('f', 999)) # LatLatComp
        fid.write(struct.pack('f', 999)) # LongLatComp
        fid.write(struct.pack('f', 999)) # VertVertComp
        fid.write(struct.pack('f', 999)) # LatVertComp
        fid.write(struct.pack('f', 999)) # LongVertComp
#        fid.flush()

#        bladed2 = np.ndarray((shape[0], shape[2], shape[1], 3), dtype=np.int16)
#        for i in xrange(shape[0]):
#            for k in xrange(shape[1]):
#                for j in xrange(shape[2]):
#                    fid.write(struct.pack('i', iu[i, shape[1]-j-1, k]))
#                    fid.write(struct.pack('i', iv[i, shape[1]-j-1, k]))
#                    fid.write(struct.pack('i', iw[i, shape[1]-j-1, k]))
#                    bladed2[i,k,j,0] = iu[i, shape[1]-j-1, k]
#                    bladed2[i,k,j,1] = iv[i, shape[1]-j-1, k]
#                    bladed2[i,k,j,2] = iw[i, shape[1]-j-1, k]

        # re-arrange array for bladed format
        bladed = np.ndarray((shape[0], shape[2], shape[1], 3), dtype=np.int16)
        bladed[:,:,:,0] = iu[:,::-1,:]
        bladed[:,:,:,1] = iv[:,::-1,:]
        bladed[:,:,:,2] = iw[:,::-1,:]
        bladed_swap_view = bladed.swapaxes(1,2)
        bladed_swap_view.tofile(fid, format='%int16')

        fid.flush()
        fid.close()

class Tests(unittest.TestCase):

    def setUp(self):
        pass

    def print_test_info(self):
        pass

    def test_reshaped(self):
        """
        Make sure we correctly reshape the array instead of the manual
        index reassignments
        """
        fpath = 'data/turb_s100_3.00w.bin'
        fid = open(fpath, 'rb')
        turb = np.fromfile(fid, 'float32', 32*32*8192)
        turb.shape
        fid.close()
        u = np.zeros((8192,32,32))

        for i in xrange(8192):
            for j in xrange(32):
                for k in xrange(32):
                    u[i,j,k] = turb[ i*1024 + j*32 + k]

        u2 = np.reshape(turb, (8192, 32, 32))

        self.assertTrue(np.alltrue(np.equal(u, u2)))

    def test_headers(self):

        fpath = 'data/'

        basename = 'turb_s100_3.00_refoctave_header'
        fid = open(fpath + basename + '.wnd', 'rb')
        R1 = struct.unpack("h",fid.read(2))[0]
        R2 = struct.unpack("h",fid.read(2))[0]
        turb = struct.unpack("i",fid.read(4))[0]
        lat = struct.unpack("f",fid.read(4))[0]
        # last line
        fid.seek(100)
        LongVertComp = struct.unpack("f",fid.read(4))[0]
        fid.close()

        basename = 'turb_s100_3.00_python_header'
        fid = open(fpath + basename + '.wnd', 'rb')
        R1_p = struct.unpack("h",fid.read(2))[0]
        R2_p = struct.unpack("h",fid.read(2))[0]
        turb_p = struct.unpack("i",fid.read(4))[0]
        lat_p = struct.unpack("f",fid.read(4))[0]
        # last line
        fid.seek(100)
        LongVertComp_p = struct.unpack("f",fid.read(4))[0]
        fid.close()

        self.assertEqual(R1, R1_p)
        self.assertEqual(R2, R2_p)
        self.assertEqual(turb, turb_p)
        self.assertEqual(lat, lat_p)
        self.assertEqual(LongVertComp, LongVertComp_p)

    def test_write_bladed(self):

        fpath = 'data/'
        turb = Turbulence()
        # write with Python
        basename = 'turb_s100_3.00'
        turb.write_bladed(fpath, basename, shape=(8192,32,32))
        python = turb.read_bladed(fpath, basename)

        # load octave
        basename = 'turb_s100_3.00_refoctave'
        octave = turb.read_bladed(fpath, basename)

        # float versions of octave
        basename = 'turb_s100_3.00_refoctave_float'
        fid = open(fpath + basename + '.wnd', 'rb')
        octave32 = np.fromfile(fid, 'float32', 8192*32*32*3)

        # find the differences
        nr_diff = (python-octave).__ne__(0).sum()
        print(nr_diff)
        print(nr_diff/len(python))

        self.assertTrue(np.alltrue(python == octave))



    def test_turbdata(self):

        shape = (8192,32,32)

        fpath = 'data/'
        basename = 'turb_s100_3.00_refoctave'
        fid = open(fpath + basename + '.wnd', 'rb')

        # check the last element of the header
        fid.seek(100)
        print(struct.unpack("f",fid.read(4))[0])
        # save in a list using struct
        items = (os.path.getsize(fpath + basename + '.wnd')-104)/2
        data_list = [struct.unpack("h",fid.read(2))[0] for k in xrange(items)]


        fid.seek(104)
        data_16 = np.fromfile(fid, 'int16', shape[0]*shape[1]*shape[2]*3)

        fid.seek(104)
        data_8 = np.fromfile(fid, 'int8', shape[0]*shape[1]*shape[2]*3)

        self.assertTrue(np.alltrue( data_16 == data_list ))
        self.assertFalse(np.alltrue( data_8 == data_list ))



    def test_compare_octave(self):
        """
        Compare the results from the original script run via octave
        """

        turb = Turbulence()
        iu, iv, iw = turb.convert2bladed('data/', 'turb_s100_3.00',
                                         shape=(8192,32,32))
        res = sio.loadmat('data/workspace.mat')
        # increase tolerances, values have a range up to 5000-10000
        # and these values will be written to an int16 format for BLADED!
        self.assertTrue(np.allclose(res['iu'], iu, rtol=1e-03, atol=1e-2))
        self.assertTrue(np.allclose(res['iv'], iv, rtol=1e-03, atol=1e-2))
        self.assertTrue(np.allclose(res['iw'], iw, rtol=1e-03, atol=1e-2))

    def test_allindices(self):
        """
        Verify that all indices are called
        """
        fpath = 'data/turb_s100_3.00w.bin'
        fid = open(fpath, 'rb')
        turb = np.fromfile(fid, 'float32', 32*32*8192)
        turb.shape
        fid.close()

        check = []
        for i in xrange(8192):
            for j in xrange(32):
                for k in xrange(32):
                    check.append(i*1024 + j*32 + k)

        qq = np.array(check)
        qdiff = np.diff(qq)

        self.assertTrue(np.alltrue(np.equal(qdiff, np.ones(qdiff.shape))))



if __name__ == '__main__':

    dummy = None
    #os.path.getsize(fpath + basename + '.wnd')
