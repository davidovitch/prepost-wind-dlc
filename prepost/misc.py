# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:09:04 2012

Library for general stuff

@author: dave
"""

from __future__ import print_function
#print(*objects, sep=' ', end='\n', file=sys.stdout)
import os
import sys
import shutil

#from xlrd import open_workbook
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd

def unique(s):
    """
    SOURCE: http://code.activestate.com/recipes/52560/
    AUTHOR: Tim Peters

    Return a list of the elements in s, but without duplicates.

    For example, unique([1,2,3,1,2,3]) is some permutation of [1,2,3],
    unique("abcabc") some permutation of ["a", "b", "c"], and
    unique(([1, 2], [2, 3], [1, 2])) some permutation of
    [[2, 3], [1, 2]].

    For best speed, all sequence elements should be hashable.  Then
    unique() will usually work in linear time.

    If not possible, the sequence elements should enjoy a total
    ordering, and if list(s).sort() doesn't raise TypeError it's
    assumed that they do enjoy a total ordering.  Then unique() will
    usually work in O(N*log2(N)) time.

    If that's not possible either, the sequence elements must support
    equality-testing.  Then unique() will usually work in quadratic
    time.
    """

    n = len(s)
    if n == 0:
        return []

    # Try using a dict first, as that's the fastest and will usually
    # work.  If it doesn't work, it will usually fail quickly, so it
    # usually doesn't cost much to *try* it.  It requires that all the
    # sequence elements be hashable, and support equality comparison.
    u = {}
    try:
        for x in s:
            u[x] = 1
    except TypeError:
        del u  # move on to the next method
    else:
        return u.keys()

    # We can't hash all the elements.  Second fastest is to sort,
    # which brings the equal elements together; then duplicates are
    # easy to weed out in a single pass.
    # NOTE:  Python's list.sort() was designed to be efficient in the
    # presence of many duplicate elements.  This isn't true of all
    # sort functions in all languages or libraries, so this approach
    # is more effective in Python than it may be elsewhere.
    try:
        t = list(s)
        t.sort()
    except TypeError:
        del t  # move on to the next method
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti += 1
            i += 1
        return t[:lasti]

    # Brute force is all that's left.
    u = []
    for x in s:
        if x not in u:
            u.append(x)
    return u

def CoeffDeter(obs, model):
    """
    Coefficient of determination
    ============================

    https://en.wikipedia.org/wiki/Coefficient_of_determination

    Parameters
    ----------

    obs : ndarray(n) or list
        The observed dataset

    model : ndarray(n), list or scalar
        The fitted dataset

    Returns
    -------

    R2 : float
        The coefficient of determination, varies between 1 for a perfect fit,
        and 0 for the worst possible fit ever

    """

    if type(obs).__name__ == 'list':
        obs = np.array(obs)

    SS_tot = np.sum(np.power( (obs - obs.mean()), 2 ))
    SS_err = np.sum(np.power( (obs - model), 2 ))
    R2 = 1 - (SS_err/SS_tot)

    return R2


def calc_sample_rate(time, rel_error=1e-4):
    """
    the sample rate should be constant throughout the measurement serie
    define the maximum allowable relative error on the local sample rate

    rel_error = 1e-4 # 0.0001 = 0.01%
    """
    deltas = np.diff(time)
    # the sample rate should be constant throughout the measurement serie
    # define the maximum allowable relative error on the local sample rate
    if not (deltas.max() - deltas.min())/deltas.max() <  rel_error:
        print('Sample rate not constant, max, min values:', end='')
        print('%1.6f, %1.6f' % (1/deltas.max(), 1/deltas.min()))
#        raise AssertionError
    return 1/deltas.mean()

def findIntersection(fun1, fun2, x0):
    """
    Find Intersection points of two functions
    =========================================

    Find the intersection between two random callable functions.
    The other alternative is that they are not callable, but are just numpy
    arrays describing the functions.

    Parameters
    ----------

    fun1 : calable
        Function 1, should return a scalar and have one argument

    fun2 : calable
        Function 2, should return a scalar and have one argument

    x0 : float
        Initial guess for sp.optimize.fsolve

    Returns
    -------



    """
    return sp.optimize.fsolve(lambda x : fun1(x) - fun2(x), x0)

# TODO: replace this with some of the pyrain functions
def find0(array, xi=0, yi=1, verbose=False, zerovalue=0.0):
    """
    Find single zero crossing
    =========================

    Find the point where a x-y dataset crosses zero. This method can only
    handle one zero crossing point.

    Parameters
    ----------
    array : ndarray
        should be 2D, with a least 2 columns and 2 rows

    xi : int, default=0
        index of the x values on array[:,xi]

    yi : int, default=1
        index of the y values on array[:,yi]

    zerovalue : float, default=0
        Set tot non zero to find the corresponding crossing.

    verbose : boolean, default=False
        if True intermediate results are printed. Usefull for debugging

    Returns
    -------
    y0 : float
        if no x0=0 exists, the result will be an interpolation between
        the two points around 0.

    y0i : int
        index leading to y0 in the input array. In case y0 was the
        result of an interpolation, the result is the one closest to x0=0

    """

    # Determine the two points where aoa=0 lies in between
    # take all the negative values, the maximum is the one closest to 0
    try:
        neg0i = np.abs(array[array[:,xi].__le__(zerovalue),xi]).argmax()
    # This method will fail if there is no zero crossing (not enough data)
    # in other words: does the given data range span from negative, to zero to
    # positive?
    except ValueError:
        print('Given data range does not include zero crossing.')
        return 0,0

    # find the points closest to zero, sort on absolute values
    isort = np.argsort(np.abs(array[:,xi]-zerovalue))
    if verbose:
        print(array[isort,:])
    # find the points closest to zero on both ends of the axis
    neg0i = isort[0]
    sign = int(np.sign(array[neg0i,xi]))
    # only search for ten points
    for i in xrange(1,20):
        # first time we switch sign, we have it
        if int(np.sign(array[isort[i],xi])) is not sign:
            pos0i = isort[i]
            break

    try:
        pos0i
    except NameError:
        print('Given data range does not include zero crossing.')
        return 0,0

    # find the value closest to zero on the positive side
#    pos0i = neg0i +1

    if verbose:
        print('0_negi, 0_posi', neg0i, pos0i)
        print('x[neg0i], x[pos0i]', array[neg0i,xi], array[pos0i,xi])

    # check if x=0 is an actual point of the series
    if np.allclose(array[neg0i,xi], 0):
        y0 = array[neg0i,yi]
        if verbose:
            prec = ' 01.08f'
            print('y0:', format(y0, prec))
            print('x0:', format(array[neg0i,xi], prec))
    # check if x=0 is an actual point of the series
    elif np.allclose(array[pos0i,xi], 0):
        y0 = array[pos0i,yi]
        if verbose:
            prec = ' 01.08f'
            print('y0:', format(y0, prec))
            print('x0:', format(array[pos0i,xi], prec))
    # if not very close to zero, interpollate to find the zero point
    else:
        y1 = array[neg0i,yi]
        y2 = array[pos0i,yi]
        x1 = array[neg0i,xi]
        x2 = array[pos0i,xi]
        y0 = (-x1*(y2-y1)/(x2-x1)) + y1

        if verbose:
            prec = ' 01.08f'
            print('y0:', format(y0, prec))
            print('y1, y2', format(y1, prec), format(y2, prec))
            print('x1, x2', format(x1, prec), format(x2, prec))

    # return the index closest to the value of AoA zero
    if abs(array[neg0i,0]) > abs(array[pos0i,0]):
        y0i = pos0i
    else:
        y0i = neg0i

    return y0, y0i

def remove_items(list, value):
    """Remove items from list
    The given list wil be returned withouth the items equal to value.
    Empty ('') is allowed. So this is een extension on list.remove()
    """
    # remove list entries who are equal to value
    ind_del = []
    for i in xrange(len(list)):
        if list[i] == value:
            # add item at the beginning of the list
            ind_del.insert(0, i)

    # remove only when there is something to remove
    if len(ind_del) > 0:
        for k in ind_del:
            del list[k]

    return list

class DictDB(object):
    """
    A dictionary based database class
    =================================

    Each tag corresponds to a row and each value holds another tag holding
    the tables values, or for the current row the column values.

    Each tag should hold a dictionary for which the subtags are the same for
    each row entry. Otherwise you have columns appearing and dissapearing.
    That is not how a database is expected to behave.
    """

    def __init__(self, dict_db):
        """
        """
        # TODO: data checks to see if the dict can qualify as a database
        # in this context

        self.dict_db = dict_db

    def search(self, dict_search):
        """
        Search a dictionary based database
        ==================================

        Searching on based keys having a certain value.

        Parameters
        ----------

        search_dict : dictionary
            Keys are the column names. If the values match the ones in the
            database, the respective row gets selected. Each tag is hence
            a unique row identifier. In case the value is a list (or it will
            be faster if it is a set), all the list entries are considered as
            a go.
        """
        self.dict_sel = dict()

        # browse through all the rows
        for row in self.dict_db:
            # and for each search value, check if the row holds the requested
            # column value
            init = True
            alltrue = True
            for col_search, val_search in dict_search.items():
                # for backwards compatibility, convert val_search to list
                if not type(val_search).__name__ in ['set', 'list']:
                    # conversion to set is more costly than what you gain
                    # by target in set([]) compared to target in []
                    # conclusion: keep it as a list
                    val_search = [val_search]

                # all items should be true
                # if the key doesn't exists, it is not to be considered
                try:
                    if self.dict_db[row][col_search] in val_search:
                        if init or alltrue:
                            alltrue = True
                    else:
                        alltrue = False
                except KeyError:
                    alltrue = False
                init = False
            # all search criteria match, save the row
            if alltrue:
                self.dict_sel[row] = self.dict_db[row]

    # TODO: merge with search into a more general search/select method?
    # shouldn't I be moving to a proper database with queries?
    def search_key(self, dict_search):
        """
        Search for a string in dictionary keys
        ======================================

        Searching based on the key of the dictionaries, not the values

        Parameters
        ----------

        searchdict : dict
            As key the search string, as value the operator: True for inclusive
            and False for exclusive. Operator is AND.

        """

        self.dict_sel = dict()

        # browse through all the rows
        for row in self.dict_db:
            # and see for each row if its name contains the search strings
            init = True
            alltrue = True
            for col_search, inc_exc in dict_search.iteritems():
                # is it inclusive the search string or exclusive?
                if (row.find(col_search) > -1) == inc_exc:
                    if init:
                        alltrue = True
                else:
                    alltrue = False
                    break
                init = False
            # all search criteria matched, save the row
            if alltrue:
                self.dict_sel[row] = self.dict_db[row]

class DictDiff(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values

    Source
    ------

    Basic idea of the magic is based on following stackoverflow question
    http://stackoverflow.com/questions/1165352/
    fast-comparison-between-two-python-dictionary
    """
    def __init__(self, current_dict, past_dict):
        self.current_d = current_dict
        self.past_d    = past_dict
        self.set_current  = set(current_dict.keys())
        self.set_past     = set(past_dict.keys())
        self.intersect    = self.set_current.intersection(self.set_past)
    def added(self):
        return self.set_current - self.intersect
    def removed(self):
        return self.set_past - self.intersect
    def changed(self):
        #set(o for o in self.intersect if self.past_d[o] != self.current_d[o])
        # which is the  similar (exept for the extension) as below
        olist = []
        for o in self.intersect:
            # if we have a numpy array
            if type(self.past_d[o]).__name__ == 'ndarray':
                if not np.allclose(self.past_d[o], self.current_d[o]):
                    olist.append(o)
            elif self.past_d[o] != self.current_d[o]:
                olist.append(o)
        return set(olist)

    def unchanged(self):
        t=set(o for o in self.intersect if self.past_d[o] == self.current_d[o])
        return t

def fit_exp(time, data, checkplot=True, method='linear', func=None, C0=0.0):
    """
    Note that all values in data have to be possitive for this method to work!
    """

    def fit_exp_linear(t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    def fit_exp_nonlinear(t, y):
        # The model function, f(x, ...). It must take the independent variable
        # as the first argument and the parameters to fit as separate remaining
        # arguments.
        opt_parms, parm_cov = sp.optimize.curve_fit(model_func,t,y)
        A, K, C = opt_parms
        return A, K, C

    def model_func(t, A, K, C):
        return A * np.exp(K * t) + C

    # Linear fit
    if method == 'linear':
#        if data.min() < 0.0:
#            msg = 'Linear exponential fitting only works for positive values'
#            raise ValueError, msg
        A, K = fit_exp_linear(time, data, C=C0)
        fit = model_func(time, A, K, C0)
        C = C0

    # Non-linear Fit
    elif method == 'nonlinear':
        A, K, C = fit_exp_nonlinear(time, data)
        fit = model_func(time, A, K, C)

    if checkplot:
        plt.figure()
        plt.plot(time, data, 'ro', label='data')
        plt.plot(time, fit, 'b', label=method)
        plt.legend(bbox_to_anchor=(0.9, 1.1), ncol=2)
        plt.grid()

    return fit, A, K, C

def curve_fit_exp(time, data, checkplot=True, weights=None):
    """
    This code is based on a StackOverflow question/answer:
    http://stackoverflow.com/questions/3938042/
    fitting-exponential-decay-with-no-initial-guessing

    A*e**(K*t) + C
    """

    def fit_exp_linear(t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    def fit_exp_nonlinear(t, y):
        # The model function, f(x, ...). It must take the independent variable
        # as the first argument and the parameters to fit as separate remaining
        # arguments.
        opt_parms, parm_cov = sp.optimize.curve_fit(model_func,t,y)
        A, K, C = opt_parms
        return A, K, C

    def model_func(t, A, K, C):
        return A * np.exp(K * t) + C

    C0 = 0

    ## Actual parameters
    #A0, K0, C0 = 2.5, -4.0, 0.0
    ## Generate some data based on these
    #tmin, tmax = 0, 0.5
    #num = 20
    #t = np.linspace(tmin, tmax, num)
    #y = model_func(t, A0, K0, C0)
    ## Add noise
    #noisy_y = y + 0.5 * (np.random.random(num) - 0.5)

    # Linear fit
    A_lin, K_lin = fit_exp_linear(time, data, C=C0)
    fit_lin = model_func(time, A_lin, K_lin, C0)

    # Non-linear Fit
    A_nonlin, K_nonlin, C = fit_exp_nonlinear(time, data)
    fit_nonlin = model_func(time, A_nonlin, K_nonlin, C)

    # and plot
    if checkplot:
        plt.figure()
        plt.plot(time, data, 'ro', label='data')
        plt.plot(time, fit_lin, 'b', label='linear')
        plt.plot(time[::-1], fit_nonlin, 'g', label='nonlinear')
        plt.legend(bbox_to_anchor=(0.9, 1.0), ncol=3)
        plt.grid()

    return

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

def to_lower_case(proot):
    """
    Rename all the files in the subfolders of proot to lower case, and
    also the subfolder name when it the folder name starts with DLC
    """
    # find all dlc defintions in the subfolders
    for root, dirs, files in os.walk(proot):
        for fname in files:
            orig = os.path.join(root, fname)
            rename = os.path.join(root, fname.lower())
            os.rename(orig, rename)
        base = root.split(os.path.sep)[-1]
        if base[:3] == 'DLC':
            new = root.replace(base, base.lower())
            os.rename(root, new)

def read_excel(ftarget, sheetname, row_sel=[], col_sel=[], data_fmt='list'):
    """
    Read a MS Excel spreadsheet (older implementation?)

    Parameters
    ----------

    ftarget

    sheentame

    row_sel : list, default=[]

    col_sel : list, default=[]

    data_fmt : 'list', 'ndarray'

    Source based on:
    http://stackoverflow.com/questions/3239207/
    how-can-i-open-an-excel-file-in-python
    http://stackoverflow.com/questions/3241039/
    how-do-i-extract-specific-lines-of-data-from-a-huge-excel-sheet-using-python
    """

    book = open_workbook(ftarget, on_demand=True)
    sheet = book.sheet_by_name(sheetname)

    # load the whole worksheet
    if len(row_sel) < 1:
        # load each row as a list of the columns
        if data_fmt == 'list':
            for i in xrange(sheet.nrows):
                rows = [cell.value for cell in sheet.row(i)]
        # load as a numpy array, what if there are text values?
        elif data_fmt == 'ndarray':
            msg = 'Loading complete worksheet only works as a list'
            raise UserWarning, msg

    # load selection of the worksheet
    else:
        # load each row as a list of the columns
        if data_fmt == 'list':
            rows = []
            for rowi in row_sel:
                rows.append([sheet.row(rowi)[coli].value for coli in col_sel])
        # load as a numpy array, what if there are text values?
        elif data_fmt == 'ndarray':
            # initialize the array
            rows = np.ndarray( (len(row_sel),len(col_sel)), order='F')
            ii,jj = 0,0
            # IndeError is thrown if we reach the end
            # note that if we have an index error because of wrong selection
            # you are on your own in finding out what goes wrong
            try:
                for rowi in row_sel:
                    for coli in col_sel:
                        rows[jj,ii] = sheet.row(rowi)[coli].value
                        ii += 1
                    jj += 1
                    ii = 0
            except IndexError:
                # crop array correspondingly
                rows = rows[:jj,:]

    book.unload_sheet(sheetname)
    return rows

def read_excel_files(proot, fext='.xlsx', pignore=None, sheet='Sheet1'):
    """
    Read recursively all MS Excel files with extension "fext". Only the
    default name for the first sheet (Sheet1) of the Excel file is considered.

    Parameters
    ----------

    proot : string
        Path that will be recursively explored for the presence of files
        that have file extension "fext"

    fext : string, default='.xlsx'
        File extension of the Excel files that should be loaded

    pignore : string, default=None
        Specify which folders should not be considered

    sheet : string, default='Sheet1'
        Name of the Excel sheet to be considered. Sheet1 is the default sheet
        name given by Excel.

    Returns
    -------

    df_list : list
        A list of pandas DataFrames. Each DataFrame corresponds to the
        contents of a single Excel file that was found in proot or one of
        its sub-directories

    """

    df_list = {}
    # find all dlc defintions in the subfolders
    for root, dirs, files in os.walk(proot):
        current_dir = root.split(os.path.sep)[-1]
        if pignore is not None and current_dir.find(pignore) > -1:
            continue
        for file_name in files:
            if not file_name[-5:] == fext:
                continue
            f_target = os.path.join(root, file_name)
            print(f_target, end='')
            try:
                xl = pd.ExcelFile(f_target)
                df = xl.parse(sheet)
                df_list[f_target.replace('.xlsx', '')] = df
                print('')
            except:
                print('     XXXXX ERROR COULD NOT READ')

    return df_list

def convert_xlsx2csv(fpath, sheet='Sheet1', fext='.xlsx'):
    """
    Convert xlsx load case definitions to csv so we can track them with git
    """

    for root, dirs, files in os.walk(fpath):
        for file_name in files:
            if not file_name[-5:] == fext:
                continue
            fxlsx = os.path.join(root, file_name)
            print(fxlsx)
            xl = pd.ExcelFile(fxlsx)
            df = xl.parse(sheet)
            fcsv = fxlsx.replace(fext, '.csv')
            df.to_csv(fcsv, sep=';')

if __name__ == '__main__':

    pass
