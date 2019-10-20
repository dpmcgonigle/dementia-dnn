#! /usr/bin/env python
import time
import argparse
import os, sys
import inspect
from datetime import datetime
import traceback
from functools import wraps

#   package directories and imports from within this software package
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
BASEDIR = os.path.join(SCRIPTDIR, os.pardir)
sys.path.append(BASEDIR)

############################################################################################
#   UTILITY FUNCTIONS
#       get_parser              -   Get command-line args parser pre-loaded with --debug for debug
#       debug_functions         -   Parse command-line args to get the list of functs to debug
#       debug                   -   Simple debugging print statements for specified functions
#       timeit                  -   decorator to time functions
#       parse_arg               -   if argument is comma-separated list, break it into list
############################################################################################

############################################################################################
def get_parser():
    """Return command line options as an argparse.parse_args() object for utility functions."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=str, default=None, help='Comma-sep func list (no spaces); e.g. func1,func2; "all" for all functions')
    # If you want to call this get_args() from a Jupyter Notebook, you need to uncomment -f line. Them's the rules.
    #parser.add_argument('-f', '--file', help='Path for input file.')
    return parser
#   END get_parser
############################################################################################

############################################################################################
def debug_functions(opt):
    """Parse the command-line arguments to get the list of functions to debug.

    Args:
    opt         -- argparse.parse_args() object containing opt.debug, a comma-separated list (NO SPACES) of functions.
                   Use "main" for main program body execution debugging, and "all" for all functions.

    Examples: 
        ./some_script.py <other args> --debug func1,func2,main
        ./another_script.py <other args> --debug all
    """
    #   Get command line options
    if opt.debug:
        functionList = opt.debug.split(',')
        return functionList
    else:
        return None
#   END debug_functions
############################################################################################

############################################################################################
def debug(inputStr, opt=None, functions=None):
    """Simple debugging print statements that can be turned off or on with the debug variable. 
    
    Args:
    inputStr    -- a string to print if debugging is turned on for a given function (based on DebugFunctions()).
    opt         -- argparse.parse_args() object containing opt.debug, a comma-separated list (NO SPACES) of functions.
                   Use "main" for main program body execution debugging, and "all" for all functions.
    functions   -- a list of (string) function names you want to debug
    
    Examples: 
        Debug("Some output", functions=["get_source_dir","load_image"]), 
        Debug("Some other output", opt=argparseObj)
    """
    if not opt:
        return

    #   if the --debug option isn't used and no functions are provided, don't do anything!
    if not opt.debug and not functions:
        return
    elif not functions:
        functions = debug_functions(opt)

    #callerFunction = inspect.stack()[1].function
    callerFunction = inspect.stack()[1][3]
    if callerFunction == '<module>':
        callerFunction = 'main'
    if 'all' in functions or callerFunction in functions:
        timeStamp = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
        print('%s %s(): %s' % (timeStamp, callerFunction, inputStr))
# END print_debug
############################################################################################

#   timeit
############################################################################################
def timeit(method):
    """
    Decorator function to time another function.
    From : https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    """
    @wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed
#   END timeit
###########################################################################################

#   parse_arg
###########################################################################################
def parse_arg(arg, dtype=None):
    """Return a list of items if arg is comma-separated list, or a single item."""
    if isinstance(arg,str):
        if type:
            return [dtype(x) for x in arg.split(',')]
        else:
            return list(arg.split(','))
    return [dtype(arg)] if dtype else [arg]
#   END parse_arg
###########################################################################################

#######
#######     MAIN
#######
if __name__ == '__main__':
    searchDict = {'NAME':'OKC', 'ACRONYMS':'[OKC]'}
    fname = os.path.join(BASEDIR, 'teams', 'teams.json')
    print(str(query_json_hashlist(fname, matchAll=False, **searchDict)))
