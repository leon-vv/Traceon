import os
from threading import Thread

import numpy as np
import pickle

from .backend import DEBUG
from . import logging

class Saveable:
    def write(self, filename):
        """Write a mesh to a file. The pickle module will be used
        to save the Geometry object.

        Args:
            filename: name of the file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def read(filename):
        """Read a geometry from disk (previously saved with the write method)
        
        Args:
            filename: the name of the file.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)


def get_number_of_threads():
    
    threads = os.environ.get('TRACEON_THREADS')

    if threads is None:
        # Use all available physical CPU's
        # To really count the number of physical cores, we would need
        # a module like psutil. But I don't want to pull in an external
        # dependency for such a triviality. Here we simply assume that
        # we have two threads per logical core, which I think is correct
        # for at least modern Intel and AMD CPU's.
        cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        threads = cpu_count // 2
    
    # os.cpu_count() might be 1 on old CPU's and in virtual machines.
    # Of course at least one thread is needed to run the computations.
    if threads == 0:
        return 1
     
    return threads

def split_collect(f, array):
    
    if DEBUG:
        logging.log_debug(f'Running function \'{f.__name__}\' on a single thread since DEBUG=True')
        return [f(array)]
    
    args = np.array_split(array, get_number_of_threads())
     
    results = [None]*len(args)
    
    def set_result(index):
        results[index] = f(args[index])
    
    threads = [Thread(target=set_result, args=(i,)) for i in range(len(args))]
     
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results
            






