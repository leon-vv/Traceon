import os
from threading import Thread

import numpy as np
import pickle

from .backend import DEBUG

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
    
    return threads

def collect_multi_threaded(f, args):
    
    if DEBUG:
        print('Running on a single thread since DEBUG=True')
        return [f(a) for a in args]
    
    results = [None]*len(args)
    
    def set_result(index):
        results[index] = f(args[index])
    
    threads = [Thread(target=set_result, args=(i,)) for i in range(len(args))]
     
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results
 
def split_collect(f, array):
    
    if DEBUG:
        print('Running on a single thread since DEBUG=True')
        return [f(array)]

    splitted = np.array_split(array, get_number_of_threads())
    return collect_multi_threaded(f, splitted)
            






