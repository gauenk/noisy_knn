"""
Timer module
"""

import time
import functools
from contextlib import ContextDecorator

VERBOSE_TIMING = False

class Timer():
    """A simple timer."""
    def __init__(self,name="timer"):
        self.name = name
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def __call__(self, func):
        """Support using Timer as a decorator"""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper_timer

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.tic()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.toc()
        if VERBOSE_TIMING:
            print("[%s]: dtime = %2.2e"%(self.name,self.diff))

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
    def __str__(self):
        return "-=-=- Timer Info -=-=-\n\
        Total Time: {}\n\
        Calls: {}\n\
        Start Time: {}\n\
        Diff: {}\n\
        Average Time: {}\n".format(self.total_time,
                                   self.calls,
                                   self.start_time,
                                   self.diff,
                                   self.average_time)


import time

class ExpTimer():

    def __init__(self):
        self.name = None # support decorator use
        self.times = []
        self.names = []
        self.start_times = []

    def __str__(self):
        msg = "--- Exp Times ---"
        for k,v in self.items():
            msg += "\n%s: %2.3f\n" % (k,v)
        return msg

    def __getitem__(self,name):
        idx = self.names.index(name)
        total_time = self.times[idx]
        return total_time

    def items(self):
        names = ["timer_%s" % name for name in self.names]
        return zip(names,self.times)

    def start(self,name):
        if name in self.names:
            raise ValueError("Name [%s] already in list." % name)
        self.names.append(name)
        start_time = time.perf_counter()
        self.start_times.append(start_time)

    def stop(self,name):
        end_time = time.perf_counter() # at start
        idx = self.names.index(name)
        start_time = self.start_times[idx]
        exec_time = end_time - start_time
        self.times.append(exec_time)

    """

    Support using ExpTimer as a decorator

    """
    def __call__(self,name):
        self.name = name

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start(self.name)
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop(self.name)
        self.name = None
