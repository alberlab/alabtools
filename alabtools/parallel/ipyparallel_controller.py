from __future__ import print_function, division

import threading
import multiprocessing
import traceback
import os
import sys
import zmq
import sqlite3
import time
import six
from .parallel_controller import ParallelController
import logging  # check dependencies!
from tqdm import tqdm


logger = logging.getLogger(__name__)  # check dependencies!

class IppFunctionWrapper(object):
    """This class wraps a function to be executed in a child process.
    
    A process is given as an argument to the constructor (inner), and
    a timeout (in seconds) as an optional argument.
    """
    
    def __init__(self, inner, timeout=None):
        self.inner = inner  # the function to be executed
        self.timeout = timeout  # timeout in seconds

    def run(self, *args, **kwargs):
        """This method runs on a child process and executes the function with
        the given arguments.
        Results are put in a queue (self._q) to be retrieved by the parent.
        If the process ends successfully, a tuple (0, result, execution_time)
        is put in the queue.
        Otherwise, a tuple (-1, error_message, None) is put in the queue.
        """
        try:
            from time import time
            tstart = time()
            res = self.inner(*args, **kwargs)
            self._q.put( (0, res, time()-tstart) )  # put result in queue (0 means success)
        except:  # if an error occurs, put the error message in the queue
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            self._q.put( (-1, tb_str, None) )

    def __call__(self, *args, **kwargs):
        """Creates the Queue attribute and starts the child process using the run method.
        """
        try:
            # probably not needed to import multiprocessing again
            import multiprocessing
            try:  # queue is imported differently in python 2 and 3
                from Queue import Empty  # python 2
            except:
                from queue import Empty  # python 3

            # Could we initialize _q in __init__?
            # create a queue to communicate with the child process
            self._q = multiprocessing.Queue()
            # create the child process
            p = multiprocessing.Process(target=self.run, args=args, kwargs=kwargs)
            p.start()  # start the child process
            # Get the result from the Queue, waiting for the specified timeout
            rval = self._q.get(block=True, timeout=self.timeout)
            p.join()  # Wait for the child process to finish
        # If the timeout is exceeded, terminate the child process and return an error message
        except Empty:
            rval = (-1, 'Processing time exceeded (%f)' % self.timeout, None)
            p.terminate()
        # If any other error occurs, terminate the child process and return the error message
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            rval = (-1, tb_str, None)
        return rval



class BasicIppController(ParallelController):
    """A basic controller with ipyparallel.
    
    Uses ipyparallel to create a client that connects to the ipcluster.
    Then it creates a load-balanced view to distribute the tasks to the engines.
    Tasks are executed using the IppFunctionWrapper class.
    Results are returned as a list.
    
    Inherits from ParallelController.
    """
    
    def __init__(self, timeout=None, max_tasks=-1):
        # a taks is a unit of work that can be executed in parallel
        self.max_tasks = max_tasks  # maximum number of tasks to be executed in parallel
        self.timeout=timeout  # timeout in seconds

    def map(self, parallel_task, args):
        """Overrides the map method of the ParallelController class.
        Distributes the tasks to the engines using the load-balanced view.
        
        @param parallel_task: the function to be executed in parallel
        @param args: a list of arguments to be passed to the function
        @return: a list of results
        """
        
        # Should be imported at the top of the file
        from ipyparallel import Client, TimeoutError

        # Sets the chunksize, i.e. the number of tasks to be executed in parallel
        # If the number of tasks (len(args)) is less than the maximum number of tasks,
        # the chunksize is set to 1.
        chunksize = 1
        # If max_tasks is given (i.e. not -1), and less than the number of tasks,
        # the chunksize is set to the number of tasks divided by the maximum number of tasks.
        if self.max_tasks > 0 and len(args) > self.max_tasks:
            chunksize = len(args) // self.max_tasks  # divided and rounded down to int
            if chunksize*self.max_tasks < len(args):
                chunksize += 1  # add 1 if there is still space for more tasks
        
        # Creates the ipyparallel client
        client = None
        try:
            client = Client()  # connect to the ipcluster
        except TimeoutError:
            raise RuntimeError('Cannot connect to the ipyparallel client. Is it running?')
        
        # asynchronous means that the results are returned as soon as they are available,
        # while synchronous means that the results are returned only when all the tasks are finished.
        ar = None  # async result, AsyncResult object
        try:
            # use cloudpickle to serialize objects
            # in this context, serialization means converting an object (i.e. a combination of
            # code and data) into a stream of bytes for transmission or storage.
            client[:].use_cloudpickle()
            # create a load-balanced view, i.e. a view that distributes tasks to the engines
            lbv = client.load_balanced_view()
            # execute the tasks in parallel, using the IppFunctionWrapper class
            ar = lbv.map_async(
                IppFunctionWrapper(parallel_task, self.timeout),
                args,
                chunksize=chunksize
            )
            
            # wait for the tasks to finish and get the results
            try:
                r = []  # list of results
                # iterate over the async results
                for k, z in enumerate(tqdm(ar, desc="(IPYPARALLEL)", total=len(args))):
                    if z[0] == -1:  # if an error occurs
                        logger.error(z[1])
                        engine = ar.engine_id[k]
                        client.abort(ar)
                        client.close()
                        raise RuntimeError('remote failure (task %d of %d on engine %d)' % (k+1, len(ar), engine))
                    elif z[0] == 0:  # if the task is successful
                        r.append(z[1])
            except KeyboardInterrupt:  # if the user interrupts the execution
                client.abort(ar)
                raise
        
        # Close client and resources
        finally:
            # always close the client to release resources
            if ar:
                client.abort(ar)
            if client:
                client.close()
        return r
