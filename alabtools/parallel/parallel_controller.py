'''
The reduce function should always accept an iterable
The map function should always return an iterable
'''

# Import the tqdm library to show a progress bar during serial processing
from tqdm import tqdm

class ParallelController(object):
    """Base class for parallel controllers.
    
    Maps parallel jobs into workers (i.e. processes or threads),
    and then reduces the results.
    """

    def __init__(self):
        """
        A parallel controller that map parallel jobs into workers
        """
        # empty constructor, just to make it explicit

    def setup(self):
        """Defines a setup function that is called before the map_reduce function.
        """
        pass

    def map(self, parallel_task, args):
        """Defines a map function for mapping parallel jobs to workers
        (i.e. processes or threads).
        
        @param parallel_task: a function to be executed in parallel
        @param args: a list of arguments to be passed to the parallel task
        
        @return: a list of results from the parallel tasks
        """
        raise NotImplementedError()

    def reduce(self, reduce_task, outs):
        """Defines a reduce function for reducing the results of the parallel tasks
           (i.e. combining the results of the parallel tasks).
           
           @param reduce_task: a function to be executed to reduce the results.
           @param outs: a list of results from the parallel tasks."""
        return reduce_task(outs)

    def map_reduce(self, parallel_task, reduce_task, args):
        """Combines the map and reduce functions into a single function.
        
           @param parallel_task: a function to be executed in parallel.
           @param reduce_task: a function to be executed to reduce the results.
           @param args: a list of arguments to be passed to the parallel task.
        """
        return self.reduce(reduce_task, self.map(parallel_task, args))

    def teardown(self):
        """Defines a teardown function that is called after the map_reduce function.
           It is used to clean up the parallel controller (e.g. close the pool of workers).
        """
        pass

class SerialController(ParallelController):
    """A serial controller that executes the parallel tasks in serial.
    """
    
    def map(self, parallel_task, args):
        """Executes the parallel tasks in serial.
        
           @param parallel_task: a function to be executed.
           @param args: a list of arguments to be passed to the parallel task."""
        return [parallel_task(a) for a in tqdm(args, desc="(SERIAL)")]


# There is already a map_reduce function in the ParallelController class,
# so this function is probably not needed anymore
def map_reduce(parallel_task, reduce_function, args, controller):
    controller.setup()
    result = controller.map_reduce(parallel_task, reduce_function, args)
    controller.teardown()
    return result
