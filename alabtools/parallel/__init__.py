from __future__ import division, print_function

from .parallel_controller import SerialController
    
from .ipyparallel_controller import BasicIppController


# map controller type to controller class
controller_class = {
    "serial" : SerialController,
    "ipyparallel" : BasicIppController,
    "ipyparallel_basic" : BasicIppController, 
}

def Controller(cfg):
    """Return a parallel controller object based on the configuration file.
       
       cfg must have a get method. The second argument of get is the default
       (i.e. if the key is not found).
    
       @param cfg: configuration object with get method
       @return: parallel controller object
    """
    # get the parallel configuration (default is empty dict)
    parallel_cfg = cfg.get("parallel", dict())
    
    # get the controller type (default is ipyparallel)
    pctype = parallel_cfg.get("controller", "ipyparallel")
    
    # get the controller options (default is empty dict)
    pcopts = parallel_cfg.get("controller_options", dict()).get(pctype, dict())
    
    return controller_class[pctype](**pcopts)
