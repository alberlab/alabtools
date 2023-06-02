from .ctreplication import CtRep

def parallel_function(segmentID, cfg, temp_dir):
    # This function is executed in parallel
    return None

def reduce_function(out_names, cfg, tempdir):
    # This function is executed on the master node
    return None
