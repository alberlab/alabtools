import numpy as np
import time
import os
from alabtools import CtFile
from alabtools import CtEnvelope

# load the ct file
ct_name = 'takei_comb.ct'
ct_name = os.path.join(os.getcwd(), ct_name)
ct = CtFile(ct_name)
print(ct.coordinates.shape)
ct.close()

# create the ctenv file
ctenv_name = 'takei_comb.ctenv'
ctenv_name = os.path.join(os.getcwd(), ctenv_name)
ctenv = CtEnvelope(ctenv_name, 'w')

# define the ctenv configuration for the run
config = {'ct_name': ct_name,
          'parallel': {'controller': 'ipyparallel'},
          'fit parameters': {'alpha': 0.0005,
                             'force': False,
                             'delta_alpha': 0.0001}}

# run the ctenv calculation
t1 = time.time()
ctenv.run(config)
t2 = time.time()

# print the execution time
print('Execution time: {} s'.format(t2 - t1))

# print the results
print(ctenv.ncell)
print(ctenv.fitted)
print(ctenv.ct_fit)
print(ctenv.alpha)
print(ctenv.volume)

# save the ctenv file
ctenv.save()
