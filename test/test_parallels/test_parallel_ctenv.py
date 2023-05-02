import numpy as np
import time
import os
from alabtools.imaging import CtFile
from alabtools.imaging import CtEnvelope

print('\n')

# print working directory and file directory
print('Current working directory:\n{}\n\n'.format(os.getcwd()))
print('File directory:\n{}\n\n'.format(os.path.dirname(os.path.realpath(__file__))))

# change working directory to file directory
print('Changing working directory to file directory...\n\n')
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# print working directory and file directory
print('Current working directory:\n{}\n\n'.format(os.getcwd()))
print('File directory:\n{}\n\n'.format(os.path.dirname(os.path.realpath(__file__))))

# load the ct file
ct_name = 'takei_rep1.ct'
ct_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), ct_name)
ct = CtFile(ct_name)
print(ct.coordinates.shape)
ct.close()

# create the ctenv file
ctenv_name = 'takei_rep1.ctenv'
ctenv_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), ctenv_name)
ctenv = CtEnvelope(ctenv_name, 'w')

# define the ctenv configuration for the run
config = {'ct_name': ct_name,
          'parallel': {'controller': 'serial'},
          'fit parameters': {'alpha': 0.0005,
                             'force': False,
                             'delta_alpha': 0.0001,
                             'thresh': 1100,
                             'min_neigh': 20}}

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
