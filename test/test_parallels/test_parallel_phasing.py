import numpy as np
import time
import os
from alabtools.imaging.ctfile import CtFile
from alabtools.imaging.phasing import WSPhaser

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

ct_name = 'takei_rep1.ct'
ct_name = os.path.join(os.getcwd(), ct_name)
ct = CtFile(ct_name)  # load the ct file
print(ct.coordinates.shape)
ct.close()

config = {'ct_name': ct_name,
          'parallel': {'controller': 'serial'},
          'ncluster': {'#': 2, 'chrX': 1},
          'additional_parameters': {'st': 1.2, 'ot': 2.5}}
phaser = WSPhaser(config)

t1 = time.time()
ct_phased = phaser.run()
t2 = time.time()

print('Execution time: {} s'.format(t2 - t1))

print(ct_phased.coordinates.shape)
print(np.max(ct_phased.nspot), ct_phased.nspot_max)
print(np.max(ct_phased.ncopy), ct_phased.ncopy_max)

ct_phased.close()
