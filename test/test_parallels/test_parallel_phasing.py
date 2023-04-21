import numpy as np
import time
import os
from alabtools.imaging.ctfile import CtFile
from alabtools.imaging.phasing import WSPhaser


ct_name = 'ct_takei_comb.ct'
ct_name = os.path.join(os.getcwd(), ct_name)
ct = CtFile(ct_name)  # load the ct file
print(ct.coordinates.shape)
ct.close()

config = {'ct_name': ct_name,
          'parallel': {'controller': 'ipyparallel'},
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
