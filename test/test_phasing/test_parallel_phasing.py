import numpy as np
import time
from alabtools.imaging.ctfile import CtFile
from alabtools.imaging.phasing import WSPhaser


ct_name = 'ct_test.ct'
ct = CtFile(ct_name)  # load the ct file
print(ct.coordinates.shape)
print(type(ct.cell_labels))
ct.close()

config = {'ct_name': ct_name,
          'parallel': {'controller': 'serial'},
          'additional_parameters': {'st': 1.2, 'ot': 2.5}}
phaser = WSPhaser(config)

t1 = time.time()
phaser.phasing()
t2 = time.time()

print(phaser.phase.shape)
print('Execution time: {} s'.format(t2 - t1))
