import numpy as np
from alabtools.imaging import CtFile
from alabtools.phasing import TakeiPhaser
import time


ct_name = 'ct_takei_comb.ct'
ct = CtFile(ct_name)  # load the ct file
print(ct.coordinates.shape)
print(type(ct.cell_labels))
ct.close()

config = {'ct_name': ct_name, 'parallel': {'controller': 'ipyparallel'}}
phaser = TakeiPhaser(config)

t1 = time.time()
phaser.phasing()
t2 = time.time()

print(phaser.phase.shape)
print('Execution time: {} s'.format(t2 - t1))
