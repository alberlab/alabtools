import os
from alabtools import CtEnvelope

# set working directory to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ctenv = CtEnvelope('test.ctenv', 'w')
print(ctenv.pio)
ctenv.save()

ctenv1 = CtEnvelope('test.ctenv', 'r')
print(ctenv1.pio)

